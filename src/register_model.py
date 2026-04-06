"""
Register a locally-trained model into the Snowflake Model Registry.

Can be run as a script:
    python -m src.register_model --model-path artifacts/readmission_model.joblib --version V2

Or imported:
    from src.register_model import register_model
"""

import json
import os

import joblib
import pandas as pd
from snowflake.ml.registry import Registry
from snowflake.ml.model import task as ml_task

from src.config import (
    CONFIG, FEATURE_COLUMNS, MODEL_NAME, MODEL_VERSION, get_session,
)


def register_model(
    model_path: str = "artifacts/readmission_model.joblib",
    metadata_path: str = "artifacts/model_metadata.json",
    test_data_path: str = "artifacts/test_data.csv",
    version: str | None = None,
) -> None:
    """
    Load a local .joblib model and register it in the Snowflake Model Registry.

    Args:
        model_path:     Path to the serialized sklearn model.
        metadata_path:  Path to model_metadata.json.
        test_data_path: Path to test CSV (used for sample_input_data).
        version:        Version name override (defaults to config MODEL_VERSION).
    """
    version = version or MODEL_VERSION
    session = get_session()
    session.sql(f"USE SCHEMA {CONFIG['schemas']['model_registry']}").collect()

    model = joblib.load(model_path)

    with open(metadata_path) as f:
        metadata = json.load(f)

    test_df = pd.read_csv(test_data_path)
    sample_input = test_df[FEATURE_COLUMNS].head(100)

    registry = Registry(
        session=session,
        database_name=CONFIG["database"],
        schema_name=CONFIG["schemas"]["model_registry"],
    )

    mv = registry.log_model(
        model=model,
        model_name=MODEL_NAME,
        version_name=version,
        sample_input_data=sample_input,
        conda_dependencies=["scikit-learn"],
        comment=(
            f"30-day hospital readmission predictor. "
            f"GradientBoostingClassifier, {len(FEATURE_COLUMNS)} features. "
            f"ROC AUC={metadata['model_metrics']['roc_auc']}"
        ),
        metrics={
            "roc_auc": metadata["model_metrics"]["roc_auc"],
            "average_precision": metadata["model_metrics"]["average_precision"],
            "training_samples": metadata.get("training_samples", 0),
            "test_samples": metadata.get("test_samples", 0),
            "n_features": len(FEATURE_COLUMNS),
        },
        task=ml_task.Task.TABULAR_BINARY_CLASSIFICATION,
    )

    print(f"Registered {MODEL_NAME} {version} in {CONFIG['database']}.{CONFIG['schemas']['model_registry']}")

    # Quick verification
    preds = mv.run(sample_input.head(5), function_name="predict")
    print(f"Verification — predict on 5 rows returned {len(preds)} results")

    session.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register model in Snowflake")
    parser.add_argument("--model-path", default="artifacts/readmission_model.joblib")
    parser.add_argument("--metadata-path", default="artifacts/model_metadata.json")
    parser.add_argument("--test-data-path", default="artifacts/test_data.csv")
    parser.add_argument("--version", default=None)
    args = parser.parse_args()

    register_model(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        test_data_path=args.test_data_path,
        version=args.version,
    )
