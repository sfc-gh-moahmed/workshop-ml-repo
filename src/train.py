"""
Model training pipeline for 30-day hospital readmission prediction.

Can be run as a script:
    python -m src.train --artifacts-dir ./artifacts

Or imported:
    from src.train import train_model
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from src.config import FEATURE_COLUMNS, TARGET


def train_model(
    training_df: pd.DataFrame,
    artifacts_dir: str = "artifacts",
) -> dict:
    """
    Train a GradientBoostingClassifier on the provided feature DataFrame.

    Args:
        training_df: DataFrame containing FEATURE_COLUMNS + TARGET.
        artifacts_dir: Directory to save model and metadata.

    Returns:
        Dict with model metrics and artifact paths.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    X = training_df[FEATURE_COLUMNS].fillna(0)
    y = training_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    # Save model
    model_path = os.path.join(artifacts_dir, "readmission_model.joblib")
    joblib.dump(model, model_path)

    # Save train/test splits
    train_out = X_train.copy()
    train_out[TARGET] = y_train.values
    train_out.to_csv(os.path.join(artifacts_dir, "training_data.csv"), index=False)

    test_out = X_test.copy()
    test_out[TARGET] = y_test.values
    test_out.to_csv(os.path.join(artifacts_dir, "test_data.csv"), index=False)

    # Save metadata
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "model_metrics": {"roc_auc": round(auc, 4), "average_precision": round(ap, 4)},
        "training_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
    }
    with open(os.path.join(artifacts_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"ROC AUC: {auc:.4f}  |  Average Precision: {ap:.4f}")

    return {
        "model": model,
        "model_path": model_path,
        "metrics": metadata["model_metrics"],
        "training_samples": metadata["training_samples"],
        "test_samples": metadata["test_samples"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train readmission model locally")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--data-csv", default="artifacts/training_data.csv",
                        help="Path to pre-engineered feature CSV (with target)")
    args = parser.parse_args()

    df = pd.read_csv(args.data_csv)
    train_model(df, artifacts_dir=args.artifacts_dir)
