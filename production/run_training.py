"""
Production entry point: Train model + register in Snowflake Model Registry.

Designed to be executed inside Snowflake via:
    EXECUTE IMMEDIATE FROM @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO/branches/main/production/run_training.py
    USING (VERSION => 'V2');

Or locally:
    HEALTHCARE_ML_ENV=PROD python -m production.run_training --version V2
"""

import os
import sys

# When executed via EXECUTE IMMEDIATE FROM, Snowflake provides a Snowpark session
# as `_snowpark_session`. When run locally, we create our own session.


def main(session=None, version: str = "V1"):
    """
    End-to-end: read data from Snowflake tables, train locally, register model.
    """
    # Add project root to path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.config import CONFIG, FEATURE_COLUMNS, TARGET, get_session
    from src.train import train_model
    from src.register_model import register_model

    close_session = session is None
    if session is None:
        session = get_session()

    # -----------------------------------------------------------------------
    # Step 1: Read feature-engineered data from Snowflake
    # -----------------------------------------------------------------------
    print("Reading training data from Snowflake...")
    session.sql(f"USE SCHEMA {CONFIG['schemas']['raw_data']}").collect()
    training_df = session.table(
        f"{CONFIG['database']}.{CONFIG['schemas']['raw_data']}.TRAINING_DATA"
    ).to_pandas()
    print(f"Training data: {training_df.shape[0]} rows, {training_df.shape[1]} columns")

    # -----------------------------------------------------------------------
    # Step 2: Train model locally
    # -----------------------------------------------------------------------
    artifacts_dir = os.path.join(project_root, "artifacts")
    result = train_model(training_df, artifacts_dir=artifacts_dir)
    print(f"Training complete — ROC AUC: {result['metrics']['roc_auc']}")

    # -----------------------------------------------------------------------
    # Step 3: Register in Snowflake Model Registry
    # -----------------------------------------------------------------------
    register_model(
        model_path=result["model_path"],
        metadata_path=os.path.join(artifacts_dir, "model_metadata.json"),
        test_data_path=os.path.join(artifacts_dir, "test_data.csv"),
        version=version,
    )

    if close_session:
        session.close()

    print(f"Pipeline complete — model registered as {version}")
    return f"Model registered as {version}"


# ---------------------------------------------------------------------------
# Snowflake EXECUTE IMMEDIATE FROM entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=os.environ.get("MODEL_VERSION", "V1"))
    args = parser.parse_args()

    main(version=args.version)
