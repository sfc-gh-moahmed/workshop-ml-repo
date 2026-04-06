"""
Production entry point: Run batch inference scoring.

Designed to be executed inside Snowflake via:
    EXECUTE IMMEDIATE FROM @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO/branches/main/production/run_batch_inference.py
    USING (VERSION => 'V1');

Or scheduled via Snowflake TASK (see production/tasks/setup_tasks.sql).

Or locally:
    HEALTHCARE_ML_ENV=PROD python -m production.run_batch_inference --version V1
"""

import os
import sys


def main(session=None, version: str = "V1"):
    """
    Score all patients in the Feature View and write results to BATCH_PREDICTIONS.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.batch_inference import run_batch_inference

    print(f"Starting batch inference with model version {version} ...")
    run_batch_inference(version=version)
    print("Batch inference complete.")
    return "Batch inference complete"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=os.environ.get("MODEL_VERSION", "V1"))
    args = parser.parse_args()

    main(version=args.version)
