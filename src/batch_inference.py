"""
Batch inference using Snowflake Model Registry run_batch() on SPCS.

Can be run as a script:
    python -m src.batch_inference --version V1

Or imported:
    from src.batch_inference import run_batch_inference
"""

import time

from snowflake.ml.registry import Registry

from src.config import (
    CONFIG, FEATURE_COLUMNS, MODEL_NAME, MODEL_VERSION, get_session,
)


def run_batch_inference(
    version: str | None = None,
    compute_pool: str | None = None,
) -> None:
    """
    Run distributed batch inference via run_batch() on SPCS.

    Reads features from the Feature View Dynamic Table, scores all rows,
    and writes results to HEALTHCARE_ML.INFERENCE.BATCH_PREDICTIONS.
    """
    from snowflake.ml.model._client.model.batch_inference_specs import (
        OutputSpec, SaveMode, JobSpec,
    )

    version = version or MODEL_VERSION
    compute_pool = compute_pool or CONFIG["compute_pool"]

    session = get_session()
    session.sql(f"USE SCHEMA {CONFIG['schemas']['inference']}").collect()

    # Load model from registry
    registry = Registry(
        session=session,
        database_name=CONFIG["database"],
        schema_name=CONFIG["schemas"]["model_registry"],
    )
    mv = registry.get_model(MODEL_NAME).version(version)
    print(f"Model loaded: {MODEL_NAME} {version}")

    # Read features from the Dynamic Table backing the Feature View
    feature_table = (
        f"{CONFIG['database']}.{CONFIG['schemas']['feature_store']}"
        f".PATIENT_CLINICAL_FEATURES$V1"
    )
    feature_df = session.table(feature_table)
    features_only = feature_df.select(FEATURE_COLUMNS)
    row_count = features_only.count()
    print(f"Input rows: {row_count}")

    # Ensure output stage exists
    session.sql(f"""
        CREATE STAGE IF NOT EXISTS {CONFIG['database']}.{CONFIG['schemas']['inference']}.BATCH_OUTPUT
        COMMENT = 'Batch inference output stage'
    """).collect()

    # Launch batch inference
    output_path = f"@{CONFIG['database']}.{CONFIG['schemas']['inference']}.BATCH_OUTPUT/readmission_predictions/"
    print(f"Starting run_batch() on {compute_pool} ...")

    job = mv.run_batch(
        compute_pool=compute_pool,
        X=features_only,
        output_spec=OutputSpec(stage_location=output_path, mode=SaveMode.OVERWRITE),
        job_spec=JobSpec(function_name="predict_proba"),
    )

    # Poll until done
    while True:
        status = job.status
        print(f"  Status: {status}")
        if status in ("DONE", "FAILED", "CANCELLED"):
            break
        time.sleep(30)

    if status == "DONE":
        results_df = session.read.parquet(output_path)
        results_df.write.save_as_table(
            f"{CONFIG['database']}.{CONFIG['schemas']['inference']}.BATCH_PREDICTIONS",
            mode="overwrite",
        )
        count = session.sql(
            f"SELECT COUNT(*) AS CNT FROM {CONFIG['database']}.{CONFIG['schemas']['inference']}.BATCH_PREDICTIONS"
        ).collect()[0]["CNT"]
        print(f"Batch predictions saved: {count} rows")
    else:
        print(f"Job ended with status: {status}")

    session.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run batch inference on SPCS")
    parser.add_argument("--version", default=None)
    parser.add_argument("--compute-pool", default=None)
    args = parser.parse_args()

    run_batch_inference(version=args.version, compute_pool=args.compute_pool)
