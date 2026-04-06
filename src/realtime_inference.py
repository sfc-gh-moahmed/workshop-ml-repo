"""
Real-time inference using Online Feature Store + Model Registry.

Can be imported for use in APIs or Streamlit apps:
    from src.realtime_inference import predict_readmission_risk
"""

import time

import pandas as pd
from snowflake.ml.feature_store import FeatureStore
from snowflake.ml.registry import Registry

from src.config import (
    CONFIG, FEATURE_COLUMNS, MODEL_NAME, MODEL_VERSION, get_session,
)


def predict_readmission_risk(
    patient_id: str,
    session=None,
    version: str | None = None,
) -> dict:
    """
    Predict 30-day readmission risk for a single patient in real time.

    1. Retrieves latest features from Online Feature Store
    2. Runs inference via the registered model
    3. Returns risk score + risk level

    Args:
        patient_id: The PATIENT_ID to score.
        session:    Optional existing Snowpark session (created if None).
        version:    Model version override.

    Returns:
        Dict with patient_id, readmission_probability, risk_level, response_time_ms.
    """
    version = version or MODEL_VERSION
    close_session = session is None
    if session is None:
        session = get_session()

    total_start = time.time()

    # Feature Store + Model
    fs = FeatureStore(
        session=session,
        database=CONFIG["database"],
        name=CONFIG["schemas"]["feature_store"],
        default_warehouse=CONFIG["warehouse"],
    )
    registry = Registry(
        session=session,
        database_name=CONFIG["database"],
        schema_name=CONFIG["schemas"]["model_registry"],
    )
    mv = registry.get_model(MODEL_NAME).version(version)
    fv = fs.get_feature_view("PATIENT_CLINICAL_FEATURES", "V1")

    # Retrieve features from Online Feature Store
    spine = session.create_dataframe([[patient_id]], schema=["PATIENT_ID"])
    features = fs.retrieve_feature_values(
        spine_df=spine, features=[fv], spine_timestamp_col=None
    ).to_pandas()

    if features.empty:
        return {"patient_id": patient_id, "error": "No features found"}

    # Run inference
    available = [c for c in FEATURE_COLUMNS if c in features.columns]
    proba = mv.run(features[available], function_name="predict_proba")
    # predict_proba returns [P(class_0), P(class_1)] — last column is readmission prob
    readmission_prob = float(proba.iloc[0, -1])

    total_ms = round((time.time() - total_start) * 1000, 1)

    if close_session:
        session.close()

    return {
        "patient_id": patient_id,
        "readmission_probability": round(readmission_prob, 4),
        "risk_level": (
            "HIGH" if readmission_prob > 0.5
            else ("MEDIUM" if readmission_prob > 0.3 else "LOW")
        ),
        "response_time_ms": total_ms,
    }
