"""
Configuration for Healthcare Readmission ML Pipeline.

Supports DEV and PROD environments via the HEALTHCARE_ML_ENV environment variable.
Defaults to DEV when not set.

Usage:
    from src.config import get_session, CONFIG
    session = get_session()
"""

import os
from snowflake.snowpark import Session


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
ENV = os.environ.get("HEALTHCARE_ML_ENV", "DEV").upper()

# ---------------------------------------------------------------------------
# Per-environment settings
# ---------------------------------------------------------------------------
_ENVIRONMENTS = {
    "DEV": {
        "connection_name": "DEMO",
        "role": "ACCOUNTADMIN",
        "database": "HEALTHCARE_ML",
        "warehouse": "HEALTHCARE_ML_WH",
        "schemas": {
            "raw_data": "RAW_DATA",
            "feature_store": "FEATURE_STORE",
            "model_registry": "MODEL_REGISTRY",
            "inference": "INFERENCE",
        },
        "compute_pool": "DEMO_POOL_CPU",
    },
    "PROD": {
        "connection_name": os.environ.get("SNOWFLAKE_CONNECTION", "DEMO"),
        "role": "ML_ENGINEER",
        "database": "HEALTHCARE_ML",
        "warehouse": "HEALTHCARE_ML_PROD_WH",
        "schemas": {
            "raw_data": "RAW_DATA",
            "feature_store": "FEATURE_STORE",
            "model_registry": "MODEL_REGISTRY",
            "inference": "INFERENCE",
        },
        "compute_pool": "PROD_POOL_CPU",
    },
}

CONFIG = _ENVIRONMENTS[ENV]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
MODEL_NAME = "READMISSION_PREDICTOR"
MODEL_VERSION = os.environ.get("MODEL_VERSION", "V1")

# ---------------------------------------------------------------------------
# Feature column list (single source of truth)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "AGE", "GENDER_ENC", "INSURANCE_ENC", "HAS_PCP_FLAG",
    "LENGTH_OF_STAY", "NUM_PROCEDURES", "NUM_DIAGNOSES",
    "DIAGNOSIS_RISK_SCORE", "DISPOSITION_RISK_SCORE", "ED_ADMISSION",
    "HEART_RATE", "SYSTOLIC_BP", "DIASTOLIC_BP", "TEMPERATURE",
    "RESPIRATORY_RATE", "O2_SATURATION", "BLOOD_GLUCOSE", "CREATININE",
    "HEMOGLOBIN", "WBC_COUNT", "SODIUM", "POTASSIUM", "BNP",
    "ABNORMAL_HR", "ABNORMAL_BP", "LOW_O2", "HIGH_CREATININE",
    "LOW_HEMOGLOBIN", "HIGH_BNP", "ABNORMAL_GLUCOSE",
    "PRIOR_ADMISSIONS_6M", "PRIOR_READMISSIONS", "AVG_PRIOR_LOS",
]

TARGET = "READMITTED_30D"


# ---------------------------------------------------------------------------
# Session helper
# ---------------------------------------------------------------------------
def get_session() -> Session:
    """Create a Snowpark session using the current environment config."""
    session = Session.builder.config("connection_name", CONFIG["connection_name"]).create()
    session.sql(f"USE ROLE {CONFIG['role']}").collect()
    session.sql(f"USE DATABASE {CONFIG['database']}").collect()
    session.sql(f"USE WAREHOUSE {CONFIG['warehouse']}").collect()
    return session
