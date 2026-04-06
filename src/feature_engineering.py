"""
Feature engineering for 30-day hospital readmission prediction.

Contains both the Snowflake SQL feature query (for Feature Store / Dynamic Tables)
and the local pandas-based feature engineering (for local training).
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Risk score mappings (shared across local + Snowflake feature engineering)
# ---------------------------------------------------------------------------
DIAGNOSIS_RISK = {
    "HEART_FAILURE": 3, "SEPSIS": 3, "COPD": 2, "RENAL_FAILURE": 2,
    "ACUTE_MI": 2, "DIABETES_COMPLICATIONS": 2, "STROKE": 2,
    "PNEUMONIA": 1, "GI_BLEED": 1, "HIP_FRACTURE": 1,
}

DISPOSITION_RISK = {
    "HOME": 0, "HOME_HEALTH": 1, "REHAB": 1, "SNF": 2, "AMA": 3,
}

# ---------------------------------------------------------------------------
# Snowflake SQL feature query (used by Feature View Dynamic Table)
# ---------------------------------------------------------------------------
FEATURE_VIEW_SQL = """
SELECT
    a.PATIENT_ID,
    a.ADMISSION_ID,
    TO_TIMESTAMP(a.DISCHARGE_DATE) AS EVENT_TIMESTAMP,
    -- Patient demographics
    p.AGE,
    CASE WHEN p.GENDER = 'M' THEN 1 ELSE 0 END AS GENDER_ENC,
    CASE p.INSURANCE_TYPE
        WHEN 'MEDICAID' THEN 0 WHEN 'MEDICARE' THEN 1
        WHEN 'PRIVATE' THEN 2 WHEN 'SELF_PAY' THEN 3
    END AS INSURANCE_ENC,
    CASE WHEN p.HAS_PCP THEN 1 ELSE 0 END AS HAS_PCP_FLAG,
    -- Admission features
    a.LENGTH_OF_STAY,
    a.NUM_PROCEDURES,
    a.NUM_DIAGNOSES,
    CASE a.PRIMARY_DIAGNOSIS
        WHEN 'HEART_FAILURE' THEN 3 WHEN 'SEPSIS' THEN 3
        WHEN 'COPD' THEN 2 WHEN 'RENAL_FAILURE' THEN 2
        WHEN 'ACUTE_MI' THEN 2 WHEN 'DIABETES_COMPLICATIONS' THEN 2 WHEN 'STROKE' THEN 2
        ELSE 1
    END AS DIAGNOSIS_RISK_SCORE,
    CASE a.DISCHARGE_DISPOSITION
        WHEN 'HOME' THEN 0 WHEN 'HOME_HEALTH' THEN 1 WHEN 'REHAB' THEN 1
        WHEN 'SNF' THEN 2 WHEN 'AMA' THEN 3
    END AS DISPOSITION_RISK_SCORE,
    a.ED_ADMISSION,
    -- Clinical measurements at discharge
    c.HEART_RATE, c.SYSTOLIC_BP, c.DIASTOLIC_BP, c.TEMPERATURE,
    c.RESPIRATORY_RATE, c.O2_SATURATION, c.BLOOD_GLUCOSE, c.CREATININE,
    c.HEMOGLOBIN, c.WBC_COUNT, c.SODIUM, c.POTASSIUM, c.BNP,
    -- Clinical abnormality flags
    CASE WHEN c.HEART_RATE > 100 OR c.HEART_RATE < 60 THEN 1 ELSE 0 END AS ABNORMAL_HR,
    CASE WHEN c.SYSTOLIC_BP > 160 OR c.SYSTOLIC_BP < 90 THEN 1 ELSE 0 END AS ABNORMAL_BP,
    CASE WHEN c.O2_SATURATION < 93 THEN 1 ELSE 0 END AS LOW_O2,
    CASE WHEN c.CREATININE > 1.5 THEN 1 ELSE 0 END AS HIGH_CREATININE,
    CASE WHEN c.HEMOGLOBIN < 10 THEN 1 ELSE 0 END AS LOW_HEMOGLOBIN,
    CASE WHEN c.BNP > 500 THEN 1 ELSE 0 END AS HIGH_BNP,
    CASE WHEN c.BLOOD_GLUCOSE > 200 OR c.BLOOD_GLUCOSE < 70 THEN 1 ELSE 0 END AS ABNORMAL_GLUCOSE,
    -- Historical features (window functions)
    COALESCE(COUNT(*) OVER (
        PARTITION BY a.PATIENT_ID ORDER BY a.ADMIT_DATE
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS PRIOR_ADMISSIONS_6M,
    COALESCE(SUM(a.READMITTED_30D) OVER (
        PARTITION BY a.PATIENT_ID ORDER BY a.ADMIT_DATE
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS PRIOR_READMISSIONS,
    COALESCE(AVG(a.LENGTH_OF_STAY) OVER (
        PARTITION BY a.PATIENT_ID ORDER BY a.ADMIT_DATE
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ), 0) AS AVG_PRIOR_LOS,
    -- Target
    a.READMITTED_30D
FROM {database}.RAW_DATA.ADMISSIONS a
JOIN {database}.RAW_DATA.PATIENTS p ON a.PATIENT_ID = p.PATIENT_ID
JOIN {database}.RAW_DATA.CLINICAL_MEASUREMENTS c
    ON a.ADMISSION_ID = c.ADMISSION_ID AND a.PATIENT_ID = c.PATIENT_ID
"""


def get_feature_view_sql(database: str = "HEALTHCARE_ML") -> str:
    """Return the feature SQL with the database name injected."""
    return FEATURE_VIEW_SQL.format(database=database)


# ---------------------------------------------------------------------------
# Local pandas feature engineering (mirrors the SQL logic)
# ---------------------------------------------------------------------------
def engineer_features(
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    clinical: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join raw tables and compute the 33 model features locally.
    Returns a DataFrame with FEATURE_COLUMNS + TARGET ready for training.
    """
    df = admissions.merge(patients, on="PATIENT_ID", how="left")
    df = df.merge(clinical, on=["ADMISSION_ID", "PATIENT_ID"], how="left")
    df["ADMIT_DATE"] = pd.to_datetime(df["ADMIT_DATE"])
    df["DISCHARGE_DATE"] = pd.to_datetime(df["DISCHARGE_DATE"])
    df = df.sort_values(["PATIENT_ID", "ADMIT_DATE"])

    # Historical features per patient
    def _historical(group):
        group = group.copy()
        pa, pr, al = [], [], []
        for _, row in group.iterrows():
            prior = group[group["ADMIT_DATE"] < row["ADMIT_DATE"]]
            prior_6m = prior[(row["ADMIT_DATE"] - prior["ADMIT_DATE"]).dt.days <= 180]
            pa.append(len(prior_6m))
            pr.append(int(prior["READMITTED_30D"].sum()) if len(prior) > 0 else 0)
            al.append(float(prior["LENGTH_OF_STAY"].mean()) if len(prior) > 0 else 0.0)
        group["PRIOR_ADMISSIONS_6M"] = pa
        group["PRIOR_READMISSIONS"] = pr
        group["AVG_PRIOR_LOS"] = al
        return group

    df = df.groupby("PATIENT_ID", group_keys=False).apply(_historical)

    # Encode categoricals
    df["DIAGNOSIS_RISK_SCORE"] = df["PRIMARY_DIAGNOSIS"].map(DIAGNOSIS_RISK)
    df["DISPOSITION_RISK_SCORE"] = df["DISCHARGE_DISPOSITION"].map(DISPOSITION_RISK)
    df["GENDER_ENC"] = (df["GENDER"] == "M").astype(int)

    insurance_order = ["MEDICAID", "MEDICARE", "PRIVATE", "SELF_PAY"]
    df["INSURANCE_ENC"] = df["INSURANCE_TYPE"].map(
        {v: i for i, v in enumerate(insurance_order)}
    )
    df["HAS_PCP_FLAG"] = df["HAS_PCP"].astype(int)

    # Clinical abnormality flags
    df["ABNORMAL_HR"] = ((df["HEART_RATE"] > 100) | (df["HEART_RATE"] < 60)).astype(int)
    df["ABNORMAL_BP"] = ((df["SYSTOLIC_BP"] > 160) | (df["SYSTOLIC_BP"] < 90)).astype(int)
    df["LOW_O2"] = (df["O2_SATURATION"] < 93).astype(int)
    df["HIGH_CREATININE"] = (df["CREATININE"] > 1.5).astype(int)
    df["LOW_HEMOGLOBIN"] = (df["HEMOGLOBIN"] < 10).astype(int)
    df["HIGH_BNP"] = (df["BNP"] > 500).astype(int)
    df["ABNORMAL_GLUCOSE"] = ((df["BLOOD_GLUCOSE"] > 200) | (df["BLOOD_GLUCOSE"] < 70)).astype(int)

    return df
