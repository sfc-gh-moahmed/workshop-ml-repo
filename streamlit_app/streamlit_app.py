"""
Healthcare 30-Day Readmission Prediction -- End-to-End ML Pipeline Demo
========================================================================
Interactive Streamlit in Snowflake app that walks through each phase of the
ML lifecycle: raw data, feature engineering, model registry, batch inference,
real-time inference, Git integration, and production scheduling.

Each step shows the exact SQL/Python that runs, executes it live, and displays
the results.
"""

import streamlit as st
import pandas as pd
import time

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Healthcare ML Pipeline",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
from snowflake.snowpark.context import get_active_session
session = get_active_session()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DB = "HEALTHCARE_ML"
WH = "HEALTHCARE_ML_WH"
FEATURE_TABLE = f"{DB}.FEATURE_STORE.\"PATIENT_CLINICAL_FEATURES$V1\""
MODEL_NAME = "READMISSION_PREDICTOR"
GIT_REPO = f"{DB}.GIT_INTEGRATION.HEALTHCARE_ML_REPO"

WORKFLOW_STEPS = [
    "Overview",
    "1. Raw Data",
    "2. Feature Store",
    "3. Model Registry",
    "4. Batch Inference",
    "5. Real-Time Inference",
    "6. Git Integration",
    "7. Production Tasks",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_sql(sql: str, show_code: bool = True) -> pd.DataFrame:
    """Execute SQL, optionally show it, return DataFrame."""
    if show_code:
        st.code(sql.strip(), language="sql")
    try:
        result = session.sql(sql)
        rows = result.collect()
        if not rows:
            return pd.DataFrame()
        col_names = list(rows[0].as_dict().keys())
        data = [list(row.as_dict().values()) for row in rows]
        df = pd.DataFrame(data, columns=col_names)
        return df
    except Exception as e:
        st.error(f"Query error: {e}")
        return pd.DataFrame()


def show_result(df: pd.DataFrame, max_rows: int = 200):
    """Display a dataframe result with row count."""
    if df.empty:
        st.info("No rows returned.")
        return
    st.caption(f"{len(df):,} row(s) returned")
    st.dataframe(df.head(max_rows), use_container_width=True)


def metric_row(metrics: dict):
    """Display a row of metric cards."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("ML Pipeline Workflow")
st.sidebar.markdown("Walk through each phase of the end-to-end healthcare readmission prediction pipeline.")
step = st.sidebar.radio("Select Phase", WORKFLOW_STEPS, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**GitHub:** [healthcare-readmission-ml](https://github.com/sfc-gh-moahmed/healthcare-readmission-ml)"
)
st.sidebar.markdown(f"**Database:** `{DB}`")
st.sidebar.markdown(f"**Warehouse:** `{WH}`")

# =========================================================================
# OVERVIEW
# =========================================================================
if step == "Overview":
    st.title("End-to-End ML on Snowflake")
    st.subheader("Healthcare 30-Day Readmission Prediction")

    st.markdown("""
    This app demonstrates the **complete ML lifecycle** running natively on Snowflake:

    | Phase | Snowflake Feature | What It Does |
    |-------|------------------|--------------|
    | **Raw Data** | Tables | 5,000 patients, 10,772 admissions, clinical vitals |
    | **Feature Store** | Dynamic Tables + Online Store | 33 engineered features, auto-refreshing |
    | **Model Registry** | Model Registry | GradientBoostingClassifier, versioned, governed |
    | **Batch Inference** | SPCS + Ray | Distributed scoring across all patients |
    | **Real-Time Inference** | Online Feature Store + mv.run() | Sub-second single-patient predictions |
    | **Git Integration** | Git Repository | Code synced from GitHub, executable from SQL |
    | **Production Tasks** | Snowflake Tasks | Scheduled fetch + scoring pipeline |

    **Use the sidebar** to walk through each phase. Every button runs **live SQL** against the
    `HEALTHCARE_ML` database.
    """)

    st.markdown("### Architecture")
    st.code("""
  LOCAL DEVELOPMENT              GITHUB                    SNOWFLAKE
  -----------------              ------                    ---------
  notebooks/          git push   main branch   GIT FETCH   GIT REPOSITORY object
  src/              ----------> feature/*    ---------->  @.../branches/main/
  production/                    tags/                     EXECUTE IMMEDIATE FROM

  Experiment in                    |                       TASK: GIT_FETCH
  notebooks                        | PR + Review           TASK: BATCH_SCORING
  Promote to src/                  v
  Test locally                  Merge to main             MODEL REGISTRY
                                                          FEATURE STORE
                                                          BATCH_PREDICTIONS
    """, language="text")

# =========================================================================
# STEP 1: RAW DATA
# =========================================================================
elif step == "1. Raw Data":
    st.title("Phase 1: Raw Data")
    st.markdown("""
    Three source tables loaded into `HEALTHCARE_ML.RAW_DATA`:
    - **PATIENTS** -- demographics (age, gender, insurance, PCP status)
    - **ADMISSIONS** -- hospital visits (diagnoses, length of stay, procedures)
    - **CLINICAL_MEASUREMENTS** -- vitals and lab values at discharge
    """)

    # Table counts
    if st.button("Show Table Row Counts", key="raw_counts"):
        sql = """
SELECT 'PATIENTS' AS TABLE_NAME, COUNT(*) AS ROW_COUNT FROM HEALTHCARE_ML.RAW_DATA.PATIENTS
UNION ALL
SELECT 'ADMISSIONS', COUNT(*) FROM HEALTHCARE_ML.RAW_DATA.ADMISSIONS
UNION ALL
SELECT 'CLINICAL_MEASUREMENTS', COUNT(*) FROM HEALTHCARE_ML.RAW_DATA.CLINICAL_MEASUREMENTS
UNION ALL
SELECT 'TRAINING_DATA', COUNT(*) FROM HEALTHCARE_ML.RAW_DATA.TRAINING_DATA
UNION ALL
SELECT 'TEST_DATA', COUNT(*) FROM HEALTHCARE_ML.RAW_DATA.TEST_DATA
ORDER BY TABLE_NAME"""
        df = run_sql(sql)
        show_result(df)

    st.markdown("---")

    st.subheader("Explore Raw Tables")
    table_choice = st.selectbox("Select table", ["PATIENTS", "ADMISSIONS", "CLINICAL_MEASUREMENTS"])

    if st.button("Preview Table", key="raw_preview"):
        sql = f"SELECT * FROM {DB}.RAW_DATA.{table_choice} LIMIT 50"
        df = run_sql(sql)
        show_result(df)

    st.markdown("---")
    st.subheader("Target Variable Distribution")
    if st.button("Show Readmission Rate", key="readmit_rate"):
        sql = """
SELECT
    CASE WHEN READMITTED_30D = 1 THEN 'Readmitted' ELSE 'Not Readmitted' END AS STATUS,
    COUNT(*) AS COUNT,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS PERCENTAGE
FROM HEALTHCARE_ML.RAW_DATA.ADMISSIONS
GROUP BY READMITTED_30D
ORDER BY READMITTED_30D"""
        df = run_sql(sql)
        show_result(df)

    st.markdown("---")
    st.subheader("Diagnosis Distribution")
    if st.button("Show Top Diagnoses", key="diag_dist"):
        sql = """
SELECT PRIMARY_DIAGNOSIS, COUNT(*) AS ADMISSION_COUNT,
       ROUND(AVG(READMITTED_30D) * 100, 1) AS READMISSION_RATE_PCT
FROM HEALTHCARE_ML.RAW_DATA.ADMISSIONS
GROUP BY PRIMARY_DIAGNOSIS
ORDER BY ADMISSION_COUNT DESC"""
        df = run_sql(sql)
        show_result(df)

# =========================================================================
# STEP 2: FEATURE STORE
# =========================================================================
elif step == "2. Feature Store":
    st.title("Phase 2: Feature Store")
    st.markdown("""
    The Feature Store uses a **Dynamic Table** that automatically joins the 3 raw tables
    and engineers 33 features. It refreshes every **5 minutes**. An **Online Feature Store**
    layer provides sub-second key-value lookups with a 1-minute target lag.

    **Feature categories:** Demographics (4), Admission (6), Clinical Vitals (13),
    Abnormality Flags (7), Historical (3)
    """)

    st.subheader("Feature View SQL")
    if st.button("Show Feature Engineering SQL", key="fv_sql"):
        sql_display = """-- This SQL drives the Dynamic Table backing the Feature Store
SELECT
    a.PATIENT_ID,
    a.ADMISSION_ID,
    TO_TIMESTAMP(a.DISCHARGE_DATE) AS EVENT_TIMESTAMP,
    -- Demographics
    p.AGE,
    CASE WHEN p.GENDER = 'M' THEN 1 ELSE 0 END AS GENDER_ENC,
    CASE p.INSURANCE_TYPE
        WHEN 'MEDICAID' THEN 0 WHEN 'MEDICARE' THEN 1
        WHEN 'PRIVATE' THEN 2 WHEN 'SELF_PAY' THEN 3
    END AS INSURANCE_ENC,
    CASE WHEN p.HAS_PCP THEN 1 ELSE 0 END AS HAS_PCP_FLAG,
    -- Admission features
    a.LENGTH_OF_STAY, a.NUM_PROCEDURES, a.NUM_DIAGNOSES,
    CASE a.PRIMARY_DIAGNOSIS
        WHEN 'HEART_FAILURE' THEN 3 WHEN 'SEPSIS' THEN 3
        WHEN 'COPD' THEN 2 WHEN 'RENAL_FAILURE' THEN 2 ...
    END AS DIAGNOSIS_RISK_SCORE,
    -- Clinical abnormality flags
    CASE WHEN c.BNP > 500 THEN 1 ELSE 0 END AS HIGH_BNP,
    CASE WHEN c.CREATININE > 1.5 THEN 1 ELSE 0 END AS HIGH_CREATININE,
    -- Historical (window functions)
    COUNT(*) OVER (PARTITION BY a.PATIENT_ID ORDER BY a.ADMIT_DATE
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        AS PRIOR_ADMISSIONS_6M
FROM RAW_DATA.ADMISSIONS a
JOIN RAW_DATA.PATIENTS p ON a.PATIENT_ID = p.PATIENT_ID
JOIN RAW_DATA.CLINICAL_MEASUREMENTS c ON a.ADMISSION_ID = c.ADMISSION_ID"""
        st.code(sql_display, language="sql")

    st.markdown("---")
    st.subheader("Dynamic Table Status")
    if st.button("Check Dynamic Table Refresh", key="dt_status"):
        sql = """SHOW DYNAMIC TABLES LIKE '%PATIENT_CLINICAL%' IN SCHEMA HEALTHCARE_ML.FEATURE_STORE"""
        df = run_sql(sql)
        if not df.empty:
            display_cols = [c for c in ["name", "rows", "scheduling_state", "target_lag",
                                        "refresh_mode", "data_timestamp", "warehouse"] if c in df.columns]
            if display_cols:
                st.dataframe(df[display_cols], use_container_width=True)
            else:
                show_result(df)

    st.markdown("---")
    st.subheader("Preview Feature Store Data")
    if st.button("Show Engineered Features (sample)", key="fv_preview"):
        sql = f'SELECT * FROM {FEATURE_TABLE} LIMIT 20'
        df = run_sql(sql)
        show_result(df)

    st.markdown("---")
    st.subheader("Feature Statistics")
    if st.button("Show Feature Summary Stats", key="fv_stats"):
        sql = f"""SELECT
    COUNT(*) AS TOTAL_ROWS,
    COUNT(DISTINCT PATIENT_ID) AS UNIQUE_PATIENTS,
    ROUND(AVG(AGE), 1) AS AVG_AGE,
    ROUND(AVG(LENGTH_OF_STAY), 1) AS AVG_LOS,
    ROUND(AVG(PRIOR_ADMISSIONS_6M), 2) AS AVG_PRIOR_ADMISSIONS,
    ROUND(AVG(HIGH_BNP), 3) AS HIGH_BNP_RATE,
    ROUND(AVG(HIGH_CREATININE), 3) AS HIGH_CREATININE_RATE,
    ROUND(AVG(LOW_O2), 3) AS LOW_O2_RATE
FROM {FEATURE_TABLE}"""
        df = run_sql(sql)
        show_result(df)

# =========================================================================
# STEP 3: MODEL REGISTRY
# =========================================================================
elif step == "3. Model Registry":
    st.title("Phase 3: Model Registry")
    st.markdown("""
    The trained **GradientBoostingClassifier** (scikit-learn, 200 estimators) is registered
    in the Snowflake Model Registry. Registration auto-detects inference functions:
    `PREDICT`, `PREDICT_PROBA`, `PREDICT_LOG_PROBA`, `DECISION_FUNCTION`, `EXPLAIN`.
    """)

    st.subheader("Registration Code")
    if st.button("Show Model Registration Python", key="reg_code"):
        st.code("""
# Load the trained model
import joblib
model = joblib.load('artifacts/readmission_model.joblib')

# Prepare sample input for schema inference
sample_input = test_df[FEATURE_COLUMNS].head(100)

# Register in Snowflake
from snowflake.ml.registry import Registry
from snowflake.ml.model import task as ml_task

registry = Registry(session=session,
                    database_name="HEALTHCARE_ML",
                    schema_name="MODEL_REGISTRY")

mv = registry.log_model(
    model=model,
    model_name="READMISSION_PREDICTOR",
    version_name="V1",
    sample_input_data=sample_input,
    conda_dependencies=["scikit-learn"],
    metrics={"roc_auc": 0.552, "average_precision": 0.290},
    task=ml_task.Task.TABULAR_BINARY_CLASSIFICATION
)
        """, language="python")

    st.markdown("---")
    st.subheader("Registered Models")
    if st.button("Show Models in Registry", key="show_models"):
        sql = "SHOW MODELS IN HEALTHCARE_ML.MODEL_REGISTRY"
        df = run_sql(sql)
        show_result(df)

    st.markdown("---")
    st.subheader("Model Versions & Metrics")
    if st.button("Show Model Version Details", key="model_versions"):
        sql = """SHOW VERSIONS IN MODEL HEALTHCARE_ML.MODEL_REGISTRY.READMISSION_PREDICTOR"""
        df = run_sql(sql)
        if not df.empty:
            import json
            display_df = df[["name", "comment", "metadata", "created_on"]].copy()
            try:
                metadata = json.loads(display_df["metadata"].iloc[0]) if display_df["metadata"].iloc[0] else {}
                metrics = metadata.get("metrics", {})
                display_df["ROC_AUC"] = metrics.get("roc_auc")
                display_df["AVG_PRECISION"] = metrics.get("average_precision")
                display_df["N_FEATURES"] = metrics.get("n_features")
                display_df["TRAINING_SAMPLES"] = metrics.get("training_samples")
                display_df["TEST_SAMPLES"] = metrics.get("test_samples")
                display_df = display_df.drop(columns=["metadata"])
            except:
                pass
            display_df.columns = [c.upper() for c in display_df.columns]
            show_result(display_df)
        else:
            show_result(df)

    st.markdown("---")
    st.subheader("Model Functions")
    if st.button("List Auto-Detected Functions", key="model_funcs"):
        sql = """SELECT
    FUNCTION_NAME, RETURN_TYPE
FROM TABLE(HEALTHCARE_ML.INFORMATION_SCHEMA.MODEL_VERSION_FUNCTIONS(
    MODEL_NAME => 'READMISSION_PREDICTOR',
    VERSION_NAME => 'V1'
))
ORDER BY FUNCTION_NAME"""
        df = run_sql(sql)
        if df.empty:
            st.info("Function listing not available via this method. The model has auto-detected functions: PREDICT, PREDICT_PROBA, PREDICT_LOG_PROBA, DECISION_FUNCTION, EXPLAIN.")
        else:
            show_result(df)

# =========================================================================
# STEP 4: BATCH INFERENCE
# =========================================================================
elif step == "4. Batch Inference":
    st.title("Phase 4: Batch Inference")
    st.markdown("""
    Batch inference scores **all patients** using the registered model. Two methods:

    | Method | How | Best For |
    |--------|-----|----------|
    | `mv.run()` | Warehouse compute | Up to ~100K rows |
    | `mv.run_batch()` | SPCS + Ray (distributed) | Millions of rows |

    Results are written to `HEALTHCARE_ML.INFERENCE.BATCH_PREDICTIONS`.
    """)

    st.subheader("Batch Inference Code (SPCS)")
    if st.button("Show Batch Inference Python", key="batch_code"):
        st.code("""
from snowflake.ml.model._client.model.batch_inference_specs import (
    OutputSpec, SaveMode, JobSpec
)

# Get model from registry
mv = registry.get_model("READMISSION_PREDICTOR").version("V1")

# Read features from Dynamic Table
features_df = session.table("HEALTHCARE_ML.FEATURE_STORE.\\"PATIENT_CLINICAL_FEATURES$V1\\"")
features_only = features_df.select(FEATURE_COLUMNS)

# Launch distributed batch inference on SPCS
job = mv.run_batch(
    compute_pool="DEMO_POOL_CPU",
    X=features_only,
    output_spec=OutputSpec(
        stage_location="@HEALTHCARE_ML.INFERENCE.BATCH_OUTPUT/predictions/",
        mode=SaveMode.OVERWRITE
    ),
    job_spec=JobSpec(function_name="predict_proba")
)

# Results written to stage, then materialized as table
results_df = session.read.parquet(output_path)
results_df.write.save_as_table(
    "HEALTHCARE_ML.INFERENCE.BATCH_PREDICTIONS",
    mode="overwrite"
)
        """, language="python")

    st.markdown("---")
    st.subheader("View Batch Predictions")
    if st.button("Show Prediction Results", key="batch_results"):
        sql = f"SELECT * FROM {DB}.INFERENCE.BATCH_PREDICTIONS LIMIT 50"
        df = run_sql(sql)
        show_result(df)

    if st.button("Prediction Summary Statistics", key="batch_stats"):
        sql = f"""SELECT
    COUNT(*) AS TOTAL_SCORED,
    ROUND(AVG(PROB_READMITTED), 4) AS AVG_READMISSION_PROB,
    ROUND(MIN(PROB_READMITTED), 4) AS MIN_PROB,
    ROUND(MAX(PROB_READMITTED), 4) AS MAX_PROB,
    SUM(CASE WHEN PROB_READMITTED > 0.5 THEN 1 ELSE 0 END) AS HIGH_RISK_COUNT,
    ROUND(SUM(CASE WHEN PROB_READMITTED > 0.5 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1)
        AS HIGH_RISK_PCT
FROM {DB}.INFERENCE.BATCH_PREDICTIONS"""
        df = run_sql(sql)
        show_result(df)

    st.markdown("---")
    st.subheader("Risk Distribution")
    if st.button("Show Risk Tier Breakdown", key="risk_tiers"):
        sql = f"""SELECT
    CASE
        WHEN PROB_READMITTED >= 0.7 THEN 'HIGH (>=0.7)'
        WHEN PROB_READMITTED >= 0.4 THEN 'MEDIUM (0.4-0.7)'
        ELSE 'LOW (<0.4)'
    END AS RISK_TIER,
    COUNT(*) AS PATIENT_COUNT,
    ROUND(AVG(PROB_READMITTED), 4) AS AVG_PROBABILITY
FROM {DB}.INFERENCE.BATCH_PREDICTIONS
GROUP BY RISK_TIER
ORDER BY AVG_PROBABILITY DESC"""
        df = run_sql(sql)
        show_result(df)

# =========================================================================
# STEP 5: REAL-TIME INFERENCE
# =========================================================================
elif step == "5. Real-Time Inference":
    st.title("Phase 5: Real-Time Inference")
    st.markdown("""
    For single-patient lookups, the **Online Feature Store** provides sub-second
    feature retrieval (1-min target lag). Combined with `mv.run()`, this enables
    real-time risk scoring at the point of care.
    """)

    st.subheader("Real-Time Inference Code")
    if st.button("Show Real-Time Python", key="rt_code"):
        st.code("""
from snowflake.ml.feature_store import FeatureStore

fs = FeatureStore(session=session,
                  database="HEALTHCARE_ML",
                  name="FEATURE_STORE")

# Get the feature view
fv = fs.get_feature_view("PATIENT_CLINICAL_FEATURES", "V1")

# Single-patient lookup via Online Feature Store
spine_df = session.create_dataframe(
    [{"PATIENT_ID": "P0042", "ADMISSION_ID": "A00123"}]
)
features = fv.retrieve_feature_values(spine_df)

# Score with the registered model
mv = registry.get_model("READMISSION_PREDICTOR").version("V1")
prediction = mv.run(features.select(FEATURE_COLUMNS),
                    function_name="predict_proba")
        """, language="python")

    st.markdown("---")
    st.subheader("Look Up a Patient")

    patient_sql = f"""SELECT DISTINCT PATIENT_ID
FROM {FEATURE_TABLE}
ORDER BY PATIENT_ID
LIMIT 100"""
    patient_list = session.sql(patient_sql).to_pandas()["PATIENT_ID"].tolist()

    selected_patient = st.selectbox("Select Patient ID", patient_list[:50])

    if st.button("Retrieve Features for Patient", key="rt_lookup"):
        sql = f"""SELECT *
FROM {FEATURE_TABLE}
WHERE PATIENT_ID = '{selected_patient}'
ORDER BY EVENT_TIMESTAMP DESC
LIMIT 1"""
        df = run_sql(sql)
        show_result(df)

    if st.button("Score Patient Now (Real-Time)", key="rt_predict"):
        with st.spinner("Running real-time inference..."):
            sql = f"""
WITH patient_features AS (
    SELECT * FROM {FEATURE_TABLE}
    WHERE PATIENT_ID = '{selected_patient}'
    ORDER BY EVENT_TIMESTAMP DESC
    LIMIT 1
)
SELECT 
    PATIENT_ID,
    ADMISSION_ID,
    READMITTED_30D AS ACTUAL_OUTCOME,
    {DB}.MODEL_REGISTRY.READMISSION_PREDICTOR!PREDICT_PROBA(
        AGE, GENDER_ENC, INSURANCE_ENC, HAS_PCP_FLAG,
        LENGTH_OF_STAY, NUM_PROCEDURES, NUM_DIAGNOSES, DIAGNOSIS_RISK_SCORE,
        DISPOSITION_RISK_SCORE, ED_ADMISSION, HEART_RATE, SYSTOLIC_BP, DIASTOLIC_BP,
        TEMPERATURE, RESPIRATORY_RATE, O2_SATURATION, BLOOD_GLUCOSE, CREATININE,
        HEMOGLOBIN, WBC_COUNT, SODIUM, POTASSIUM, BNP, ABNORMAL_HR, ABNORMAL_BP,
        LOW_O2, HIGH_CREATININE, LOW_HEMOGLOBIN, HIGH_BNP, ABNORMAL_GLUCOSE,
        PRIOR_ADMISSIONS_6M, PRIOR_READMISSIONS, AVG_PRIOR_LOS
    ) AS PREDICTION_RESULT
FROM patient_features"""
            df = run_sql(sql)
            if not df.empty:
                show_result(df)
                st.markdown("**Interpretation:** The `PREDICTION_RESULT` column contains the probability scores from the model.")

# =========================================================================
# STEP 6: GIT INTEGRATION
# =========================================================================
elif step == "6. Git Integration":
    st.title("Phase 6: Git Integration")
    st.markdown("""
    Snowflake connects directly to GitHub via a **Git Repository** object.
    The repo is accessible as a stage (`@repo/branches/main/`), and Python scripts
    can be executed with `EXECUTE IMMEDIATE FROM`.
    """)

    st.subheader("Git Integration Setup SQL")
    if st.button("Show Setup Commands", key="git_setup"):
        st.code("""
-- 1. API Integration (account-level, one-time)
CREATE OR REPLACE API INTEGRATION GITHUB_API_INTEGRATION
    API_PROVIDER = git_https_api
    API_ALLOWED_PREFIXES = ('https://github.com/sfc-gh-moahmed/')
    ALLOWED_AUTHENTICATION_SECRETS = ALL
    ENABLED = TRUE;

-- 2. Git Repository object
CREATE OR REPLACE GIT REPOSITORY
    HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO
    ORIGIN = 'https://github.com/sfc-gh-moahmed/healthcare-readmission-ml.git'
    API_INTEGRATION = GITHUB_API_INTEGRATION;

-- 3. Fetch latest code
ALTER GIT REPOSITORY
    HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO FETCH;

-- 4. Execute Python from Git
EXECUTE IMMEDIATE FROM
    @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO
    /branches/main/production/run_batch_inference.py;
        """, language="sql")

    st.markdown("---")
    st.subheader("Browse Repository")

    if st.button("Fetch Latest from GitHub", key="git_fetch"):
        sql = f"ALTER GIT REPOSITORY {GIT_REPO} FETCH"
        with st.spinner("Fetching from GitHub..."):
            run_sql(sql)
        st.success("Fetch complete -- Snowflake now has the latest code from GitHub.")

    if st.button("Show Branches", key="git_branches"):
        sql = f"SHOW GIT BRANCHES IN {GIT_REPO}"
        df = run_sql(sql)
        show_result(df)

    if st.button("List Files on Main Branch", key="git_list_main"):
        sql = f"LIST @{GIT_REPO}/branches/main/"
        df = run_sql(sql)
        show_result(df)

    browse_dir = st.selectbox("Browse directory", [
        "production/", "src/", "notebooks/", "scripts/", "artifacts/"
    ])
    if st.button("List Directory Contents", key="git_list_dir"):
        sql = f"LIST @{GIT_REPO}/branches/main/{browse_dir}"
        df = run_sql(sql)
        show_result(df)

    st.markdown("---")
    st.subheader("Read File Contents")
    file_path = st.text_input("File path (relative to repo root)",
                              value="src/config.py")
    if st.button("Read File from Git", key="git_read_file"):
        sql = f"SELECT $1 AS FILE_CONTENT FROM @{GIT_REPO}/branches/main/{file_path}"
        df = run_sql(sql)
        if not df.empty:
            content = "\n".join(df["FILE_CONTENT"].astype(str).tolist())
            st.code(content, language="python")

# =========================================================================
# STEP 7: PRODUCTION TASKS
# =========================================================================
elif step == "7. Production Tasks":
    st.title("Phase 7: Production Scheduling")
    st.markdown("""
    Two chained **Snowflake Tasks** automate the pipeline:

    1. **GIT_FETCH_TASK** -- runs every 60 minutes, pulls latest code from GitHub
    2. **BATCH_SCORING_TASK** -- runs *after* each fetch, executes batch inference from Git

    This ensures every scoring run uses the latest committed production code.
    """)

    st.subheader("Task Definitions")
    if st.button("Show Task SQL", key="task_sql"):
        st.code("""
-- Task 1: Keep code in sync (every 60 min)
CREATE OR REPLACE TASK HEALTHCARE_ML.TASKS.GIT_FETCH_TASK
    WAREHOUSE = HEALTHCARE_ML_WH
    SCHEDULE  = '60 MINUTE'
AS
    ALTER GIT REPOSITORY
        HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO FETCH;

-- Task 2: Score patients (chained after fetch)
CREATE OR REPLACE TASK HEALTHCARE_ML.TASKS.BATCH_SCORING_TASK
    WAREHOUSE = HEALTHCARE_ML_WH
    AFTER HEALTHCARE_ML.TASKS.GIT_FETCH_TASK
AS
    EXECUTE IMMEDIATE FROM
        @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO
        /branches/main/production/run_batch_inference.py;

-- Enable tasks (child first, then root)
ALTER TASK BATCH_SCORING_TASK RESUME;
ALTER TASK GIT_FETCH_TASK RESUME;
        """, language="sql")

    st.markdown("---")
    st.subheader("Task Status")
    if st.button("Show Current Tasks", key="show_tasks"):
        sql = f"SHOW TASKS IN {DB}.TASKS"
        df = run_sql(sql)
        show_result(df)

    st.markdown("---")
    st.subheader("Task Run History")
    task_name = st.selectbox("Select Task", ["GIT_FETCH_TASK", "BATCH_SCORING_TASK"])
    if st.button("Show Run History (last 24h)", key="task_history"):
        sql = f"""SELECT NAME, STATE, SCHEDULED_TIME, COMPLETED_TIME,
       ERROR_CODE, ERROR_MESSAGE
FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
    TASK_NAME => '{task_name}',
    SCHEDULED_TIME_RANGE_START => DATEADD('HOUR', -24, CURRENT_TIMESTAMP())
))
ORDER BY SCHEDULED_TIME DESC
LIMIT 20"""
        df = run_sql(sql)
        show_result(df)

    st.markdown("---")
    st.subheader("RBAC -- Team Access Control")
    if st.button("Show Grants to ML_ENGINEER Role", key="rbac_grants"):
        sql = "SHOW GRANTS TO ROLE ML_ENGINEER"
        df = run_sql(sql)
        show_result(df)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Healthcare 30-Day Readmission Prediction | "
    "Powered by Snowflake ML Platform | "
    "github.com/sfc-gh-moahmed/healthcare-readmission-ml"
)
