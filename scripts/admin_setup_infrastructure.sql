/*
=============================================================================
  Healthcare Readmission ML — Admin Infrastructure Setup
  Creates all databases, schemas, warehouses, and source tables
=============================================================================
  Run as: ACCOUNTADMIN
  Prerequisites: None
  
  This script sets up the Snowflake infrastructure needed for the 
  Healthcare 30-Day Readmission ML pipeline. After running this script,
  proceed to load data via CSV upload (PUT + COPY INTO) or run 
  Notebook 02 to upload from artifacts/.
=============================================================================
*/

USE ROLE ACCOUNTADMIN;

-- ==========================================================================
-- 1. CREATE DATABASE
-- ==========================================================================
CREATE DATABASE IF NOT EXISTS HEALTHCARE_ML
    COMMENT = 'Healthcare 30-day readmission prediction ML pipeline';

-- ==========================================================================
-- 2. CREATE SCHEMAS
-- ==========================================================================
CREATE SCHEMA IF NOT EXISTS HEALTHCARE_ML.RAW_DATA
    COMMENT = 'Source EHR data: patients, admissions, clinical measurements';

CREATE SCHEMA IF NOT EXISTS HEALTHCARE_ML.FEATURE_STORE
    COMMENT = 'Snowflake Feature Store: entities, feature views, dynamic tables';

CREATE SCHEMA IF NOT EXISTS HEALTHCARE_ML.MODEL_REGISTRY
    COMMENT = 'Snowflake Model Registry: trained models and versions';

CREATE SCHEMA IF NOT EXISTS HEALTHCARE_ML.INFERENCE
    COMMENT = 'Batch and real-time inference outputs';

CREATE SCHEMA IF NOT EXISTS HEALTHCARE_ML.GIT_INTEGRATION
    COMMENT = 'Git repository integration objects for ML pipeline code';

CREATE SCHEMA IF NOT EXISTS HEALTHCARE_ML.TASKS
    COMMENT = 'Scheduled pipeline tasks (git fetch, batch scoring)';

-- ==========================================================================
-- 3. CREATE WAREHOUSE
-- ==========================================================================
CREATE WAREHOUSE IF NOT EXISTS HEALTHCARE_ML_WH
    WAREHOUSE_SIZE = 'SMALL'
    AUTO_SUSPEND = 120
    AUTO_RESUME = TRUE
    COMMENT = 'Warehouse for Healthcare ML pipeline (training, inference, feature engineering)';

-- ==========================================================================
-- 4. CREATE SOURCE TABLES — RAW_DATA
-- ==========================================================================
USE SCHEMA HEALTHCARE_ML.RAW_DATA;

-- Table 1: PATIENTS — Patient demographics
CREATE TABLE IF NOT EXISTS HEALTHCARE_ML.RAW_DATA.PATIENTS (
    PATIENT_ID    VARCHAR(10)   NOT NULL,
    AGE           INTEGER,
    GENDER        VARCHAR(1),
    INSURANCE_TYPE VARCHAR(20),
    ZIP_CODE      VARCHAR(10),
    HAS_PCP       BOOLEAN,
    CONSTRAINT PK_PATIENTS PRIMARY KEY (PATIENT_ID)
);

-- Table 2: ADMISSIONS — Hospital admission records
CREATE TABLE IF NOT EXISTS HEALTHCARE_ML.RAW_DATA.ADMISSIONS (
    ADMISSION_ID          VARCHAR(10)   NOT NULL,
    PATIENT_ID            VARCHAR(10)   NOT NULL,
    ADMIT_DATE            DATE,
    DISCHARGE_DATE        DATE,
    LENGTH_OF_STAY        INTEGER,
    PRIMARY_DIAGNOSIS     VARCHAR(50),
    NUM_PROCEDURES        INTEGER,
    NUM_DIAGNOSES         INTEGER,
    DISCHARGE_DISPOSITION VARCHAR(20),
    ED_ADMISSION          INTEGER,
    READMITTED_30D        INTEGER,
    CONSTRAINT PK_ADMISSIONS PRIMARY KEY (ADMISSION_ID)
);

-- Table 3: CLINICAL_MEASUREMENTS — Vitals and labs at discharge
CREATE TABLE IF NOT EXISTS HEALTHCARE_ML.RAW_DATA.CLINICAL_MEASUREMENTS (
    ADMISSION_ID      VARCHAR(10)   NOT NULL,
    PATIENT_ID        VARCHAR(10)   NOT NULL,
    MEASUREMENT_DATE  DATE,
    HEART_RATE        FLOAT,
    SYSTOLIC_BP       FLOAT,
    DIASTOLIC_BP      FLOAT,
    TEMPERATURE       FLOAT,
    RESPIRATORY_RATE  FLOAT,
    O2_SATURATION     FLOAT,
    BLOOD_GLUCOSE     FLOAT,
    CREATININE        FLOAT,
    HEMOGLOBIN        FLOAT,
    WBC_COUNT         FLOAT,
    SODIUM            FLOAT,
    POTASSIUM         FLOAT,
    BNP               FLOAT
);

-- Table 4: TRAINING_DATA — Pre-engineered features for model training
CREATE TABLE IF NOT EXISTS HEALTHCARE_ML.RAW_DATA.TRAINING_DATA (
    AGE                   INTEGER,
    GENDER_ENC            INTEGER,
    INSURANCE_ENC         INTEGER,
    HAS_PCP_FLAG          INTEGER,
    LENGTH_OF_STAY        INTEGER,
    NUM_PROCEDURES        INTEGER,
    NUM_DIAGNOSES         INTEGER,
    DIAGNOSIS_RISK_SCORE  INTEGER,
    DISPOSITION_RISK_SCORE INTEGER,
    ED_ADMISSION          INTEGER,
    HEART_RATE            FLOAT,
    SYSTOLIC_BP           FLOAT,
    DIASTOLIC_BP          FLOAT,
    TEMPERATURE           FLOAT,
    RESPIRATORY_RATE      FLOAT,
    O2_SATURATION         FLOAT,
    BLOOD_GLUCOSE         FLOAT,
    CREATININE            FLOAT,
    HEMOGLOBIN            FLOAT,
    WBC_COUNT             FLOAT,
    SODIUM                FLOAT,
    POTASSIUM             FLOAT,
    BNP                   FLOAT,
    ABNORMAL_HR           INTEGER,
    ABNORMAL_BP           INTEGER,
    LOW_O2                INTEGER,
    HIGH_CREATININE       INTEGER,
    LOW_HEMOGLOBIN        INTEGER,
    HIGH_BNP              INTEGER,
    ABNORMAL_GLUCOSE      INTEGER,
    PRIOR_ADMISSIONS_6M   INTEGER,
    PRIOR_READMISSIONS    INTEGER,
    AVG_PRIOR_LOS         FLOAT,
    READMITTED_30D        INTEGER
);

-- Table 5: TEST_DATA — Pre-engineered features for model evaluation
CREATE TABLE IF NOT EXISTS HEALTHCARE_ML.RAW_DATA.TEST_DATA (
    AGE                   INTEGER,
    GENDER_ENC            INTEGER,
    INSURANCE_ENC         INTEGER,
    HAS_PCP_FLAG          INTEGER,
    LENGTH_OF_STAY        INTEGER,
    NUM_PROCEDURES        INTEGER,
    NUM_DIAGNOSES         INTEGER,
    DIAGNOSIS_RISK_SCORE  INTEGER,
    DISPOSITION_RISK_SCORE INTEGER,
    ED_ADMISSION          INTEGER,
    HEART_RATE            FLOAT,
    SYSTOLIC_BP           FLOAT,
    DIASTOLIC_BP          FLOAT,
    TEMPERATURE           FLOAT,
    RESPIRATORY_RATE      FLOAT,
    O2_SATURATION         FLOAT,
    BLOOD_GLUCOSE         FLOAT,
    CREATININE            FLOAT,
    HEMOGLOBIN            FLOAT,
    WBC_COUNT             FLOAT,
    SODIUM                FLOAT,
    POTASSIUM             FLOAT,
    BNP                   FLOAT,
    ABNORMAL_HR           INTEGER,
    ABNORMAL_BP           INTEGER,
    LOW_O2                INTEGER,
    HIGH_CREATININE       INTEGER,
    LOW_HEMOGLOBIN        INTEGER,
    HIGH_BNP              INTEGER,
    ABNORMAL_GLUCOSE      INTEGER,
    PRIOR_ADMISSIONS_6M   INTEGER,
    PRIOR_READMISSIONS    INTEGER,
    AVG_PRIOR_LOS         FLOAT,
    READMITTED_30D        INTEGER
);

-- ==========================================================================
-- 5. CREATE FILE FORMAT AND STAGE FOR CSV UPLOAD
-- ==========================================================================
CREATE FILE FORMAT IF NOT EXISTS HEALTHCARE_ML.RAW_DATA.CSV_FORMAT
    TYPE = 'CSV'
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    SKIP_HEADER = 1
    NULL_IF = ('', 'NULL', 'None')
    TRIM_SPACE = TRUE;

CREATE STAGE IF NOT EXISTS HEALTHCARE_ML.RAW_DATA.DATA_UPLOAD_STAGE
    FILE_FORMAT = HEALTHCARE_ML.RAW_DATA.CSV_FORMAT
    COMMENT = 'Stage for uploading CSV source data files';

-- ==========================================================================
-- 6. DATA LOADING INSTRUCTIONS
-- ==========================================================================
/*
  After creating the infrastructure above, load data using one of:

  OPTION A: PUT + COPY INTO (from local machine via SnowSQL or Snowsight)
  -----------------------------------------------------------------------
  -- Upload CSVs to stage
  PUT file:///path/to/artifacts/patients.csv @HEALTHCARE_ML.RAW_DATA.DATA_UPLOAD_STAGE/patients AUTO_COMPRESS=TRUE;
  PUT file:///path/to/artifacts/admissions.csv @HEALTHCARE_ML.RAW_DATA.DATA_UPLOAD_STAGE/admissions AUTO_COMPRESS=TRUE;
  PUT file:///path/to/artifacts/clinical.csv @HEALTHCARE_ML.RAW_DATA.DATA_UPLOAD_STAGE/clinical AUTO_COMPRESS=TRUE;
  PUT file:///path/to/artifacts/training_data.csv @HEALTHCARE_ML.RAW_DATA.DATA_UPLOAD_STAGE/training AUTO_COMPRESS=TRUE;
  PUT file:///path/to/artifacts/test_data.csv @HEALTHCARE_ML.RAW_DATA.DATA_UPLOAD_STAGE/test AUTO_COMPRESS=TRUE;

  -- Load into tables
  COPY INTO HEALTHCARE_ML.RAW_DATA.PATIENTS FROM @HEALTHCARE_ML.RAW_DATA.DATA_UPLOAD_STAGE/patients
      FILE_FORMAT = HEALTHCARE_ML.RAW_DATA.CSV_FORMAT ON_ERROR = 'CONTINUE';

  COPY INTO HEALTHCARE_ML.RAW_DATA.ADMISSIONS FROM @HEALTHCARE_ML.RAW_DATA.DATA_UPLOAD_STAGE/admissions
      FILE_FORMAT = HEALTHCARE_ML.RAW_DATA.CSV_FORMAT ON_ERROR = 'CONTINUE';

  COPY INTO HEALTHCARE_ML.RAW_DATA.CLINICAL_MEASUREMENTS FROM @HEALTHCARE_ML.RAW_DATA.DATA_UPLOAD_STAGE/clinical
      FILE_FORMAT = HEALTHCARE_ML.RAW_DATA.CSV_FORMAT ON_ERROR = 'CONTINUE';

  COPY INTO HEALTHCARE_ML.RAW_DATA.TRAINING_DATA FROM @HEALTHCARE_ML.RAW_DATA.DATA_UPLOAD_STAGE/training
      FILE_FORMAT = HEALTHCARE_ML.RAW_DATA.CSV_FORMAT ON_ERROR = 'CONTINUE';

  COPY INTO HEALTHCARE_ML.RAW_DATA.TEST_DATA FROM @HEALTHCARE_ML.RAW_DATA.DATA_UPLOAD_STAGE/test
      FILE_FORMAT = HEALTHCARE_ML.RAW_DATA.CSV_FORMAT ON_ERROR = 'CONTINUE';

  OPTION B: Run Notebook 02 (02_snowflake_setup.ipynb)
  -----------------------------------------------------------------------
  The notebook uses Snowpark to upload CSVs from artifacts/ and also
  creates the Feature Store entity, feature view (Dynamic Table), and
  enables the Online Feature Store.

  OPTION C: Snowsight Upload
  -----------------------------------------------------------------------
  Use Snowsight UI: Data > Add Data > Load Data into Table.
  Select each CSV and map to the corresponding table.
*/

-- ==========================================================================
-- 7. VERIFICATION
-- ==========================================================================
SELECT 'HEALTHCARE_ML infrastructure setup complete' AS STATUS;

SHOW SCHEMAS IN DATABASE HEALTHCARE_ML;

SELECT TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME, ROW_COUNT
FROM HEALTHCARE_ML.INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'RAW_DATA'
ORDER BY TABLE_NAME;
