# Healthcare 30-Day Readmission Prediction — Project Documentation

## Project Overview

This project demonstrates an end-to-end ML workflow using Snowflake's ML platform:
1. **Local training** in Jupyter notebooks (scikit-learn)
2. **Model deployment** to Snowflake Model Registry
3. **Feature management** via Snowflake Feature Store (backed by Dynamic Tables)
4. **Batch inference** using distributed `run_batch()` on Snowpark Container Services (SPCS)
5. **Real-time inference** using the Online Feature Store + warehouse inference

**Use Case:** Predict whether a hospital patient will be readmitted within 30 days of discharge, using demographics, clinical vitals, lab values, and admission history.

---

## Architecture Diagram

```
Local Environment                    Snowflake
┌─────────────────┐     ┌──────────────────────────────────────────────────┐
│  Jupyter         │     │                                                  │
│  Notebooks       │     │  ┌─────────────┐    ┌──────────────────┐        │
│                  │     │  │  RAW_DATA    │    │  FEATURE_STORE   │        │
│  01: Train model │────▶│  │  - PATIENTS  │───▶│  - Dynamic Table │        │
│  02: Upload data │     │  │  - ADMISSIONS│    │  - Online Store  │        │
│  03: Register    │     │  │  - CLINICAL  │    │  (1-min refresh) │        │
│  04: Batch inf.  │     │  └─────────────┘    └────────┬─────────┘        │
│  05: RT inf.     │     │                              │                  │
│                  │     │  ┌─────────────────┐         │                  │
│  GradientBoosting│     │  │ MODEL_REGISTRY  │         │                  │
│  Classifier      │────▶│  │ READMISSION_    │◀────────┘                  │
│  (scikit-learn)  │     │  │ PREDICTOR V1    │                            │
└─────────────────┘     │  └───────┬─────────┘                            │
                        │          │                                       │
                        │    ┌─────┴──────┐                                │
                        │    │            │                                 │
                        │  ┌─▼──────┐  ┌──▼──────────┐                    │
                        │  │Batch   │  │Real-time     │                    │
                        │  │run_batch│  │mv.run() +   │                    │
                        │  │on SPCS │  │Online Feature│                    │
                        │  │(Ray)   │  │Store         │                    │
                        │  └────────┘  └──────────────┘                    │
                        └──────────────────────────────────────────────────┘
```

---

## Process Workflow Diagrams

### End-to-End Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        END-TO-END ML PIPELINE                            │
└──────────────────────────────────────────────────────────────────────────┘

  PHASE 1: LOCAL DEVELOPMENT              PHASE 2: SNOWFLAKE DEPLOYMENT
  ─────────────────────────                ────────────────────────────

  ┌───────────────┐                        ┌──────────────────────┐
  │ Raw Data       │                        │ Snowflake RAW_DATA   │
  │ (synthetic)    │───── CSV upload ──────▶│ schema               │
  └───────┬───────┘                        └──────────┬───────────┘
          │                                           │
          ▼                                           ▼
  ┌───────────────┐                        ┌──────────────────────┐
  │ Feature        │                        │ Feature Store        │
  │ Engineering    │                        │ (Dynamic Table)      │
  │ (pandas)       │                        │ auto-refreshes SQL   │
  └───────┬───────┘                        └──────────┬───────────┘
          │                                           │
          ▼                                           ├──────────────────┐
  ┌───────────────┐                                   │                  │
  │ Model Training │                                   ▼                  ▼
  │ (scikit-learn) │                        ┌─────────────────┐  ┌──────────────┐
  └───────┬───────┘                        │ Online Feature   │  │ Offline       │
          │                                │ Store            │  │ Feature Store │
          ▼                                │ (key-value,      │  │ (Dynamic      │
  ┌───────────────┐                        │  1-min lag)      │  │  Table scan)  │
  │ Save Artifacts │                        └────────┬────────┘  └──────┬───────┘
  │ - model.joblib │                                 │                  │
  │ - metadata.json│                                 │                  │
  │ - CSVs         │                                 │                  │
  └───────┬───────┘                                  │                  │
          │                                          │                  │
          │         ┌──────────────────────┐         │                  │
          └────────▶│ Model Registry       │         │                  │
   log_model()      │ (READMISSION_        │         │                  │
                    │  PREDICTOR V1)       │◀────────┘                  │
                    └──────────┬───────────┘                            │
                               │                                        │
                         ┌─────┴──────┐                                 │
                         │            │                                  │
                         ▼            ▼                                  │
              ┌──────────────┐  ┌──────────────┐                        │
              │ mv.run()     │  │ mv.run_batch()│◀───────────────────────┘
              │ (warehouse)  │  │ (SPCS / Ray) │
              │ real-time    │  │ distributed  │
              └──────┬───────┘  └──────┬───────┘
                     │                 │
                     ▼                 ▼
              ┌──────────────┐  ┌──────────────┐
              │ Single-patient│  │ Parquet files │
              │ risk score   │  │ on stage      │
              │ (API response)│  │ (bulk table)  │
              └──────────────┘  └──────────────┘
```

### Model Registration Workflow (Joblib to Registry)

```
  LOCAL MACHINE                           SNOWFLAKE
  ─────────────                           ─────────

  readmission_model.joblib
         │
         ▼
  ┌──────────────────┐
  │ joblib.load()    │   Load the .joblib file back into
  │                  │   a live Python model object
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Prepare          │   Create a pandas DataFrame with
  │ sample_input_data│   100 rows of real feature data
  │ (pandas DF)      │   (used for schema inference)
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐    Snowpark Session     ┌─────────────────────┐
  │ registry         │ ──────────────────────▶ │ Snowflake Model     │
  │ .log_model(      │                         │ Registry             │
  │   model=model,   │    1. Serializes model  │                     │
  │   model_name=.., │    2. Uploads to stage  │ snow://model/       │
  │   version=..,    │    3. Detects functions  │  HEALTHCARE_ML.     │
  │   sample_input=..│    4. Creates UDFs       │  MODEL_REGISTRY.    │
  │   conda_deps=.., │    5. Stores metadata    │  READMISSION_       │
  │   metrics=..,    │                         │  PREDICTOR/V1/      │
  │   task=..        │                         │  ├── model.pkl      │
  │ )                │                         │  ├── model.yaml     │
  └────────┬─────────┘                         │  ├── conda.yml      │
           │                                   │  ├── functions/      │
           ▼                                   │  │   ├── predict.py  │
  ┌──────────────────┐                         │  │   ├── predict_    │
  │ Returns:         │                         │  │   │   proba.py    │
  │ ModelVersion obj │                         │  │   ├── explain.py  │
  │ (mv)             │                         │  │   └── ...         │
  │                  │                         │  └── MANIFEST.yml    │
  │ mv.run()         │                         └─────────────────────┘
  │ mv.run_batch()   │
  └──────────────────┘
```

### Inference Decision Tree

```
                    ┌─────────────────────────┐
                    │ Need to run inference?   │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ How many rows?           │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
         < 1 row             1-100K rows        > 100K rows
         (single patient)    (medium batch)     (large batch)
              │                  │                   │
              ▼                  ▼                   ▼
    ┌─────────────────┐ ┌───────────────┐ ┌──────────────────┐
    │ Online Feature   │ │ mv.run()      │ │ mv.run_batch()   │
    │ Store + mv.run() │ │ (warehouse)   │ │ (SPCS + Ray)     │
    │                  │ │               │ │                  │
    │ - Sub-second     │ │ - Simple      │ │ - Distributed    │
    │   feature lookup │ │ - No setup    │ │ - Multi-node     │
    │ - Warehouse      │ │ - Good for    │ │ - Parquet output │
    │   inference      │ │   scheduled   │ │ - First run slow │
    │ - Best for APIs  │ │   jobs        │ │   (image build)  │
    └─────────────────┘ └───────────────┘ └──────────────────┘
```

### Feature Store Data Flow

```
  RAW TABLES                    FEATURE STORE                    CONSUMERS
  ──────────                    ─────────────                    ─────────

  ┌──────────┐
  │ PATIENTS │──┐
  └──────────┘  │
                │    ┌────────────────────────────────┐
  ┌──────────┐  ├───▶│  Feature View SQL               │
  │ADMISSIONS│──┤    │  (JOIN + window functions +     │     ┌──────────────┐
  └──────────┘  │    │   CASE encodings)               │────▶│ Dynamic Table │
                │    └────────────────────────────────┘     │ (refreshes    │
  ┌──────────┐  │                                           │  every 5 min) │
  │ CLINICAL │──┘                                           └──────┬───────┘
  └──────────┘                                                     │
                                                          ┌────────┴────────┐
                                                          │                 │
                                                          ▼                 ▼
                                                 ┌──────────────┐  ┌──────────────┐
                                                 │ Online Store  │  │ Batch read   │
                                                 │ (key-value)   │  │ (full scan)  │
                                                 │ 1-min sync    │  │              │
                                                 └──────┬───────┘  └──────┬───────┘
                                                        │                 │
                                                        ▼                 ▼
                                                 ┌──────────────┐  ┌──────────────┐
                                                 │ Real-time    │  │ Batch        │
                                                 │ inference    │  │ inference    │
                                                 │ (1 patient)  │  │ (all rows)  │
                                                 └──────────────┘  └──────────────┘
```

### Diagram Summary

The diagrams above illustrate the key workflows in this project:

1. **End-to-End Pipeline Flow** — The complete path from local development through Snowflake deployment. Raw data and the trained model are uploaded independently. The Feature Store materializes features via a Dynamic Table that feeds both the Online Store (for real-time lookups) and batch reads (for bulk inference). The Model Registry serves both `mv.run()` (warehouse) and `mv.run_batch()` (SPCS/Ray).

2. **Model Registration Workflow** — Shows exactly what happens when you take a local `.joblib` file and register it. The key insight: you load the file into a live Python object with `joblib.load()`, prepare a `sample_input_data` DataFrame, and call `registry.log_model()`. Internally, Snowflake serializes with `cloudpickle`, infers the schema, auto-generates wrapper functions (`predict`, `predict_proba`, `explain`, etc.), and uploads everything to an internal model stage (`snow://model/...`).

3. **Inference Decision Tree** — How to choose the right inference method. Single-patient lookups use the Online Feature Store + `mv.run()`. Medium batches (up to ~100K rows) use `mv.run()` directly on the warehouse. Large-scale jobs (millions of rows) use `mv.run_batch()` on SPCS, which spins up a Ray cluster across multiple container nodes.

4. **Feature Store Data Flow** — Three raw tables (PATIENTS, ADMISSIONS, CLINICAL) are joined and transformed via the Feature View SQL into a single Dynamic Table with 33 engineered features. That Dynamic Table forks into two serving paths: the Online Store (key-value optimized, 1-min sync lag) for real-time point lookups, and direct table scans for batch inference.

---

## How to Import a Local Joblib Model into Snowflake Model Registry

This project uses the **native scikit-learn model object** approach — the simplest and most common
path for getting a locally-trained model into Snowflake. Below is a detailed guide covering this
approach and the alternative **Custom Model** approach.

### Approach 1: Native Model Object (Used in This Project)

This works for any model from a supported ML framework (scikit-learn, XGBoost, LightGBM, PyTorch,
TensorFlow, Hugging Face, etc.). You pass the live Python model object directly to `log_model()`.

#### Prerequisites

```
pip install snowflake-ml-python>=1.22.0 joblib
```

A Snowpark session connected to Snowflake with a database and schema for the registry.

#### Step-by-Step Process

**Step 1: Load the joblib file into a live model object**

```python
import joblib
model = joblib.load('artifacts/readmission_model.joblib')

# Verify it loaded correctly
print(type(model))  # <class 'sklearn.ensemble._gb.GradientBoostingClassifier'>
print(model.n_estimators)  # 200
```

The `joblib.load()` call deserializes the `.joblib` file back into a fully functional
scikit-learn model object in local memory. This is the same object you had after calling
`model.fit()` during training.

> **Important:** You do NOT upload the `.joblib` file directly to Snowflake. The Model Registry
> needs a live Python object, not a file path. The registry handles its own serialization
> internally (using `cloudpickle`).

**Step 2: Prepare sample input data**

```python
import pandas as pd
test_df = pd.read_csv('artifacts/test_data.csv')
FEATURE_COLUMNS = [...]  # your 33 feature column names

sample_input = test_df[FEATURE_COLUMNS].head(100)
```

The `sample_input_data` parameter is critical — the registry uses it to:
- Infer the input schema (column names, data types)
- Auto-detect which model methods exist (`predict`, `predict_proba`, etc.)
- Generate the function signatures for warehouse inference

Without it, the registry cannot create callable inference functions.

**Step 3: Connect to Snowflake and initialize the Registry**

```python
from snowflake.snowpark import Session
from snowflake.ml.registry import Registry

session = Session.builder.config("connection_name", "DEMO").create()
session.sql("USE DATABASE HEALTHCARE_ML").collect()
session.sql("USE SCHEMA MODEL_REGISTRY").collect()

registry = Registry(
    session=session,
    database_name="HEALTHCARE_ML",
    schema_name="MODEL_REGISTRY"
)
```

**Step 4: Log the model**

```python
from snowflake.ml.model import task as ml_task

mv = registry.log_model(
    model=model,                              # Live Python object (NOT a file path)
    model_name="READMISSION_PREDICTOR",       # Name in the registry
    version_name="V1",                        # Version identifier
    sample_input_data=sample_input,           # pandas DF for schema inference
    conda_dependencies=["scikit-learn"],      # Runtime dependencies
    comment="30-day readmission prediction",  # Human-readable description
    metrics={                                 # Tracked metrics (queryable later)
        "roc_auc": 0.552,
        "average_precision": 0.290,
    },
    task=ml_task.Task.TABULAR_BINARY_CLASSIFICATION  # Enables EXPLAIN function
)
```

**What happens inside `log_model()`:**

1. **Serialization** — The model object is serialized with `cloudpickle` and saved as `model.pkl`
2. **Schema inference** — `sample_input` is passed through `model.predict()`, `model.predict_proba()`, etc. to discover available methods and their output shapes
3. **Function generation** — For each detected method, a Python wrapper function is generated (e.g., `predict.py`, `predict_proba.py`)
4. **Dependency resolution** — `conda_dependencies` are combined with the registry's own requirements into `conda.yml`
5. **Upload** — All artifacts are uploaded to a Snowflake internal model stage: `snow://model/DB.SCHEMA.MODEL_NAME/versions/VERSION/`
6. **Registration** — Metadata (metrics, comment, task type) is stored in the model catalog

**Step 5: Use the registered model**

```python
# Retrieve the model version
mv = registry.get_model("READMISSION_PREDICTOR").version("V1")

# List auto-detected functions
mv.show_functions()
# ['PREDICT', 'PREDICT_PROBA', 'PREDICT_LOG_PROBA', 'DECISION_FUNCTION', 'EXPLAIN']

# Warehouse inference (returns pandas DataFrame)
predictions = mv.run(test_features, function_name="predict_proba")

# Distributed batch inference (runs on SPCS)
from snowflake.ml.model._client.model.batch_inference_specs import OutputSpec, SaveMode, JobSpec
job = mv.run_batch(
    compute_pool="DEMO_POOL_CPU",
    X=snowpark_df,
    output_spec=OutputSpec(stage_location="@MY_STAGE/output/", mode=SaveMode.OVERWRITE),
    job_spec=JobSpec(function_name="predict_proba")
)
```

#### What the Native Approach Supports

| Framework | Supported | Auto-detected Functions |
|-----------|-----------|------------------------|
| scikit-learn | Yes | predict, predict_proba, predict_log_proba, decision_function, explain |
| XGBoost | Yes | predict, predict_proba |
| LightGBM | Yes | predict, predict_proba |
| PyTorch | Yes | forward (mapped to predict) |
| TensorFlow/Keras | Yes | predict |
| Hugging Face | Yes | Varies by pipeline type |

---

### Approach 2: Custom Model (For Non-Standard Models or Custom Logic)

Use a **Custom Model** when your inference logic does not fit neatly into a single
`model.predict()` call — for example:

- Ensemble of multiple models
- Pre/post-processing that must run alongside inference
- Models from unsupported frameworks
- Business logic mixed with ML (e.g., rule-based overrides)
- Models that need to load additional artifacts (lookup tables, tokenizers, etc.)

#### When to Use Custom Model vs Native

| Scenario | Use Native | Use Custom Model |
|----------|-----------|-----------------|
| Single scikit-learn/XGBoost model | Yes | No |
| Model + preprocessing pipeline | Maybe (if using sklearn Pipeline) | Yes (if preprocessing is separate) |
| Ensemble of 3 models with voting | No | Yes |
| Model + business rules | No | Yes |
| Non-supported framework (e.g., ONNX) | No | Yes |
| Need to load auxiliary files | No | Yes |

#### Custom Model Implementation

```python
from snowflake.ml.model import custom_model
import pandas as pd
import joblib
import json

class ReadmissionPredictor(custom_model.CustomModel):
    """Custom model that wraps a GradientBoostingClassifier with
    pre-processing and post-processing logic."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        # Load model from artifacts (uploaded with the custom model)
        self.model = joblib.load(context.path("model_artifact"))
        # Load any additional config
        with open(context.path("metadata"), "r") as f:
            self.metadata = json.load(f)

    @custom_model.inference_api
    def predict_readmission(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Custom inference method with post-processing."""
        # Run model prediction
        probabilities = self.model.predict_proba(input_df.values)

        # Post-processing: add risk categorization
        results = pd.DataFrame({
            "READMISSION_PROBABILITY": probabilities[:, 1],
            "RISK_LEVEL": pd.cut(
                probabilities[:, 1],
                bins=[0, 0.3, 0.5, 1.0],
                labels=["LOW", "MEDIUM", "HIGH"]
            ).astype(str)
        })
        return results
```

#### Registering a Custom Model

```python
# Create the model context with artifacts
model_context = custom_model.ModelContext(
    artifacts={
        "model_artifact": "artifacts/readmission_model.joblib",
        "metadata": "artifacts/model_metadata.json"
    }
)

# Instantiate the custom model
custom_predictor = ReadmissionPredictor(model_context)

# Register in the same way
mv = registry.log_model(
    model=custom_predictor,
    model_name="READMISSION_PREDICTOR_CUSTOM",
    version_name="V1",
    sample_input_data=sample_input,
    conda_dependencies=["scikit-learn", "joblib"],
    comment="Custom readmission model with risk categorization"
)

# Inference — note the custom function name
results = mv.run(test_features, function_name="predict_readmission")
# Returns: READMISSION_PROBABILITY, RISK_LEVEL columns
```

#### Key Differences: Native vs Custom

```
  NATIVE MODEL APPROACH                    CUSTOM MODEL APPROACH
  ─────────────────────                    ──────────────────────

  joblib.load() → model object             Define a class extending CustomModel
         │                                         │
         ▼                                         ▼
  log_model(model=model)                   @inference_api decorated methods
         │                                 define your own function signatures
         ▼                                         │
  Auto-detects: predict,                           ▼
  predict_proba, explain, etc.             ModelContext packages .joblib +
         │                                 any other artifacts (JSON, CSVs,
         ▼                                 tokenizers, lookup tables)
  mv.run(df, function_name="predict")              │
                                                   ▼
                                           log_model(model=custom_predictor)
                                                   │
                                                   ▼
                                           mv.run(df, function_name=
                                                "predict_readmission")
                                           (your custom method name)
```

### Common Gotchas

1. **Do NOT pass a file path to `log_model()`** — It expects a live Python object, not
   `"artifacts/model.joblib"`. Load the file with `joblib.load()` first.

2. **`sample_input_data` must be a pandas DataFrame** — Not a Snowpark DataFrame, not a numpy
   array. Column names must match what the model expects.

3. **`conda_dependencies` are runtime deps, not local deps** — These are installed inside
   the Snowflake execution environment. If your model needs scikit-learn 1.7, specify
   `["scikit-learn==1.7.2"]`. Omitting this causes `ModuleNotFoundError` at inference time.

4. **Version names must be unique** — You cannot overwrite V1. Either use a new version name
   (V2) or delete the existing version first with `registry.delete_model("MODEL_NAME")`.

5. **`run_batch()` import path changed in recent versions** — In `snowflake-ml-python>=1.28.0`:
   ```python
   # Correct:
   from snowflake.ml.model._client.model.batch_inference_specs import OutputSpec, SaveMode, JobSpec
   # NOT: from snowflake.ml.model import OutputSpec
   ```

6. **`function_name` goes in `JobSpec`, not `run_batch()`** — For `run_batch()`:
   ```python
   # Correct:
   job = mv.run_batch(..., job_spec=JobSpec(function_name="predict_proba"))
   # NOT: mv.run_batch(..., function_name="predict_proba")
   ```

---

## Step-by-Step Walkthrough

### Step 1: Local Model Training (`01_local_training.ipynb`)

| Detail | Value |
|--------|-------|
| **What** | Generate synthetic healthcare data and train a GradientBoostingClassifier locally |
| **Snowflake Cost** | $0.00 (runs entirely on local machine) |
| **Why** | Data scientists typically prototype and iterate on models locally before deploying. Local training avoids compute costs during experimentation. |

**What was done:**
- Generated 3 synthetic tables: `PATIENTS` (5,000 rows), `ADMISSIONS` (10,772 rows), `CLINICAL_MEASUREMENTS` (10,772 rows)
- Engineered 33 features: demographics, admission details, clinical vitals/labs, abnormality flags, historical admission patterns
- Trained a `GradientBoostingClassifier` (200 estimators, max_depth=4) on 8,617 training samples
- Evaluated on 2,155 test samples: **ROC AUC = 0.552**, **Average Precision = 0.290**
- Saved all artifacts to `./artifacts/` (model joblib, CSVs, metadata JSON, evaluation plot)

**Model Details:**
- **Algorithm:** GradientBoostingClassifier (scikit-learn)
- **Features (33):** AGE, GENDER_ENC, INSURANCE_ENC, HAS_PCP_FLAG, LENGTH_OF_STAY, NUM_PROCEDURES, NUM_DIAGNOSES, DIAGNOSIS_RISK_SCORE, DISPOSITION_RISK_SCORE, ED_ADMISSION, HEART_RATE, SYSTOLIC_BP, DIASTOLIC_BP, TEMPERATURE, RESPIRATORY_RATE, O2_SATURATION, BLOOD_GLUCOSE, CREATININE, HEMOGLOBIN, WBC_COUNT, SODIUM, POTASSIUM, BNP, ABNORMAL_HR, ABNORMAL_BP, LOW_O2, HIGH_CREATININE, LOW_HEMOGLOBIN, HIGH_BNP, ABNORMAL_GLUCOSE, PRIOR_ADMISSIONS_6M, PRIOR_READMISSIONS, AVG_PRIOR_LOS
- **Target:** READMITTED_30D (binary, 25.4% positive rate)

---

### Step 2: Snowflake Setup — Data Upload, Feature Store, Online Store (`02_snowflake_setup.ipynb`)

| Detail | Value |
|--------|-------|
| **What** | Create database/schemas, upload data, set up Feature Store with Dynamic Tables, enable Online Feature Store |
| **Estimated Snowflake Cost** | ~$0.50–$1.00/day |
| **Why** | Centralizes data in Snowflake for governed access. Feature Store ensures consistent feature computation across training and inference. Dynamic Tables auto-refresh features. Online Store enables low-latency lookups. |

**Snowflake Objects Created:**

| Object | Type | Location | Purpose |
|--------|------|----------|---------|
| `HEALTHCARE_ML` | Database | — | Project container |
| `RAW_DATA` | Schema | `HEALTHCARE_ML` | Raw EHR tables |
| `FEATURE_STORE` | Schema | `HEALTHCARE_ML` | Feature Store objects |
| `MODEL_REGISTRY` | Schema | `HEALTHCARE_ML` | Model versions |
| `INFERENCE` | Schema | `HEALTHCARE_ML` | Batch output stage/tables |
| `HEALTHCARE_ML_WH` | Warehouse | — | XS warehouse, 60s auto-suspend |
| `PATIENTS` | Table | `RAW_DATA` | 5,000 patient demographics |
| `ADMISSIONS` | Table | `RAW_DATA` | 10,772 hospital admissions |
| `CLINICAL_MEASUREMENTS` | Table | `RAW_DATA` | 10,772 vitals/lab records |
| `TRAINING_DATA` | Table | `RAW_DATA` | 8,617 pre-engineered training rows |
| `TEST_DATA` | Table | `RAW_DATA` | 2,155 pre-engineered test rows |
| `PATIENT` | Entity | `FEATURE_STORE` | Join key: PATIENT_ID |
| `PATIENT_CLINICAL_FEATURES$V1` | Dynamic Table | `FEATURE_STORE` | 10,772 rows, 33 features, 5-min FULL refresh |
| `READMISSION_TRAINING_V1` | Dataset | `FEATURE_STORE` | Training dataset with lineage |

**How It Works:**

- **Feature Store:** The `FeatureStore` API registers entities (e.g., `PATIENT`) and feature views. A Feature View is backed by a SQL query that joins raw tables and computes features. Snowflake materializes this as a **Dynamic Table** that auto-refreshes.

- **Dynamic Table:** The `PATIENT_CLINICAL_FEATURES$V1` Dynamic Table runs the feature engineering SQL every 5 minutes in FULL refresh mode. This ensures downstream consumers always have fresh features without manual ETL.

- **Online Feature Store:** Enabled with `OnlineConfig(enable=True, target_lag='1 minute')`. This creates a low-latency, key-value optimized replica of the feature data, synchronized from the Dynamic Table. It enables sub-second feature lookups by patient ID for real-time inference.

**Cost Breakdown:**
| Component | Estimated Cost | Notes |
|-----------|---------------|-------|
| `HEALTHCARE_ML_WH` (XS) | ~$0.05/query burst | 1 credit/hr, auto-suspends in 60s |
| Dynamic Table refresh | ~$0.10–$0.20/day | XS compute every 5 min for small dataset |
| Online Feature Store | ~$0.20–$0.50/day | Serverless, charged per data volume synced |
| Storage (all tables) | <$0.01/day | ~50MB total compressed |

---

### Step 3: Model Registration (`03_model_registry.ipynb`)

| Detail | Value |
|--------|-------|
| **What** | Register the locally-trained model in Snowflake Model Registry and verify warehouse inference |
| **Estimated Snowflake Cost** | ~$0.10 (one-time upload + test queries) |
| **Why** | Model Registry provides versioning, governance, lineage tracking, and makes the model callable as a SQL function or Python API within Snowflake. No need to manage model artifacts externally. |

**What was done:**
- Loaded the local `readmission_model.joblib` and registered it via `registry.log_model()`
- Snowflake auto-detected 5 model functions: `PREDICT`, `PREDICT_PROBA`, `PREDICT_LOG_PROBA`, `DECISION_FUNCTION`, `EXPLAIN`
- Stored model metrics (ROC AUC, AP) and metadata
- Verified warehouse inference: **100% match** with local predictions on 20 test samples

**How It Works:**
- `Registry.log_model()` serializes the model, uploads it to a Snowflake internal stage, and creates callable functions
- `mv.run(df, function_name="predict")` runs inference on a Snowpark DataFrame using the warehouse
- The model is versioned — you can register V2, V3, etc. and compare metrics
- SQL access: `SHOW MODELS IN SCHEMA HEALTHCARE_ML.MODEL_REGISTRY` lists all registered models

**Key Code:**
```python
from snowflake.ml.registry import Registry
registry = Registry(session=session, database_name="HEALTHCARE_ML", schema_name="MODEL_REGISTRY")
mv = registry.log_model(model=model, model_name="READMISSION_PREDICTOR", version_name="V1",
                         sample_input_data=sample_input, conda_dependencies=["scikit-learn"],
                         task=ml_task.Task.TABULAR_BINARY_CLASSIFICATION)
# Inference:
predictions = mv.run(test_features, function_name="predict_proba")
```

---

### Step 4: Batch Inference with `run_batch()` on SPCS (`04_batch_inference.ipynb`)

| Detail | Value |
|--------|-------|
| **What** | Run distributed batch inference on all 10,772 admissions using SPCS compute pools |
| **Estimated Snowflake Cost** | ~$1.50–$3.00 per run |
| **Why** | `run_batch()` distributes inference across multiple nodes via Ray on SPCS. For large datasets (millions of rows), this is significantly faster than warehouse inference. It also runs on dedicated compute, avoiding warehouse contention. |

**What was done:**
- Created output stage `@HEALTHCARE_ML.INFERENCE.BATCH_OUTPUT`
- Loaded all features from the Feature View Dynamic Table
- Launched `mv.run_batch()` targeting `DEMO_POOL_CPU` compute pool
- Job builds a Docker image (first run), starts a Ray cluster, distributes predictions
- Results written to stage as Parquet files, then loaded into `BATCH_PREDICTIONS` table

**How It Works:**
- `run_batch()` is fundamentally different from `mv.run()`:
  - `mv.run()` → Runs on the Snowflake warehouse (single-node, good for <100K rows)
  - `mv.run_batch()` → Runs on SPCS compute pool (multi-node Ray cluster, good for millions+ rows)
- On first invocation, Snowflake builds a Docker image with all model dependencies (scikit-learn, etc.) — this takes ~10-15 minutes
- Subsequent runs reuse the cached image and complete much faster
- Output is written to a Snowflake stage in Parquet format

**Key Code:**
```python
from snowflake.ml.model._client.model.batch_inference_specs import OutputSpec, SaveMode, JobSpec
job = mv.run_batch(
    compute_pool="DEMO_POOL_CPU",
    X=features_only_df,
    output_spec=OutputSpec(
        stage_location="@HEALTHCARE_ML.INFERENCE.BATCH_OUTPUT/readmission_predictions/",
        mode=SaveMode.OVERWRITE
    ),
    job_spec=JobSpec(function_name="predict_proba")
)
```

**Cost Breakdown:**
| Component | Estimated Cost | Notes |
|-----------|---------------|-------|
| `DEMO_POOL_CPU` (CPU_X64_S) | ~$1.00–$2.00/run | 1 node minimum, billed per second while active |
| Docker image build (first run) | ~$0.50 | ~15 min of compute |
| Stage storage | <$0.01 | Parquet output ~200KB |

---

### Step 5: Real-Time Inference with Online Feature Store (`05_realtime_inference.ipynb`)

| Detail | Value |
|--------|-------|
| **What** | Retrieve features from Online Feature Store by patient ID and run real-time prediction |
| **Estimated Snowflake Cost** | ~$0.01–$0.05 per prediction |
| **Why** | In a clinical setting, when a patient is being discharged, the system needs an immediate readmission risk score. The Online Feature Store provides sub-second feature lookups, and `mv.run()` gives instant predictions. |

**What was done:**
- Retrieved features for 5 patients from Online Feature Store in **0.274 seconds**
- All 33 features returned correctly per patient
- Ran predictions via `mv.run()`:
  - P00001: 21.4% readmission probability (LOW risk)
  - P00002: 21.1% readmission probability (LOW risk)
  - P00003: 11.0% readmission probability (LOW risk)
- Built a `predict_readmission_risk()` function simulating a clinical decision support API
- Per-patient response time: ~18-20 seconds (includes warehouse cold-start; subsequent calls are faster)

**How It Works:**
- `fs.retrieve_feature_values(spine_df, features=[fv])` does a key-value lookup in the Online Feature Store
- Unlike querying the Dynamic Table directly, the Online Store is optimized for point lookups by entity key (PATIENT_ID)
- The Online Store syncs from the Dynamic Table with a 1-minute target lag
- Combined with `mv.run()`, this gives a complete real-time inference pipeline

**Key Code:**
```python
fs = FeatureStore(session=session, database="HEALTHCARE_ML", name="FEATURE_STORE",
                  default_warehouse="HEALTHCARE_ML_WH")
fv = fs.get_feature_view("PATIENT_CLINICAL_FEATURES", "V1")

# Retrieve features for a patient
spine_df = session.create_dataframe([[patient_id]], schema=["PATIENT_ID"])
features = fs.retrieve_feature_values(spine_df=spine_df, features=[fv], spine_timestamp_col=None)

# Predict
prediction = mv.run(model_input, function_name="predict_proba")
```

**Cost Breakdown:**
| Component | Estimated Cost | Notes |
|-----------|---------------|-------|
| Online Feature Store lookup | ~$0.001/query | Serverless, per-request pricing |
| Warehouse inference (mv.run) | ~$0.01–$0.05/query | XS warehouse, cold start adds latency |
| Total per prediction | ~$0.01–$0.05 | Dominated by warehouse compute time |

---

## Total Cost Summary

| Component | One-Time Cost | Recurring Cost (Daily) | Notes |
|-----------|--------------|----------------------|-------|
| Local training | $0.00 | — | Runs on laptop |
| Data upload | ~$0.05 | — | XS warehouse for a few minutes |
| Model registration | ~$0.10 | — | Upload + verification queries |
| Dynamic Table refresh | — | ~$0.10–$0.20 | Every 5 min, XS compute |
| Online Feature Store | — | ~$0.20–$0.50 | Serverless sync |
| Warehouse (HEALTHCARE_ML_WH) | — | $0.00–$0.50 | Only when queries run; auto-suspends |
| Batch inference (per run) | ~$1.50–$3.00 | — | SPCS compute pool |
| Real-time inference | — | ~$0.01–$0.05/query | On-demand |
| Storage | — | <$0.05 | ~50MB total |
| **TOTAL (idle)** | — | **~$0.30–$0.75/day** | DT refresh + Online Store only |
| **TOTAL (active use)** | — | **~$1.00–$5.00/day** | With queries + batch runs |

> **Note:** Costs are estimates based on Snowflake credit pricing (~$2–$4/credit depending on edition). Actual costs depend on Snowflake edition, cloud provider, region, and usage patterns. The SPCS compute pool (`DEMO_POOL_CPU`) has a 1-hour auto-suspend and should be manually suspended when not in use to avoid idle charges.

---

## Snowflake Components — Why Each One Matters

### Model Registry
- **What:** Centralized repository for ML models within Snowflake
- **Why we need it:** Without it, models live as files on someone's laptop or in an external MLflow server. The Model Registry keeps models versioned, governed, and directly callable from SQL or Python within Snowflake. No external infrastructure to manage.
- **How it works:** `log_model()` serializes your scikit-learn/XGBoost/PyTorch model, stores it in a Snowflake stage, and auto-generates callable functions. `mv.run()` deserializes and runs inference on the warehouse.

### Feature Store
- **What:** Managed feature computation and serving layer
- **Why we need it:** Features must be computed consistently between training and inference (train-serve skew is a top ML production failure). The Feature Store computes features once via SQL, materializes them in a Dynamic Table, and serves them for both batch and real-time inference.
- **How it works:** You define entities (PATIENT) and feature views (SQL queries). Snowflake materializes feature views as Dynamic Tables that auto-refresh on a schedule.

### Dynamic Tables
- **What:** Declarative, auto-refreshing materialized views
- **Why we need it:** Raw data changes as new admissions arrive. Dynamic Tables automatically re-compute features without manual ETL jobs or orchestration. The 5-minute refresh ensures features are always current.
- **How it works:** You define the SQL transformation. Snowflake tracks data dependencies and refreshes the table on the configured schedule (5 min in our case). FULL refresh mode recomputes all rows; INCREMENTAL mode (for simpler transformations) only processes new/changed rows.

### Online Feature Store
- **What:** Low-latency key-value store for feature serving
- **Why we need it:** Dynamic Tables are optimized for analytical queries (scans), not point lookups. When a real-time system needs features for one patient, the Online Store provides sub-second key-value lookups instead of scanning the entire table.
- **How it works:** Snowflake maintains an internal key-value store synced from the Dynamic Table with a configurable lag (1 minute in our case). `retrieve_feature_values()` does a direct key lookup.

### Snowpark Container Services (SPCS)
- **What:** Managed container compute within Snowflake
- **Why we need it:** `mv.run()` on a warehouse is single-node and limited for large datasets. `run_batch()` uses SPCS to spin up a Ray cluster across multiple container nodes, distributing inference across them. This scales to millions of rows.
- **How it works:** On first call, Snowflake builds a Docker image with model dependencies, pushes it to an internal registry, then launches containers on the compute pool. Ray distributes the data across nodes, each running inference in parallel.

---

## Project File Structure

```
healthcare-readmission-ml/
├── 01_local_training.ipynb          # Synthetic data generation + model training
├── 02_snowflake_setup.ipynb         # Data upload, Feature Store, Dynamic Tables, Online Store
├── 03_model_registry.ipynb          # Model registration and warehouse inference verification
├── 04_batch_inference.ipynb         # Distributed batch inference via run_batch() on SPCS
├── 05_realtime_inference.ipynb      # Real-time inference via Online Feature Store
├── PROJECT_DOCUMENTATION.md         # This file
├── requirements.txt                 # Python dependencies
└── artifacts/
    ├── readmission_model.joblib     # Trained GradientBoostingClassifier (415KB)
    ├── patients.csv                 # 5,000 patient records
    ├── admissions.csv               # 10,772 admission records
    ├── clinical.csv                 # 10,772 clinical measurement records
    ├── training_data.csv            # 8,617 pre-engineered training rows
    ├── test_data.csv                # 2,155 pre-engineered test rows
    ├── model_metadata.json          # Feature columns, metrics, configuration
    └── model_evaluation.png         # Confusion matrix + feature importance plot
```

---

## Snowflake Connection Details

| Setting | Value |
|---------|-------|
| Connection profile | `DEMO` |
| Account | `sfsenorthamerica-moizahmeddemo` |
| Role | `ACCOUNTADMIN` |
| Database | `HEALTHCARE_ML` |
| Warehouse | `HEALTHCARE_ML_WH` (XS, 60s auto-suspend) |
| Compute Pool | `DEMO_POOL_CPU` (CPU_X64_S, 1-10 nodes) |

---

## Verification Status

| Notebook | Component | Status | Evidence |
|----------|-----------|--------|----------|
| 01 | Local model training | VERIFIED | ROC AUC=0.552, model serialization round-trip confirmed |
| 02 | Data upload to Snowflake | VERIFIED | Row counts match: 5000/10772/10772/8617/2155 |
| 02 | Feature Store + Dynamic Table | VERIFIED | 10,772 rows, 33 features, ACTIVE refresh state |
| 02 | Online Feature Store | VERIFIED | Enabled with 1-min target lag |
| 03 | Model Registry | VERIFIED | 5 functions auto-registered, 100% prediction match vs local |
| 04 | Batch inference (run_batch) | VERIFIED | 10,772 predictions on SPCS (Ray), 507 high-risk, avg prob 25.6% |
| 05 | Real-time inference | VERIFIED | Feature retrieval 0.274s, predictions correct for 5 patients |
