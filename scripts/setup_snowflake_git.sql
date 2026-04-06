-- =============================================================================
-- Healthcare ML — Snowflake Git Integration Setup
-- =============================================================================
--
-- This script configures Snowflake to connect to the healthcare-readmission-ml
-- Git repository, enabling the data science team to:
--
--   1. Pull/fetch the latest code directly into Snowflake
--   2. Browse repo files from SQL or the Snowsight UI
--   3. Execute production Python scripts via EXECUTE IMMEDIATE FROM
--   4. Schedule recurring batch inference via Snowflake TASKs
--
-- PREREQUISITE: You already have the GITHUB_RESEARCH_CHOP_EDU_API configured for
-- https://github.research.chop.edu/analytics/. This script reuses that integration
-- (CHOP's GitHub Enterprise API integration with allowed prefix https://github.research.chop.edu/analytics).
--
-- Run this script once to set up the integration. After that, use
--   ALTER GIT REPOSITORY ... FETCH
-- to pull the latest code whenever you push changes.
-- =============================================================================

USE ROLE ACCOUNTADMIN;
USE DATABASE HEALTHCARE_ML;

-- -----------------------------------------------------------------------------
-- Step 1: Create a schema for Git integration objects
-- -----------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS HEALTHCARE_ML.GIT_INTEGRATION
    COMMENT = 'Git repository integration objects for ML pipeline code';

USE SCHEMA GIT_INTEGRATION;

-- -----------------------------------------------------------------------------
-- Step 2: (Optional) Create a Git secret for private repos
-- -----------------------------------------------------------------------------
-- The GITHUB_RESEARCH_CHOP_EDU_API allows HTTPS access to repos under
-- https://github.research.chop.edu/analytics/. If the repo is PUBLIC, no secret
-- is needed. For PRIVATE repos, create a secret with a GitHub PAT:
--
-- CREATE OR REPLACE SECRET HEALTHCARE_ML.GIT_INTEGRATION.GITHUB_SECRET
--     TYPE = password
--     USERNAME = '<your_chop_github_username>'
--     PASSWORD = '<YOUR_GITHUB_PERSONAL_ACCESS_TOKEN>';
--
-- Then add GIT_CREDENTIALS below.

-- -----------------------------------------------------------------------------
-- Step 3: Create the Git Repository object
-- -----------------------------------------------------------------------------
-- This connects Snowflake to the GitHub repo. Once created, Snowflake
-- maintains a read-only clone that you can browse and execute files from.

CREATE OR REPLACE GIT REPOSITORY HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO
    ORIGIN = 'https://github.research.chop.edu/analytics/healthcare-readmission-ml.git'
    API_INTEGRATION = GITHUB_RESEARCH_CHOP_EDU_API
    -- GIT_CREDENTIALS = HEALTHCARE_ML.GIT_INTEGRATION.GITHUB_SECRET  -- uncomment for private repos
    COMMENT = 'Healthcare 30-day readmission ML pipeline — notebooks, src, and production code';

-- -----------------------------------------------------------------------------
-- Step 4: Fetch the latest code from the remote
-- -----------------------------------------------------------------------------
ALTER GIT REPOSITORY HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO FETCH;

-- -----------------------------------------------------------------------------
-- Step 5: Verify — browse the repository contents
-- -----------------------------------------------------------------------------

-- Show all branches
SHOW GIT BRANCHES IN HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO;

-- List files on the main branch (top level)
LIST @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO/branches/main/;

-- Browse the production directory
LIST @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO/branches/main/production/;

-- Browse the src directory
LIST @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO/branches/main/src/;

-- Browse notebooks
LIST @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO/branches/main/notebooks/;

-- =============================================================================
-- Step 6: Execute production scripts from the Git repo
-- =============================================================================
--
-- EXECUTE IMMEDIATE FROM runs a Python file directly from the Git stage.
-- This is how you run production code without copying files manually.

-- Example: Run batch inference (model V1)
-- EXECUTE IMMEDIATE FROM @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO/branches/main/production/run_batch_inference.py
--     USING (VERSION => 'V1');

-- Example: Train + register a new model version
-- EXECUTE IMMEDIATE FROM @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO/branches/main/production/run_training.py
--     USING (VERSION => 'V2');

-- =============================================================================
-- Step 7: Grant access to the data science team
-- =============================================================================
-- Create a role for data scientists who need to pull code and run experiments

CREATE ROLE IF NOT EXISTS ML_ENGINEER;

-- Grant access to the Git repo (read-only — they can fetch and execute, not push)
GRANT USAGE ON DATABASE HEALTHCARE_ML TO ROLE ML_ENGINEER;
GRANT USAGE ON SCHEMA HEALTHCARE_ML.GIT_INTEGRATION TO ROLE ML_ENGINEER;
GRANT READ ON GIT REPOSITORY HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO TO ROLE ML_ENGINEER;

-- Grant access to the ML schemas
GRANT USAGE ON SCHEMA HEALTHCARE_ML.RAW_DATA TO ROLE ML_ENGINEER;
GRANT USAGE ON SCHEMA HEALTHCARE_ML.FEATURE_STORE TO ROLE ML_ENGINEER;
GRANT USAGE ON SCHEMA HEALTHCARE_ML.MODEL_REGISTRY TO ROLE ML_ENGINEER;
GRANT USAGE ON SCHEMA HEALTHCARE_ML.INFERENCE TO ROLE ML_ENGINEER;

GRANT SELECT ON ALL TABLES IN SCHEMA HEALTHCARE_ML.RAW_DATA TO ROLE ML_ENGINEER;
GRANT SELECT ON ALL TABLES IN SCHEMA HEALTHCARE_ML.FEATURE_STORE TO ROLE ML_ENGINEER;

-- Grant warehouse access
GRANT USAGE ON WAREHOUSE HEALTHCARE_ML_WH TO ROLE ML_ENGINEER;

-- Grant model registry access (read + write for registering new versions)
GRANT CREATE MODEL ON SCHEMA HEALTHCARE_ML.MODEL_REGISTRY TO ROLE ML_ENGINEER;

-- Grant inference schema access (write for batch predictions)
GRANT CREATE TABLE ON SCHEMA HEALTHCARE_ML.INFERENCE TO ROLE ML_ENGINEER;
GRANT CREATE STAGE ON SCHEMA HEALTHCARE_ML.INFERENCE TO ROLE ML_ENGINEER;

-- Assign to users
-- GRANT ROLE ML_ENGINEER TO USER <data_scientist_username>;

-- =============================================================================
-- TEAM WORKFLOW: How the data science team uses this setup
-- =============================================================================
--
-- ┌─────────────────────────────────────────────────────────────────────┐
-- │                     DATA SCIENCE TEAM WORKFLOW                     │
-- ├─────────────────────────────────────────────────────────────────────┤
-- │                                                                    │
-- │  LOCAL DEVELOPMENT (Git repo on laptop)                           │
-- │  ─────────────────────────────────────────                        │
-- │  1. git clone the healthcare-readmission-ml repo                  │
-- │  2. Create a feature branch:                                      │
-- │       git checkout -b feature/improve-bnp-threshold               │
-- │  3. Edit notebooks in notebooks/ to experiment                    │
-- │  4. When ready, extract changes to src/ modules                   │
-- │  5. Test locally:                                                 │
-- │       python -m src.train --artifacts-dir ./artifacts             │
-- │       python -m src.register_model --version V2_TEST              │
-- │  6. Commit and push:                                              │
-- │       git add . && git commit -m "improve BNP threshold"          │
-- │       git push origin feature/improve-bnp-threshold               │
-- │  7. Open a Pull Request into main                                 │
-- │                                                                    │
-- │  CODE REVIEW + MERGE                                              │
-- │  ────────────────────                                             │
-- │  8. Team reviews the PR (code diff, metrics comparison)           │
-- │  9. Merge to main                                                 │
-- │                                                                    │
-- │  SNOWFLAKE PICKS UP THE NEW CODE                                  │
-- │  ────────────────────────────────                                 │
-- │  10. In Snowflake (or via promotion script):                      │
-- │        ALTER GIT REPOSITORY                                       │
-- │          HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO FETCH;  │
-- │                                                                    │
-- │  11. The next scheduled TASK automatically runs the new code      │
-- │      OR run manually:                                             │
-- │        EXECUTE IMMEDIATE FROM                                     │
-- │          @...HEALTHCARE_ML_REPO/branches/main/production/         │
-- │            run_batch_inference.py                                  │
-- │          USING (VERSION => 'V2');                                  │
-- │                                                                    │
-- │  MODEL PROMOTION                                                  │
-- │  ─────────────────                                                │
-- │  12. For formal promotion, use scripts/promote_to_prod.sh which   │
-- │      tags the release, optionally pushes to a separate prod repo, │
-- │      and triggers the Git fetch in Snowflake.                     │
-- │                                                                    │
-- └─────────────────────────────────────────────────────────────────────┘
--
-- =============================================================================
-- USEFUL COMMANDS REFERENCE
-- =============================================================================
--
-- Fetch latest code after a push:
--   ALTER GIT REPOSITORY HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO FETCH;
--
-- List all files on main:
--   LIST @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO/branches/main/;
--
-- List files on a feature branch:
--   LIST @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO/branches/feature/my-branch/;
--
-- Execute a Python script from a specific branch:
--   EXECUTE IMMEDIATE FROM
--     @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO/branches/feature/my-branch/production/run_training.py
--     USING (VERSION => 'V2_TEST');
--
-- View file contents (read a SQL file):
--   SELECT $1 FROM @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO/branches/main/scripts/setup_snowflake_git.sql;
--
-- Show repo metadata:
--   DESCRIBE GIT REPOSITORY HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO;
