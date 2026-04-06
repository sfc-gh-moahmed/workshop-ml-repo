-- =============================================================================
-- Healthcare ML — Snowflake Tasks for Scheduled Pipeline Execution
-- =============================================================================
--
-- These TASKs run production Python scripts directly from the Git repository.
-- Every execution uses the latest code fetched from the repo — no manual
-- file copying required.
--
-- Prerequisites:
--   1. Run scripts/setup_snowflake_git.sql to create the Git repo object
--   2. Push your code to the main branch of the remote repo
--   3. Run ALTER GIT REPOSITORY ... FETCH at least once
--
-- Architecture:
--
--   ┌──────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
--   │  GitHub Repo     │      │  Snowflake Git Repo  │      │  Snowflake Tasks │
--   │  (main branch)   │─────>│  (fetched clone)     │─────>│  (scheduled)     │
--   │                  │ FETCH│                      │ EXEC │                  │
--   │  production/     │      │  @.../branches/main/ │      │  GIT_FETCH_TASK  │
--   │    run_*.py      │      │    production/       │      │  BATCH_SCORING   │
--   └──────────────────┘      └──────────────────────┘      └──────────────────┘
--
-- =============================================================================

USE ROLE ACCOUNTADMIN;
USE DATABASE HEALTHCARE_ML;

CREATE SCHEMA IF NOT EXISTS HEALTHCARE_ML.TASKS
    COMMENT = 'Scheduled pipeline tasks';

USE SCHEMA TASKS;

-- =============================================================================
-- Task 1: GIT FETCH — Pull latest code from GitHub
-- =============================================================================
-- Runs every 60 minutes to keep the Snowflake Git clone in sync.
-- This ensures that when batch scoring runs, it always uses the latest
-- production code.

CREATE OR REPLACE TASK HEALTHCARE_ML.TASKS.GIT_FETCH_TASK
    WAREHOUSE = HEALTHCARE_ML_WH
    SCHEDULE  = '60 MINUTE'
    COMMENT   = 'Fetch latest code from healthcare-readmission-ml GitHub repo'
AS
    ALTER GIT REPOSITORY HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO FETCH;

-- =============================================================================
-- Task 2: BATCH SCORING — Run daily batch inference
-- =============================================================================
-- Runs after each Git fetch to score all patients using the latest model.
-- Uses EXECUTE IMMEDIATE FROM to run the Python script directly from the
-- Git repo stage.
--
-- This task depends on GIT_FETCH_TASK (runs after fetch completes).

CREATE OR REPLACE TASK HEALTHCARE_ML.TASKS.BATCH_SCORING_TASK
    WAREHOUSE = HEALTHCARE_ML_WH
    AFTER     HEALTHCARE_ML.TASKS.GIT_FETCH_TASK
    COMMENT   = 'Run batch inference using latest code from Git repo'
AS
    EXECUTE IMMEDIATE FROM @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO/branches/main/production/run_batch_inference.py;

-- =============================================================================
-- Task 3: (Optional) RETRAIN — Scheduled model retraining
-- =============================================================================
-- Uncomment to enable weekly retraining. This reads the latest data from
-- RAW_DATA, trains a new model, and registers it in the Model Registry.
-- The VERSION is incremented automatically based on the date.

-- CREATE OR REPLACE TASK HEALTHCARE_ML.TASKS.RETRAIN_TASK
--     WAREHOUSE = HEALTHCARE_ML_WH
--     SCHEDULE  = 'USING CRON 0 2 * * SUN America/Los_Angeles'  -- Sundays at 2am PT
--     COMMENT   = 'Weekly model retraining from latest data'
-- AS
--     EXECUTE IMMEDIATE FROM @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO/branches/main/production/run_training.py;

-- =============================================================================
-- Enable / Disable Tasks
-- =============================================================================

-- Start the task tree (GIT_FETCH_TASK is the root; BATCH_SCORING follows)
ALTER TASK HEALTHCARE_ML.TASKS.BATCH_SCORING_TASK RESUME;
ALTER TASK HEALTHCARE_ML.TASKS.GIT_FETCH_TASK RESUME;

-- To pause the pipeline:
-- ALTER TASK HEALTHCARE_ML.TASKS.GIT_FETCH_TASK SUSPEND;
-- ALTER TASK HEALTHCARE_ML.TASKS.BATCH_SCORING_TASK SUSPEND;

-- =============================================================================
-- Monitoring
-- =============================================================================

-- Check task run history
-- SELECT *
-- FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
--     TASK_NAME => 'GIT_FETCH_TASK',
--     SCHEDULED_TIME_RANGE_START => DATEADD('HOUR', -24, CURRENT_TIMESTAMP())
-- ))
-- ORDER BY SCHEDULED_TIME DESC;

-- SELECT *
-- FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
--     TASK_NAME => 'BATCH_SCORING_TASK',
--     SCHEDULED_TIME_RANGE_START => DATEADD('HOUR', -24, CURRENT_TIMESTAMP())
-- ))
-- ORDER BY SCHEDULED_TIME DESC;
