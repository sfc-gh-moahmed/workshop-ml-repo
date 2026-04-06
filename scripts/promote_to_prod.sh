#!/usr/bin/env bash
# =============================================================================
# promote_to_prod.sh — Promote a model version to production
# =============================================================================
#
# This script handles the full promotion flow:
#   1. Validates the current branch is clean and on main
#   2. Tags the commit with the model version
#   3. (Optional) Copies src/ + production/ to a separate production Git repo
#   4. Pushes the tag
#   5. Triggers ALTER GIT REPOSITORY ... FETCH in Snowflake
#
# Usage:
#   ./scripts/promote_to_prod.sh V2
#   ./scripts/promote_to_prod.sh V3 --prod-repo /path/to/prod-healthcare-ml
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
VERSION="${1:?Usage: promote_to_prod.sh <VERSION> [--prod-repo <path>]}"
PROD_REPO=""
SNOWFLAKE_CONNECTION="${SNOWFLAKE_CONNECTION:-DEMO}"

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prod-repo)
            PROD_REPO="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  Model Promotion: ${VERSION}"
echo "=============================================="
echo ""

# ---------------------------------------------------------------------------
# Step 1: Validate working tree
# ---------------------------------------------------------------------------
echo "--- Step 1: Validating working tree ---"

cd "$PROJECT_ROOT"

# Check for uncommitted changes
if ! git diff --quiet HEAD 2>/dev/null; then
    echo "ERROR: Uncommitted changes detected. Commit or stash before promoting."
    git status --short
    exit 1
fi

CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: ${CURRENT_BRANCH}"

if [[ "$CURRENT_BRANCH" != "main" ]]; then
    echo "WARNING: Not on 'main' branch. Production promotions should come from main."
    read -rp "Continue anyway? [y/N] " confirm
    if [[ "$confirm" != "y" ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "Working tree is clean."
echo ""

# ---------------------------------------------------------------------------
# Step 2: Tag the commit
# ---------------------------------------------------------------------------
echo "--- Step 2: Tagging commit ---"

TAG_NAME="model-${VERSION}"
COMMIT_SHA=$(git rev-parse --short HEAD)

if git tag -l "$TAG_NAME" | grep -q "$TAG_NAME"; then
    echo "WARNING: Tag '${TAG_NAME}' already exists."
    read -rp "Overwrite? [y/N] " confirm
    if [[ "$confirm" != "y" ]]; then
        echo "Aborted."
        exit 1
    fi
    git tag -d "$TAG_NAME"
fi

git tag -a "$TAG_NAME" -m "Promote model version ${VERSION} (commit ${COMMIT_SHA})"
echo "Tagged ${COMMIT_SHA} as ${TAG_NAME}"
echo ""

# ---------------------------------------------------------------------------
# Step 3: (Optional) Copy to production repo
# ---------------------------------------------------------------------------
if [[ -n "$PROD_REPO" ]]; then
    echo "--- Step 3: Copying to production repo ---"

    if [[ ! -d "$PROD_REPO" ]]; then
        echo "ERROR: Production repo not found at ${PROD_REPO}"
        exit 1
    fi

    # Copy production-ready code
    rsync -av --delete \
        "$PROJECT_ROOT/src/" \
        "$PROD_REPO/src/"

    rsync -av --delete \
        "$PROJECT_ROOT/production/" \
        "$PROD_REPO/production/"

    cp "$PROJECT_ROOT/requirements.txt" "$PROD_REPO/requirements.txt"
    cp "$PROJECT_ROOT/artifacts/model_metadata.json" "$PROD_REPO/artifacts/model_metadata.json" 2>/dev/null || true

    # Commit and push in the production repo
    cd "$PROD_REPO"
    git add -A
    git commit -m "Promote model ${VERSION} from healthcare-readmission-ml (${COMMIT_SHA})" || echo "No changes to commit"
    git tag -a "$TAG_NAME" -m "Model version ${VERSION}" 2>/dev/null || echo "Tag already exists in prod repo"
    git push origin main --tags
    echo "Production repo updated and pushed."
    echo ""
    cd "$PROJECT_ROOT"
else
    echo "--- Step 3: Skipped (no --prod-repo specified) ---"
    echo "  Tip: To promote to a separate production repo, run:"
    echo "    ./scripts/promote_to_prod.sh ${VERSION} --prod-repo /path/to/prod-repo"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 4: Push tags to origin
# ---------------------------------------------------------------------------
echo "--- Step 4: Pushing tags ---"

git push origin "$TAG_NAME" 2>/dev/null && echo "Tag pushed to origin." || echo "WARNING: Could not push tag (no remote configured or auth issue)"
echo ""

# ---------------------------------------------------------------------------
# Step 5: Trigger Snowflake Git fetch
# ---------------------------------------------------------------------------
echo "--- Step 5: Fetching latest code in Snowflake ---"

# Use the snow CLI or snowsql to run the FETCH command
if command -v snow &>/dev/null; then
    snow sql -c "$SNOWFLAKE_CONNECTION" -q \
        "ALTER GIT REPOSITORY HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO FETCH;" \
        && echo "Snowflake Git repo fetched successfully." \
        || echo "WARNING: Could not fetch. Run manually in Snowflake."
else
    echo "snow CLI not found. Run this manually in Snowflake:"
    echo ""
    echo "  ALTER GIT REPOSITORY HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO FETCH;"
    echo ""
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Promotion Complete"
echo "=============================================="
echo ""
echo "  Model version:  ${VERSION}"
echo "  Git tag:         ${TAG_NAME}"
echo "  Commit:          ${COMMIT_SHA}"
echo "  Branch:          ${CURRENT_BRANCH}"
if [[ -n "$PROD_REPO" ]]; then
echo "  Prod repo:       ${PROD_REPO}"
fi
echo ""
echo "  Next steps:"
echo "    - Verify in Snowflake:"
echo "        LIST @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO/branches/main/production/;"
echo ""
echo "    - Run batch inference with the new version:"
echo "        EXECUTE IMMEDIATE FROM"
echo "          @HEALTHCARE_ML.GIT_INTEGRATION.HEALTHCARE_ML_REPO/branches/main/production/run_batch_inference.py"
echo "          USING (VERSION => '${VERSION}');"
echo ""
