#!/usr/bin/env bash
# One-time setup of DagsHub as DVC remote + MLflow tracking server.
# Usage: bash scripts/setup_dagshub.sh <dagshub-user> <repo-name> <dagshub-token>
# The token is written ONLY to git-ignored files (.env and .dvc/config.local).
set -euo pipefail
if [ $# -ne 3 ]; then
  echo "usage: bash scripts/setup_dagshub.sh <dagshub-user> <repo-name> <dagshub-token>"; exit 1
fi
USER_NAME="$1"; REPO="$2"; TOKEN="$3"

dvc remote add -d -f origin "https://dagshub.com/${USER_NAME}/${REPO}.dvc"
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "${USER_NAME}"
dvc remote modify origin --local password "${TOKEN}"

cat > .env <<ENV
MLFLOW_TRACKING_URI=https://dagshub.com/${USER_NAME}/${REPO}.mlflow
MLFLOW_TRACKING_USERNAME=${USER_NAME}
MLFLOW_TRACKING_PASSWORD=${TOKEN}
REQUIRE_REMOTE_TRACKING=1
ENV

echo "DVC remote  -> https://dagshub.com/${USER_NAME}/${REPO}.dvc  (credentials in .dvc/config.local)"
echo "MLflow      -> https://dagshub.com/${USER_NAME}/${REPO}.mlflow  (credentials in .env)"
echo "Next: git add .dvc/config && git commit -m 'Configure DagsHub DVC remote'"
