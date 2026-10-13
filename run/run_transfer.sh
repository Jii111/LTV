#!/bin/bash
# Usage: ./run_cross_model.sh [config_path]

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HF_DATASETS_CACHE="$PROJECT_DIR/.cache"
export HF_HOME="$PROJECT_DIR/.cache"
export LOG_DIR="$PROJECT_DIR/.log"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

mkdir -p "$PROJECT_DIR/.log"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_DIR/.log/cross_model_${TIMESTAMP}.log"
CONFIG_PATH="${1:-${REPO_ROOT}/config/config_transfer.py}"

echo "Logging to: $LOG_FILE"
echo "Project directory: $PROJECT_DIR"
echo "Config: $CONFIG_PATH"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Starting cross-model LTV run" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

CMD=(python "${REPO_ROOT}/run/run_transfer.py" --config_path "${CONFIG_PATH}")
echo "Command: ${CMD[*]}" | tee -a "$LOG_FILE"

if "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
  echo "==========================================" | tee -a "$LOG_FILE"
  echo "✓ Cross-model run completed" | tee -a "$LOG_FILE"
  echo "==========================================" | tee -a "$LOG_FILE"
else
  echo "" | tee -a "$LOG_FILE"
  echo "==========================================" | tee -a "$LOG_FILE"
  echo "✗ Cross-model run failed!" | tee -a "$LOG_FILE"
  echo "==========================================" | tee -a "$LOG_FILE"
  exit 1
fi
