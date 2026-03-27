#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")

export HF_DATASETS_CACHE="$PROJECT_DIR/.cache"
export HF_HOME="$PROJECT_DIR/.cache"
export LOG_DIR="$PROJECT_DIR/.log"

# Require HF_TOKEN for gated models (e.g. Llama 3.1)
export HF_TOKEN="${HF_TOKEN:?'Set HF_TOKEN env var for Llama gated access'}"

CONFIG_PATH=${1:-"config/config_comparison_exp.py"}

mkdir -p "$PROJECT_DIR/.log"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_DIR/.log/comparison_exp_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo "Project directory: $PROJECT_DIR"
echo "Config: $CONFIG_PATH"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Starting comparison experiment" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

CMD="cd $PROJECT_DIR && python run/run_comparison_exp.py --config_path $CONFIG_PATH"
echo "Command: $CMD" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

if eval $CMD 2>&1 | tee -a "$LOG_FILE"; then
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Comparison experiment completed successfully!" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
else
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Comparison experiment failed!" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    exit 1
fi