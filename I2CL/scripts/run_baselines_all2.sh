#!/bin/bash

# Run baseline suite (zero-shot, few-shot, M1, M2, M2-adaptive)
# Usage: ./I2CL/scripts/run_baselines.sh [config_path]
# Example: ./I2CL/scripts/run_baselines.sh configs/config_m2.py

SCRIPT_DIR=$(dirname "$(realpath "$0")")
I2CL_DIR=$(dirname "$SCRIPT_DIR")
PROJECT_DIR=$(dirname "$I2CL_DIR")

export HF_DATASETS_CACHE="$PROJECT_DIR/.cache"
export HF_HOME="$PROJECT_DIR/.cache"
export LOG_DIR="$I2CL_DIR/.log"

CONFIG_PATH=${1:-"configs/config_baseline_all5.py"}

mkdir -p "$I2CL_DIR/.log"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$I2CL_DIR/.log/baselines_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo "Project directory: $PROJECT_DIR"
echo "I2CL directory: $I2CL_DIR"
echo "Config: $CONFIG_PATH"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Starting baseline pipeline" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

CMD="cd $I2CL_DIR && python run_baselines_all.py --config_path $CONFIG_PATH"
echo "Command: $CMD" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

if eval $CMD 2>&1 | tee -a "$LOG_FILE"; then
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "✓ Baseline pipeline completed successfully!" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
else
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "✗ Baseline pipeline failed!" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    exit 1
fi
