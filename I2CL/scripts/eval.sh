#!/bin/bash

# Evaluation script for PEFT models using YAML config
# Usage: ./I2CL/scripts/eval.sh [config_path]
# Example: ./I2CL/scripts/eval.sh configs/yamls/eval_config.yaml
#          ./I2CL/scripts/eval.sh     # Uses default config

# Get script directory and project root
SCRIPT_DIR=$(dirname "$(realpath "$0")")
I2CL_DIR=$(dirname "$SCRIPT_DIR")
PROJECT_DIR=$(dirname "$I2CL_DIR")

# Set cache directories (same as train.sh)
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache"
export HF_HOME="$PROJECT_DIR/.cache"
export LOG_DIR="$I2CL_DIR/.log"

# Get config path (default to configs/yamls/config_prompt_tuned.yaml)
CONFIG_PATH=${1:-"configs/yamls/config_prompt_tuned.yaml"}

# Create log directory if it doesn't exist
mkdir -p "$I2CL_DIR/.log"

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$I2CL_DIR/.log/eval_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo "Project directory: $PROJECT_DIR"
echo "I2CL directory: $I2CL_DIR"

echo "==========================================" | tee -a "$LOG_FILE"
echo "Starting PEFT Model Evaluation Pipeline" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Config: $CONFIG_PATH" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Build Python command
CMD="cd $I2CL_DIR && python run_prompt_tuned_with_yaml.py --config_path $CONFIG_PATH"

echo "Command: $CMD" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Run evaluation
if eval $CMD 2>&1 | tee -a "$LOG_FILE"; then
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "✓ All evaluations completed successfully!" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
else
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "✗ Evaluation pipeline failed!" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    exit 1
fi