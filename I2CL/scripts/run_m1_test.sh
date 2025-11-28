#!/bin/bash

cd "$(dirname "$0")/.." || exit 1

CONFIG_PATH="configs/config_m1.py"
mkdir -p .log
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=".log/m1_test_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "M1 (Task Vector) Test" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Config: $CONFIG_PATH" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if python run_m1.py --config_path $CONFIG_PATH 2>&1 | tee -a "$LOG_FILE"; then
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "✓ M1 test completed successfully!" | tee -a "$LOG_FILE"
    echo "Results saved in: exps/taskvector_shot/" | tee -a "$LOG_FILE"
    echo "Log saved in: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
else
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "✗ M1 test failed!" | tee -a "$LOG_FILE"
    echo "Check log: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    exit 1
fi
