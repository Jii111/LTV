#!/bin/bash

# M3 Experiment Test Script
# Runs M3 with current config settings

# Set working directory
cd "$(dirname "$0")/.." || exit 1

# Configuration
CONFIG_PATH="configs/config_m3.py"

# Create log directory
mkdir -p .log

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=".log/m3_test_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"

echo "========================================" | tee -a "$LOG_FILE"
echo "M3 (Feature-based Query-adaptive) Test" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Config: $CONFIG_PATH" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run M3
if python run_m3.py --config_path $CONFIG_PATH 2>&1 | tee -a "$LOG_FILE"; then
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "✓ M3 test completed successfully!" | tee -a "$LOG_FILE"
    echo "Results saved in: exps/m3/" | tee -a "$LOG_FILE"
    echo "Log saved in: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
else
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "✗ M3 test failed!" | tee -a "$LOG_FILE"
    echo "Check log: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    exit 1
fi
