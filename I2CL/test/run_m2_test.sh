#!/bin/bash

# M2 Experiment Test Script
# Runs M2 with current config settings

# Set working directory
cd "$(dirname "$0")/.." || exit 1

# Configuration
CONFIG_PATH="configs/config_m2.py"

# Create log directory
mkdir -p .log

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=".log/m2_test_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"

echo "========================================" | tee -a "$LOG_FILE"
echo "M2 (Mask-based Multi-layer) Test" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Config: $CONFIG_PATH" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run M2
if python run_m2.py --config_path $CONFIG_PATH 2>&1 | tee -a "$LOG_FILE"; then
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "✓ M2 test completed successfully!" | tee -a "$LOG_FILE"
    echo "Results saved in: exps/m2/" | tee -a "$LOG_FILE"
    echo "Log saved in: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
else
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "✗ M2 test failed!" | tee -a "$LOG_FILE"
    echo "Check log: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    exit 1
fi
