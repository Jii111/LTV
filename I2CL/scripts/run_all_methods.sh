#!/bin/bash

cd "$(dirname "$0")/.." || exit 1

LOG_DIR=".log"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

run_and_log() {
    local name=$1
    local script=$2
    local config=$3
    local log_file="${LOG_DIR}/run_$(echo "$name" | tr ' ' '_' | tr '()' '__')_${TIMESTAMP}.log"
    echo ""
    echo "========================================"
    echo "Running $name"
    echo "Log: $log_file"
    echo "========================================"
    if [ -z "$config" ]; then
        python "$script" 2>&1 | tee "$log_file"
    else
        python "$script" --config_path "$config" 2>&1 | tee "$log_file"
    fi
}

run_and_log "M1 (Task Vector)" "run_m1.py" "configs/config_m1.py"
run_and_log "M2 (Mask-based)" "run_m2.py" "configs/config_m2.py"
run_and_log "M3 (Feature-based)" "run_m3.py" "configs/config_m3.py"
run_and_log "M2-Adaptive (Linear)" "run_m2_adaptive.py" "configs/config_m2_adaptive.py"
run_and_log "M3 Low-Rank" "run_m3_lowrank.py" "configs/config_m3_lowrank.py"
run_and_log "M3 Linearized" "run_m3_linearized.py" "configs/config_m3_linearized.py"

echo ""
echo "All experiments finished. Logs saved under $LOG_DIR and results under exps/ directories."
