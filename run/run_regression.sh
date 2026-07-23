#!/usr/bin/env bash
# Run regression experiments.
# All settings (model, gpu, datasets, ...) are in config/config_regression.py
#
# Usage:
#   bash scripts/run_regression.sh

SCRIPT_DIR=$(dirname "$(realpath "$0")")
I2CL_DIR=$(dirname "$SCRIPT_DIR")
PROJECT_DIR=$(dirname "$I2CL_DIR")

export HF_DATASETS_CACHE="$PROJECT_DIR/.cache"
export HF_HOME="$PROJECT_DIR/.cache"
export PYTHONPATH="${I2CL_DIR}:${PYTHONPATH}"

mkdir -p "$I2CL_DIR/.log"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$I2CL_DIR/.log/regression_${TIMESTAMP}.log"
CONFIG="$I2CL_DIR/config/config_regression.py"

# Read model, gpu, datasets from config
MODEL=$(python3 -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('cfg', '$CONFIG')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(m.config.get('model', 'Qwen/Qwen2.5-7B'))
")
# CUDA_VISIBLE_DEVICES가 이미 외부에서 설정된 경우 우선 사용, 아니면 config 값 사용
_CONFIG_GPU=$(python3 -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('cfg', '$CONFIG')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(m.config.get('gpu', '0'))
")
GPU="${CUDA_VISIBLE_DEVICES:-$_CONFIG_GPU}"
DATASETS=$(python3 -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('cfg', '$CONFIG')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(' '.join(m.config.get('datasets', ['linear_regression', 'relu_regression'])))
")

echo "Model:    $MODEL"    | tee -a "$LOG_FILE"
echo "GPU:      $GPU"      | tee -a "$LOG_FILE"
echo "Datasets: $DATASETS" | tee -a "$LOG_FILE"
echo "Config:   $CONFIG"   | tee -a "$LOG_FILE"
echo "Log:      $LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

for DATASET in $DATASETS; do
    echo "" | tee -a "$LOG_FILE"
    echo ">> Dataset: $DATASET" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"

    if cd "$I2CL_DIR" && CUDA_VISIBLE_DEVICES=$GPU python run/run_regression.py \
        --config  "$CONFIG" \
        --model   "$MODEL" \
        --dataset "$DATASET" \
        --gpu     0 \
        2>&1 | tee -a "$LOG_FILE"; then
        echo "✓ $DATASET done" | tee -a "$LOG_FILE"
    else
        echo "✗ $DATASET FAILED" | tee -a "$LOG_FILE"
        exit 1
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "All regression experiments completed." | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
