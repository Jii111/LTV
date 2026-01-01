SCRIPT_DIR=$(dirname "$(realpath "$0")")
I2CL_DIR=$(dirname "$SCRIPT_DIR")
PROJECT_DIR=$(dirname "$I2CL_DIR")

export HF_DATASETS_CACHE="$PROJECT_DIR/.cache"
export HF_HOME="$PROJECT_DIR/.cache"
export LOG_DIR="$I2CL_DIR/.log"

mkdir -p "$I2CL_DIR/.log"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$I2CL_DIR/.log/baselines_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo "Project directory: $PROJECT_DIR"
echo "I2CL directory: $I2CL_DIR"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Starting baseline pipeline" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

CMD="cd $I2CL_DIR && python run_baselines_all_extra.py \
  --exp_name exps/mainexp2 \
  --models Qwen/Qwen2.5-14B \
  --datasets dbpedia \
  --shot_per_class 30 \
  --bs 1 \
  --run_num 5 \
  --seed 42
"

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
