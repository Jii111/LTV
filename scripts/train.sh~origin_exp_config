#!/bin/bash

# Training script for LVLM with Accelerate & DeepSpeed
# Usage: ./scripts/train.sh [gpu_ids] [wandb_key]
# Example: ./scripts/train.sh 0,1,2,3 YOUR_WANDB_KEY
#          ./scripts/train.sh 0        # Single GPU, no wandb
# Note: DeepSpeed config should be specified in the training config YAML file
#       under trainer.deepspeed: "path/to/ds_config.json"

# Get script directory and project root
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
PARENT_DIR=$(dirname "$PROJECT_DIR")

# Set cache directories
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache"
export HF_HOME="$PROJECT_DIR/.cache"
export LOG_DIR="$PROJECT_DIR/.log"
export DEEPSPEED_COMM=nccl
export NCCL_DEBUG=INFO

# Get GPU configuration (default to 0)
DEVICES=${1:-"0"}

# Get wandb key (optional)
WANDB_KEY=${2:-"3314a9f18c06914b9c333abc68130f93f2cb1a23"}

# Create log directory if it doesn't exist
mkdir -p "$PROJECT_DIR/.log"

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_DIR/.log/train_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo "Project directory: $PROJECT_DIR"
echo "Parent directory: $PARENT_DIR"

# Configuration files to train
CFG_PATHS=(
#    "$PROJECT_DIR/train_config/qwen-2.5-7b-agnews-1.0e-1-500-bs8.yaml"
#    "$PROJECT_DIR/train_config/qwen-2.5-7b-dbpedia-1.0e-1-500-bs8.yaml"
#    "$PROJECT_DIR/train_config/qwen-2.5-7b-sst5-1.0e-1-500-bs8.yaml"
#    "$PROJECT_DIR/train_config/qwen-2.5-7b-sst2-1.0e-1-500-bs8.yaml"
#    "$PROJECT_DIR/train_config/qwen-2.5-7b-trec-1.0e-1-500-bs8.yaml"
#    "$PROJECT_DIR/train_config/qwen-2.5-7b-mr-1.0e-1-500-bs8.yaml"
#    "$PROJECT_DIR/train_config/qwen-2.5-7b-hate_speech18-1.0e-1-500-bs8.yaml"
#    "$PROJECT_DIR/train_config/qwen-2.5-7b-subj-1.0e-1-500-bs8.yaml"
    "$PROJECT_DIR/train_config/llama-3.1-8b-agnews-1.0e-1-500-bs8.yaml"
    "$PROJECT_DIR/train_config/llama-3.1-8b-dbpedia-1.0e-1-500-bs8.yaml"
    "$PROJECT_DIR/train_config/llama-3.1-8b-sst5-1.0e-1-500-bs8.yaml"
    "$PROJECT_DIR/train_config/llama-3.1-8b-sst2-1.0e-1-500-bs8.yaml"
    "$PROJECT_DIR/train_config/llama-3.1-8b-trec-1.0e-1-500-bs8.yaml"
    "$PROJECT_DIR/train_config/llama-3.1-8b-mr-1.0e-1-500-bs8.yaml"
    "$PROJECT_DIR/train_config/llama-3.1-8b-hate_speech18-1.0e-1-500-bs8.yaml"
    "$PROJECT_DIR/train_config/llama-3.1-8b-subj-1.0e-1-500-bs8.yaml"
)

echo "==========================================" | tee -a "$LOG_FILE"
echo "Starting LVLM Training Pipeline" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "GPUs: $DEVICES" | tee -a "$LOG_FILE"
echo "Total configs: ${#CFG_PATHS[@]}" | tee -a "$LOG_FILE"
if [ -n "$WANDB_KEY" ]; then
    echo "W&B logging: Enabled" | tee -a "$LOG_FILE"
else
    echo "W&B logging: Disabled" | tee -a "$LOG_FILE"
fi
echo "==========================================" | tee -a "$LOG_FILE"

# Loop through each config file
for i in "${!CFG_PATHS[@]}"
do
    CFG_PATH="${CFG_PATHS[$i]}"

    echo "" | tee -a "$LOG_FILE"
    echo "[$((i+1))/${#CFG_PATHS[@]}] Currently Running with Config: $CFG_PATH" | tee -a "$LOG_FILE"

    # Validate config file exists
    if [ ! -f "$CFG_PATH" ]; then
        echo "Error: Configuration file not found: $CFG_PATH" | tee -a "$LOG_FILE"
        echo "Skipping..." | tee -a "$LOG_FILE"
        continue
    fi

    echo "==========================================" | tee -a "$LOG_FILE"

    # Calculate number of GPUs from DEVICES
    NUM_GPUS=$(echo "$DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')

    echo "Number of GPUs: $NUM_GPUS" | tee -a "$LOG_FILE"

    # Build command using accelerate launch
    # DeepSpeed will be automatically used if specified in TrainingArguments (deepspeed parameter in config)
    # Force NCCL backend to avoid MPI dependency
    # CMD="CUDA_VISIBLE_DEVICES=$DEVICES ACCELERATE_USE_DEEPSPEED=true accelerate launch"
    CMD="CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch"

    # Add accelerate arguments
    CMD="$CMD --num_processes=$NUM_GPUS"

    # Multi-GPU settings
    if [ "$NUM_GPUS" -gt 1 ]; then
        CMD="$CMD"
        echo "Using multi-GPU training" | tee -a "$LOG_FILE"
        echo "DeepSpeed will be used if 'deepspeed' is specified in trainer config" | tee -a "$LOG_FILE"
    fi

    # Add the training script
    CMD="$CMD $PROJECT_DIR/train.py --cfg-path $CFG_PATH"

    # Add wandb key if provided
    if [ -n "$WANDB_KEY" ]; then
        CMD="$CMD --wandb-key $WANDB_KEY"
    fi

    echo "Command: $CMD" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"

    # Run training
    if eval $CMD 2>&1 | tee -a "$LOG_FILE"; then
        echo "✓ Config $((i+1))/${#CFG_PATHS[@]} completed successfully" | tee -a "$LOG_FILE"
    else
        echo "✗ Config $((i+1))/${#CFG_PATHS[@]} failed" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        echo "==========================================" | tee -a "$LOG_FILE"
        echo "Training pipeline failed!" | tee -a "$LOG_FILE"
        echo "==========================================" | tee -a "$LOG_FILE"
        exit 1
    fi

    echo "==========================================" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "All trainings completed successfully!" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
