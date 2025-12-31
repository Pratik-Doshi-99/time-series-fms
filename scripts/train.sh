#!/bin/bash
# Training script with file-based logging and GPU monitoring

# Setup
PROJECT_DIR="/home/pratikdoshi/projects/time-series-fms"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG="$LOG_DIR/training_${TIMESTAMP}.log"
GPU_LOG="$LOG_DIR/gpu_${TIMESTAMP}.csv"
LATEST_LOG="$LOG_DIR/latest.log"
LATEST_GPU_LOG="$LOG_DIR/latest_gpu.csv"

# Create logs directory
mkdir -p "$LOG_DIR"

# Create/update symlinks to latest logs
ln -sf "training_${TIMESTAMP}.log" "$LATEST_LOG"
ln -sf "gpu_${TIMESTAMP}.csv" "$LATEST_GPU_LOG"

# Cleanup function to kill GPU logger on exit
cleanup() {
    if [ ! -z "$GPU_LOGGER_PID" ]; then
        echo "Stopping GPU logger (PID: $GPU_LOGGER_PID)..."
        kill $GPU_LOGGER_PID 2>/dev/null
        wait $GPU_LOGGER_PID 2>/dev/null
    fi
}

# Register cleanup function
trap cleanup EXIT INT TERM

# Log startup
echo "========================================"
echo "Training started at $(date)"
echo "Training log: $RUN_LOG"
echo "GPU log: $GPU_LOG"
echo "Monitor training: tail -f $LATEST_LOG"
echo "Monitor GPU: tail -f $LATEST_GPU_LOG"
echo "========================================"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Activate virtual environment
source /home/pratikdoshi/projects/torch_env/bin/activate

# Start GPU logger in background
python infra/log_usage_metrics.py "$GPU_LOG" >> "$LOG_DIR/gpu_logger_${TIMESTAMP}.log" 2>&1 &
GPU_LOGGER_PID=$!
echo "Started GPU logger (PID: $GPU_LOGGER_PID)"

# Run training - all output goes to log file
# -u flag for unbuffered output (immediate writes to log file)
python -u main.py \
    --num_layers 4 \
    --model_dim 128 \
    --hidden_dim 512 \
    --num_heads 8 \
    --epochs 16 \
    --max_training_length 256 \
    --batch_size 1536 \
    --val_batch_size 4096 \
    --optimizer adam \
    --validate_every 50 \
    --save_every 1000 \
    --base_model_name "tsfm" \
    --samples_per_file 10000 \
    --warmup_steps 1000 \
    --train_dir synth-data-train \
    --val_dir synth-data-val \
    --device "cuda:0" \
    --run_name "TSFM-RunABCD-Step2000Onwards" \
    --base_dir "TSFM-4L-256Model-1024Hidden" \
    --autoreg_expansion_factor 1023 \
    --pretrained_model "TSFM-4L-256Model-1024Hidden/tsfm_2000.pt" \
    --start_global_step 2000 \
    >> "$RUN_LOG" 2>&1

# Training finished
TRAIN_EXIT_CODE=$?
echo "========================================"
echo "Training ended at $(date) with exit code: $TRAIN_EXIT_CODE"
echo "========================================"

exit $TRAIN_EXIT_CODE
