#!/bin/bash
# Benchmark script with file-based logging and GPU monitoring

# Setup
PROJECT_DIR="/home/pratikdoshi/projects/time-series/time-series-fms"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG="$LOG_DIR/benchmark_${TIMESTAMP}.log"
GPU_LOG="$LOG_DIR/benchmark_gpu_${TIMESTAMP}.csv"
LATEST_LOG="$LOG_DIR/latest_benchmark.log"
LATEST_GPU_LOG="$LOG_DIR/latest_benchmark_gpu.csv"

# Create logs directory
mkdir -p "$LOG_DIR"

# Create/update symlinks to latest logs
ln -sf "benchmark_${TIMESTAMP}.log" "$LATEST_LOG"
ln -sf "benchmark_gpu_${TIMESTAMP}.csv" "$LATEST_GPU_LOG"

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
echo "Benchmark started at $(date)"
echo "Benchmark log: $RUN_LOG"
echo "GPU log: $GPU_LOG"
echo "Monitor benchmark: tail -f $LATEST_LOG"
echo "Monitor GPU: tail -f $LATEST_GPU_LOG"
echo "========================================"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Activate virtual environment
source /home/pratikdoshi/projects/torch_env/bin/activate

# Start GPU logger in background
python infra/log_usage_metrics.py "$GPU_LOG" >> "$LOG_DIR/benchmark_gpu_logger_${TIMESTAMP}.log" 2>&1 &
GPU_LOGGER_PID=$!
echo "Started GPU logger (PID: $GPU_LOGGER_PID)"

# Run benchmark - all output goes to log file
# -u flag for unbuffered output (immediate writes to log file)
python -u benchmarks/run_benchmark.py \
    --model-path "../tsfm-archived/TSFM-ArchFix/tsfm_10432.pt" \
    --benchmark-dir "benchmark_data" \
    --output "benchmark_results_${TIMESTAMP}.json" \
    --context-length 128 \
    --device "cuda:0" \
    --num-layers 4 \
    --model-dim 128 \
    --num-heads 8 \
    --hidden-dim 512 \
    --num-classes 100 \
    >> "$RUN_LOG" 2>&1

# Benchmark finished
BENCHMARK_EXIT_CODE=$?
echo "========================================"
echo "Benchmark ended at $(date) with exit code: $BENCHMARK_EXIT_CODE"
echo "========================================"

exit $BENCHMARK_EXIT_CODE
