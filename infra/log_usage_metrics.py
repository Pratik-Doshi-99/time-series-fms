import subprocess
import json
import time
import csv
import sys
import datetime
import os

# Configuration
LOG_FILE = "gpu_log.csv"
INTERVAL_SEC = 1
FLUSH_INTERVAL_SEC = 10  # Force write to disk every 10 seconds

def check_rocm_smi():
    """Verifies that rocm-smi is installed."""
    try:
        subprocess.run(["rocm-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("Error: 'rocm-smi' not found. Please ensure ROCm is installed and in your PATH.")
        sys.exit(1)

def get_gpu_metrics():
    """Fetches GPU metrics using rocm-smi --json."""
    try:
        cmd = [
            "rocm-smi",
            "--showuse",
            "--showmeminfo", "vram", "gtt",
            "--showpower",
            "--showtemp",
            "--json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        # We assume card0 for your single APU setup.
        card_data = data.get("card0")
        
        if not card_data:
            return None

        # Extract specific fields with safe fallbacks
        usage_pct = card_data.get("GPU use (%)", 0)
        
        # Memory: Prioritize GTT (System RAM) for Unified Memory systems
        gtt_used_bytes = int(card_data.get("GTT Total Used Memory (B)", 0))
        mem_gb = gtt_used_bytes / (1024**3)
        
        power_w = card_data.get("Average Graphics Package Power (W)", 
                  card_data.get("Average Power (W)", 0.0))
        
        temp_c = card_data.get("Temperature (Sensor edge) (C)", 
                 card_data.get("Temperature (Sensor junction) (C)", 0.0))

        return {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "usage_pct": float(usage_pct),
            "memory_gb": float(mem_gb),
            "power_w": float(power_w),
            "temp_c": float(temp_c)
        }

    except (json.JSONDecodeError, ValueError):
        # Return None so we skip this tick without crashing
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def main():
    check_rocm_smi()
    
    print(f"Logging GPU metrics to '{LOG_FILE}' every {INTERVAL_SEC}s.")
    print(f"Data will flush to disk every {FLUSH_INTERVAL_SEC}s.")
    print("Press Ctrl+C to stop.")
    print("-" * 65)
    print(f"{'Timestamp':<20} | {'GPU Use':<10} | {'Mem (GB)':<10} | {'Power (W)':<10} | {'Temp (C)':<10}")
    print("-" * 65)

    file_exists = os.path.isfile(LOG_FILE)
    last_flush_time = time.time()
    
    try:
        # buffer_size=1 implies line buffering, but we control flush manually below
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "usage_pct", "memory_gb", "power_w", "temp_c"])
            
            if not file_exists:
                writer.writeheader()
                f.flush()

            while True:
                metrics = get_gpu_metrics()
                
                if metrics:
                    # 1. Write to internal Python buffer
                    writer.writerow(metrics)
                    
                    # 2. Check if it is time to flush to disk
                    if time.time() - last_flush_time >= FLUSH_INTERVAL_SEC:
                        f.flush()
                        os.fsync(f.fileno())  # Ensure OS writes to physical disk
                        last_flush_time = time.time()
                    
                    # 3. Print to console (always)
                    print(f"{metrics['timestamp']:<20} | {metrics['usage_pct']:>8.1f} % | {metrics['memory_gb']:>8.2f} | {metrics['power_w']:>8.1f} | {metrics['temp_c']:>8.1f}")
                
                time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\nLogging stopped by user. Flushing remaining data...")
        # Context manager exit will automatically close and flush the file
        sys.exit(0)

if __name__ == "__main__":
    main()