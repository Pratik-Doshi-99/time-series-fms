import os
import sys
import subprocess
import re
import math

def read_sysfs(path):
    """Reads a single value from a sysfs path."""
    try:
        with open(path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "Not Found"
    except PermissionError:
        return "Permission Denied"

def get_physical_ram():
    """Gets total physical RAM in Bytes."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    # MemTotal:       131960240 kB
                    parts = line.split()
                    return int(parts[1]) * 1024
    except:
        return 0

def get_ttm_info():
    """Checks TTM (Translation Table Manager) limits."""
    print(f"{'='*20} KERNEL TTM LIMITS {'='*20}")
    
    # These are the critical parameters for the 50% cap
    paths = {
        "ttm.pages_limit": "/sys/module/ttm/parameters/pages_limit",
        "ttm.page_pool_size": "/sys/module/ttm/parameters/page_pool_size",
        "amdgpu.gttsize": "/sys/module/amdgpu/parameters/gttsize",
        "amdgpu.vm_size": "/sys/module/amdgpu/parameters/vm_size"
    }

    ram_bytes = get_physical_ram()
    print(f"Physical System RAM: {ram_bytes / (1024**3):.2f} GB")

    for name, path in paths.items():
        val = read_sysfs(path)
        print(f"{name:<20}: {val}", end="")
        
        # Analyze values
        if val.isdigit():
            int_val = int(val)
            if "pages_limit" in name:
                # Convert pages (4KB) to GB
                gb_limit = (int_val * 4096) / (1024**3)
                percent = (int_val * 4096) / ram_bytes * 100
                print(f"  (~{gb_limit:.2f} GB | {percent:.1f}% of RAM)")
                
                if percent < 60:
                    print(f"    [WARNING] TTM limit is near 50%. This is likely the bottleneck.")
                else:
                    print(f"    [OK] TTM limit > 60%, fix likely active.")
            elif "gttsize" in name:
                 # gttsize is usually in MB or -1 (auto)
                 if int_val == -1:
                     print(" (Auto)")
                 else:
                     print(f" ({int_val/1024:.2f} GB)")
            else:
                print() # Newline for others
        else:
            print()

def get_rocm_pools():
    """Parses rocminfo to find the actual HSA pools."""
    print(f"\n{'='*20} ROCM HSA POOLS {'='*20}")
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True)
        output = result.stdout
    except FileNotFoundError:
        print("Error: 'rocminfo' command not found. Is ROCm installed?")
        return

    # Looking for Pool Info
    # Pool 1 often represents the system/GTT memory accessible to GPU
    pools = re.findall(r'Pool (\d+).*?Size:\s+(\d+)\(', output, re.DOTALL | re.MULTILINE)
    
    found_gpu_pool = False
    for pool_idx, size_kb in pools:
        size_gb = int(size_kb) * 1024 / (1024**3)
        print(f"HSA Pool {pool_idx:<2}: {size_gb:.2f} GB", end="")
        
        # Heuristic: The pool near 62GB on a 128GB system is the restricted one
        if 60 < size_gb < 70:
            print("  <-- This looks like the RESTRICTED pool (approx 50%)")
            found_gpu_pool = True
        elif size_gb > 100:
            print("  <-- This looks like the UNLOCKED pool (full RAM access)")
            found_gpu_pool = True
        else:
            print()

def check_pytorch():
    """Checks what PyTorch actually sees."""
    print(f"\n{'='*20} PYTORCH VISIBILITY {'='*20}")
    try:
        import torch
        if not torch.cuda.is_available():
            print("PyTorch is installed but CUDA/ROCm is not available.")
            return
        
        device = 0
        props = torch.cuda.get_device_properties(device)
        total_mem_gb = props.total_memory / (1024**3)
        
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Device Name:     {props.name}")
        print(f"Total Visible:   {total_mem_gb:.2f} GB")
        
        if total_mem_gb < 70:
             print("\n[CONCLUSION] PyTorch is still capped at ~50%.")
             print("Action: Apply 'ttm.pages_limit' kernel parameter.")
        else:
             print("\n[CONCLUSION] SUCCESS! PyTorch sees > 100GB.")
             
    except ImportError:
        print("PyTorch not installed in this environment.")
    except Exception as e:
        print(f"Error checking PyTorch: {e}")

if __name__ == "__main__":
    get_ttm_info()
    get_rocm_pools()
    check_pytorch()
    print("\n" + "="*60)