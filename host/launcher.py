"""
GPU-aware Launcher for Megakernel

Handles:
1. GPU detection and compile flag selection
2. CUDA kernel compilation via nvcc
3. Loading compiled binary and calling it via ctypes
4. Weight loading and pointer setup
"""

import os
import subprocess
import ctypes
import json
import struct
import numpy as np
import torch

# Model constants (must match config.cuh)
HIDDEN_DIM = 1024
NUM_LAYERS = 28
VOCAB_SIZE = 151936


def detect_gpu():
    """Detect the GPU and return info dict."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    gpu_info = {
        "name": props.name,
        "sm_count": props.multi_processor_count,
        "total_memory_gb": props.total_mem / 1e9,
        "compute_capability": f"{props.major}.{props.minor}",
        "cc_major": props.major,
        "cc_minor": props.minor,
        "l2_cache_mb": getattr(props, 'l2_cache_size', 0) / (1024 * 1024),
    }

    # Determine GPU class and arch flag
    cc = props.major * 10 + props.minor
    if cc >= 90:
        gpu_info["arch_flag"] = "sm_90"
        gpu_info["class"] = "H100"
        gpu_info["peak_bw_gbps"] = 3350.0
    elif cc >= 89:
        gpu_info["arch_flag"] = "sm_89"
        gpu_info["class"] = "L4/RTX4090"
        gpu_info["peak_bw_gbps"] = 300.0  # L4
    elif cc >= 80:
        gpu_info["arch_flag"] = "sm_80"
        gpu_info["class"] = "A100"
        gpu_info["peak_bw_gbps"] = 1555.0 if props.total_mem < 50e9 else 2039.0
    elif cc >= 75:
        gpu_info["arch_flag"] = "sm_75"
        gpu_info["class"] = "T4"
        gpu_info["peak_bw_gbps"] = 300.0
    else:
        gpu_info["arch_flag"] = f"sm_{cc}"
        gpu_info["class"] = "Unknown"
        gpu_info["peak_bw_gbps"] = 200.0

    return gpu_info


def compile_kernel(src_dir="src", output="megakernel", gpu_info=None):
    """Compile the CUDA kernel with nvcc."""
    if gpu_info is None:
        gpu_info = detect_gpu()

    arch = gpu_info["arch_flag"]
    src_file = os.path.join(src_dir, "megakernel.cu")
    output_binary = output

    cmd = [
        "nvcc",
        "-O3",
        f"-arch={arch}",
        "-std=c++17",
        "--use_fast_math",
        "-lineinfo",
        f"-I{src_dir}",
        src_file,
        "-o", output_binary,
    ]

    print(f"Compiling kernel for {gpu_info['class']} ({arch})...")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation FAILED:\n{result.stderr}")
        raise RuntimeError("nvcc compilation failed")

    print(f"Compilation successful: {output_binary}")
    return output_binary


def compile_shared_library(src_dir="src", output="megakernel.so", gpu_info=None):
    """Compile as shared library for ctypes loading."""
    if gpu_info is None:
        gpu_info = detect_gpu()

    arch = gpu_info["arch_flag"]
    src_file = os.path.join(src_dir, "megakernel.cu")

    cmd = [
        "nvcc",
        "-O3",
        f"-arch={arch}",
        "-std=c++17",
        "--use_fast_math",
        "-shared",
        "-Xcompiler", "-fPIC",
        "-DMEGAKERNEL_LIBRARY_MODE",
        f"-I{src_dir}",
        src_file,
        "-o", output,
    ]

    print(f"Compiling shared library for {gpu_info['class']} ({arch})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation FAILED:\n{result.stderr}")
        raise RuntimeError("nvcc compilation failed")

    print(f"Compiled: {output}")
    return output


def run_standalone(binary_path="megakernel"):
    """Run the standalone binary (for info/testing)."""
    result = subprocess.run([f"./{binary_path}"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.returncode == 0


def setup_weight_pointers(gpu_buffer, base_ptr, offsets):
    """
    Given the GPU buffer and offsets, create a structured mapping
    of weight pointers that matches the ModelWeights struct.

    Returns a dict with all pointer values (as integers).
    """
    ptrs = {}

    ptrs["embedding"] = base_ptr + offsets["embedding"]
    ptrs["final_norm"] = base_ptr + offsets["final_norm"]

    for i in range(NUM_LAYERS):
        ptrs[f"layer.{i}.attn_norm"] = base_ptr + offsets[f"layer.{i}.attn_norm"]
        ptrs[f"layer.{i}.w_q"] = base_ptr + offsets[f"layer.{i}.w_q"]
        ptrs[f"layer.{i}.w_k"] = base_ptr + offsets[f"layer.{i}.w_k"]
        ptrs[f"layer.{i}.w_v"] = base_ptr + offsets[f"layer.{i}.w_v"]
        ptrs[f"layer.{i}.w_o"] = base_ptr + offsets[f"layer.{i}.w_o"]
        ptrs[f"layer.{i}.ffn_norm"] = base_ptr + offsets[f"layer.{i}.ffn_norm"]
        ptrs[f"layer.{i}.w_gate"] = base_ptr + offsets[f"layer.{i}.w_gate"]
        ptrs[f"layer.{i}.w_up"] = base_ptr + offsets[f"layer.{i}.w_up"]
        ptrs[f"layer.{i}.w_down"] = base_ptr + offsets[f"layer.{i}.w_down"]

    return ptrs


def print_gpu_info(gpu_info):
    """Pretty-print GPU information."""
    print(f"GPU: {gpu_info['name']}")
    print(f"  Class: {gpu_info['class']}")
    print(f"  Compute Capability: {gpu_info['compute_capability']}")
    print(f"  SMs: {gpu_info['sm_count']}")
    print(f"  Memory: {gpu_info['total_memory_gb']:.1f} GB")
    print(f"  Peak Bandwidth: {gpu_info['peak_bw_gbps']:.0f} GB/s")
    print(f"  Arch Flag: {gpu_info['arch_flag']}")

    # Theoretical max tok/s for Qwen3-0.6B
    model_bytes = 926e6  # ~926 MB in BF16
    theoretical_max = gpu_info['peak_bw_gbps'] * 1e9 / model_bytes
    print(f"  Theoretical max tok/s (Qwen3-0.6B BF16): {theoretical_max:.0f}")


if __name__ == "__main__":
    gpu_info = detect_gpu()
    print_gpu_info(gpu_info)
    print()
    compile_kernel(gpu_info=gpu_info)
    print()
    run_standalone()
