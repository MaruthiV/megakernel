"""
Benchmark Suite for Megakernel

Provides comprehensive benchmarking with:
- Context position sweep
- Comparison against baselines
- Bandwidth utilization calculation
- Formatted output tables
"""

import time
import torch
import numpy as np


# Model constants
HIDDEN_DIM = 1024
NUM_LAYERS = 28
VOCAB_SIZE = 151936


def estimate_model_bytes(dtype="bf16"):
    """Estimate total model weight bytes for bandwidth calculation."""
    bytes_per_param = {"bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
    bpp = bytes_per_param.get(dtype, 2)

    # Approximate parameter count for Qwen3-0.6B
    per_layer = (
        HIDDEN_DIM * HIDDEN_DIM +          # Q
        HIDDEN_DIM * (HIDDEN_DIM // 2) +   # K
        HIDDEN_DIM * (HIDDEN_DIM // 2) +   # V
        HIDDEN_DIM * HIDDEN_DIM +          # O
        HIDDEN_DIM * 2816 +               # gate
        HIDDEN_DIM * 2816 +               # up
        2816 * HIDDEN_DIM +               # down
        HIDDEN_DIM * 2                    # norms
    )
    total = per_layer * NUM_LAYERS + VOCAB_SIZE * HIDDEN_DIM + HIDDEN_DIM

    return int(total * bpp)


def get_peak_bandwidth():
    """Get peak memory bandwidth for current GPU."""
    props = torch.cuda.get_device_properties(0)
    cc = props.major * 10 + props.minor

    if cc >= 90:
        return 3350.0  # H100 SXM
    elif cc >= 80:
        mem_gb = props.total_mem / 1e9
        return 2039.0 if mem_gb > 50 else 1555.0  # A100 80GB vs 40GB
    elif cc >= 75:
        return 300.0   # T4
    else:
        return 300.0


def benchmark_hf_baseline(model, num_tokens=50, warmup=10):
    """Benchmark HuggingFace model as baseline."""
    input_ids = torch.tensor([[1]], device="cuda")

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(input_ids)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_tokens):
        with torch.no_grad():
            model(input_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return num_tokens / elapsed


def format_results_table(results):
    """Format benchmark results as a table."""
    headers = ["Position", "Tok/s", "Latency(us)", "BW(GB/s)", "BW Util(%)", "vs HF"]
    widths = [10, 10, 12, 10, 10, 8]

    # Header
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    separator = "  ".join("-" * w for w in widths)

    lines = [header_line, separator]
    for r in results:
        row = [
            str(r["position"]).ljust(widths[0]),
            f"{r['toks_per_sec']:.1f}".ljust(widths[1]),
            f"{r['latency_us']:.1f}".ljust(widths[2]),
            f"{r['bandwidth_gbps']:.1f}".ljust(widths[3]),
            f"{r['bw_utilization']:.1f}%".ljust(widths[4]),
            f"{r['speedup_vs_hf']:.1f}x".ljust(widths[5]),
        ]
        lines.append("  ".join(row))

    return "\n".join(lines)


def compute_metrics(toks_per_sec, model_bytes, peak_bw_gbps, hf_baseline_toks=None):
    """Compute derived metrics from raw throughput."""
    latency_us = 1e6 / toks_per_sec if toks_per_sec > 0 else float('inf')
    bandwidth_gbps = (model_bytes * toks_per_sec) / 1e9
    bw_utilization = bandwidth_gbps / peak_bw_gbps * 100
    speedup_vs_hf = toks_per_sec / hf_baseline_toks if hf_baseline_toks else 0

    return {
        "toks_per_sec": toks_per_sec,
        "latency_us": latency_us,
        "bandwidth_gbps": bandwidth_gbps,
        "bw_utilization": bw_utilization,
        "speedup_vs_hf": speedup_vs_hf,
    }


def print_summary(gpu_name, dtype, results, hf_baseline):
    """Print a formatted benchmark summary."""
    model_bytes = estimate_model_bytes(dtype)
    peak_bw = get_peak_bandwidth()
    theoretical_max = peak_bw * 1e9 / model_bytes

    print(f"\n{'='*70}")
    print(f"  MEGAKERNEL BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"  GPU:              {gpu_name}")
    print(f"  Model:            Qwen3-0.6B")
    print(f"  Precision:        {dtype}")
    print(f"  Batch size:       1")
    print(f"  Sampling:         greedy (argmax)")
    print(f"  Model size:       {model_bytes / 1e6:.1f} MB")
    print(f"  Peak bandwidth:   {peak_bw:.0f} GB/s")
    print(f"  Theoretical max:  {theoretical_max:.0f} tok/s")
    print(f"  HF baseline:      {hf_baseline:.1f} tok/s")
    print(f"{'='*70}\n")

    if results:
        # Best result
        best = max(results, key=lambda r: r["toks_per_sec"])
        print(f"  PEAK: {best['toks_per_sec']:.0f} tok/s "
              f"({best['bw_utilization']:.1f}% BW, "
              f"{best['speedup_vs_hf']:.1f}x vs HuggingFace)")
        print()
        print(format_results_table(results))
    print(f"\n{'='*70}")
