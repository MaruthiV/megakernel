# megakernel

Skeleton implementation of a persistent megakernel project for Qwen3-0.6B.
This repo is intentionally minimal and focuses on structure and interfaces.

## Layout
- `kernels/` CUDA sources
- `python/megakernel/` Python launcher + autotuning
- `configs/` model and GPU configs
- `bench/` benchmark harness

## Quick start (planned)
1. Install deps: Python, PyTorch with CUDA.
2. Run a benchmark (this will build the extension on first run):
   - `python bench/run.py --gpu H100 --precision bf16 --pos 1,4096 --batch 1 --tokens 16`

## Status
The codebase includes a naive CUDA megakernel for a single-token, batch=1 decode
with a multi-layer loop and RoPE. It is correctness-first, float32-only, and
slow (limited parallelism; attention uses per-(head, dim) threads).
Note: attention uses `float4` vectorized loads when `head_dim` is a multiple of 4.
Note: for `kv_layout=0`, attention uses shared-memory tiling over sequence positions.

## Weight dict expected by `run_decode`
If you pass `weights`, provide a dict with:
- `rms_attn`: [layers, hidden]
- `rms_ffn`: [layers, hidden]
- `w_qkv`: [layers, hidden, 3*hidden]
- `w_o`: [layers, hidden, hidden]
- `w_gate`: [layers, hidden, 4*hidden]
- `w_up`: [layers, hidden, 4*hidden]
- `w_down`: [layers, 4*hidden, hidden]

## Correctness check
Run a simple reference comparison (GPU required):
- `python bench/verify.py --pos 0`

## Multi-step decode
You can run multiple positions by providing a sequence of embeddings:
- `from megakernel import run_decode_steps`
This now uses a single CUDA kernel over the sequence (persistent per-call).
`run_decode_steps` returns `(outputs, kv_cache)` where `kv_cache` can be fed back in.
Batch mode is supported by looping over batch for correctness (not optimized).

## Persistent mode (experimental)
Enable a multi-block persistent launch (currently correctness-only; extra blocks are idle):
- set `persistent: true` and `persistent_blocks: N` in the config
This now uses role-split phases with global barriers; only QKV/ATTN/MLP roles do work.
Prefetch role now touches weight slices to warm cache (best-effort).
Prefetch blocks are excluded from compute barriers to allow overlap (experimental).
Persistent mode currently assumes exactly 4 blocks (one per role) for correctness.
ATTN now writes to a global buffer so MLP can consume it across blocks.
ATTN role now supports multiple blocks (each handles a subset of heads).
QKV and MLP now also support multiple blocks (partitioned by output dimension).
QKV now writes Q to a global buffer so ATTN can read across blocks.
Spin-wait uses exponential backoff to reduce sync overhead (experimental).
Shared-memory tile size reduced to 16 to improve occupancy.
You can set `threads_per_block` in the config to tune kernel launch.

## KV quantization (experimental)
Set `kv_quant: int8` in the config to store an INT8 KV cache. The CUDA kernel
now supports an int8 KV path (kv_layout=0 only) with per-head scales.
Persistent mode also supports int8 KV for kv_layout=0.

## BF16 path (experimental)
The non-persistent seq kernel supports BF16 tensors (compute is still float).

## WMMA micro-bench
Test BF16 WMMA throughput:
- `python bench/wmma_bench.py --m 256 --n 256 --k 256 --iters 50`

## WMMA prefill demo
Demonstrate BF16 WMMA GEMMs for QKV/MLP (prefill-style):
- `python bench/prefill_wmma_demo.py --tokens 16`

## Prefill kernel demo
Run WMMA-backed prefill kernels:
- `python bench/prefill_kernel_demo.py --tokens 16`

## Prefill pipeline demo
Full prefill-style path with WMMA GEMMs + torch attention:
- `python bench/prefill_pipeline_demo.py --tokens 8`

## Qwen3 HF weight loader (experimental)
If you have `transformers` installed, you can map HF weights to the kernel format:
- `from megakernel import load_qwen3_weights_hf`
- `model_cfg, weights = load_qwen3_weights_hf(\"Qwen/Qwen3-0.6B\")`
You can request BF16 weights by passing `dtype=torch.bfloat16`.

Limitations:
- Only supports `num_heads * head_dim == hidden`
- Supports GQA when `num_heads` is divisible by `num_key_value_heads`

## HF correctness check
Requires `transformers` and a GPU:
- `python bench/hf_verify.py --model Qwen/Qwen3-0.6B --pos 0`
Multi-position HF correctness check:
- `python bench/hf_verify_multi.py --model Qwen/Qwen3-0.6B --pos 0 --tokens 4`
Persistent FP32 correctness check:
- `python bench/persistent_verify_fp32.py --tokens 4 --pos 0 --persistent-blocks 4`
Persistent int8 KV correctness check (kv_layout=0 only):
- `python bench/persistent_verify_int8.py --tokens 4 --pos 0 --persistent-blocks 4`
Persistent kv_layout=1 correctness check:
- `python bench/persistent_verify_kv_layout1.py --tokens 4 --pos 0 --persistent-blocks 4`

## RoPE check
Simple RoPE sanity check (GPU required):
- `python bench/rope_check.py --pos 0 --head-dim 64`
HF RoPE parity check (best-effort, requires transformers + GPU):
- `python bench/rope_hf_verify.py --model Qwen/Qwen3-0.6B --pos 0 --head-dim 64`

## Sampling benchmark
Greedy sampling on GPU:
- `python bench/sample_bench.py --vocab 151936 --iters 100`

## Profiling
See `bench/profile_note.md` and run:
- `python bench/profile_run.py --model-config configs/qwen3_0p6b_bf16.yaml --tokens 16 --pos 0`
Enable NVTX ranges by setting `nvtx: true` in the config.

## Validation checklist
Run the full correctness suite (GPU required):
- `python bench/validate_all.py --tokens 4`

## Autotune stub
Runs a stub autotune and writes candidate settings:
- `python bench/autotune.py --model-config configs/qwen3_0p6b_bf16.yaml`
Autotune run (persistent blocks sweep):
- `python bench/autotune_run.py --model-config configs/qwen3_0p6b_bf16.yaml --tokens 16 --pos 0`
Apply autotune result to config:
- `python bench/autotune_apply.py --config configs/qwen3_0p6b_bf16.yaml --autotune-json bench/results/autotune.json`
