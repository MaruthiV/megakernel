# Megakernel: Persistent CUDA Kernel for Qwen3-0.6B Inference

A single persistent CUDA kernel that runs the entire Qwen3-0.6B forward pass on-GPU, targeting maximum single-token decode throughput on Google Colab GPUs (A100, H100, T4).

## Architecture

Unlike standard frameworks that launch hundreds of separate CUDA kernels per token, this project fuses the entire transformer forward pass into **one persistent kernel** that stays resident on the GPU. Key techniques:

- **Atomic barriers** instead of cooperative groups — ~0.5-2us sync vs ~3-5us for `grid.sync()`, enabling 70%+ bandwidth utilization
- **Redundant RMSNorm** — all blocks compute norm independently, eliminating 56 barriers per token
- **Productive spin** — idle blocks prefetch next-phase weights into L2 while attention blocks work
- **Fused LM-head + argmax** — on-device sampling, no CPU readback during generation
- **Vectorized 128-bit loads** with L1 bypass for streaming weight data
- **Online softmax** for single-pass attention without materializing the full attention matrix
- **GQA support** — 2 Q-heads share 1 KV-head, matching Qwen3-0.6B's architecture

## Performance Targets

| GPU | Peak BW (GB/s) | Model Size | Roofline (tok/s) | Target (tok/s) | BW Util |
|-----|----------------|------------|-------------------|----------------|---------|
| A100 40GB | 1,555 | 926 MB | 1,679 | 1,200+ | 71%+ |
| H100 80GB | 3,350 | 926 MB | 3,617 | 2,600+ | 71%+ |
| T4 16GB | 300 | 926 MB | 324 | 200+ | 62%+ |

## Quick Start (Google Colab)

1. Open `megakernel_colab.ipynb` in Google Colab
2. Select **Runtime > Change runtime type > A100** (or H100/T4)
3. Run all cells top to bottom

The notebook handles everything: GPU detection, weight download, kernel compilation, correctness validation, and benchmarking.

## Project Structure

```
├── megakernel_colab.ipynb     # Main entry point (run in Colab)
├── src/
│   ├── megakernel.cu          # Persistent kernel + benchmark harness
│   ├── config.cuh             # Model constants + GPU auto-tune
│   ├── barriers.cuh           # Atomic barrier primitives
│   ├── gemv.cuh               # BF16 GEMV (fused QKV, fused gate+up+SiLU)
│   ├── attention.cuh          # GQA + online softmax + RoPE
│   ├── rmsnorm.cuh            # Redundant RMSNorm
│   ├── sampling.cuh           # Fused LM-head + argmax
│   ├── kv_cache.cuh           # Coalesced KV cache layout
│   └── utils.cuh              # Fast math intrinsics, vectorized loads
├── host/
│   ├── weight_loader.py       # HuggingFace → flat binary conversion
│   ├── launcher.py            # GPU detection + nvcc compilation
│   ├── benchmark.py           # Throughput measurement + formatting
│   └── validate.py            # Correctness validation vs PyTorch
├── Makefile                   # Multi-arch build (sm_75, sm_80, sm_90)
└── RESEARCH.md                # Technical analysis + references
```

## Building Locally

Requires CUDA toolkit with `nvcc` and PyTorch (for GPU detection):

```bash
# Auto-detect GPU and build
make

# Build for a specific architecture
make ARCH=sm_80

# Build shared library (for ctypes integration)
make lib

# Build for all Colab GPU types
make all-arch
```

## Model Details

**Qwen3-0.6B** (BF16):
- Hidden dim: 1024, Layers: 28
- Q-heads: 16, KV-heads: 8 (GQA ratio 2:1), Head dim: 64
- MLP: SwiGLU with intermediate dim 2816
- Vocab: 151,936 (tied embeddings)
- Total weights: ~926 MB in BF16

## Kernel Architecture

Each token generation step runs 5 phases per layer across all persistent blocks:

1. **RMSNorm + QKV GEMV** — all blocks compute input norm redundantly, then split QKV output rows → barrier
2. **RoPE + Attention** — 16 blocks compute attention (one per Q-head), idle blocks prefetch next weights → partial barrier
3. **O-projection GEMV** — all blocks → barrier
4. **RMSNorm + Gate+Up GEMV + SiLU** — fused MLP first half → barrier
5. **Down-projection + residual** — MLP second half → barrier

After 28 layers: final RMSNorm → fused LM-head argmax → next token written to device memory.

## References

- [MegaQwen](https://elliotarledge.com/blog/megaqwen) — original cooperative megakernel (530 tok/s, RTX 3090)
- [1k tok/s kernel](https://blog.alpindale.net) — non-cooperative persistent approach (1,000 tok/s, RTX 5090)
- [Hazy Research "No Bubbles"](https://hazyresearch.stanford.edu/blog/2025-05-20-megakernel) — instruction-interpreter megakernel (<1ms, H100)
