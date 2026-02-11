# MegaKernel Research: Beating MegaQwen and the 1k tok/s Barrier

## Executive Summary

We're building a persistent megakernel for Qwen3-0.6B that fuses the entire decode forward pass into a single CUDA kernel. Our targets:

| GPU | BF16 Target | FP8 Target | Theoretical Roofline (BF16) |
|-----|------------|------------|---------------------------|
| H100 SXM | 2,600+ tok/s | 4,500+ tok/s | ~3,450 tok/s |
| A100 40GB | 1,200+ tok/s | 2,200+ tok/s | ~1,600 tok/s |
| T4 | 200+ tok/s | N/A (no FP8) | ~309 tok/s |

For comparison:
- MegaQwen (cooperative, RTX 3090): **530 tok/s** short context, 158 tok/s long context
- Alpindale (persistent, RTX 5090): **1,000 tok/s** short context
- Hazy Research (instruction interpreter, H100): **<1 ms/token** on Llama-1B

---

## 1. Landscape Analysis: What Exists and Why

### 1.1 MegaQwen (Elliot Arledge)
- **Repo**: github.com/Infatoshi/megakernels
- **Architecture**: 82 cooperative CUDA blocks, all operations fused into one `cudaLaunchCooperativeKernel`
- **Sync**: ~225 `grid.sync()` calls per token (8 per layer x 28 layers + extras)
- **Key innovation**: Block divergence + L2 prefetch. During attention (16 blocks working), 66 idle blocks prefetch MLP weights into L2 cache. This was the +2x win.
- **Results**: 530 tok/s short context, 158 tok/s long context (RTX 3090)
- **Critical finding**: NOT bandwidth-bound. Only 5% bandwidth utilization (~47 GB/s of 936 GB/s). **Synchronization-bound** - grid.sync() dominates wall time.
- **Failed optimizations**: Warp producer/consumer split (0%), shared memory caching (0%), cp.async double-buffering (+1%)

### 1.2 Alpindale's 1k tok/s Kernel (RTX 5090)
- **Architecture**: 128 persistent thread blocks x 512 threads, NON-cooperative launch
- **Sync**: Custom atomic barriers with monotonic generation counters + flag-based partial barriers for attention-only phases
- **Key innovations**:
  1. Atomic barriers instead of cooperative `grid.sync()` (~2.2us vs ~3us per barrier)
  2. Flag-based partial barriers for attention: only 16 blocks participate, saving ~56us/step
  3. Productive spin: 112 idle blocks prefetch next-phase weights during attention (~23 MB of O/gate/up/down projection weights)
  4. Redundant RMSNorm: all 128 blocks compute norm independently (eliminates barrier)
  5. `fence.acq_rel.gpu` instead of `__threadfence()` (lighter on Blackwell)
  6. `ptx_exp2(x * 1.4427f)` -> `ex2.approx.ftz.f32` (10x faster than expf())
  7. L1-bypass hint: `ld.global.L1::no_allocate.v4.b32` for weights
  8. Two-phase LM head argmax achieving ~1,500 GB/s (90% theoretical)
- **Results**: 1,000 tok/s (71.2% of 1,674 GB/s effective bandwidth on 5090)
- **Optimization progression**: 494 -> 813 -> 890 -> 905 -> 1,000 tok/s
- **Barrier overhead**: Still 33.6% of per-layer time

### 1.3 Hazy Research "No Bubbles" Megakernel
- **Architecture**: Instruction-and-interpreter model. Pre-computed instruction schedule, each SM executes a sequence of fine-grained instructions.
- **Key innovations**:
  1. Shared memory paging: H100's 213KB SMEM divided into 13 x 16KB pages, explicitly managed
  2. Counter-based sync in global memory (similar to atomic barriers but more granular)
  3. MLP intermediates processed in 4 chunks with individual counters (more parallelism)
  4. Weight loading pipelined across instructions (loads for instruction N+1 begin during instruction N)
- **Results**: <1ms per forward pass on H100 (78% bandwidth utilization), <680us on B200
- **Key insight**: "Memory pipeline bubbles" from discrete kernel launches waste 20-50% of bandwidth even with CUDA graphs
- **Code**: github.com/HazyResearch/Megakernels

### 1.4 Mirage (CMU/Zhihao Jia)
- **Approach**: Compiler that automatically generates megakernels from model definitions
- **Repo**: github.com/mirage-project/mirage
- **Insight**: Treats megakernel generation as a search/compilation problem rather than hand-coding

---

## 2. Qwen3-0.6B Architecture Details

```
hidden_size:           1024
num_hidden_layers:     28
num_attention_heads:   16
num_key_value_heads:   8  (GQA, 2:1 ratio)
head_dim:              64
intermediate_size:     2816 (SwiGLU MLP)
vocab_size:            151,936
max_position_embeddings: 32768
rope_theta:            1,000,000
rms_norm_eps:          1e-6
tie_word_embeddings:   true (LM head = embedding^T)
```

### Per-Layer Weight Sizes (BF16)
| Weight | Shape | Size (bytes) |
|--------|-------|-------------|
| W_q | [1024, 1024] | 2,097,152 |
| W_k | [1024, 512] | 1,048,576 |
| W_v | [1024, 512] | 1,048,576 |
| W_o | [1024, 1024] | 2,097,152 |
| W_gate | [1024, 2816] | 5,767,168 |
| W_up | [1024, 2816] | 5,767,168 |
| W_down | [2816, 1024] | 5,767,168 |
| RMSNorm x2 | [1024] x2 | 4,096 |
| **Layer total** | | **23,596,956 (~22.5 MB)** |

**All 28 layers**: ~630 MB in BF16
**Embedding/LM head**: 151,936 x 1024 x 2 = ~296 MB in BF16
**Total model**: ~926 MB in BF16 (~463 MB in FP8)

### KV Cache Per Token
```
Per layer: 2 * 8 heads * 64 dim * 2 bytes = 2,048 bytes
All layers: 2,048 * 28 = 57,344 bytes (~56 KB per token)
At 4096 context: ~229 MB
```

### Per-Layer Decode Flow (batch=1)
```
Input: x [1024]  (2 KB in BF16)

1. RMSNorm(x) -> x_norm [1024]
2. Q = x_norm @ W_q  [1024x1024] -> q [1024] (reshaped to [16, 64])
3. K = x_norm @ W_k  [1024x512]  -> k [512]  (reshaped to [8, 64])
4. V = x_norm @ W_v  [1024x512]  -> v [512]  (reshaped to [8, 64])
5. RoPE(Q, K) with position encoding
6. Append K, V to KV cache
7. Attention: for each of 16 Q heads (2 Q heads share 1 KV head):
   score = Q_head @ K_cache^T / sqrt(64)  -> [seq_len]
   weights = softmax(score)
   attn_out = weights @ V_cache -> [64]
8. O = concat(attn_outs) @ W_o [1024x1024] -> [1024]
9. residual = x + O
10. RMSNorm(residual) -> h [1024]
11. gate = h @ W_gate [1024x2816] -> [2816]
12. up   = h @ W_up   [1024x2816] -> [2816]
13. mlp_out = SiLU(gate) * up
14. down = mlp_out @ W_down [2816x1024] -> [1024]
15. output = residual + down
```

---

## 3. GPU Specifications (Colab Available)

### H100 SXM (Best case on Colab - rare but available with Pro+)
- Architecture: Hopper (SM 9.0)
- SMs: 132
- Memory: 80 GB HBM3
- Bandwidth: 3,350 GB/s
- L2 Cache: 50 MB
- Tensor Cores: 528 (4th gen, native FP8)
- Peak FP16: 495 TFLOPS (dense)
- Peak FP8: 989 TOPS (dense)
- **Special features**: TMA, thread block clusters, distributed shared memory, FP8 native
- **Theoretical max (BF16)**: 3,350 / 0.926 = **3,617 tok/s**
- **Theoretical max (FP8)**: 3,350 / 0.463 = **7,235 tok/s**

### A100 40GB SXM (Common on Colab Pro)
- Architecture: Ampere (SM 8.0)
- SMs: 108
- Memory: 40 GB HBM2e
- Bandwidth: 1,555 GB/s (40GB variant)
- L2 Cache: 40 MB
- Tensor Cores: 432 (3rd gen, no FP8)
- Peak BF16: 312 TFLOPS
- **Special features**: cp.async, large L2, INT8 tensor cores
- **Theoretical max (BF16)**: 1,555 / 0.926 = **1,679 tok/s**
- **Theoretical max (INT8)**: 1,555 / 0.463 = **3,358 tok/s**

### T4 (Free tier, for development/testing)
- Architecture: Turing (SM 7.5)
- SMs: 40
- Memory: 16 GB GDDR6
- Bandwidth: 300 GB/s
- L2 Cache: 4 MB
- **Theoretical max (BF16)**: 300 / 0.926 = **324 tok/s**

### Colab Limitations
- A100 is typically 40GB variant (not 80GB, so 1,555 GB/s not 2,039 GB/s)
- Possible clock throttling under sustained load
- Single GPU only (no NVLink/multi-GPU)
- Network-mounted disk (slow model loading, prefer RAM)
- Runtime limits: ~12h max session, ~24h with Pro+

---

## 4. Roofline Analysis

### Why Decode is Pure Bandwidth
```
Arithmetic Intensity = 2 * batch_size / sizeof(weight_dtype)

For batch=1, BF16: AI = 2 * 1 / 2 = 1 FLOP/byte
For batch=1, FP8:  AI = 2 * 1 / 1 = 2 FLOPs/byte

H100 compute-to-BW ratio: 495,000 GFLOPS / 3,350 GB/s = 148 FLOPs/byte

We need 148 FLOPs/byte to be compute-bound.
At 1-2 FLOPs/byte, we are 74-148x below compute roofline.
=> ENTIRELY MEMORY BANDWIDTH BOUND
```

### What Steals Bandwidth
At perfect efficiency, every byte of bandwidth goes to reading model weights. In practice:
1. **Synchronization overhead**: Barriers stall all threads, wasting cycles
2. **KV cache reads**: Additional bandwidth for attention (grows with context)
3. **L2 misses**: Not all weight accesses hit L2 on re-reads
4. **Memory access inefficiency**: Non-coalesced accesses, bank conflicts
5. **Instruction overhead**: Address computation, loop control
6. **Sampling/output**: LM head + argmax

### Bandwidth Budget Per Token (BF16, short context)
```
Must-read bytes:
  28 layers x 22.5 MB = 630 MB (layer weights)
  296 MB (LM head / embedding)
  ~2 KB per layer (activations from L2, negligible)
  ~56 KB (KV cache at position 0, negligible)
  ----------------------------------------
  Total: ~926 MB

On H100: 926 MB / 3,350 GB/s = 276 us -> 3,623 tok/s theoretical
On A100: 926 MB / 1,555 GB/s = 595 us -> 1,681 tok/s theoretical
On T4:   926 MB / 300 GB/s  = 3,087 us -> 324 tok/s theoretical
```

### Efficiency Targets
Based on what alpindale achieved (71.2% on 5090) and Hazy Research (78% on H100):

| GPU | 70% efficiency | 80% efficiency | 90% efficiency |
|-----|---------------|----------------|----------------|
| H100 | 2,536 tok/s | 2,898 tok/s | 3,261 tok/s |
| A100 | 1,177 tok/s | 1,345 tok/s | 1,513 tok/s |
| T4 | 227 tok/s | 259 tok/s | 291 tok/s |

---

## 5. Technical Deep Dive: Key Techniques

### 5.1 Persistent Kernel Architecture
Instead of launching hundreds of kernels per token, launch ONE kernel that stays resident:
```
blocks stay alive -> process all 28 layers -> emit token -> loop
```

**Why it wins**: Eliminates kernel launch overhead (3-10us each, ~100+ launches per token = 300-1000us saved), keeps activations in L2/registers, enables cross-operation prefetching.

**Launch configuration**: Must ensure all blocks fit simultaneously. Rule of thumb: num_blocks <= num_SMs * max_blocks_per_SM. Typical: 1 block per SM or 2 blocks per SM.

### 5.2 Atomic Barriers vs Cooperative Groups

**Cooperative groups (MegaQwen)**:
- `cudaLaunchCooperativeKernel` + `grid.sync()`
- ~2-5us per sync on consumer GPUs
- 225 syncs/token = 450-1125us overhead (!)
- Guaranteed correctness by driver
- Limited to occupancy-1 blocks per SM

**Atomic barriers (alpindale, our approach)**:
- Regular kernel launch, blocks stay persistent
- Custom sense-reversing barrier with monotonic generation counter
- ~0.5-2us per sync
- Must manually guarantee all blocks are simultaneously resident
- Can do PARTIAL barriers (huge win for attention phases)

**Our implementation should use**:
```cuda
// Sense-reversing atomic barrier with monotonic generation
__device__ void grid_barrier(
    int* barrier_counter,      // global memory
    int* barrier_generation,   // global memory
    int num_blocks,
    int* local_gen             // per-block tracking
) {
    __syncthreads(); // intra-block sync first

    if (threadIdx.x == 0) {
        int gen = atomicAdd(barrier_counter, 1);
        if (gen == num_blocks - 1) {
            // Last block to arrive
            *barrier_counter = 0;
            __threadfence(); // ensure counter reset is visible
            atomicAdd(barrier_generation, 1);
        }
        // Wait: use > comparison (monotonic) to avoid ABA
        while (atomicAdd(barrier_generation, 0) <= *local_gen) {
            // spin (optionally __nanosleep for power)
        }
        (*local_gen)++;
    }
    __syncthreads(); // ensure all threads see the barrier completion
}
```

**Flag-based partial barriers** (for attention-only sync):
```cuda
// Only 16 attention blocks participate
__device__ void attention_barrier(int* flag, int layer, int num_attn_blocks) {
    if (threadIdx.x == 0) {
        atomicAdd(flag, 1);
        while (atomicAdd(flag, 0) < num_attn_blocks * (layer + 1)) {
            // spin
        }
    }
    __syncthreads();
}
```

### 5.3 Block Organization Strategy

For Qwen3-0.6B with hidden_dim=1024:
- **QKV GEMV**: All blocks participate. Each block computes a slice of the output.
- **Attention**: 16 blocks (one per Q head). Remaining blocks do productive spin (prefetch).
- **O projection**: All blocks participate.
- **Gate+Up GEMV**: All blocks participate (can fuse gate and up into one pass).
- **Down GEMV**: All blocks participate.

**Block count selection**:
- H100: 132 SMs -> use 132 blocks (1 per SM) or 128 for power-of-2 convenience
- A100: 108 SMs -> use 108 blocks
- T4: 40 SMs -> use 40 blocks
- Auto-detect at launch time!

### 5.4 Weight Loading Optimization

**Vectorized loads** (128-bit / uint4):
```cuda
uint4* w_ptr = reinterpret_cast<uint4*>(weight_row + offset);
uint4 w = w_ptr[threadIdx.x]; // 16 bytes per thread per load
// 32 threads * 16 bytes = 512 bytes per warp per load (4 cache lines)
```

**L1 bypass for weights** (don't pollute L1 with streaming weight data):
```cuda
// PTX: ld.global.L1::no_allocate.v4.b32
asm volatile("ld.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3}, [%4];"
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
    : "l"(ptr));
```

**__ldg() for read-only cache path**:
```cuda
float val = __ldg(&weights[idx]); // Uses texture/read-only cache
```

### 5.5 Productive Spin (Idle Block Prefetching)

During attention (only 16 blocks active, reading ~400KB KV cache),
memory subsystem is 95%+ idle. Use idle blocks to prefetch next weights:

```cuda
if (block_is_idle_during_attention) {
    // Prefetch O-proj, gate, up, down weights for current layer
    // ~23 MB of weights warmed into L2
    for (int offset = threadIdx.x * 128; offset < weight_bytes; offset += blockDim.x * 128) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(weight_ptr + offset));
    }
}
```

This was the difference between 905 and 1000 tok/s in alpindale's kernel.

### 5.6 Redundant RMSNorm (Eliminate Barriers)

Instead of one block computing RMSNorm + barrier + all blocks use result:
ALL blocks compute RMSNorm independently.

Cost: 1024-element reduction per block (trivial, ~1us)
Benefit: Eliminates one grid barrier per norm (saves ~2-5us x 56 norms = 112-280us)

```cuda
// Every block does this independently
__device__ void redundant_rmsnorm(float* input, float* weight, float* output, int dim) {
    // Read input from global/L2 (same address, hits L2)
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = input[i];
        sum_sq += val * val;
    }
    // Warp reduction + block reduction for sum_sq
    sum_sq = block_reduce_sum(sum_sq);
    float rms = rsqrtf(sum_sq / dim + 1e-6f);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output[i] = input[i] * rms * weight[i];
    }
}
```

### 5.7 On-Device Sampling (No CPU Readback)

**Fused LM head + argmax** (alpindale's approach):
```
Phase 1: Multiple blocks scan vocab rows, each finds local (max_logit, max_index)
Phase 2: Single block tree-reduces across block-level results
Output: single token_id written to device memory
```

This avoids materializing the full 151,936-element logit vector.

### 5.8 Fast Math Intrinsics
```cuda
// Fast exp2 approximation (10x faster than expf)
__device__ float fast_exp(float x) {
    return __expf(x);  // Or even faster:
    // asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x * 1.4427f));
}

// Fast rsqrt for RMSNorm
__device__ float fast_rsqrt(float x) {
    return rsqrtf(x);  // Maps to hardware rsqrt
}
```

---

## 6. How We Beat Both MegaQwen and the 1k Post

### 6.1 Our Advantages

1. **Better GPUs**: H100 has 3,350 GB/s (2x the 5090's ~1,674 GB/s). Even the A100 at 1,555 GB/s outperforms a 3090.
2. **We can learn from both implementations' mistakes and successes**
3. **H100-specific features**: TMA, thread block clusters, FP8 native, larger L2 (50MB)

### 6.2 Architecture: Non-Cooperative Persistent Kernel (Like Alpindale, Not MegaQwen)

Definitive choice: **atomic barriers, NOT cooperative groups**. The 71.2% bandwidth utilization vs MegaQwen's 5% makes this obvious.

### 6.3 Key Improvements Over Alpindale

**A. Hierarchical Barriers (H100 only)**
Thread block clusters enable cluster-level sync at ~100-200ns (vs ~1-2us for grid atomics).
Use cluster barriers for tightly-coupled phases, grid atomics only for layer boundaries.
```
Cluster sync: 100-200 ns (for within-phase ops)
Grid atomic: 500-2000 ns (for phase boundaries)
```
Could cut barrier overhead from 33.6% to ~15-20%.

**B. TMA for Weight Loading (H100 only)**
Tensor Memory Accelerator offloads address computation from CUDA cores:
- Frees warps to do compute while hardware unit handles data movement
- Native tiled addressing
- Can sustain near-peak HBM bandwidth with minimal warp overhead

**C. FP8 Weights (H100 only, INT8 on A100)**
Halves model size from 926 MB to 463 MB -> doubles theoretical throughput.
Per-channel scaling maintains quality. FP8 E4M3 has enough range for Qwen3 weights.

**D. Chunked MLP with Finer-Grained Sync (From Hazy Research)**
Instead of waiting for ALL of gate+up to complete before starting down-proj:
Process MLP intermediates in 4 chunks, each with its own counter.
Blocks that finish chunk 0 of gate+up can start computing chunk 0 of SiLU+down
while other blocks still compute chunk 3 of gate+up.

**E. Shared Memory Paging (From Hazy Research)**
Divide SMEM into pages, explicitly manage allocation across phases.
Enables immediate weight loading as SMEM pages free up from previous operations.

**F. Auto-Tuning Launch Parameters**
```python
def auto_tune(gpu_name):
    if "H100" in gpu_name:
        return {"blocks": 132, "threads": 512, "attn_blocks": 16}
    elif "A100" in gpu_name:
        return {"blocks": 108, "threads": 512, "attn_blocks": 16}
    elif "T4" in gpu_name:
        return {"blocks": 40, "threads": 256, "attn_blocks": 16}
```

### 6.4 Long-Context Fix (MegaQwen's Achilles Heel)

MegaQwen drops from 530 -> 158 tok/s at longer contexts. Our fixes:

1. **KV cache quantization**: FP8 KV cache (half the bandwidth per cached token)
2. **Coalesced KV layout**: Store KV as [layer, head, seq_len, dim] with seq_len dimension contiguous for coalesced attention reads
3. **Split-K attention at long context**: When seq_len > threshold, distribute KV reads across multiple blocks instead of one block per head
4. **L2 residency for recent KV**: Pin recent KV entries in L2 using `cudaAccessPolicyWindow`

---

## 7. Implementation Plan

### Phase 1: Baseline Reproduction (Week 1)
- Clone MegaQwen repo, build and run on Colab T4/A100/H100
- Record benchmark numbers at multiple context positions
- Profile with Nsight Compute to understand bottlenecks per GPU
- Understand the existing code structure thoroughly

### Phase 2: Non-Cooperative Persistent Kernel (Week 2-3)
- Rewrite kernel launch from cooperative to regular persistent
- Implement sense-reversing atomic barriers
- Implement flag-based partial barriers for attention phases
- Implement productive spin / idle-block prefetching
- Implement redundant RMSNorm
- Target: match or exceed MegaQwen numbers on same GPU class

### Phase 3: H100/A100 Optimizations (Week 3-4)
- Auto-tune launch parameters per GPU
- Implement L1-bypass weight loads
- 128-bit vectorized loads everywhere
- Fast math intrinsics (fast exp, rsqrt)
- On-device argmax sampling (eliminate CPU readback)
- Target: 70%+ bandwidth utilization

### Phase 4: Advanced Optimizations (Week 4-5)
- FP8 weight support (H100) / INT8 (A100)
- KV cache quantization to FP8
- Coalesced KV cache layout
- Split-K attention for long context
- Chunked MLP with finer-grained sync
- TMA weight loading (H100 only)
- Thread block clusters with hierarchical barriers (H100 only)
- Target: 80%+ bandwidth utilization, good long-context performance

### Phase 5: Benchmarking & Documentation (Week 5-6)
- Comprehensive benchmarks: every GPU, multiple context lengths, BF16 vs FP8
- Comparison table vs MegaQwen, TensorRT-LLM, vLLM
- Profile-guided final tuning
- Blog post / documentation

---

## 8. Key References

- MegaQwen Blog: https://elliotarledge.com/blog/megaqwen
- MegaQwen Code: https://github.com/Infatoshi/megakernels
- 1k tok/s Blog: https://blog.alpindale.net/posts/5090_decode_optimization/
- Hazy Research "No Bubbles": https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- Hazy Research Code: https://github.com/HazyResearch/Megakernels
- Mirage Compiler: https://github.com/mirage-project/mirage
- "We Bought the Whole GPU": https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main
