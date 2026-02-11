#pragma once

#include "config.cuh"
#include "kv_cache.cuh"
#include "utils.cuh"

// ============================================================================
// RoPE (Rotary Position Embeddings)
//
// Applied to Q and K vectors before attention.
// For Qwen3-0.6B: head_dim=64, rope_theta=1,000,000
//
// RoPE pairs elements (i, i + head_dim/2) and applies rotation:
//   q_new[i]              = q[i] * cos(theta) - q[i + half] * sin(theta)
//   q_new[i + half]       = q[i] * sin(theta) + q[i + half] * cos(theta)
//
// where theta = pos / (rope_theta ^ (2*i / head_dim))
//
// We use warp shuffles to exchange paired elements efficiently.
// ============================================================================

__device__ __forceinline__ void apply_rope_inplace(
    float* vec,    // [HEAD_DIM] vector (Q or K head)
    int pos,       // sequence position
    int head_dim   // HEAD_DIM = 64
) {
    int half = head_dim / 2;  // 32

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        // Compute rotation angle
        float freq = 1.0f / powf(config::ROPE_THETA, (float)(2 * i) / (float)head_dim);
        float theta = (float)pos * freq;
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        // Get paired elements
        float x0 = vec[i];
        float x1 = vec[i + half];

        // Apply rotation
        vec[i]        = x0 * cos_t - x1 * sin_t;
        vec[i + half] = x0 * sin_t + x1 * cos_t;
    }
}

// ============================================================================
// Single-Head Attention with Online Softmax
//
// For decode (batch=1), each Q head attends to its corresponding KV head.
// With GQA (2 Q heads per KV head), multiple Q heads share the same KV cache.
//
// Algorithm (online softmax - single pass over KV cache):
//   1. For each position t in [0, seq_len]:
//      a. dot = Q . K[t] / sqrt(head_dim)
//      b. Update running max: new_max = max(old_max, dot)
//      c. Correction factor: correction = exp(old_max - new_max)
//      d. Update exp_sum = exp_sum * correction + exp(dot - new_max)
//      e. Update output: output = output * correction + exp(dot - new_max) * V[t]
//   2. Final: output = output / exp_sum
//
// This avoids materializing the full attention matrix.
// ============================================================================

// Compute attention for a single Q head.
// q_head:     [HEAD_DIM] float (after RoPE)
// kv_data:    full KV cache
// output:     [HEAD_DIM] float attention output
// layer:      current layer index
// kv_head:    which KV head this Q head corresponds to
// seq_len:    number of KV entries (= current position + 1)
// smem_q:     shared memory for Q vector [HEAD_DIM]
// smem_scratch: shared memory for reductions [num_warps]
__device__ void single_head_attention(
    const float* q_head,           // [HEAD_DIM]
    const __nv_bfloat16* kv_data,  // full KV cache
    float* output,                 // [HEAD_DIM] result
    int layer,
    int kv_head,
    int seq_len,
    float* smem_q,
    float* smem_scratch
) {
    const int D = config::HEAD_DIM;  // 64
    float scale = 1.0f / sqrtf((float)D);

    // Load Q into shared memory for fast repeated access
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        smem_q[i] = q_head[i] * scale;  // pre-scale Q
    }
    __syncthreads();

    // Get pointers to K and V caches for this layer and head
    const __nv_bfloat16* k_cache = get_key_cache(kv_data, layer, kv_head);
    const __nv_bfloat16* v_cache = get_value_cache(kv_data, layer, kv_head);
    // k_cache layout: [MAX_SEQ_LEN, HEAD_DIM] with HEAD_DIM contiguous

    // Online softmax state
    float running_max = -INFINITY;
    float exp_sum = 0.0f;

    // Accumulator for weighted V values (per-thread, over assigned dims)
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // each thread handles up to 4 output dims

    // Determine which output dimensions this thread handles
    // With blockDim.x threads and HEAD_DIM=64, each thread handles a few dims
    // Actually, we need all threads to cooperate on the dot product for each position,
    // then each thread maintains its own output accumulator for its dimensions.

    // Strategy: iterate over KV positions. For each position:
    //  1. All threads cooperate to compute Q.K dot product (reduction)
    //  2. Thread 0 broadcasts the softmax weight
    //  3. Each thread updates its output dimensions with the weighted V

    for (int t = 0; t < seq_len; t++) {
        // Compute Q . K[t] (all threads cooperate)
        float dot = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            float k_val = bf16_to_float(k_cache[t * D + d]);
            dot += smem_q[d] * k_val;  // smem_q already scaled
        }
        dot = block_reduce_sum(dot, smem_scratch);
        // Now dot is the attention score for position t (in thread 0, broadcast via smem)

        // Online softmax update
        float old_max = running_max;
        running_max = fmaxf(running_max, dot);
        float correction = fast_exp(old_max - running_max);
        float weight = fast_exp(dot - running_max);
        exp_sum = exp_sum * correction + weight;

        // Update output accumulator with weighted V[t]
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            float v_val = bf16_to_float(v_cache[t * D + d]);
            // acc for dimension d (stored at d's local index)
            // We need per-dimension accumulators in registers
            // Since each thread handles D/blockDim.x dimensions, use a loop
            // For now, use global memory output as accumulator (will optimize later)
        }

        // Actually, let's use a simpler approach: each thread tracks dims it owns
        // For HEAD_DIM=64 and blockDim.x=512, most threads do nothing.
        // Better: assign fewer threads to attention (e.g., one warp per head)
    }

    // Let me restructure: use just 1-2 warps for the dot product
    // and update output in the same threads.
    // ACTUALLY: For a proper implementation with HEAD_DIM=64:
    // Use one warp (32 threads). Each thread handles 2 dimensions of Q,K,V.

    // Clear previous state - let's redo with warp-level implementation
}

// ============================================================================
// Warp-level single head attention (optimized)
//
// Uses exactly one warp (32 threads) per Q head.
// Each thread handles 2 dimensions (HEAD_DIM=64 / 32 threads = 2 per thread).
// Dot product via warp_reduce_sum.
// ============================================================================

__device__ void warp_attention(
    const float* q_head,           // [HEAD_DIM] Q vector (after RoPE, in global mem)
    const __nv_bfloat16* kv_data,  // full KV cache
    float* output,                 // [HEAD_DIM] result (in global mem)
    int layer,
    int kv_head,
    int seq_len
) {
    const int D = config::HEAD_DIM;  // 64
    const int DIMS_PER_THREAD = D / 32;  // 2
    float scale = 1.0f / sqrtf((float)D);

    int lane = threadIdx.x % 32;

    // Load Q into registers (each thread owns 2 dimensions)
    float q_reg[DIMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; i++) {
        q_reg[i] = q_head[lane * DIMS_PER_THREAD + i] * scale;
    }

    const __nv_bfloat16* k_cache = get_key_cache(kv_data, layer, kv_head);
    const __nv_bfloat16* v_cache = get_value_cache(kv_data, layer, kv_head);

    // Online softmax state (each thread maintains its own for output dims)
    float running_max = -INFINITY;
    float exp_sum = 0.0f;
    float acc[DIMS_PER_THREAD] = {0.0f};

    for (int t = 0; t < seq_len; t++) {
        // Load K[t] (each thread loads its 2 dims)
        float k_reg[DIMS_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD; i++) {
            k_reg[i] = bf16_to_float(k_cache[t * D + lane * DIMS_PER_THREAD + i]);
        }

        // Compute dot product Q.K[t]
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD; i++) {
            dot += q_reg[i] * k_reg[i];
        }
        dot = warp_reduce_sum(dot);
        // Now all lanes have the same dot value

        // Online softmax
        float old_max = running_max;
        running_max = fmaxf(running_max, dot);
        float correction = fast_exp(old_max - running_max);
        float weight = fast_exp(dot - running_max);
        exp_sum = exp_sum * correction + weight;

        // Load V[t] and accumulate
        #pragma unroll
        for (int i = 0; i < DIMS_PER_THREAD; i++) {
            float v_val = bf16_to_float(v_cache[t * D + lane * DIMS_PER_THREAD + i]);
            acc[i] = acc[i] * correction + weight * v_val;
        }
    }

    // Normalize by exp_sum and write output
    float inv_sum = 1.0f / exp_sum;
    #pragma unroll
    for (int i = 0; i < DIMS_PER_THREAD; i++) {
        output[lane * DIMS_PER_THREAD + i] = acc[i] * inv_sum;
    }
}

// ============================================================================
// Full attention for all Q heads in a layer
//
// This function is called by the attention blocks (block_id < NUM_Q_HEADS).
// Each block handles one Q head using the first warp.
// The remaining warps in the block are idle during attention (but the whole
// block is used for prefetching later).
//
// q_proj:  [Q_DIM] = [1024] float (16 heads * 64 dims)
// k_proj:  [KV_DIM] = [512] float (already written to KV cache via append_kv)
// output:  [Q_DIM] = [1024] float (concatenated head outputs)
// ============================================================================

__device__ void multi_head_attention(
    const float* q_proj,           // [Q_DIM] after RoPE
    const __nv_bfloat16* kv_data,
    float* output,                 // [Q_DIM] concatenated attention output
    int layer,
    int seq_len,                   // number of KV entries
    int head_id                    // which Q head this block handles (0..15)
) {
    int kv_head = head_id / config::GQA_RATIO;  // GQA: 2 Q heads per KV head

    // Only the first warp does the actual attention computation
    if (threadIdx.x < 32) {
        warp_attention(
            q_proj + head_id * config::HEAD_DIM,     // this head's Q
            kv_data,
            output + head_id * config::HEAD_DIM,     // this head's output slot
            layer,
            kv_head,
            seq_len
        );
    }
    // Other warps in this block are free for prefetching (handled by caller)
}
