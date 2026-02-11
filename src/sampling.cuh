#pragma once

#include "config.cuh"
#include "utils.cuh"

// ============================================================================
// On-Device Sampling: Fused LM Head + Argmax
//
// The LM head uses tied embeddings: logit[v] = hidden @ embedding[v]
// where embedding is [VOCAB_SIZE, HIDDEN_DIM].
//
// For greedy decoding, we want argmax(logits).
// Instead of materializing all 151,936 logits and then reducing,
// we fuse the GEMV with argmax: each block computes partial logits
// for its vocab rows and tracks the local (max_val, max_idx).
// Then a single-block reduction finds the global argmax.
//
// This avoids writing 151,936 * 4 = ~593 KB of logits to global memory.
//
// Two-phase approach (from alpindale):
//   Phase 1: Each block scans its vocab chunk, emits (max_logit, max_index)
//   Phase 2: Single block reduces across all block-level results
// ============================================================================

// Phase 1: Each block computes logits for a chunk of the vocabulary
// and finds the local argmax.
//
// embedding: [VOCAB_SIZE, HIDDEN_DIM] BF16 (shared with token embedding)
// hidden:    [HIDDEN_DIM] float (after final RMSNorm)
// block_results_val: [num_blocks] float  (output: max logit per block)
// block_results_idx: [num_blocks] int    (output: argmax index per block)
__device__ void lm_head_argmax_phase1(
    const __nv_bfloat16* __restrict__ embedding,
    const float* __restrict__ hidden,
    float* block_results_val,
    int* block_results_idx,
    float* smem_input,      // [HIDDEN_DIM] for caching hidden state
    float* smem_vals,       // [num_warps] for reductions
    int* smem_idxs,         // [num_warps] for reductions
    int num_blocks
) {
    int vocab_per_block = (config::VOCAB_SIZE + num_blocks - 1) / num_blocks;
    int v_start = blockIdx.x * vocab_per_block;
    int v_end = min(v_start + vocab_per_block, config::VOCAB_SIZE);

    // Load hidden state into shared memory (all blocks read same data, hits L2)
    for (int i = threadIdx.x; i < config::HIDDEN_DIM; i += blockDim.x) {
        smem_input[i] = hidden[i];
    }
    __syncthreads();

    float best_val = -INFINITY;
    int best_idx = v_start;

    // Each thread processes multiple vocab rows
    for (int v = v_start + threadIdx.x; v < v_end; v += blockDim.x) {
        // Compute logit = embedding[v] . hidden
        const __nv_bfloat16* emb_row = embedding + (size_t)v * config::HIDDEN_DIM;
        float dot = 0.0f;

        // Vectorized load of embedding row
        for (int k = 0; k < config::HIDDEN_DIM; k += 8) {
            __nv_bfloat16 w[8];
            *reinterpret_cast<uint4*>(w) = ldg_uint4(&emb_row[k]);

            dot += bf16_to_float(w[0]) * smem_input[k + 0];
            dot += bf16_to_float(w[1]) * smem_input[k + 1];
            dot += bf16_to_float(w[2]) * smem_input[k + 2];
            dot += bf16_to_float(w[3]) * smem_input[k + 3];
            dot += bf16_to_float(w[4]) * smem_input[k + 4];
            dot += bf16_to_float(w[5]) * smem_input[k + 5];
            dot += bf16_to_float(w[6]) * smem_input[k + 6];
            dot += bf16_to_float(w[7]) * smem_input[k + 7];
        }

        if (dot > best_val) {
            best_val = dot;
            best_idx = v;
        }
    }

    // Block-level argmax reduction
    MaxValIdx result = block_reduce_argmax(best_val, best_idx, smem_vals, smem_idxs);

    if (threadIdx.x == 0) {
        block_results_val[blockIdx.x] = result.val;
        block_results_idx[blockIdx.x] = result.idx;
    }
}

// Phase 2: Single block reduces across all block-level results
// Only block 0 calls this after a grid barrier.
__device__ void lm_head_argmax_phase2(
    const float* block_results_val,  // [num_blocks]
    const int* block_results_idx,    // [num_blocks]
    int* output_token,               // single int: the chosen token
    float* smem_vals,
    int* smem_idxs,
    int num_blocks
) {
    float best_val = -INFINITY;
    int best_idx = 0;

    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        float v = block_results_val[i];
        if (v > best_val) {
            best_val = v;
            best_idx = block_results_idx[i];
        }
    }

    MaxValIdx result = block_reduce_argmax(best_val, best_idx, smem_vals, smem_idxs);

    if (threadIdx.x == 0) {
        *output_token = result.idx;
    }
}

// ============================================================================
// Host-side allocation for sampling buffers
// ============================================================================

struct SamplingBuffers {
    float* block_results_val;  // [max_blocks]
    int* block_results_idx;    // [max_blocks]
    int* output_token;         // single int
};

__host__ inline SamplingBuffers allocate_sampling_buffers(int max_blocks) {
    SamplingBuffers buf;
    cudaMalloc(&buf.block_results_val, max_blocks * sizeof(float));
    cudaMalloc(&buf.block_results_idx, max_blocks * sizeof(int));
    cudaMalloc(&buf.output_token, sizeof(int));
    return buf;
}

__host__ inline void free_sampling_buffers(SamplingBuffers& buf) {
    cudaFree(buf.block_results_val);
    cudaFree(buf.block_results_idx);
    cudaFree(buf.output_token);
}
