#pragma once

#include "config.cuh"
#include "utils.cuh"

// ============================================================================
// Redundant RMSNorm
//
// ALL blocks compute RMSNorm independently on the same input vector.
// This eliminates a grid barrier per norm (saves ~2-5us per norm).
//
// Cost: Each block reads 1024 BF16 values (~2 KB) from L2 and does
//       1024 FMA operations - trivial overhead.
// Benefit: Eliminates 2 grid barriers per layer * 28 layers = 56 barriers.
//          At 2-5us each, saves 112-280us per token.
// ============================================================================

// Compute RMSNorm: output = (input / rms(input)) * weight
// input:  [HIDDEN_DIM] in global memory (same address for all blocks, hits L2)
// weight: [HIDDEN_DIM] norm weight (read-only, __ldg cache)
// output: [HIDDEN_DIM] in global memory (each block writes its own copy,
//         or all blocks write the same result to the same address)
//
// smem_scratch: shared memory for block reduction, size [num_warps]
__device__ __forceinline__ void rmsnorm(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    float* output,  // float intermediate for use by subsequent GEMV
    float* smem_scratch,
    int dim
) {
    // Step 1: Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = bf16_to_float(__ldg(&input[i]));
        sum_sq += val * val;
    }

    // Step 2: Block-level reduction for sum of squares
    sum_sq = block_reduce_sum(sum_sq, smem_scratch);

    // Step 3: Compute scaling factor
    float rms_inv = fast_rsqrt(sum_sq / (float)dim + config::RMS_NORM_EPS);

    // Step 4: Normalize and scale
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = bf16_to_float(__ldg(&input[i]));
        float w = bf16_to_float(__ldg(&weight[i]));
        output[i] = val * rms_inv * w;
    }
}

// Per-head QK-norm: applies RMSNorm to each head independently
// Used by Qwen3 on Q and K vectors after projection, before RoPE.
// vec: [num_heads * HEAD_DIM] float (in-place)
// weight: [HEAD_DIM] bf16 (shared across all heads)
__device__ __forceinline__ void qk_rmsnorm_inplace(
    float* vec,
    const __nv_bfloat16* __restrict__ weight,
    int num_heads,
    int head_dim,
    float* smem_scratch
) {
    // Each head is normalized independently
    for (int h = 0; h < num_heads; h++) {
        float* head_vec = vec + h * head_dim;

        // Compute sum of squares for this head
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float val = head_vec[i];
            sum_sq += val * val;
        }
        sum_sq = block_reduce_sum(sum_sq, smem_scratch);
        float rms_inv = fast_rsqrt(sum_sq / (float)head_dim + config::RMS_NORM_EPS);

        // Normalize and scale
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float w = bf16_to_float(__ldg(&weight[i]));
            head_vec[i] = head_vec[i] * rms_inv * w;
        }
        __syncthreads();
    }
}

// Variant that reads float input (for when input is already in float)
__device__ __forceinline__ void rmsnorm_float_in(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    float* output,
    float* smem_scratch,
    int dim
) {
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = input[i];
        sum_sq += val * val;
    }

    sum_sq = block_reduce_sum(sum_sq, smem_scratch);
    float rms_inv = fast_rsqrt(sum_sq / (float)dim + config::RMS_NORM_EPS);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = input[i];
        float w = bf16_to_float(__ldg(&weight[i]));
        output[i] = val * rms_inv * w;
    }
}
