#pragma once

#include "config.cuh"
#include "utils.cuh"

// ============================================================================
// GEMV: Matrix-Vector Multiply for BF16 Weights
//
// For batch=1 decode, all weight multiplications are GEMV:
//   output[M] = weight[M, K] @ input[K]
//
// Parallelization strategy:
//   - Grid of blocks, each block handles (M / num_blocks) output rows
//   - Within a block, threads cooperate on the K dimension
//   - Vectorized 128-bit loads for weights (8 BF16 values per load)
//   - __ldg() for read-only cache path on weight data
//   - Accumulate in float32 for numerical stability
//
// For Qwen3-0.6B, the GEMV dimensions are:
//   QKV:      [1024, 2048] (fused Q+K+V)
//   O proj:   [1024, 1024]
//   Gate+Up:  [1024, 5632] (fused gate + up)
//   Down:     [2816, 1024]
//   LM head:  [1024, 151936] (embedding^T, done separately)
// ============================================================================

// GEMV: output[out_start..out_end] = weight[out_start..out_end, :K] @ input[:K]
// Each block computes a contiguous slice of the output vector.
//
// weight:    [M, K] in BF16, row-major (each row is one output element)
// input:     [K] float32 (already normalized by RMSNorm)
// output:    [M] float32
// smem_input: shared memory buffer [K] for the input vector (loaded once by the block)
//
// This function handles the slice assigned to the calling block.
__device__ __forceinline__ void gemv_block(
    const __nv_bfloat16* __restrict__ weight,  // full weight matrix [M, K]
    const float* __restrict__ input,            // input vector [K]
    float* __restrict__ output,                 // output vector [M]
    float* smem_input,                          // shared memory for input caching
    float* smem_scratch,                        // shared memory for reductions
    int M,                                      // output dimension
    int K,                                      // input dimension
    int num_blocks,                             // total blocks participating
    int block_id                                // this block's ID (0-indexed)
) {
    // Compute this block's row range
    int rows_per_block = (M + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, M);

    // Step 1: Cooperatively load input vector into shared memory
    // This is only K=1024 or K=2816 floats (4-11 KB), fits easily in SMEM
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        smem_input[i] = input[i];
    }
    __syncthreads();

    // Step 2: Each thread computes dot products for its assigned rows
    // With 512 threads per block and typically 8-10 rows per block,
    // we use multiple threads per row for the K-dimension reduction
    for (int row = row_start; row < row_end; row++) {
        // Pointer to this row's weights
        const __nv_bfloat16* w_row = weight + (size_t)row * K;

        // Each thread accumulates a partial sum over K dimension
        float acc = 0.0f;

        // Vectorized loads: 8 BF16 values (16 bytes) per iteration
        int k = threadIdx.x * 8;
        for (; k + 7 < K; k += blockDim.x * 8) {
            // Load 8 BF16 weights via read-only cache
            __nv_bfloat16 w[8];
            uint4 loaded = ldg_uint4(&w_row[k]);
            *reinterpret_cast<uint4*>(w) = loaded;

            // Multiply-accumulate with input from shared memory
            acc += bf16_to_float(w[0]) * smem_input[k + 0];
            acc += bf16_to_float(w[1]) * smem_input[k + 1];
            acc += bf16_to_float(w[2]) * smem_input[k + 2];
            acc += bf16_to_float(w[3]) * smem_input[k + 3];
            acc += bf16_to_float(w[4]) * smem_input[k + 4];
            acc += bf16_to_float(w[5]) * smem_input[k + 5];
            acc += bf16_to_float(w[6]) * smem_input[k + 6];
            acc += bf16_to_float(w[7]) * smem_input[k + 7];
        }

        // Handle remaining elements
        for (; k < K; k += blockDim.x) {
            if (k < K) {
                acc += bf16_to_float(__ldg(&w_row[k])) * smem_input[k];
            }
        }

        // Reduce across threads in the block
        acc = block_reduce_sum(acc, smem_scratch);

        // Thread 0 writes the result
        if (threadIdx.x == 0) {
            output[row] = acc;
        }
        __syncthreads();  // ensure smem_scratch is free for next row
    }
}

// ============================================================================
// Fused QKV GEMV
//
// Computes Q, K, V projections in a single pass over the input.
// output_q [Q_DIM], output_k [KV_DIM], output_v [KV_DIM]
//
// Weight layout: W_q [HIDDEN_DIM, Q_DIM] || W_k [HIDDEN_DIM, KV_DIM] || W_v [HIDDEN_DIM, KV_DIM]
// Can be stored as a single fused matrix [HIDDEN_DIM, Q_DIM + 2*KV_DIM]
// or computed separately. We do separate for clarity and because
// different blocks can handle Q vs K vs V for better load balancing.
// ============================================================================

__device__ __forceinline__ void gemv_qkv(
    const __nv_bfloat16* __restrict__ w_q,   // [HIDDEN_DIM, Q_DIM]
    const __nv_bfloat16* __restrict__ w_k,   // [HIDDEN_DIM, KV_DIM]
    const __nv_bfloat16* __restrict__ w_v,   // [HIDDEN_DIM, KV_DIM]
    const float* __restrict__ input,          // [HIDDEN_DIM] normalized
    float* __restrict__ output_q,             // [Q_DIM] = [1024]
    float* __restrict__ output_k,             // [KV_DIM] = [512]
    float* __restrict__ output_v,             // [KV_DIM] = [512]
    float* smem_input,
    float* smem_scratch,
    int num_blocks,
    int block_id
) {
    // Total output dimension = Q_DIM + KV_DIM + KV_DIM = 2048
    // Split rows across blocks
    int total_rows = config::Q_DIM + 2 * config::KV_DIM;  // 2048
    int rows_per_block = (total_rows + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, total_rows);

    // Load input into shared memory
    for (int i = threadIdx.x; i < config::HIDDEN_DIM; i += blockDim.x) {
        smem_input[i] = input[i];
    }
    __syncthreads();

    // Process each row in this block's range
    for (int row = row_start; row < row_end; row++) {
        // Determine which matrix and output buffer this row belongs to
        const __nv_bfloat16* w_row;
        float* out_ptr;
        int out_idx;

        if (row < config::Q_DIM) {
            // Q projection
            w_row = w_q + (size_t)row * config::HIDDEN_DIM;
            out_ptr = output_q;
            out_idx = row;
        } else if (row < config::Q_DIM + config::KV_DIM) {
            // K projection
            int k_row = row - config::Q_DIM;
            w_row = w_k + (size_t)k_row * config::HIDDEN_DIM;
            out_ptr = output_k;
            out_idx = k_row;
        } else {
            // V projection
            int v_row = row - config::Q_DIM - config::KV_DIM;
            w_row = w_v + (size_t)v_row * config::HIDDEN_DIM;
            out_ptr = output_v;
            out_idx = v_row;
        }

        // Dot product with vectorized loads
        float acc = 0.0f;
        int k = threadIdx.x * 8;
        for (; k + 7 < config::HIDDEN_DIM; k += blockDim.x * 8) {
            __nv_bfloat16 w[8];
            *reinterpret_cast<uint4*>(w) = ldg_uint4(&w_row[k]);

            acc += bf16_to_float(w[0]) * smem_input[k + 0];
            acc += bf16_to_float(w[1]) * smem_input[k + 1];
            acc += bf16_to_float(w[2]) * smem_input[k + 2];
            acc += bf16_to_float(w[3]) * smem_input[k + 3];
            acc += bf16_to_float(w[4]) * smem_input[k + 4];
            acc += bf16_to_float(w[5]) * smem_input[k + 5];
            acc += bf16_to_float(w[6]) * smem_input[k + 6];
            acc += bf16_to_float(w[7]) * smem_input[k + 7];
        }
        for (; k < config::HIDDEN_DIM; k += blockDim.x) {
            if (k < config::HIDDEN_DIM) {
                acc += bf16_to_float(__ldg(&w_row[k])) * smem_input[k];
            }
        }

        acc = block_reduce_sum(acc, smem_scratch);

        if (threadIdx.x == 0) {
            out_ptr[out_idx] = acc;
        }
        __syncthreads();
    }
}

// ============================================================================
// Fused Gate + Up GEMV with SiLU activation
//
// Computes: silu(input @ W_gate) * (input @ W_up)
// Both projections are [HIDDEN_DIM, INTERMEDIATE_DIM]
// Output: [INTERMEDIATE_DIM] float
//
// We fuse gate and up by interleaving: each block processes matched rows
// from both gate and up matrices, then applies SiLU fusion.
// ============================================================================

__device__ __forceinline__ void gemv_gate_up_silu(
    const __nv_bfloat16* __restrict__ w_gate,  // [HIDDEN_DIM, INTERMEDIATE_DIM]
    const __nv_bfloat16* __restrict__ w_up,    // [HIDDEN_DIM, INTERMEDIATE_DIM]
    const float* __restrict__ input,            // [HIDDEN_DIM] normalized
    float* __restrict__ output,                 // [INTERMEDIATE_DIM] = silu(gate) * up
    float* smem_input,
    float* smem_scratch,
    int num_blocks,
    int block_id
) {
    int rows_per_block = (config::INTERMEDIATE_DIM + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, config::INTERMEDIATE_DIM);

    // Load input
    for (int i = threadIdx.x; i < config::HIDDEN_DIM; i += blockDim.x) {
        smem_input[i] = input[i];
    }
    __syncthreads();

    for (int row = row_start; row < row_end; row++) {
        const __nv_bfloat16* g_row = w_gate + (size_t)row * config::HIDDEN_DIM;
        const __nv_bfloat16* u_row = w_up + (size_t)row * config::HIDDEN_DIM;

        float gate_acc = 0.0f;
        float up_acc = 0.0f;

        // Compute both dot products simultaneously
        int k = threadIdx.x * 8;
        for (; k + 7 < config::HIDDEN_DIM; k += blockDim.x * 8) {
            __nv_bfloat16 gw[8], uw[8];
            *reinterpret_cast<uint4*>(gw) = ldg_uint4(&g_row[k]);
            *reinterpret_cast<uint4*>(uw) = ldg_uint4(&u_row[k]);

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float inp = smem_input[k + j];
                gate_acc += bf16_to_float(gw[j]) * inp;
                up_acc += bf16_to_float(uw[j]) * inp;
            }
        }
        for (; k < config::HIDDEN_DIM; k += blockDim.x) {
            if (k < config::HIDDEN_DIM) {
                float inp = smem_input[k];
                gate_acc += bf16_to_float(__ldg(&g_row[k])) * inp;
                up_acc += bf16_to_float(__ldg(&u_row[k])) * inp;
            }
        }

        // Reduce both accumulators
        // We need two separate reductions - interleave using smem
        gate_acc = block_reduce_sum(gate_acc, smem_scratch);

        // Save gate result, then reduce up
        float gate_val = 0.0f;
        if (threadIdx.x == 0) {
            gate_val = gate_acc;
        }

        up_acc = block_reduce_sum(up_acc, smem_scratch);

        // Thread 0 applies SiLU fusion and writes output
        if (threadIdx.x == 0) {
            output[row] = silu(gate_val) * up_acc;
        }
        __syncthreads();
    }
}
