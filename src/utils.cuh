#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ============================================================================
// Vectorized Load Helpers
// ============================================================================

// Load 8 BF16 values (16 bytes) at once via uint4
__device__ __forceinline__ void load_bf16x8(
    const __nv_bfloat16* ptr,
    __nv_bfloat16 out[8]
) {
    uint4 tmp = *reinterpret_cast<const uint4*>(ptr);
    *reinterpret_cast<uint4*>(out) = tmp;
}

// Load via read-only cache (__ldg) for weight data
__device__ __forceinline__ uint4 ldg_uint4(const void* ptr) {
    return __ldg(reinterpret_cast<const uint4*>(ptr));
}

// Load 4 floats via read-only cache
__device__ __forceinline__ float4 ldg_float4(const float* ptr) {
    return __ldg(reinterpret_cast<const float4*>(ptr));
}

// ============================================================================
// BF16 <-> Float Conversion
// ============================================================================

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

// ============================================================================
// Fast Math Intrinsics
// ============================================================================

// Fast exponential using hardware ex2.approx instruction
// exp(x) = exp2(x * log2(e)) = exp2(x * 1.4427f)
__device__ __forceinline__ float fast_exp(float x) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x * 1.4427f));
    return result;
}

// Fast reciprocal square root (maps to hardware rsqrt)
__device__ __forceinline__ float fast_rsqrt(float x) {
    return rsqrtf(x);
}

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + fast_exp(-x));
}

// ============================================================================
// Warp-Level Reductions
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// ============================================================================
// Block-Level Reductions (using shared memory)
// ============================================================================

// Requires shared memory of size [num_warps] floats
__device__ __forceinline__ float block_reduce_sum(float val, float* smem_scratch) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;

    // Warp-level reduction first
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    if (lane_id == 0) {
        smem_scratch[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across all warps
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? smem_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();

    // Broadcast result from thread 0
    if (threadIdx.x == 0) {
        smem_scratch[0] = val;
    }
    __syncthreads();

    return smem_scratch[0];
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem_scratch) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;

    val = warp_reduce_max(val);

    if (lane_id == 0) {
        smem_scratch[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < num_warps) ? smem_scratch[lane_id] : -INFINITY;
        val = warp_reduce_max(val);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        smem_scratch[0] = val;
    }
    __syncthreads();

    return smem_scratch[0];
}

// ============================================================================
// Block-level argmax reduction
// Returns both the max value and its index
// ============================================================================

struct MaxValIdx {
    float val;
    int idx;
};

__device__ __forceinline__ MaxValIdx warp_reduce_argmax(float val, int idx) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        int other_idx = __shfl_xor_sync(0xFFFFFFFF, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    return {val, idx};
}

// smem_vals and smem_idxs should have size [num_warps]
__device__ __forceinline__ MaxValIdx block_reduce_argmax(
    float val, int idx,
    float* smem_vals, int* smem_idxs
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;

    MaxValIdx result = warp_reduce_argmax(val, idx);

    if (lane_id == 0) {
        smem_vals[warp_id] = result.val;
        smem_idxs[warp_id] = result.idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? smem_vals[lane_id] : -INFINITY;
        int i = (lane_id < num_warps) ? smem_idxs[lane_id] : 0;
        result = warp_reduce_argmax(v, i);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        smem_vals[0] = result.val;
        smem_idxs[0] = result.idx;
    }
    __syncthreads();

    return {smem_vals[0], smem_idxs[0]};
}

// ============================================================================
// L2 Prefetch Helper
// ============================================================================

__device__ __forceinline__ void prefetch_l2(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
}

// Prefetch a range of memory into L2 cache
// Each thread prefetches different cache lines
__device__ __forceinline__ void prefetch_range_l2(
    const void* base,
    size_t bytes,
    int threads_per_block
) {
    const char* ptr = reinterpret_cast<const char*>(base);
    // Each cache line is 128 bytes
    for (size_t offset = threadIdx.x * 128; offset < bytes; offset += threads_per_block * 128) {
        prefetch_l2(ptr + offset);
    }
}
