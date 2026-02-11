#pragma once

#include "config.cuh"
#include "utils.cuh"

// ============================================================================
// KV Cache Management
//
// Layout: [NUM_LAYERS, 2, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM]
//   - Dimension 1: 0=key, 1=value
//   - seq_len dimension is contiguous for coalesced attention reads
//   - BF16 storage (FP8 quantization is a stretch goal)
//
// For Qwen3-0.6B:
//   Per token per layer: 2 * 8 * 64 * 2 bytes = 2048 bytes
//   Per token all layers: 2048 * 28 = 57,344 bytes (~56 KB)
//   At MAX_SEQ_LEN=4096: 56 KB * 4096 = ~229 MB
// ============================================================================

struct KVCache {
    __nv_bfloat16* data;  // [NUM_LAYERS, 2, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM]
    int current_pos;       // current sequence position (0-indexed)
};

// Compute byte offset into KV cache
__device__ __forceinline__ size_t kv_offset(
    int layer, int kv_type, int head, int pos, int dim
) {
    // kv_type: 0=key, 1=value
    return ((((size_t)layer * 2 + kv_type) * config::NUM_KV_HEADS + head)
            * config::MAX_SEQ_LEN + pos) * config::HEAD_DIM + dim;
}

// Get pointer to key cache for a specific layer and head
__device__ __forceinline__ const __nv_bfloat16* get_key_cache(
    const __nv_bfloat16* kv_data, int layer, int head
) {
    return kv_data + kv_offset(layer, 0, head, 0, 0);
}

// Get pointer to value cache for a specific layer and head
__device__ __forceinline__ const __nv_bfloat16* get_value_cache(
    const __nv_bfloat16* kv_data, int layer, int head
) {
    return kv_data + kv_offset(layer, 1, head, 0, 0);
}

// ============================================================================
// Append K, V to cache at current position
//
// Called by a single block (or cooperatively by all blocks with row splitting).
// k_proj: [KV_DIM] = [512] float (from QKV GEMV output)
// v_proj: [KV_DIM] = [512] float
//
// After RoPE is applied to K, we write K and V to the cache.
// ============================================================================

__device__ __forceinline__ void append_kv(
    __nv_bfloat16* kv_data,
    const float* k_proj,   // [KV_DIM] = [NUM_KV_HEADS * HEAD_DIM] after RoPE
    const float* v_proj,   // [KV_DIM]
    int layer,
    int pos
) {
    // Each thread writes a few elements
    for (int i = threadIdx.x; i < config::KV_DIM; i += blockDim.x) {
        int head = i / config::HEAD_DIM;
        int dim = i % config::HEAD_DIM;

        // Write key
        size_t k_idx = kv_offset(layer, 0, head, pos, dim);
        kv_data[k_idx] = float_to_bf16(k_proj[i]);

        // Write value
        size_t v_idx = kv_offset(layer, 1, head, pos, dim);
        kv_data[v_idx] = float_to_bf16(v_proj[i]);
    }
}

// ============================================================================
// Host allocation
// ============================================================================

__host__ inline KVCache allocate_kv_cache() {
    KVCache cache;
    size_t total_bytes = (size_t)config::NUM_LAYERS * 2 * config::NUM_KV_HEADS
                         * config::MAX_SEQ_LEN * config::HEAD_DIM * sizeof(__nv_bfloat16);
    cudaMalloc(&cache.data, total_bytes);
    cudaMemset(cache.data, 0, total_bytes);
    cache.current_pos = 0;
    return cache;
}

__host__ inline void free_kv_cache(KVCache& cache) {
    cudaFree(cache.data);
}
