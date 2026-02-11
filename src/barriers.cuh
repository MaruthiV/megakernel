#pragma once

#include <cuda_runtime.h>

// ============================================================================
// Atomic Barrier Primitives for Persistent Megakernel
//
// These replace cooperative groups grid.sync() with much lower overhead.
// Key insight from alpindale's 1k tok/s kernel: use monotonic generation
// counters to prevent ABA race conditions.
//
// On A100, atomic barriers cost ~0.5-2us vs ~3-5us for grid.sync().
// With 4 barriers per layer * 28 layers = 112 barriers, saving 1-3us each
// saves 112-336us per token.
// ============================================================================

// State for grid-wide barrier (allocated in global memory)
struct BarrierState {
    int counter;        // atomically incremented by arriving blocks
    int generation;     // monotonically increasing generation counter
};

// State for partial barrier (attention-only sync)
struct PartialBarrierState {
    int flag;           // monotonic counter for lightweight sync
};

// ============================================================================
// Grid-wide atomic barrier
//
// All num_blocks blocks must call this. Uses monotonic generation counter
// to prevent ABA problems. Each block tracks its local generation.
//
// Memory ordering: __threadfence() ensures all prior writes from this block
// are visible to other blocks before the barrier releases.
// ============================================================================

__device__ __forceinline__ void grid_barrier(
    BarrierState* barrier,
    int num_blocks,
    int* local_gen  // per-block, stored in shared memory or registers
) {
    __syncthreads();  // ensure all threads in this block are at the barrier

    if (threadIdx.x == 0) {
        // Ensure all prior global memory writes from this block are visible
        __threadfence();

        // Atomically increment the arrival counter
        int arrived = atomicAdd(&barrier->counter, 1);

        if (arrived == num_blocks - 1) {
            // Last block to arrive: reset counter and advance generation
            barrier->counter = 0;
            __threadfence();  // ensure counter reset is visible before generation bump
            atomicAdd(&barrier->generation, 1);
        }

        // Wait until generation advances past our local tracking
        // Using > (not ==) for monotonic safety
        int my_gen = *local_gen;
        while (atomicAdd(&barrier->generation, 0) <= my_gen) {
            // Spin. On sm_80+ we could use __nanosleep() to save power,
            // but it adds latency. For maximum throughput, busy-spin.
#if __CUDA_ARCH__ >= 800
            __nanosleep(2);  // minimal sleep to reduce memory bus contention
#endif
        }
        (*local_gen)++;
    }

    __syncthreads();  // ensure all threads see the barrier completion
}

// ============================================================================
// Flag-based partial barrier for attention phases
//
// Only a subset of blocks (the attention blocks) participate.
// Uses a monotonic counter: after layer L completes, flag = (L+1) * num_participants.
//
// Non-attention blocks do NOT call this - they do productive spin instead.
// ============================================================================

__device__ __forceinline__ void attention_signal(
    PartialBarrierState* barrier,
    int expected_total  // num_attn_blocks * (layer + 1)
) {
    __syncthreads();

    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(&barrier->flag, 1);

        // Wait until all attention blocks for this layer have arrived
        while (atomicAdd(&barrier->flag, 0) < expected_total) {
#if __CUDA_ARCH__ >= 800
            __nanosleep(2);
#endif
        }
    }

    __syncthreads();
}

// ============================================================================
// KV readiness signal
//
// Block 0 signals that KV cache has been written for the current layer.
// Attention blocks wait on this before starting attention computation.
// ============================================================================

__device__ __forceinline__ void signal_kv_ready(
    PartialBarrierState* barrier,
    int layer
) {
    if (threadIdx.x == 0) {
        __threadfence();  // ensure KV cache writes are visible
        atomicExch(&barrier->flag, layer + 1);
    }
}

__device__ __forceinline__ void wait_kv_ready(
    PartialBarrierState* barrier,
    int layer
) {
    if (threadIdx.x == 0) {
        while (atomicAdd(&barrier->flag, 0) < layer + 1) {
#if __CUDA_ARCH__ >= 800
            __nanosleep(2);
#endif
        }
    }
    __syncthreads();
}

// ============================================================================
// Host-side initialization
// ============================================================================

struct MegakernelBarriers {
    BarrierState* grid_barriers;          // one per phase that needs full grid sync
    PartialBarrierState* attn_barrier;    // attention completion signal
    PartialBarrierState* kv_barrier;      // KV readiness signal
    int* block_local_gens;                // [num_blocks * num_grid_barriers] local gen counters
};

// Number of full-grid barriers per layer:
// 1: after QKV GEMV
// 2: after O projection
// 3: after gate+up GEMV + SiLU
// 4: after down projection
constexpr int GRID_BARRIERS_PER_LAYER = 4;
constexpr int TOTAL_GRID_BARRIERS = GRID_BARRIERS_PER_LAYER; // we reuse barriers across layers

__host__ inline MegakernelBarriers allocate_barriers(int num_blocks) {
    MegakernelBarriers b;

    cudaMalloc(&b.grid_barriers, TOTAL_GRID_BARRIERS * sizeof(BarrierState));
    cudaMemset(b.grid_barriers, 0, TOTAL_GRID_BARRIERS * sizeof(BarrierState));

    cudaMalloc(&b.attn_barrier, sizeof(PartialBarrierState));
    cudaMemset(b.attn_barrier, 0, sizeof(PartialBarrierState));

    cudaMalloc(&b.kv_barrier, sizeof(PartialBarrierState));
    cudaMemset(b.kv_barrier, 0, sizeof(PartialBarrierState));

    // Each block needs a local generation counter per grid barrier
    cudaMalloc(&b.block_local_gens, num_blocks * TOTAL_GRID_BARRIERS * sizeof(int));
    cudaMemset(b.block_local_gens, 0, num_blocks * TOTAL_GRID_BARRIERS * sizeof(int));

    return b;
}

__host__ inline void free_barriers(MegakernelBarriers& b) {
    cudaFree(b.grid_barriers);
    cudaFree(b.attn_barrier);
    cudaFree(b.kv_barrier);
    cudaFree(b.block_local_gens);
}

// Reset barriers between generation runs
__host__ inline void reset_barriers(MegakernelBarriers& b, int num_blocks) {
    cudaMemset(b.grid_barriers, 0, TOTAL_GRID_BARRIERS * sizeof(BarrierState));
    cudaMemset(b.attn_barrier, 0, sizeof(PartialBarrierState));
    cudaMemset(b.kv_barrier, 0, sizeof(PartialBarrierState));
    cudaMemset(b.block_local_gens, 0, num_blocks * TOTAL_GRID_BARRIERS * sizeof(int));
}
