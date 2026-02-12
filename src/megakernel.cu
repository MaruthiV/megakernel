#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#include "config.cuh"
#include "barriers.cuh"
#include "utils.cuh"
#include "rmsnorm.cuh"
#include "gemv.cuh"
#include "kv_cache.cuh"
#include "attention.cuh"
#include "sampling.cuh"

// ============================================================================
// Activation Buffers (global memory, shared across blocks via barriers)
//
// These are small vectors (1024 or 2816 floats) that live in L2.
// Each phase writes its output here; the next phase reads it.
// We double-buffer to allow some overlap.
// ============================================================================

struct ActivationBuffers {
    float* x;              // [HIDDEN_DIM] current hidden state
    float* x_norm;         // [HIDDEN_DIM] after RMSNorm
    float* q_proj;         // [Q_DIM] Q projection output
    float* k_proj;         // [KV_DIM] K projection output
    float* v_proj;         // [KV_DIM] V projection output
    float* attn_out;       // [Q_DIM] attention output (concatenated heads)
    float* o_proj;         // [HIDDEN_DIM] after O projection
    float* ffn_norm;       // [HIDDEN_DIM] after FFN RMSNorm
    float* mlp_inter;      // [INTERMEDIATE_DIM] after gate+up+silu
    float* mlp_out;        // [HIDDEN_DIM] after down projection
    float* final_norm;     // [HIDDEN_DIM] after final RMSNorm
};

__host__ inline ActivationBuffers allocate_activations() {
    ActivationBuffers act;
    cudaMalloc(&act.x,          config::HIDDEN_DIM * sizeof(float));
    cudaMalloc(&act.x_norm,     config::HIDDEN_DIM * sizeof(float));
    cudaMalloc(&act.q_proj,     config::Q_DIM * sizeof(float));
    cudaMalloc(&act.k_proj,     config::KV_DIM * sizeof(float));
    cudaMalloc(&act.v_proj,     config::KV_DIM * sizeof(float));
    cudaMalloc(&act.attn_out,   config::Q_DIM * sizeof(float));
    cudaMalloc(&act.o_proj,     config::HIDDEN_DIM * sizeof(float));
    cudaMalloc(&act.ffn_norm,   config::HIDDEN_DIM * sizeof(float));
    cudaMalloc(&act.mlp_inter,  config::INTERMEDIATE_DIM * sizeof(float));
    cudaMalloc(&act.mlp_out,    config::HIDDEN_DIM * sizeof(float));
    cudaMalloc(&act.final_norm, config::HIDDEN_DIM * sizeof(float));
    return act;
}

__host__ inline void free_activations(ActivationBuffers& act) {
    cudaFree(act.x);
    cudaFree(act.x_norm);
    cudaFree(act.q_proj);
    cudaFree(act.k_proj);
    cudaFree(act.v_proj);
    cudaFree(act.attn_out);
    cudaFree(act.o_proj);
    cudaFree(act.ffn_norm);
    cudaFree(act.mlp_inter);
    cudaFree(act.mlp_out);
    cudaFree(act.final_norm);
}

// ============================================================================
// THE PERSISTENT MEGAKERNEL
//
// Single kernel launch, all blocks stay resident, processes one token per
// iteration of the outer loop. Blocks synchronize via atomic barriers.
//
// Per-layer flow (5 phases, 4 full-grid barriers):
//   Phase 1: RMSNorm(x) -> QKV GEMV                    [all blocks] -> BARRIER 0
//   Phase 2: RoPE + KV append + Attention / Prefetch    [attn blocks / idle blocks]
//            -> PARTIAL BARRIER (attn blocks only)       -> BARRIER 1
//   Phase 3: O projection + residual                     [all blocks] -> BARRIER 2
//   Phase 4: RMSNorm + Gate+Up GEMV + SiLU             [all blocks] -> BARRIER 3
//   Phase 5: Down projection + residual                  [all blocks] -> (next layer's BARRIER 0)
//
// After 28 layers: Final RMSNorm + LM head argmax
// ============================================================================

__global__ void persistent_megakernel(
    // Model weights
    ModelWeights weights,
    // KV cache
    __nv_bfloat16* kv_data,
    // Activation buffers
    ActivationBuffers act,
    // Barrier state
    BarrierState* grid_barriers,      // [TOTAL_GRID_BARRIERS]
    PartialBarrierState* attn_barrier,
    PartialBarrierState* kv_barrier,
    int* block_local_gens,            // [num_blocks * TOTAL_GRID_BARRIERS]
    // Sampling buffers
    float* sampling_block_vals,
    int* sampling_block_idxs,
    int* output_token,
    // Generation params
    int input_token,
    int current_pos,
    int num_blocks_total
) {
    // Shared memory allocation:
    // - smem_input: [max(HIDDEN_DIM, INTERMEDIATE_DIM)] for GEMV input caching
    // - smem_scratch: [num_warps] for reductions
    // - smem_idxs: [num_warps] for argmax reductions
    extern __shared__ char shared_mem[];
    int num_warps = blockDim.x / 32;

    float* smem_input = reinterpret_cast<float*>(shared_mem);
    float* smem_scratch = smem_input + config::INTERMEDIATE_DIM;  // after largest input buffer
    int* smem_idxs = reinterpret_cast<int*>(smem_scratch + num_warps);

    // Per-block barrier generation counters (in shared memory for faster access)
    __shared__ int local_gens[TOTAL_GRID_BARRIERS];
    if (threadIdx.x < TOTAL_GRID_BARRIERS) {
        local_gens[threadIdx.x] = block_local_gens[blockIdx.x * TOTAL_GRID_BARRIERS + threadIdx.x];
    }
    __syncthreads();

    int bid = blockIdx.x;
    bool is_attn_block = (bid < config::NUM_Q_HEADS);

    // ========================================================================
    // Step 0: Embedding lookup
    // ========================================================================
    // All blocks cooperatively read the embedding for the input token
    const __nv_bfloat16* emb_row = weights.embedding + (size_t)input_token * config::HIDDEN_DIM;
    for (int i = threadIdx.x; i < config::HIDDEN_DIM; i += blockDim.x) {
        act.x[i] = bf16_to_float(__ldg(&emb_row[i]));
    }

    // Grid barrier to ensure embedding is fully written
    grid_barrier(&grid_barriers[0], num_blocks_total, &local_gens[0]);

    // ========================================================================
    // Main loop: 28 transformer layers
    // ========================================================================
    for (int layer = 0; layer < config::NUM_LAYERS; layer++) {
        const LayerWeights& lw = weights.layers[layer];

        // ====================================================================
        // Phase 1: RMSNorm + QKV GEMV
        // All blocks compute RMSNorm redundantly, then split QKV output rows
        // ====================================================================

        // Redundant RMSNorm: every block computes the same result
        rmsnorm_float_in(act.x, lw.attn_norm, act.x_norm, smem_scratch, config::HIDDEN_DIM);
        __syncthreads();

        // QKV GEMV: split 2048 output rows across all blocks
        gemv_qkv(
            lw.w_q, lw.w_k, lw.w_v,
            act.x_norm,
            act.q_proj, act.k_proj, act.v_proj,
            smem_input, smem_scratch,
            num_blocks_total, bid
        );

        // BARRIER 0: QKV projection complete
        grid_barrier(&grid_barriers[0], num_blocks_total, &local_gens[0]);

        // ====================================================================
        // Phase 2: RoPE + KV cache append + Attention (attn blocks)
        //          Productive spin / prefetch (idle blocks)
        // ====================================================================

        if (bid == 0) {
            // Block 0: Apply RoPE to Q and K, append to KV cache
            // RoPE on all Q heads
            for (int h = 0; h < config::NUM_Q_HEADS; h++) {
                apply_rope_inplace(act.q_proj + h * config::HEAD_DIM, current_pos, config::HEAD_DIM);
            }
            // RoPE on all K heads
            for (int h = 0; h < config::NUM_KV_HEADS; h++) {
                apply_rope_inplace(act.k_proj + h * config::HEAD_DIM, current_pos, config::HEAD_DIM);
            }
            __syncthreads();

            // Append K, V to cache
            append_kv(kv_data, act.k_proj, act.v_proj, layer, current_pos);

            // Signal KV is ready
            signal_kv_ready(kv_barrier, layer);
        }

        if (is_attn_block) {
            // Wait for KV to be ready
            wait_kv_ready(kv_barrier, layer);

            // Compute attention for this Q head
            int head_id = bid;  // block 0..15 -> head 0..15
            multi_head_attention(
                act.q_proj,
                kv_data,
                act.attn_out,
                layer,
                current_pos + 1,  // seq_len includes current token
                head_id
            );
        } else {
            // Idle blocks: productive spin - prefetch O-proj, gate, up, down weights
            // O projection weights
            prefetch_range_l2(lw.w_o, config::W_O_BYTES, blockDim.x);
            // Gate weights
            prefetch_range_l2(lw.w_gate, config::W_GATE_BYTES, blockDim.x);
            // Up weights
            prefetch_range_l2(lw.w_up, config::W_UP_BYTES, blockDim.x);
            // Down weights
            prefetch_range_l2(lw.w_down, config::W_DOWN_BYTES, blockDim.x);
        }

        // BARRIER 1: Attention complete, all blocks ready for O-proj
        grid_barrier(&grid_barriers[1], num_blocks_total, &local_gens[1]);

        // ====================================================================
        // Phase 3: O projection + residual add
        // output = x + attn_out @ W_o
        // ====================================================================

        gemv_block(
            lw.w_o, act.attn_out, act.o_proj,
            smem_input, smem_scratch,
            config::HIDDEN_DIM, config::Q_DIM,
            num_blocks_total, bid
        );

        // BARRIER 2: O projection complete
        grid_barrier(&grid_barriers[2], num_blocks_total, &local_gens[2]);

        // Residual connection: x = x + o_proj (all blocks do this redundantly, it's tiny)
        for (int i = threadIdx.x; i < config::HIDDEN_DIM; i += blockDim.x) {
            act.x[i] = act.x[i] + act.o_proj[i];
        }
        __syncthreads();

        // ====================================================================
        // Phase 4: RMSNorm + Gate+Up GEMV + SiLU
        // ====================================================================

        rmsnorm_float_in(act.x, lw.ffn_norm, act.ffn_norm, smem_scratch, config::HIDDEN_DIM);
        __syncthreads();

        gemv_gate_up_silu(
            lw.w_gate, lw.w_up,
            act.ffn_norm,
            act.mlp_inter,
            smem_input, smem_scratch,
            num_blocks_total, bid
        );

        // BARRIER 3: Gate+Up+SiLU complete
        grid_barrier(&grid_barriers[3], num_blocks_total, &local_gens[3]);

        // ====================================================================
        // Phase 5: Down projection + residual
        // ====================================================================

        gemv_block(
            lw.w_down, act.mlp_inter, act.mlp_out,
            smem_input, smem_scratch,
            config::HIDDEN_DIM, config::INTERMEDIATE_DIM,
            num_blocks_total, bid
        );

        // Need barrier before residual add (to ensure mlp_out is complete)
        // We reuse BARRIER 0 from the next layer iteration
        // For the last layer, we add an explicit barrier below
        if (layer < config::NUM_LAYERS - 1) {
            grid_barrier(&grid_barriers[0], num_blocks_total, &local_gens[0]);
        } else {
            // Last layer: use barrier 0 for the final sync
            grid_barrier(&grid_barriers[0], num_blocks_total, &local_gens[0]);
        }

        // Residual: x = x + mlp_out
        for (int i = threadIdx.x; i < config::HIDDEN_DIM; i += blockDim.x) {
            act.x[i] = act.x[i] + act.mlp_out[i];
        }
        __syncthreads();
    }

    // ========================================================================
    // Post-layers: Final RMSNorm
    // ========================================================================
    grid_barrier(&grid_barriers[1], num_blocks_total, &local_gens[1]);

    rmsnorm_float_in(act.x, weights.final_norm, act.final_norm, smem_scratch, config::HIDDEN_DIM);
    __syncthreads();

    grid_barrier(&grid_barriers[2], num_blocks_total, &local_gens[2]);

    // ========================================================================
    // LM Head + Argmax (on-device sampling)
    // ========================================================================

    // Phase 1: Each block computes logits for its vocab chunk and finds local max
    lm_head_argmax_phase1(
        weights.embedding,  // tied weights
        act.final_norm,
        sampling_block_vals,
        sampling_block_idxs,
        smem_input,
        smem_scratch,
        smem_idxs,
        num_blocks_total
    );

    grid_barrier(&grid_barriers[3], num_blocks_total, &local_gens[3]);

    // Phase 2: Block 0 reduces across all block results
    if (bid == 0) {
        lm_head_argmax_phase2(
            sampling_block_vals,
            sampling_block_idxs,
            output_token,
            smem_scratch,
            smem_idxs,
            num_blocks_total
        );
    }

    // Save local generation counters back to global memory for next kernel invocation
    __syncthreads();
    if (threadIdx.x < TOTAL_GRID_BARRIERS) {
        block_local_gens[blockIdx.x * TOTAL_GRID_BARRIERS + threadIdx.x] = local_gens[threadIdx.x];
    }
}

// ============================================================================
// Host-Side Launcher
// ============================================================================

struct MegakernelState {
    ModelWeights weights;
    KVCache kv_cache;
    ActivationBuffers activations;
    MegakernelBarriers barriers;
    SamplingBuffers sampling;
    LaunchConfig launch_cfg;
    int current_pos;
};

// Initialize all GPU-side state
MegakernelState init_megakernel(
    const __nv_bfloat16* h_weights,  // host pointer to flat weight buffer
    int device_id = 0
) {
    MegakernelState state;
    state.launch_cfg = get_launch_config(device_id);
    state.current_pos = 0;

    // Allocate buffers
    state.kv_cache = allocate_kv_cache();
    state.activations = allocate_activations();
    state.barriers = allocate_barriers(state.launch_cfg.num_blocks);
    state.sampling = allocate_sampling_buffers(state.launch_cfg.num_blocks);

    // Copy weights to device (caller is responsible for layout)
    // Weight layout is defined by the Python weight_loader.py

    return state;
}

// Run one decode step: given input token, produce next token
int decode_step(MegakernelState& state, int input_token) {
    LaunchConfig& cfg = state.launch_cfg;

    // Shared memory size:
    // smem_input: max(HIDDEN_DIM, INTERMEDIATE_DIM) floats
    // smem_scratch: num_warps floats
    // smem_idxs: num_warps ints
    int num_warps = cfg.threads_per_block / 32;
    size_t smem_bytes = config::INTERMEDIATE_DIM * sizeof(float)
                      + num_warps * sizeof(float)
                      + num_warps * sizeof(int);

    // Launch the persistent kernel for one token
    persistent_megakernel<<<cfg.num_blocks, cfg.threads_per_block, smem_bytes>>>(
        state.weights,
        state.kv_cache.data,
        state.activations,
        state.barriers.grid_barriers,
        state.barriers.attn_barrier,
        state.barriers.kv_barrier,
        state.barriers.block_local_gens,
        state.sampling.block_results_val,
        state.sampling.block_results_idx,
        state.sampling.output_token,
        input_token,
        state.current_pos,
        cfg.num_blocks
    );

    // Read back the output token
    int next_token;
    cudaMemcpy(&next_token, state.sampling.output_token, sizeof(int), cudaMemcpyDeviceToHost);

    state.current_pos++;

    return next_token;
}

// Benchmark: generate num_tokens and measure throughput
void benchmark(MegakernelState& state, int prompt_token, int num_tokens, bool warmup = true) {
    // Warmup
    if (warmup) {
        reset_barriers(state.barriers, state.launch_cfg.num_blocks);
        state.current_pos = 0;
        for (int i = 0; i < 5; i++) {
            decode_step(state, prompt_token);
        }
    }

    // Reset for actual benchmark
    reset_barriers(state.barriers, state.launch_cfg.num_blocks);
    state.current_pos = 0;
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();

    int token = prompt_token;
    for (int i = 0; i < num_tokens; i++) {
        token = decode_step(state, token);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double toks_per_sec = num_tokens / (elapsed_ms / 1000.0);

    // Bandwidth utilization calculation
    double bytes_per_token = (double)config::TOTAL_WEIGHT_BYTES;
    double achieved_bw_gbps = (bytes_per_token * toks_per_sec) / 1e9;
    double utilization = achieved_bw_gbps / state.launch_cfg.mem_bandwidth_gbps * 100.0;

    printf("=== Benchmark Results ===\n");
    printf("Tokens generated:    %d\n", num_tokens);
    printf("Total time:          %.2f ms\n", elapsed_ms);
    printf("Throughput:          %.1f tok/s\n", toks_per_sec);
    printf("Per-token latency:   %.1f us\n", elapsed_ms * 1000.0 / num_tokens);
    printf("Achieved bandwidth:  %.1f GB/s\n", achieved_bw_gbps);
    printf("BW utilization:      %.1f%%\n", utilization);
    printf("Peak BW:             %.1f GB/s\n", state.launch_cfg.mem_bandwidth_gbps);
    printf("GPU SMs:             %d\n", state.launch_cfg.sm_count);
    printf("Blocks:              %d\n", state.launch_cfg.num_blocks);
    printf("Threads/block:       %d\n", state.launch_cfg.threads_per_block);
}

// Benchmark at specific context positions
void benchmark_sweep(MegakernelState& state, int prompt_token) {
    int positions[] = {1, 10, 50, 100, 500, 1000, 2000, 4096};
    int num_positions = sizeof(positions) / sizeof(positions[0]);

    printf("\n=== Context Position Sweep ===\n");
    printf("%-10s %-12s %-12s %-12s\n", "Position", "Tok/s", "Latency(us)", "BW Util(%)");
    printf("%-10s %-12s %-12s %-12s\n", "--------", "-----", "-----------", "---------");

    for (int p = 0; p < num_positions; p++) {
        int target_pos = positions[p];
        if (target_pos > config::MAX_SEQ_LEN) break;

        // Reset and fill KV cache to target position
        reset_barriers(state.barriers, state.launch_cfg.num_blocks);
        state.current_pos = 0;

        // Generate tokens up to target position
        int token = prompt_token;
        for (int i = 0; i < target_pos; i++) {
            token = decode_step(state, token);
        }
        cudaDeviceSynchronize();

        // Now benchmark a few tokens at this position
        int bench_tokens = 20;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < bench_tokens; i++) {
            token = decode_step(state, token);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double toks_per_sec = bench_tokens / (elapsed_ms / 1000.0);
        double latency_us = elapsed_ms * 1000.0 / bench_tokens;
        double bytes_per_token = (double)config::TOTAL_WEIGHT_BYTES;
        double achieved_bw = (bytes_per_token * toks_per_sec) / 1e9;
        double utilization = achieved_bw / state.launch_cfg.mem_bandwidth_gbps * 100.0;

        printf("%-10d %-12.1f %-12.1f %-12.1f\n",
               target_pos, toks_per_sec, latency_us, utilization);
    }
}

// ============================================================================
// Main entry point (standalone executable for testing)
// Actual usage goes through the Python launcher + Colab notebook.
// ============================================================================

// ============================================================================
// Library Mode: extern "C" API for Python ctypes
//
// When compiled with -DMEGAKERNEL_LIBRARY_MODE, these flat C functions are
// exported so Python can call them via ctypes. The MegakernelState is passed
// around as an opaque void* handle.
//
// Weight offset ordering (must match host/weight_loader.py save_flat_binary):
//   offsets[0]  = embedding
//   offsets[1]  = final_norm
//   offsets[2 + layer*9 + 0] = layer.{layer}.attn_norm
//   offsets[2 + layer*9 + 1] = layer.{layer}.w_q
//   offsets[2 + layer*9 + 2] = layer.{layer}.w_k
//   offsets[2 + layer*9 + 3] = layer.{layer}.w_v
//   offsets[2 + layer*9 + 4] = layer.{layer}.w_o
//   offsets[2 + layer*9 + 5] = layer.{layer}.ffn_norm
//   offsets[2 + layer*9 + 6] = layer.{layer}.w_gate
//   offsets[2 + layer*9 + 7] = layer.{layer}.w_up
//   offsets[2 + layer*9 + 8] = layer.{layer}.w_down
//   Total: 2 + 28*9 = 254 offsets
// ============================================================================

#ifdef MEGAKERNEL_LIBRARY_MODE

extern "C" {

// Initialize the megakernel state.
// d_weights: GPU pointer to the flat weight binary (already on device)
// offsets: host array of byte offsets into d_weights for each weight tensor
// num_offsets: should be 254 (2 global + 28 layers * 9 weights)
// Returns opaque handle to MegakernelState.
void* megakernel_init(void* d_weights, long long* offsets, int num_offsets) {
    MegakernelState* state = new MegakernelState();

    state->launch_cfg = get_launch_config(0);
    state->current_pos = 0;

    // Set up weight pointers from flat binary + offsets
    const char* base = reinterpret_cast<const char*>(d_weights);

    state->weights.embedding   = reinterpret_cast<const __nv_bfloat16*>(base + offsets[0]);
    state->weights.final_norm  = reinterpret_cast<const __nv_bfloat16*>(base + offsets[1]);

    for (int i = 0; i < config::NUM_LAYERS; i++) {
        int idx = 2 + i * 9;
        LayerWeights& lw = state->weights.layers[i];
        lw.attn_norm = reinterpret_cast<const __nv_bfloat16*>(base + offsets[idx + 0]);
        lw.w_q       = reinterpret_cast<const __nv_bfloat16*>(base + offsets[idx + 1]);
        lw.w_k       = reinterpret_cast<const __nv_bfloat16*>(base + offsets[idx + 2]);
        lw.w_v       = reinterpret_cast<const __nv_bfloat16*>(base + offsets[idx + 3]);
        lw.w_o       = reinterpret_cast<const __nv_bfloat16*>(base + offsets[idx + 4]);
        lw.ffn_norm  = reinterpret_cast<const __nv_bfloat16*>(base + offsets[idx + 5]);
        lw.w_gate    = reinterpret_cast<const __nv_bfloat16*>(base + offsets[idx + 6]);
        lw.w_up      = reinterpret_cast<const __nv_bfloat16*>(base + offsets[idx + 7]);
        lw.w_down    = reinterpret_cast<const __nv_bfloat16*>(base + offsets[idx + 8]);
    }

    // Allocate GPU buffers
    state->kv_cache = allocate_kv_cache();
    state->activations = allocate_activations();
    state->barriers = allocate_barriers(state->launch_cfg.num_blocks);
    state->sampling = allocate_sampling_buffers(state->launch_cfg.num_blocks);

    printf("Megakernel initialized:\n");
    printf("  Blocks: %d, Threads/block: %d\n",
           state->launch_cfg.num_blocks, state->launch_cfg.threads_per_block);
    printf("  Peak BW: %.1f GB/s\n", state->launch_cfg.mem_bandwidth_gbps);
    printf("  Weight offsets loaded: %d\n", num_offsets);

    return static_cast<void*>(state);
}

// Run one decode step: input_token -> next token id
int megakernel_decode(void* handle, int input_token) {
    MegakernelState* state = static_cast<MegakernelState*>(handle);
    return decode_step(*state, input_token);
}

// Reset KV cache and barriers for a new generation
void megakernel_reset(void* handle) {
    MegakernelState* state = static_cast<MegakernelState*>(handle);
    reset_barriers(state->barriers, state->launch_cfg.num_blocks);
    cudaMemset(state->kv_cache.data, 0,
               (size_t)config::NUM_LAYERS * 2 * config::NUM_KV_HEADS
               * config::MAX_SEQ_LEN * config::HEAD_DIM * sizeof(__nv_bfloat16));
    state->current_pos = 0;
}

// Get current sequence position
int megakernel_get_pos(void* handle) {
    MegakernelState* state = static_cast<MegakernelState*>(handle);
    return state->current_pos;
}

// Free all GPU resources
void megakernel_free(void* handle) {
    MegakernelState* state = static_cast<MegakernelState*>(handle);
    free_kv_cache(state->kv_cache);
    free_activations(state->activations);
    free_barriers(state->barriers);
    free_sampling_buffers(state->sampling);
    delete state;
}

// Synchronize GPU (call after last decode before timing)
void megakernel_sync() {
    cudaDeviceSynchronize();
}

} // extern "C"

#else
// ============================================================================
// Standalone Mode: main() for testing (no weights, just prints info)
// ============================================================================

int main(int argc, char** argv) {
    printf("MegaKernel for Qwen3-0.6B\n");
    printf("==========================\n\n");

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s\n", prop.name);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("L2 Cache: %d MB\n", prop.l2CacheSize / (1024 * 1024));
    printf("Global Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    LaunchConfig cfg = get_launch_config(device);
    printf("Launch Config:\n");
    printf("  Blocks: %d\n", cfg.num_blocks);
    printf("  Threads/block: %d\n", cfg.threads_per_block);
    printf("  Attn blocks: %d\n", cfg.attn_blocks);
    printf("  Idle blocks: %d\n", cfg.idle_blocks);
    printf("  Peak BW: %.1f GB/s\n", cfg.mem_bandwidth_gbps);

    printf("\nModel: Qwen3-0.6B\n");
    printf("  Layers: %d\n", config::NUM_LAYERS);
    printf("  Hidden dim: %d\n", config::HIDDEN_DIM);
    printf("  Q heads: %d, KV heads: %d\n", config::NUM_Q_HEADS, config::NUM_KV_HEADS);
    printf("  Total weight bytes: %.1f MB\n", config::TOTAL_WEIGHT_BYTES / 1e6);
    printf("  Theoretical max tok/s: %.1f\n",
           cfg.mem_bandwidth_gbps * 1e9 / config::TOTAL_WEIGHT_BYTES);

    printf("\nTo run actual inference, use the Colab notebook.\n");

    return 0;
}

#endif // MEGAKERNEL_LIBRARY_MODE
