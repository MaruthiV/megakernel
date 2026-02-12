#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ============================================================================
// Qwen3-0.6B Model Configuration
// ============================================================================

namespace config {

// Model architecture constants
constexpr int HIDDEN_DIM      = 1024;
constexpr int NUM_LAYERS       = 28;
constexpr int NUM_Q_HEADS      = 16;
constexpr int NUM_KV_HEADS     = 8;
constexpr int HEAD_DIM         = 128;  // explicitly set in Qwen3 config (not hidden/heads)
constexpr int INTERMEDIATE_DIM = 3072; // MLP intermediate size (SwiGLU)
constexpr int VOCAB_SIZE       = 151936;
constexpr int MAX_SEQ_LEN      = 4096; // max context we support
constexpr float ROPE_THETA     = 1000000.0f;
constexpr float RMS_NORM_EPS   = 1e-6f;

// Derived constants
constexpr int QKV_DIM          = HIDDEN_DIM + 2 * (NUM_KV_HEADS * HEAD_DIM);
                                 // 1024 + 2*1024 = 3072
constexpr int Q_DIM            = NUM_Q_HEADS * HEAD_DIM;   // 2048
constexpr int KV_DIM           = NUM_KV_HEADS * HEAD_DIM;  // 1024
constexpr int GQA_RATIO        = NUM_Q_HEADS / NUM_KV_HEADS; // 2

// Per-layer weight sizes in bytes (BF16)
constexpr size_t W_Q_BYTES     = HIDDEN_DIM * Q_DIM * 2;       // 4 MB
constexpr size_t W_K_BYTES     = HIDDEN_DIM * KV_DIM * 2;      // 2 MB
constexpr size_t W_V_BYTES     = HIDDEN_DIM * KV_DIM * 2;      // 2 MB
constexpr size_t W_O_BYTES     = Q_DIM * HIDDEN_DIM * 2;       // 4 MB
constexpr size_t W_GATE_BYTES  = HIDDEN_DIM * INTERMEDIATE_DIM * 2; // 6 MB
constexpr size_t W_UP_BYTES    = HIDDEN_DIM * INTERMEDIATE_DIM * 2; // 6 MB
constexpr size_t W_DOWN_BYTES  = INTERMEDIATE_DIM * HIDDEN_DIM * 2; // 6 MB
constexpr size_t NORM_BYTES    = HIDDEN_DIM * 2;               // 2 KB

constexpr size_t LAYER_WEIGHT_BYTES =
    W_Q_BYTES + W_K_BYTES + W_V_BYTES + W_O_BYTES +
    W_GATE_BYTES + W_UP_BYTES + W_DOWN_BYTES +
    2 * NORM_BYTES; // attn_norm + ffn_norm

constexpr size_t EMBED_BYTES   = (size_t)VOCAB_SIZE * HIDDEN_DIM * 2; // ~296 MB
constexpr size_t FINAL_NORM_BYTES = HIDDEN_DIM * 2;

constexpr size_t TOTAL_WEIGHT_BYTES =
    NUM_LAYERS * LAYER_WEIGHT_BYTES + EMBED_BYTES + FINAL_NORM_BYTES;

// KV cache size per token per layer (BF16)
constexpr size_t KV_TOKEN_LAYER_BYTES = 2 * NUM_KV_HEADS * HEAD_DIM * 2;  // 2048 bytes
constexpr size_t KV_TOKEN_BYTES = KV_TOKEN_LAYER_BYTES * NUM_LAYERS;       // 57344 bytes

} // namespace config

// ============================================================================
// GPU Launch Configuration (auto-tuned per GPU)
// ============================================================================

struct LaunchConfig {
    int num_blocks;           // total persistent blocks
    int threads_per_block;    // threads per block
    int attn_blocks;          // blocks assigned to attention (= NUM_Q_HEADS)
    int idle_blocks;          // blocks doing productive spin during attention
    int sm_count;             // number of SMs on this GPU
    int l2_cache_bytes;       // L2 cache size
    float mem_bandwidth_gbps; // peak memory bandwidth in GB/s
};

__host__ inline LaunchConfig get_launch_config(int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    LaunchConfig cfg;
    cfg.sm_count = prop.multiProcessorCount;
    cfg.l2_cache_bytes = prop.l2CacheSize;
    cfg.attn_blocks = config::NUM_Q_HEADS; // 16

    // Set num_blocks to SM count (1 block per SM for maximum occupancy)
    cfg.num_blocks = cfg.sm_count;
    cfg.idle_blocks = cfg.num_blocks - cfg.attn_blocks;

    // Per-GPU tuning
    int cc = prop.major * 10 + prop.minor;

    if (cc >= 90) {
        // H100 (sm_90) or newer
        cfg.threads_per_block = 512;
        cfg.mem_bandwidth_gbps = 3350.0f;
    } else if (cc >= 80) {
        // A100 (sm_80) or A10/A30
        cfg.threads_per_block = 512;
        cfg.mem_bandwidth_gbps = (prop.totalGlobalMem > 50ULL * 1024 * 1024 * 1024)
            ? 2039.0f  // A100-80GB
            : 1555.0f; // A100-40GB
    } else if (cc >= 75) {
        // T4 (sm_75), RTX 20xx
        cfg.threads_per_block = 256;
        cfg.mem_bandwidth_gbps = 300.0f;
    } else {
        // Fallback
        cfg.threads_per_block = 256;
        cfg.mem_bandwidth_gbps = 300.0f;
    }

    return cfg;
}

// ============================================================================
// Weight layout offsets within a layer's weight buffer
// ============================================================================

struct LayerWeights {
    const __nv_bfloat16* attn_norm;  // [HIDDEN_DIM]
    const __nv_bfloat16* w_q;        // [HIDDEN_DIM, Q_DIM]
    const __nv_bfloat16* w_k;        // [HIDDEN_DIM, KV_DIM]
    const __nv_bfloat16* w_v;        // [HIDDEN_DIM, KV_DIM]
    const __nv_bfloat16* w_o;        // [Q_DIM, HIDDEN_DIM]
    const __nv_bfloat16* ffn_norm;   // [HIDDEN_DIM]
    const __nv_bfloat16* w_gate;     // [HIDDEN_DIM, INTERMEDIATE_DIM]
    const __nv_bfloat16* w_up;       // [HIDDEN_DIM, INTERMEDIATE_DIM]
    const __nv_bfloat16* w_down;     // [INTERMEDIATE_DIM, HIDDEN_DIM]
};

struct ModelWeights {
    const __nv_bfloat16* embedding;    // [VOCAB_SIZE, HIDDEN_DIM]
    const __nv_bfloat16* final_norm;   // [HIDDEN_DIM]
    // LM head is tied = embedding^T
    LayerWeights layers[config::NUM_LAYERS];
};
