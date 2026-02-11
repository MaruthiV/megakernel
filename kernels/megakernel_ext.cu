#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ void barrier_wait_global(int* counter, int* sense, int expected) {
  __shared__ int local_sense;
  if (threadIdx.x == 0) {
    local_sense = 1 - *sense;
    int arrived = atomicAdd(counter, 1);
    if (arrived == expected - 1) {
      atomicExch(counter, 0);
      atomicExch(sense, local_sense);
    }
  }
  __syncthreads();
  while (*sense != local_sense) {
    __nanosleep(50);
  }
  __syncthreads();
}

__device__ __forceinline__ void spin_wait(volatile int* flag, int target) {
  int backoff = 50;
  while (*flag != target) {
    __nanosleep(backoff);
    if (backoff < 800) {
      backoff *= 2;
    }
  }
}

enum BlockRole : int {
  ROLE_QKV = 0,
  ROLE_ATTN = 1,
  ROLE_MLP = 2,
  ROLE_PREFETCH = 3,
};

__device__ __forceinline__ int role_count(int total_blocks, int role_id) {
  if (total_blocks <= role_id) {
    return 0;
  }
  return (total_blocks + (4 - 1 - role_id)) / 4;
}

__device__ __forceinline__ float rope_inv_freq(int d, int head_dim, float rope_theta) {
  float exponent = (float)d / (float)head_dim;
  return powf(rope_theta, -exponent);
}

__device__ __forceinline__ void rope_apply(float* q, float* k, int head_dim, int pos, float rope_theta) {
  for (int d = 0; d + 1 < head_dim; d += 2) {
    float inv = rope_inv_freq(d, head_dim, rope_theta);
    float angle = (float)pos * inv;
    float c = cosf(angle);
    float s = sinf(angle);
    float q0 = q[d];
    float q1 = q[d + 1];
    float k0 = k[d];
    float k1 = k[d + 1];
    q[d] = q0 * c - q1 * s;
    q[d + 1] = q0 * s + q1 * c;
    k[d] = k0 * c - k1 * s;
    k[d + 1] = k0 * s + k1 * c;
  }
}

__device__ __forceinline__ int kv_index_seq_hidden(int pos, int h, int d, int head_dim, int hidden) {
  return pos * hidden + h * head_dim + d;
}

__device__ __forceinline__ int kv_index_head_dim_seq(int pos, int h, int d, int head_dim, int max_seq) {
  return h * head_dim * max_seq + d * max_seq + pos;
}

__global__ void megakernel_naive(
    const float* input,
    const float* rms_attn_weight,
    const float* rms_ffn_weight,
    const float* w_qkv,
    const float* w_o,
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    float* kv_k,
    float* kv_v,
    float* output,
    int num_layers,
    int hidden,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int position,
    int max_seq,
    int kv_layout,
    float eps,
    float rope_theta) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  // Naive single-thread megakernel for multiple layers.
  // Limitations: float32 only, batch=1, no kv quant, no parallelism.
  // Shared scratch: normed, qkv, attn_out, mlp_gate, mlp_up, reduce, head_max, head_denom, head_sum, tiles
  extern __shared__ float scratch[];
  float* normed = scratch;                         // hidden
  int kv_dim = num_kv_heads * head_dim;
  float* qkv = normed + hidden;                    // hidden + 2*kv_dim
  float* attn_out = qkv + hidden + 2 * kv_dim;     // hidden
  float* mlp_gate = attn_out + hidden;             // 4*hidden
  float* mlp_up = mlp_gate + 4 * hidden;           // 4*hidden
  float* reduce = mlp_up + 4 * hidden;             // blockDim.x
  float* head_max = reduce + blockDim.x;           // num_heads
  float* head_denom = head_max + num_heads;        // num_heads
  float* head_sum = head_denom + num_heads;        // num_heads * blockDim.x
  int tile_t = 32;
  float* k_tile = head_sum + num_heads * blockDim.x;               // tile_t * head_dim
  float* v_tile = k_tile + tile_t * head_dim;                      // tile_t * head_dim
  float* score_tile = v_tile + tile_t * head_dim;                  // tile_t

  // Initialize output with input for residual chaining.
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    output[i] = input[i];
  }
  __syncthreads();

  int last = position;
  if (last < 0) last = 0;
  if (last >= max_seq) last = max_seq - 1;

  int mlp_hidden = 4 * hidden;
  int qkv_out = hidden + 2 * kv_dim;
  int group_size = num_heads / num_kv_heads;

  for (int layer = 0; layer < num_layers; ++layer) {
    const float* rms_attn = rms_attn_weight + layer * hidden;
    const float* rms_ffn = rms_ffn_weight + layer * hidden;
    const float* wqkv = w_qkv + layer * hidden * (hidden + 2 * kv_dim);
    const float* wo = w_o + layer * hidden * hidden;
    const float* wgate = w_gate + layer * hidden * mlp_hidden;
    const float* wup = w_up + layer * hidden * mlp_hidden;
    const float* wdown = w_down + layer * mlp_hidden * hidden;

    // RMSNorm (attn)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      float v0 = output[i];
      local_sum += v0 * v0;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
      float sum = 0.0f;
      for (int i = 0; i < blockDim.x; ++i) {
        sum += reduce[i];
      }
      reduce[0] = sum / hidden;
    }
    __syncthreads();
    float inv_rms = rsqrtf(reduce[0] + eps);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      normed[i] = output[i] * inv_rms * rms_attn[i];
    }
    __syncthreads();

    // QKV projection: [hidden] x [hidden, 3*hidden]
    for (int o = threadIdx.x; o < qkv_out; o += blockDim.x) {
      float acc = 0.0f;
      const float* w = wqkv + o;
      for (int i = 0; i < hidden; ++i) {
        acc += normed[i] * w[i * qkv_out];
      }
      qkv[o] = acc;
    }
    __syncthreads();

    float* q = qkv;
    float* k = qkv + hidden;
    float* v = qkv + hidden + kv_dim;

    // RoPE on Q and K per head
    if (threadIdx.x == 0) {
    for (int h = 0; h < num_heads; ++h) {
      float* qh = q + h * head_dim;
      rope_apply(qh, qh, head_dim, position, rope_theta);
    }
    for (int h = 0; h < num_kv_heads; ++h) {
      float* kh = k + h * head_dim;
      rope_apply(kh, kh, head_dim, position, rope_theta);
    }
    }
    __syncthreads();

    // Write current K/V to cache at position with layout
    if (position >= 0 && position < max_seq) {
      for (int h = 0; h < num_kv_heads; ++h) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
          int idx = (kv_layout == 0)
                        ? kv_index_seq_hidden(position, h, d, head_dim, kv_dim)
                        : kv_index_head_dim_seq(position, h, d, head_dim, max_seq);
          kv_k[idx] = k[h * head_dim + d];
          kv_v[idx] = v[h * head_dim + d];
        }
      }
    }
    __syncthreads();

    // Attention (partial parallelism: per-(h,d) output in parallel)
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      attn_out[i] = 0.0f;
    }
    __syncthreads();

        for (int h = role_index; h < num_heads; h += role_blocks) {
          int hk = h / group_size;
      if (kv_layout == 0) {
        // Tiled max
        float local_max = -1e30f;
        for (int t0 = 0; t0 <= last; t0 += tile_t) {
          int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
          // Load K tile
          int tile_elems = tile_len * head_dim;
          for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
            int t = idx / head_dim;
            int d = idx - t * head_dim;
            int gidx = kv_index_seq_hidden(t0 + t, hk, d, head_dim, kv_dim);
            k_tile[idx] = kv_k[gidx];
          }
          __syncthreads();
          if (threadIdx.x < tile_len) {
            float dot = 0.0f;
            int t = threadIdx.x;
            for (int d = 0; d < head_dim; ++d) {
              dot += q[h * head_dim + d] * k_tile[t * head_dim + d];
            }
            dot /= sqrtf((float)head_dim);
            if (dot > local_max) {
              local_max = dot;
            }
          }
          __syncthreads();
        }
        head_sum[h * blockDim.x + threadIdx.x] = local_max;
        __syncthreads();
        if (threadIdx.x == 0) {
          float max_score = -1e30f;
          for (int i = 0; i < blockDim.x; ++i) {
            float v = head_sum[h * blockDim.x + i];
            if (v > max_score) {
              max_score = v;
            }
          }
          head_max[h] = max_score;
        }
        __syncthreads();

        // Tiled denom
        float local_denom = 0.0f;
        float max_score = head_max[h];
        for (int t0 = 0; t0 <= last; t0 += tile_t) {
          int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
          int tile_elems = tile_len * head_dim;
          for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
            int t = idx / head_dim;
            int d = idx - t * head_dim;
            int gidx = kv_index_seq_hidden(t0 + t, hk, d, head_dim, kv_dim);
            k_tile[idx] = kv_k[gidx];
          }
          __syncthreads();
          if (threadIdx.x < tile_len) {
            float dot = 0.0f;
            int t = threadIdx.x;
            for (int d = 0; d < head_dim; ++d) {
              dot += q[h * head_dim + d] * k_tile[t * head_dim + d];
            }
            dot /= sqrtf((float)head_dim);
            local_denom += expf(dot - max_score);
          }
          __syncthreads();
        }
        head_sum[h * blockDim.x + threadIdx.x] = local_denom;
        __syncthreads();
        if (threadIdx.x == 0) {
          float denom = 0.0f;
          for (int i = 0; i < blockDim.x; ++i) {
            denom += head_sum[h * blockDim.x + i];
          }
          head_denom[h] = denom;
        }
        __syncthreads();
      } else {
        // Fallback path for kv_layout=1 (no tiling)
        float local_max = -1e30f;
        for (int t = threadIdx.x; t <= last; t += blockDim.x) {
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) {
            int idx = kv_index_head_dim_seq(t, hk, d, head_dim, max_seq);
            dot += q[h * head_dim + d] * kv_k[idx];
          }
          dot /= sqrtf((float)head_dim);
          if (dot > local_max) {
            local_max = dot;
          }
        }
        head_sum[h * blockDim.x + threadIdx.x] = local_max;
        __syncthreads();
        if (threadIdx.x == 0) {
          float max_score = -1e30f;
          for (int i = 0; i < blockDim.x; ++i) {
            float v = head_sum[h * blockDim.x + i];
            if (v > max_score) {
              max_score = v;
            }
          }
          head_max[h] = max_score;
        }
        __syncthreads();

        float local_denom = 0.0f;
        float max_score = head_max[h];
        for (int t = threadIdx.x; t <= last; t += blockDim.x) {
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) {
            int idx = kv_index_head_dim_seq(t, hk, d, head_dim, max_seq);
            dot += q[h * head_dim + d] * kv_k[idx];
          }
          dot /= sqrtf((float)head_dim);
          local_denom += expf(dot - max_score);
        }
        head_sum[h * blockDim.x + threadIdx.x] = local_denom;
        __syncthreads();
        if (threadIdx.x == 0) {
          float denom = 0.0f;
          for (int i = 0; i < blockDim.x; ++i) {
            denom += head_sum[h * blockDim.x + i];
          }
          head_denom[h] = denom;
        }
        __syncthreads();
      }
    }

    // Parallelize over (head, d): each thread handles one d and loops over t.
    for (int h = 0; h < num_heads; ++h) {
      if (threadIdx.x < head_dim) {
        int d = threadIdx.x;
        float max_score = head_max[h];
        float denom = head_denom[h];
        float acc = 0.0f;
        if (kv_layout == 0) {
          for (int t0 = 0; t0 <= last; t0 += tile_t) {
            int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
            int tile_elems = tile_len * head_dim;
            for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
              int t = idx / head_dim;
              int dd = idx - t * head_dim;
              int gidx = kv_index_seq_hidden(t0 + t, hk, dd, head_dim, kv_dim);
              k_tile[idx] = kv_k[gidx];
              v_tile[idx] = kv_v[gidx];
            }
            __syncthreads();
            if (threadIdx.x < tile_len) {
              float dot = 0.0f;
              int t = threadIdx.x;
              int dd = 0;
              for (; dd + 3 < head_dim; dd += 4) {
                float4 k4 = *reinterpret_cast<const float4*>(&k_tile[t * head_dim + dd]);
                float4 q4 = *reinterpret_cast<const float4*>(&q[h * head_dim + dd]);
                dot += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
              }
              for (; dd < head_dim; ++dd) {
                dot += q[h * head_dim + dd] * k_tile[t * head_dim + dd];
              }
              dot /= sqrtf((float)head_dim);
              score_tile[t] = dot;
            }
            __syncthreads();
            for (int t = 0; t < tile_len; ++t) {
              float w = expf(score_tile[t] - max_score) / denom;
              acc += w * v_tile[t * head_dim + d];
            }
            __syncthreads();
          }
        } else {
          for (int t = 0; t <= last; ++t) {
            float dot = 0.0f;
            for (int dd = 0; dd < head_dim; ++dd) {
              int idx_k2 = kv_index_head_dim_seq(t, hk, dd, head_dim, max_seq);
              dot += q[h * head_dim + dd] * kv_k[idx_k2];
            }
            dot /= sqrtf((float)head_dim);
            float w = expf(dot - max_score) / denom;
            int idx_v = kv_index_head_dim_seq(t, hk, d, head_dim, max_seq);
            acc += w * kv_v[idx_v];
          }
        }
        attn_out[h * head_dim + d] = acc;
      }
      __syncthreads();
    }

    // Output projection
    for (int o = threadIdx.x; o < hidden; o += blockDim.x) {
      float acc = 0.0f;
      const float* w = wo + o;
      for (int i = 0; i < hidden; ++i) {
        acc += attn_out[i] * w[i * hidden];
      }
      output[o] = output[o] + acc;  // residual
    }
    __syncthreads();

    // RMSNorm (ffn)
    float local_sum2 = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      float v2 = output[i];
      local_sum2 += v2 * v2;
    }
    reduce[threadIdx.x] = local_sum2;
    __syncthreads();
    if (threadIdx.x == 0) {
      float sum2 = 0.0f;
      for (int i = 0; i < blockDim.x; ++i) {
        sum2 += reduce[i];
      }
      reduce[0] = sum2 / hidden;
    }
    __syncthreads();
    float inv_rms2 = rsqrtf(reduce[0] + eps);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      normed[i] = output[i] * inv_rms2 * rms_ffn[i];
    }
    __syncthreads();

    // MLP gate + up (SwiGLU)
    for (int o = threadIdx.x; o < mlp_hidden; o += blockDim.x) {
      float acc_gate = 0.0f;
      float acc_up = 0.0f;
      const float* wg = wgate + o;
      const float* wu = wup + o;
      for (int i = 0; i < hidden; ++i) {
        float v = normed[i];
        acc_gate += v * wg[i * mlp_hidden];
        acc_up += v * wu[i * mlp_hidden];
      }
      mlp_gate[o] = silu(acc_gate);
      mlp_up[o] = acc_up;
    }
    __syncthreads();

    // MLP down-proj (4*hidden -> hidden) + residual
    for (int o = threadIdx.x; o < hidden; o += blockDim.x) {
      float acc = 0.0f;
      const float* w = wdown + o;
      for (int i = 0; i < mlp_hidden; ++i) {
        acc += (mlp_gate[i] * mlp_up[i]) * w[i * hidden];
      }
      output[o] = output[o] + acc;
    }
    __syncthreads();
  }
}

__global__ void megakernel_naive_seq(
    const float* inputs,
    const float* rms_attn_weight,
    const float* rms_ffn_weight,
    const float* w_qkv,
    const float* w_o,
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    float* kv_k,
    float* kv_v,
    float* outputs,
    int num_layers,
    int hidden,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    int seq_len,
    int max_seq,
    int kv_layout,
    float eps,
    float rope_theta) {
  if (blockIdx.x != 0) {
    return;
  }

  int kv_dim = num_kv_heads * head_dim;
  int qkv_out = hidden + 2 * kv_dim;
  int group_size = num_heads / num_kv_heads;
  int mlp_hidden = 4 * hidden;

  // Shared scratch for one position at a time
  extern __shared__ float scratch[];
  float* normed = scratch;                         // hidden
  float* qkv = normed + hidden;                    // hidden + 2*kv_dim
  float* attn_out = qkv + qkv_out;                 // hidden
  float* mlp_gate = attn_out + hidden;             // 4*hidden
  float* mlp_up = mlp_gate + 4 * hidden;           // 4*hidden
  float* reduce = mlp_up + 4 * hidden;             // blockDim.x
  float* head_max = reduce + blockDim.x;           // num_heads
  float* head_denom = head_max + num_heads;        // num_heads
  float* head_sum = head_denom + num_heads;        // num_heads * blockDim.x
  int tile_t = 16;
  float* k_tile = head_sum + num_heads * blockDim.x;               // tile_t * head_dim
  float* v_tile = k_tile + tile_t * head_dim;                      // tile_t * head_dim
  float* score_tile = v_tile + tile_t * head_dim;                  // tile_t

  for (int tpos = 0; tpos < seq_len; ++tpos) {
    int position = start_pos + tpos;
    int last = position;
    if (last < 0) last = 0;
    if (last >= max_seq) last = max_seq - 1;

    const float* input = inputs + tpos * hidden;
    float* output = outputs + tpos * hidden;

    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      output[i] = input[i];
    }
    __syncthreads();

    for (int layer = 0; layer < num_layers; ++layer) {
      if (role == ROLE_QKV) {
        spin_wait((volatile int*)&sync_flags[2], 1);
        if (threadIdx.x == 0) {
          sync_flags[0] = 0;
          sync_flags[1] = 0;
          sync_flags[2] = 0;
        }
        __syncthreads();
      }
      const float* rms_attn = rms_attn_weight + layer * hidden;
      const float* rms_ffn = rms_ffn_weight + layer * hidden;
      const float* wqkv = w_qkv + layer * hidden * (hidden + 2 * kv_dim);
      const float* wo = w_o + layer * hidden * hidden;
      const float* wgate = w_gate + layer * hidden * mlp_hidden;
      const float* wup = w_up + layer * hidden * mlp_hidden;
      const float* wdown = w_down + layer * mlp_hidden * hidden;

      // RMSNorm (attn)
      float local_sum = 0.0f;
      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v0 = output[i];
        local_sum += v0 * v0;
      }
      reduce[threadIdx.x] = local_sum;
      __syncthreads();
      if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
          sum += reduce[i];
        }
        reduce[0] = sum / hidden;
      }
      __syncthreads();
      float inv_rms = rsqrtf(reduce[0] + eps);
      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        normed[i] = output[i] * inv_rms * rms_attn[i];
      }
      __syncthreads();

      // QKV projection
      for (int o = threadIdx.x; o < qkv_out; o += blockDim.x) {
        float acc = 0.0f;
        const float* w = wqkv + o;
        for (int i = 0; i < hidden; ++i) {
          acc += normed[i] * w[i * qkv_out];
        }
        qkv[o] = acc;
      }
      __syncthreads();

      float* q = qkv;
      float* k = qkv + hidden;
      float* v = qkv + hidden + kv_dim;

      if (threadIdx.x == 0) {
        for (int h = 0; h < num_heads; ++h) {
          float* qh = q + h * head_dim;
          rope_apply(qh, qh, head_dim, position, rope_theta);
        }
        for (int h = 0; h < num_kv_heads; ++h) {
          float* kh = k + h * head_dim;
          rope_apply(kh, kh, head_dim, position, rope_theta);
        }
      }
      __syncthreads();

      if (position >= 0 && position < max_seq) {
        for (int h = 0; h < num_kv_heads; ++h) {
          for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            int idx = (kv_layout == 0)
                          ? kv_index_seq_hidden(position, h, d, head_dim, kv_dim)
                          : kv_index_head_dim_seq(position, h, d, head_dim, max_seq);
            kv_k[idx] = k[h * head_dim + d];
            kv_v[idx] = v[h * head_dim + d];
          }
        }
      }
      __syncthreads();

      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        attn_out[i] = 0.0f;
      }
      __syncthreads();

      for (int h = 0; h < num_heads; ++h) {
        int hk = h / group_size;
        if (kv_layout == 0) {
          float local_max = -1e30f;
          for (int t0 = 0; t0 <= last; t0 += tile_t) {
            int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
            int tile_elems = tile_len * head_dim;
            for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
              int t = idx / head_dim;
              int d = idx - t * head_dim;
              int gidx = kv_index_seq_hidden(t0 + t, hk, d, head_dim, kv_dim);
              k_tile[idx] = kv_k[gidx];
            }
            __syncthreads();
            if (threadIdx.x < tile_len) {
              float dot = 0.0f;
              int t = threadIdx.x;
              for (int d = 0; d < head_dim; ++d) {
                dot += q[h * head_dim + d] * k_tile[t * head_dim + d];
              }
              dot /= sqrtf((float)head_dim);
              if (dot > local_max) {
                local_max = dot;
              }
            }
            __syncthreads();
          }
          head_sum[h * blockDim.x + threadIdx.x] = local_max;
          __syncthreads();
          if (threadIdx.x == 0) {
            float max_score = -1e30f;
            for (int i = 0; i < blockDim.x; ++i) {
              float v0 = head_sum[h * blockDim.x + i];
              if (v0 > max_score) {
                max_score = v0;
              }
            }
            head_max[h] = max_score;
          }
          __syncthreads();

          float local_denom = 0.0f;
          float max_score = head_max[h];
          for (int t0 = 0; t0 <= last; t0 += tile_t) {
            int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
            int tile_elems = tile_len * head_dim;
            for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
              int t = idx / head_dim;
              int d = idx - t * head_dim;
              int gidx = kv_index_seq_hidden(t0 + t, hk, d, head_dim, kv_dim);
              k_tile[idx] = kv_k[gidx];
            }
            __syncthreads();
            if (threadIdx.x < tile_len) {
              float dot = 0.0f;
              int t = threadIdx.x;
              for (int d = 0; d < head_dim; ++d) {
                dot += q[h * head_dim + d] * k_tile[t * head_dim + d];
              }
              dot /= sqrtf((float)head_dim);
              local_denom += expf(dot - max_score);
            }
            __syncthreads();
          }
          head_sum[h * blockDim.x + threadIdx.x] = local_denom;
          __syncthreads();
          if (threadIdx.x == 0) {
            float denom = 0.0f;
            for (int i = 0; i < blockDim.x; ++i) {
              denom += head_sum[h * blockDim.x + i];
            }
            head_denom[h] = denom;
          }
          __syncthreads();
        } else {
          float local_max = -1e30f;
          for (int t = threadIdx.x; t <= last; t += blockDim.x) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
              int idx = kv_index_head_dim_seq(t, hk, d, head_dim, max_seq);
              dot += q[h * head_dim + d] * kv_k[idx];
            }
            dot /= sqrtf((float)head_dim);
            if (dot > local_max) {
              local_max = dot;
            }
          }
          head_sum[h * blockDim.x + threadIdx.x] = local_max;
          __syncthreads();
          if (threadIdx.x == 0) {
            float max_score = -1e30f;
            for (int i = 0; i < blockDim.x; ++i) {
              float v0 = head_sum[h * blockDim.x + i];
              if (v0 > max_score) {
                max_score = v0;
              }
            }
            head_max[h] = max_score;
          }
          __syncthreads();

          float local_denom = 0.0f;
          float max_score = head_max[h];
          for (int t = threadIdx.x; t <= last; t += blockDim.x) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
              int idx = kv_index_head_dim_seq(t, hk, d, head_dim, max_seq);
              dot += q[h * head_dim + d] * kv_k[idx];
            }
            dot /= sqrtf((float)head_dim);
            local_denom += expf(dot - max_score);
          }
          head_sum[h * blockDim.x + threadIdx.x] = local_denom;
          __syncthreads();
          if (threadIdx.x == 0) {
            float denom = 0.0f;
            for (int i = 0; i < blockDim.x; ++i) {
              denom += head_sum[h * blockDim.x + i];
            }
            head_denom[h] = denom;
          }
          __syncthreads();
        }
      }

      for (int h = 0; h < num_heads; ++h) {
        int hk = h / group_size;
        if (threadIdx.x < head_dim) {
          int d = threadIdx.x;
          float max_score = head_max[h];
          float denom = head_denom[h];
          float acc = 0.0f;
          if (kv_layout == 0) {
            for (int t0 = 0; t0 <= last; t0 += tile_t) {
              int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
              int tile_elems = tile_len * head_dim;
              for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
                int t = idx / head_dim;
                int dd = idx - t * head_dim;
                int gidx = kv_index_seq_hidden(t0 + t, hk, dd, head_dim, kv_dim);
                k_tile[idx] = kv_k[gidx];
                v_tile[idx] = kv_v[gidx];
              }
              __syncthreads();
              if (threadIdx.x < tile_len) {
                float dot = 0.0f;
                int t = threadIdx.x;
                for (int dd = 0; dd < head_dim; ++dd) {
                  dot += q[h * head_dim + dd] * k_tile[t * head_dim + dd];
                }
                dot /= sqrtf((float)head_dim);
                score_tile[t] = dot;
              }
              __syncthreads();
              for (int t = 0; t < tile_len; ++t) {
                float w = expf(score_tile[t] - max_score) / denom;
                acc += w * v_tile[t * head_dim + d];
              }
              __syncthreads();
            }
          } else {
            for (int t = 0; t <= last; ++t) {
              float dot = 0.0f;
              for (int dd = 0; dd < head_dim; ++dd) {
                int idx_k2 = kv_index_head_dim_seq(t, hk, dd, head_dim, max_seq);
                dot += q[h * head_dim + dd] * kv_k[idx_k2];
              }
              dot /= sqrtf((float)head_dim);
              float w = expf(dot - max_score) / denom;
              int idx_v = kv_index_head_dim_seq(t, hk, d, head_dim, max_seq);
              acc += w * kv_v[idx_v];
            }
          }
          attn_out[h * head_dim + d] = acc;
        }
        __syncthreads();
      }

      for (int o = threadIdx.x; o < hidden; o += blockDim.x) {
        float acc = 0.0f;
        const float* w = wo + o;
        for (int i = 0; i < hidden; ++i) {
          acc += attn_out[i] * w[i * hidden];
        }
        output[o] = output[o] + acc;
      }
      __syncthreads();

      float local_sum2 = 0.0f;
      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v2 = output[i];
        local_sum2 += v2 * v2;
      }
      reduce[threadIdx.x] = local_sum2;
      __syncthreads();
      if (threadIdx.x == 0) {
        float sum2 = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
          sum2 += reduce[i];
        }
        reduce[0] = sum2 / hidden;
      }
      __syncthreads();
      float inv_rms2 = rsqrtf(reduce[0] + eps);
      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        normed[i] = output[i] * inv_rms2 * rms_ffn[i];
      }
      __syncthreads();

      for (int o = threadIdx.x; o < mlp_hidden; o += blockDim.x) {
        float acc_gate = 0.0f;
        float acc_up = 0.0f;
        const float* wg = wgate + o;
        const float* wu = wup + o;
        for (int i = 0; i < hidden; ++i) {
          float v0 = normed[i];
          acc_gate += v0 * wg[i * mlp_hidden];
          acc_up += v0 * wu[i * mlp_hidden];
        }
        mlp_gate[o] = silu(acc_gate);
        mlp_up[o] = acc_up;
      }
      __syncthreads();

      for (int o = threadIdx.x; o < hidden; o += blockDim.x) {
        float acc = 0.0f;
        const float* w = wdown + o;
        for (int i = 0; i < mlp_hidden; ++i) {
          acc += (mlp_gate[i] * mlp_up[i]) * w[i * hidden];
        }
        output[o] = output[o] + acc;
      }
      __syncthreads();
    }
  }
}

__global__ void megakernel_persistent_seq(
    const float* inputs,
    const float* rms_attn_weight,
    const float* rms_ffn_weight,
    const float* w_qkv,
    const float* w_o,
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    float* kv_k,
    float* kv_v,
    float* outputs,
    float* q_buf,
    float* attn_buf,
    int* barrier_counter,
    int* barrier_sense,
    int* sync_flags,
    int num_layers,
    int hidden,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    int seq_len,
    int max_seq,
    int kv_layout,
    float eps,
    float rope_theta,
    int total_blocks) {
  BlockRole role = (BlockRole)(blockIdx.x % 4);
  int role_index = blockIdx.x / 4;
  int role_blocks = (total_blocks + 3) / 4;
  (void)barrier_counter;
  (void)barrier_sense;
  int prefetch_blocks = role_count(total_blocks, ROLE_PREFETCH);
  (void)prefetch_blocks;
  int kv_dim = num_kv_heads * head_dim;
  int qkv_out = hidden + 2 * kv_dim;
  int group_size = num_heads / num_kv_heads;
  int mlp_hidden = 4 * hidden;

  extern __shared__ float scratch[];
  float* normed = scratch;
  float* qkv = normed + hidden;
  float* attn_out = qkv + qkv_out;
  float* mlp_gate = attn_out + hidden;
  float* mlp_up = mlp_gate + 4 * hidden;
  float* reduce = mlp_up + 4 * hidden;
  float* head_max = reduce + blockDim.x;
  float* head_denom = head_max + num_heads;
  float* head_sum = head_denom + num_heads;
  const int tile_t = 32;
  float* k_tile = head_sum + num_heads * blockDim.x;
  float* v_tile = k_tile + tile_t * head_dim;
  float* score_tile = v_tile + tile_t * head_dim;

  for (int tpos = 0; tpos < seq_len; ++tpos) {
    int position = start_pos + tpos;
    int last = position;
    if (last < 0) last = 0;
    if (last >= max_seq) last = max_seq - 1;

    const float* input = inputs + tpos * hidden;
    float* output = outputs + tpos * hidden;
    float* q_global = q_buf + tpos * hidden;
    float* attn_global = attn_buf + tpos * hidden;

    if (role == ROLE_QKV) {
      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        output[i] = input[i];
      }
      if (threadIdx.x == 0) {
        sync_flags[2] = 1;
      }
    }
    __syncthreads();

    for (int layer = 0; layer < num_layers; ++layer) {
      const float* rms_attn = rms_attn_weight + layer * hidden;
      const float* rms_ffn = rms_ffn_weight + layer * hidden;
      const float* wqkv = w_qkv + layer * hidden * (hidden + 2 * kv_dim);
      const float* wo = w_o + layer * hidden * hidden;
      const float* wgate = w_gate + layer * hidden * mlp_hidden;
      const float* wup = w_up + layer * hidden * mlp_hidden;
      const float* wdown = w_down + layer * mlp_hidden * hidden;
      volatile float* prefetch_sink = (volatile float*)reduce;

      if (role == ROLE_QKV) {
        while (sync_flags[2] == 0) {
          __nanosleep(50);
        }
        if (threadIdx.x == 0) {
          sync_flags[0] = 0;
          sync_flags[1] = 0;
          sync_flags[2] = 0;
        }
        __syncthreads();
        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
          float v0 = output[i];
          local_sum += v0 * v0;
        }
        reduce[threadIdx.x] = local_sum;
        __syncthreads();
        if (threadIdx.x == 0) {
          float sum = 0.0f;
          for (int i = 0; i < blockDim.x; ++i) {
            sum += reduce[i];
          }
          reduce[0] = sum / hidden;
        }
        __syncthreads();
        float inv_rms = rsqrtf(reduce[0] + eps);
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
          normed[i] = output[i] * inv_rms * rms_attn[i];
        }
        __syncthreads();

        for (int o = role_index; o < qkv_out; o += role_blocks) {
          float acc = 0.0f;
          const float* w = wqkv + o;
          for (int i = 0; i < hidden; ++i) {
            acc += normed[i] * w[i * qkv_out];
          }
          qkv[o] = acc;
        }
        __syncthreads();

        float* q = qkv;
        float* k = qkv + hidden;
        float* v = qkv + hidden + kv_dim;

        if (threadIdx.x == 0 && role_index == 0) {
          for (int h = 0; h < num_heads; ++h) {
            float* qh = q + h * head_dim;
            rope_apply(qh, qh, head_dim, position, rope_theta);
          }
          for (int h = 0; h < num_kv_heads; ++h) {
            float* kh = k + h * head_dim;
            rope_apply(kh, kh, head_dim, position, rope_theta);
          }
        }
        __syncthreads();

        if (position >= 0 && position < max_seq) {
          for (int h = role_index; h < num_kv_heads; h += role_blocks) {
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
              int idx = (kv_layout == 0)
                            ? kv_index_seq_hidden(position, h, d, head_dim, kv_dim)
                            : kv_index_head_dim_seq(position, h, d, head_dim, max_seq);
              kv_k[idx] = k[h * head_dim + d];
              kv_v[idx] = v[h * head_dim + d];
            }
          }
        }
        // Write Q to global buffer (partitioned)
        for (int i = role_index; i < hidden; i += role_blocks) {
          q_global[i] = q[i];
        }
        if (threadIdx.x == 0) {
          int done = atomicAdd(&sync_flags[0], 1) + 1;
          if (done == role_blocks) {
            __threadfence();
            sync_flags[0] = -1;
          }
        }
      }

      if (role == ROLE_PREFETCH) {
        // Touch a slice of weights to warm L2. This is a best-effort prefetch stub.
        float acc = 0.0f;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
          acc += wqkv[i];
        }
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
          acc += wo[i];
        }
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
          acc += wgate[i];
          acc += wup[i];
        }
        if (threadIdx.x == 0) {
          prefetch_sink[0] = acc;
        }
      }

      if (role == ROLE_ATTN) {
        spin_wait((volatile int*)&sync_flags[0], -1);
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
          attn_out[i] = 0.0f;
        }
        __syncthreads();

        for (int h = role_index; h < num_heads; h += role_blocks) {
          int hk = h / group_size;
          if (kv_layout == 0) {
            float local_max = -1e30f;
            for (int t0 = 0; t0 <= last; t0 += tile_t) {
              int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
              int tile_elems = tile_len * head_dim;
              for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
                int t = idx / head_dim;
                int d = idx - t * head_dim;
                int gidx = kv_index_seq_hidden(t0 + t, hk, d, head_dim, kv_dim);
                k_tile[idx] = kv_k[gidx];
              }
              __syncthreads();
              if (threadIdx.x < tile_len) {
                float dot = 0.0f;
                int t = threadIdx.x;
                for (int d = 0; d < head_dim; ++d) {
                  dot += q_global[h * head_dim + d] * k_tile[t * head_dim + d];
                }
                dot /= sqrtf((float)head_dim);
                if (dot > local_max) {
                  local_max = dot;
                }
              }
              __syncthreads();
            }
            head_sum[h * blockDim.x + threadIdx.x] = local_max;
            __syncthreads();
            if (threadIdx.x == 0) {
              float max_score = -1e30f;
              for (int i = 0; i < blockDim.x; ++i) {
                float v0 = head_sum[h * blockDim.x + i];
                if (v0 > max_score) {
                  max_score = v0;
                }
              }
              head_max[h] = max_score;
            }
            __syncthreads();

            float local_denom = 0.0f;
            float max_score = head_max[h];
            for (int t0 = 0; t0 <= last; t0 += tile_t) {
              int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
              int tile_elems = tile_len * head_dim;
              for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
                int t = idx / head_dim;
                int d = idx - t * head_dim;
                int gidx = kv_index_seq_hidden(t0 + t, hk, d, head_dim, kv_dim);
                k_tile[idx] = kv_k[gidx];
              }
              __syncthreads();
              if (threadIdx.x < tile_len) {
                float dot = 0.0f;
                int t = threadIdx.x;
                for (int d = 0; d < head_dim; ++d) {
                  dot += q_global[h * head_dim + d] * k_tile[t * head_dim + d];
                }
                dot /= sqrtf((float)head_dim);
                local_denom += expf(dot - max_score);
              }
              __syncthreads();
            }
            head_sum[h * blockDim.x + threadIdx.x] = local_denom;
            __syncthreads();
            if (threadIdx.x == 0) {
              float denom = 0.0f;
              for (int i = 0; i < blockDim.x; ++i) {
                denom += head_sum[h * blockDim.x + i];
              }
              head_denom[h] = denom;
            }
            __syncthreads();
          } else {
            float local_max = -1e30f;
            for (int t = threadIdx.x; t <= last; t += blockDim.x) {
              float dot = 0.0f;
              for (int d = 0; d < head_dim; ++d) {
                int idx = kv_index_head_dim_seq(t, hk, d, head_dim, max_seq);
                dot += q_global[h * head_dim + d] * kv_k[idx];
              }
              dot /= sqrtf((float)head_dim);
              if (dot > local_max) {
                local_max = dot;
              }
            }
            head_sum[h * blockDim.x + threadIdx.x] = local_max;
            __syncthreads();
            if (threadIdx.x == 0) {
              float max_score = -1e30f;
              for (int i = 0; i < blockDim.x; ++i) {
                float v0 = head_sum[h * blockDim.x + i];
                if (v0 > max_score) {
                  max_score = v0;
                }
              }
              head_max[h] = max_score;
            }
            __syncthreads();

            float local_denom = 0.0f;
            float max_score = head_max[h];
            for (int t = threadIdx.x; t <= last; t += blockDim.x) {
              float dot = 0.0f;
              for (int d = 0; d < head_dim; ++d) {
                int idx = kv_index_head_dim_seq(t, hk, d, head_dim, max_seq);
                dot += q_global[h * head_dim + d] * kv_k[idx];
              }
              dot /= sqrtf((float)head_dim);
              local_denom += expf(dot - max_score);
            }
            head_sum[h * blockDim.x + threadIdx.x] = local_denom;
            __syncthreads();
            if (threadIdx.x == 0) {
              float denom = 0.0f;
              for (int i = 0; i < blockDim.x; ++i) {
                denom += head_sum[h * blockDim.x + i];
              }
              head_denom[h] = denom;
            }
            __syncthreads();
          }
        }

        for (int h = role_index; h < num_heads; h += role_blocks) {
          int hk = h / group_size;
          if (threadIdx.x < head_dim) {
            int d = threadIdx.x;
            float max_score = head_max[h];
            float denom = head_denom[h];
            float acc = 0.0f;
            if (kv_layout == 0) {
              for (int t0 = 0; t0 <= last; t0 += tile_t) {
                int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
                int tile_elems = tile_len * head_dim;
                for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
                  int t = idx / head_dim;
                  int dd = idx - t * head_dim;
                  int gidx = kv_index_seq_hidden(t0 + t, hk, dd, head_dim, kv_dim);
                  k_tile[idx] = kv_k[gidx];
                  v_tile[idx] = kv_v[gidx];
                }
                __syncthreads();
                if (threadIdx.x < tile_len) {
                  float dot = 0.0f;
                  int t = threadIdx.x;
                  for (int dd = 0; dd < head_dim; ++dd) {
                    dot += q_global[h * head_dim + dd] * k_tile[t * head_dim + dd];
                  }
                  dot /= sqrtf((float)head_dim);
                  score_tile[t] = dot;
                }
                __syncthreads();
                for (int t = 0; t < tile_len; ++t) {
                  float w = expf(score_tile[t] - max_score) / denom;
                  acc += w * v_tile[t * head_dim + d];
                }
                __syncthreads();
              }
            } else {
              for (int t = 0; t <= last; ++t) {
                float dot = 0.0f;
                for (int dd = 0; dd < head_dim; ++dd) {
                  int idx_k2 = kv_index_head_dim_seq(t, hk, dd, head_dim, max_seq);
                  dot += q_global[h * head_dim + dd] * kv_k[idx_k2];
                }
                dot /= sqrtf((float)head_dim);
                float w = expf(dot - max_score) / denom;
                int idx_v = kv_index_head_dim_seq(t, hk, d, head_dim, max_seq);
                acc += w * kv_v[idx_v];
              }
            }
            attn_out[h * head_dim + d] = acc;
          }
          __syncthreads();
        }
        // Each ATTN block writes its own heads into global buffer.
        for (int h = role_index; h < num_heads; h += role_blocks) {
          for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            attn_global[h * head_dim + d] = attn_out[h * head_dim + d];
          }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          int done = atomicAdd(&sync_flags[1], 1) + 1;
          if (done == role_blocks) {
            __threadfence();
            sync_flags[1] = -1;  // all ATTN blocks done
          }
        }
      }

      if (role == ROLE_MLP) {
        spin_wait((volatile int*)&sync_flags[1], -1);
        for (int o = role_index; o < hidden; o += role_blocks) {
          float acc = 0.0f;
          const float* w = wo + o;
          for (int i = 0; i < hidden; ++i) {
            acc += attn_global[i] * w[i * hidden];
          }
          output[o] = output[o] + acc;
        }
        __syncthreads();

        float local_sum2 = 0.0f;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
          float v2 = output[i];
          local_sum2 += v2 * v2;
        }
        reduce[threadIdx.x] = local_sum2;
        __syncthreads();
        if (threadIdx.x == 0) {
          float sum2 = 0.0f;
          for (int i = 0; i < blockDim.x; ++i) {
            sum2 += reduce[i];
          }
          reduce[0] = sum2 / hidden;
        }
        __syncthreads();
        float inv_rms2 = rsqrtf(reduce[0] + eps);
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
          normed[i] = output[i] * inv_rms2 * rms_ffn[i];
        }
        __syncthreads();

        for (int o = role_index; o < mlp_hidden; o += role_blocks) {
          float acc_gate = 0.0f;
          float acc_up = 0.0f;
          const float* wg = wgate + o;
          const float* wu = wup + o;
          for (int i = 0; i < hidden; ++i) {
            float v0 = normed[i];
            acc_gate += v0 * wg[i * mlp_hidden];
            acc_up += v0 * wu[i * mlp_hidden];
          }
          mlp_gate[o] = silu(acc_gate);
          mlp_up[o] = acc_up;
        }
        __syncthreads();

        for (int o = role_index; o < hidden; o += role_blocks) {
          float acc = 0.0f;
          const float* w = wdown + o;
          for (int i = 0; i < mlp_hidden; ++i) {
            acc += (mlp_gate[i] * mlp_up[i]) * w[i * hidden];
          }
          output[o] = output[o] + acc;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          int done = atomicAdd(&sync_flags[2], 1) + 1;
          if (done == role_blocks) {
            __threadfence();
            sync_flags[2] = 1;
          }
        }
      }
    }
  }
}

void megakernel_forward_cuda(
    torch::Tensor input,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    torch::Tensor output,
    int64_t num_layers,
    int64_t position,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta) {
  int hidden = input.numel();
  int max_seq = (kv_layout == 0) ? kv_k.size(0) : kv_k.size(2);
  const int threads = 256;
  int kv_dim = (int)num_kv_heads * (int)head_dim;
  size_t shared_bytes = (size_t)(hidden + (hidden + 2 * kv_dim) + hidden + 8 * hidden +
                                 threads + 2 * num_heads + num_heads * threads +
                                 2 * tile_t * head_dim + tile_t) * sizeof(float);

  const int blocks = 1;
  auto stream = at::cuda::getDefaultCUDAStream();

  megakernel_naive<<<blocks, threads, shared_bytes, stream>>>(
      input.data_ptr<float>(),
      rms_attn_weight.data_ptr<float>(),
      rms_ffn_weight.data_ptr<float>(),
      w_qkv.data_ptr<float>(),
      w_o.data_ptr<float>(),
      w_gate.data_ptr<float>(),
      w_up.data_ptr<float>(),
      w_down.data_ptr<float>(),
      kv_k.data_ptr<float>(),
      kv_v.data_ptr<float>(),
      output.data_ptr<float>(),
      (int)num_layers,
      hidden,
      (int)num_heads,
      (int)num_kv_heads,
      (int)head_dim,
      (int)position,
      max_seq,
      (int)kv_layout,
      (float)eps,
      (float)rope_theta);
}

void megakernel_forward_seq_cuda(
    torch::Tensor inputs,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    torch::Tensor outputs,
    int64_t num_layers,
    int64_t start_pos,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta,
    int64_t threads_per_block) {
  int hidden = inputs.size(1);
  int seq_len = inputs.size(0);
  int max_seq = (kv_layout == 0) ? kv_k.size(0) : kv_k.size(2);
  const int threads = (int)threads_per_block;
  int kv_dim = (int)num_kv_heads * (int)head_dim;
  int tile_t = 16;
  size_t shared_bytes = (size_t)(hidden + (hidden + 2 * kv_dim) + hidden + 8 * hidden +
                                 threads + 2 * num_heads + num_heads * threads +
                                 2 * tile_t * head_dim + tile_t) * sizeof(float);
  const int blocks = 1;
  auto stream = at::cuda::getDefaultCUDAStream();

  megakernel_naive_seq<<<blocks, threads, shared_bytes, stream>>>(
      inputs.data_ptr<float>(),
      rms_attn_weight.data_ptr<float>(),
      rms_ffn_weight.data_ptr<float>(),
      w_qkv.data_ptr<float>(),
      w_o.data_ptr<float>(),
      w_gate.data_ptr<float>(),
      w_up.data_ptr<float>(),
      w_down.data_ptr<float>(),
      kv_k.data_ptr<float>(),
      kv_v.data_ptr<float>(),
      outputs.data_ptr<float>(),
      (int)num_layers,
      hidden,
      (int)num_heads,
      (int)num_kv_heads,
      (int)head_dim,
      (int)start_pos,
      seq_len,
      max_seq,
      (int)kv_layout,
      (float)eps,
      (float)rope_theta);
}

void megakernel_forward_seq_cuda_bf16(
    torch::Tensor inputs,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    torch::Tensor outputs,
    int64_t num_layers,
    int64_t start_pos,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta,
    int64_t threads_per_block) {
  int hidden = inputs.size(1);
  int seq_len = inputs.size(0);
  int max_seq = (kv_layout == 0) ? kv_k.size(0) : kv_k.size(2);
  const int threads = (int)threads_per_block;
  int kv_dim = (int)num_kv_heads * (int)head_dim;
  int tile_t = 16;
  size_t shared_bytes = (size_t)(hidden + (hidden + 2 * kv_dim) + hidden + 8 * hidden +
                                 threads + 2 * num_heads + num_heads * threads +
                                 2 * tile_t * head_dim + tile_t) * sizeof(float);
  const int blocks = 1;
  auto stream = at::cuda::getDefaultCUDAStream();

  megakernel_naive_seq_bf16<<<blocks, threads, shared_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(inputs.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(rms_attn_weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(rms_ffn_weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w_qkv.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w_o.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w_gate.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w_up.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w_down.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(kv_k.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(kv_v.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(outputs.data_ptr<at::BFloat16>()),
      (int)num_layers,
      hidden,
      (int)num_heads,
      (int)num_kv_heads,
      (int)head_dim,
      (int)start_pos,
      seq_len,
      max_seq,
      (int)kv_layout,
      (float)eps,
      (float)rope_theta);
}

void megakernel_forward_seq_cuda_int8kv(
    torch::Tensor inputs,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor outputs,
    int64_t num_layers,
    int64_t start_pos,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta,
    int64_t threads_per_block) {
  int hidden = inputs.size(1);
  int seq_len = inputs.size(0);
  int max_seq = (kv_layout == 0) ? kv_k.size(0) : kv_k.size(2);
  const int threads = (int)threads_per_block;
  int kv_dim = (int)num_kv_heads * (int)head_dim;
  int tile_t = 16;
  size_t shared_bytes = (size_t)(hidden + (hidden + 2 * kv_dim) + hidden + 8 * hidden +
                                 threads + 2 * num_heads + num_heads * threads +
                                 2 * tile_t * head_dim + tile_t) * sizeof(float);
  const int blocks = 1;
  auto stream = at::cuda::getDefaultCUDAStream();

  megakernel_naive_seq_int8kv<<<blocks, threads, shared_bytes, stream>>>(
      inputs.data_ptr<float>(),
      rms_attn_weight.data_ptr<float>(),
      rms_ffn_weight.data_ptr<float>(),
      w_qkv.data_ptr<float>(),
      w_o.data_ptr<float>(),
      w_gate.data_ptr<float>(),
      w_up.data_ptr<float>(),
      w_down.data_ptr<float>(),
      (int8_t*)kv_k.data_ptr<int8_t>(),
      (int8_t*)kv_v.data_ptr<int8_t>(),
      k_scale.data_ptr<float>(),
      v_scale.data_ptr<float>(),
      outputs.data_ptr<float>(),
      (int)num_layers,
      hidden,
      (int)num_heads,
      (int)num_kv_heads,
      (int)head_dim,
      (int)start_pos,
      seq_len,
      max_seq,
      (int)kv_layout,
      (float)eps,
      (float)rope_theta);
}

void megakernel_forward_seq_persistent_cuda(
    torch::Tensor inputs,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    torch::Tensor outputs,
    torch::Tensor q_buf,
    torch::Tensor attn_buf,
    torch::Tensor barrier_counter,
    torch::Tensor barrier_sense,
    torch::Tensor sync_flags,
    int64_t num_layers,
    int64_t start_pos,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta,
    int64_t num_blocks,
    int64_t threads_per_block) {
  int hidden = inputs.size(1);
  int seq_len = inputs.size(0);
  int max_seq = (kv_layout == 0) ? kv_k.size(0) : kv_k.size(2);
  const int threads = (int)threads_per_block;
  int kv_dim = (int)num_kv_heads * (int)head_dim;
  int tile_t = 16;
  size_t shared_bytes = (size_t)(hidden + (hidden + 2 * kv_dim) + hidden + 8 * hidden +
                                 threads + 2 * num_heads + num_heads * threads +
                                 2 * tile_t * head_dim + tile_t) * sizeof(float);
  auto stream = at::cuda::getDefaultCUDAStream();

  // Launch multi-block persistent kernel.
  megakernel_persistent_seq<<<(int)num_blocks, threads, shared_bytes, stream>>>(
      inputs.data_ptr<float>(),
      rms_attn_weight.data_ptr<float>(),
      rms_ffn_weight.data_ptr<float>(),
      w_qkv.data_ptr<float>(),
      w_o.data_ptr<float>(),
      w_gate.data_ptr<float>(),
      w_up.data_ptr<float>(),
      w_down.data_ptr<float>(),
      kv_k.data_ptr<float>(),
      kv_v.data_ptr<float>(),
      outputs.data_ptr<float>(),
      q_buf.data_ptr<float>(),
      attn_buf.data_ptr<float>(),
      barrier_counter.data_ptr<int>(),
      barrier_sense.data_ptr<int>(),
      sync_flags.data_ptr<int>(),
      (int)num_layers,
      hidden,
      (int)num_heads,
      (int)num_kv_heads,
      (int)head_dim,
      (int)start_pos,
      seq_len,
      max_seq,
      (int)kv_layout,
      (float)eps,
      (float)rope_theta,
      (int)num_blocks);
}

void megakernel_forward_seq_persistent_cuda_int8kv(
    torch::Tensor inputs,
    torch::Tensor rms_attn_weight,
    torch::Tensor rms_ffn_weight,
    torch::Tensor w_qkv,
    torch::Tensor w_o,
    torch::Tensor w_gate,
    torch::Tensor w_up,
    torch::Tensor w_down,
    torch::Tensor kv_k,
    torch::Tensor kv_v,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    torch::Tensor outputs,
    torch::Tensor q_buf,
    torch::Tensor attn_buf,
    torch::Tensor barrier_counter,
    torch::Tensor barrier_sense,
    torch::Tensor sync_flags,
    int64_t num_layers,
    int64_t start_pos,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_layout,
    double eps,
    double rope_theta,
    int64_t num_blocks,
    int64_t threads_per_block) {
  int hidden = inputs.size(1);
  int seq_len = inputs.size(0);
  int max_seq = (kv_layout == 0) ? kv_k.size(0) : kv_k.size(2);
  const int threads = (int)threads_per_block;
  int kv_dim = (int)num_kv_heads * (int)head_dim;
  size_t shared_bytes = (size_t)(hidden + (hidden + 2 * kv_dim) + hidden + 8 * hidden +
                                 threads + 2 * num_heads + num_heads * threads +
                                 2 * 32 * head_dim + 32) * sizeof(float);
  auto stream = at::cuda::getDefaultCUDAStream();

  megakernel_persistent_seq_int8kv<<<(int)num_blocks, threads, shared_bytes, stream>>>(
      inputs.data_ptr<float>(),
      rms_attn_weight.data_ptr<float>(),
      rms_ffn_weight.data_ptr<float>(),
      w_qkv.data_ptr<float>(),
      w_o.data_ptr<float>(),
      w_gate.data_ptr<float>(),
      w_up.data_ptr<float>(),
      w_down.data_ptr<float>(),
      (int8_t*)kv_k.data_ptr<int8_t>(),
      (int8_t*)kv_v.data_ptr<int8_t>(),
      k_scale.data_ptr<float>(),
      v_scale.data_ptr<float>(),
      outputs.data_ptr<float>(),
      q_buf.data_ptr<float>(),
      attn_buf.data_ptr<float>(),
      sync_flags.data_ptr<int>(),
      (int)num_layers,
      hidden,
      (int)num_heads,
      (int)num_kv_heads,
      (int)head_dim,
      (int)start_pos,
      seq_len,
      max_seq,
      (int)kv_layout,
      (float)eps,
      (float)rope_theta,
      (int)num_blocks);
  (void)barrier_counter;
  (void)barrier_sense;
}

__device__ __forceinline__ float ld_bf16(const __nv_bfloat16* p) {
  return __bfloat162float(*p);
}

__device__ __forceinline__ __nv_bfloat16 st_bf16(float v) {
  return __float2bfloat16(v);
}

__global__ void megakernel_naive_seq_bf16(
    const __nv_bfloat16* inputs,
    const __nv_bfloat16* rms_attn_weight,
    const __nv_bfloat16* rms_ffn_weight,
    const __nv_bfloat16* w_qkv,
    const __nv_bfloat16* w_o,
    const __nv_bfloat16* w_gate,
    const __nv_bfloat16* w_up,
    const __nv_bfloat16* w_down,
    __nv_bfloat16* kv_k,
    __nv_bfloat16* kv_v,
    __nv_bfloat16* outputs,
    int num_layers,
    int hidden,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    int seq_len,
    int max_seq,
    int kv_layout,
    float eps,
    float rope_theta) {
  if (blockIdx.x != 0) {
    return;
  }

  int kv_dim = num_kv_heads * head_dim;
  int qkv_out = hidden + 2 * kv_dim;
  int group_size = num_heads / num_kv_heads;
  int mlp_hidden = 4 * hidden;

  extern __shared__ float scratch[];
  float* normed = scratch;                         // hidden
  float* qkv = normed + hidden;                    // hidden + 2*kv_dim
  float* attn_out = qkv + qkv_out;                 // hidden
  float* mlp_gate = attn_out + hidden;             // 4*hidden
  float* mlp_up = mlp_gate + 4 * hidden;           // 4*hidden
  float* reduce = mlp_up + 4 * hidden;             // blockDim.x
  float* head_max = reduce + blockDim.x;           // num_heads
  float* head_denom = head_max + num_heads;        // num_heads
  float* head_sum = head_denom + num_heads;        // num_heads * blockDim.x
  const int tile_t = 32;
  float* k_tile = head_sum + num_heads * blockDim.x;               // tile_t * head_dim
  float* v_tile = k_tile + tile_t * head_dim;                      // tile_t * head_dim
  float* score_tile = v_tile + tile_t * head_dim;                  // tile_t

  for (int tpos = 0; tpos < seq_len; ++tpos) {
    int position = start_pos + tpos;
    int last = position;
    if (last < 0) last = 0;
    if (last >= max_seq) last = max_seq - 1;

    const __nv_bfloat16* input = inputs + tpos * hidden;
    __nv_bfloat16* output = outputs + tpos * hidden;

    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      output[i] = input[i];
    }
    __syncthreads();

    for (int layer = 0; layer < num_layers; ++layer) {
      const __nv_bfloat16* rms_attn = rms_attn_weight + layer * hidden;
      const __nv_bfloat16* rms_ffn = rms_ffn_weight + layer * hidden;
      const __nv_bfloat16* wqkv = w_qkv + layer * hidden * (hidden + 2 * kv_dim);
      const __nv_bfloat16* wo = w_o + layer * hidden * hidden;
      const __nv_bfloat16* wgate = w_gate + layer * hidden * mlp_hidden;
      const __nv_bfloat16* wup = w_up + layer * hidden * mlp_hidden;
      const __nv_bfloat16* wdown = w_down + layer * mlp_hidden * hidden;

      float local_sum = 0.0f;
      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v0 = ld_bf16(&output[i]);
        local_sum += v0 * v0;
      }
      reduce[threadIdx.x] = local_sum;
      __syncthreads();
      if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
          sum += reduce[i];
        }
        reduce[0] = sum / hidden;
      }
      __syncthreads();
      float inv_rms = rsqrtf(reduce[0] + eps);
      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        normed[i] = ld_bf16(&output[i]) * inv_rms * ld_bf16(&rms_attn[i]);
      }
      __syncthreads();

      for (int o = threadIdx.x; o < qkv_out; o += blockDim.x) {
        float acc = 0.0f;
        const __nv_bfloat16* w = wqkv + o;
        for (int i = 0; i < hidden; ++i) {
          acc += normed[i] * ld_bf16(&w[i * qkv_out]);
        }
        qkv[o] = acc;
      }
      __syncthreads();

      float* q = qkv;
      float* k = qkv + hidden;
      float* v = qkv + hidden + kv_dim;

      if (threadIdx.x == 0) {
        for (int h = 0; h < num_heads; ++h) {
          float* qh = q + h * head_dim;
          rope_apply(qh, qh, head_dim, position, rope_theta);
        }
        for (int h = 0; h < num_kv_heads; ++h) {
          float* kh = k + h * head_dim;
          rope_apply(kh, kh, head_dim, position, rope_theta);
        }
      }
      __syncthreads();

      if (position >= 0 && position < max_seq) {
        for (int h = 0; h < num_kv_heads; ++h) {
          for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            int idx = (kv_layout == 0)
                          ? kv_index_seq_hidden(position, h, d, head_dim, kv_dim)
                          : kv_index_head_dim_seq(position, h, d, head_dim, max_seq);
            kv_k[idx] = st_bf16(k[h * head_dim + d]);
            kv_v[idx] = st_bf16(v[h * head_dim + d]);
          }
        }
      }
      __syncthreads();

      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        attn_out[i] = 0.0f;
      }
      __syncthreads();

      for (int h = 0; h < num_heads; ++h) {
        int hk = h / group_size;
        if (kv_layout == 0) {
          float local_max = -1e30f;
          for (int t0 = 0; t0 <= last; t0 += tile_t) {
            int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
            int tile_elems = tile_len * head_dim;
            for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
              int t = idx / head_dim;
              int d = idx - t * head_dim;
              int gidx = kv_index_seq_hidden(t0 + t, hk, d, head_dim, kv_dim);
              k_tile[idx] = ld_bf16(&kv_k[gidx]);
            }
            __syncthreads();
            if (threadIdx.x < tile_len) {
              float dot = 0.0f;
              int t = threadIdx.x;
              for (int d = 0; d < head_dim; ++d) {
                dot += q[h * head_dim + d] * k_tile[t * head_dim + d];
              }
              dot /= sqrtf((float)head_dim);
              if (dot > local_max) {
                local_max = dot;
              }
            }
            __syncthreads();
          }
          head_sum[h * blockDim.x + threadIdx.x] = local_max;
          __syncthreads();
          if (threadIdx.x == 0) {
            float max_score = -1e30f;
            for (int i = 0; i < blockDim.x; ++i) {
              float v0 = head_sum[h * blockDim.x + i];
              if (v0 > max_score) {
                max_score = v0;
              }
            }
            head_max[h] = max_score;
          }
          __syncthreads();

          float local_denom = 0.0f;
          float max_score = head_max[h];
          for (int t0 = 0; t0 <= last; t0 += tile_t) {
            int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
            int tile_elems = tile_len * head_dim;
            for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
              int t = idx / head_dim;
              int d = idx - t * head_dim;
              int gidx = kv_index_seq_hidden(t0 + t, hk, d, head_dim, kv_dim);
              k_tile[idx] = ld_bf16(&kv_k[gidx]);
            }
            __syncthreads();
            if (threadIdx.x < tile_len) {
              float dot = 0.0f;
              int t = threadIdx.x;
              for (int d = 0; d < head_dim; ++d) {
                dot += q[h * head_dim + d] * k_tile[t * head_dim + d];
              }
              dot /= sqrtf((float)head_dim);
              local_denom += expf(dot - max_score);
            }
            __syncthreads();
          }
          head_sum[h * blockDim.x + threadIdx.x] = local_denom;
          __syncthreads();
          if (threadIdx.x == 0) {
            float denom = 0.0f;
            for (int i = 0; i < blockDim.x; ++i) {
              denom += head_sum[h * blockDim.x + i];
            }
            head_denom[h] = denom;
          }
          __syncthreads();
        }
      }

      for (int h = 0; h < num_heads; ++h) {
        int hk = h / group_size;
        if (threadIdx.x < head_dim) {
          int d = threadIdx.x;
          float max_score = head_max[h];
          float denom = head_denom[h];
          float acc = 0.0f;
          if (kv_layout == 0) {
            for (int t0 = 0; t0 <= last; t0 += tile_t) {
              int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
              int tile_elems = tile_len * head_dim;
              for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
                int t = idx / head_dim;
                int dd = idx - t * head_dim;
                int gidx = kv_index_seq_hidden(t0 + t, hk, dd, head_dim, kv_dim);
                k_tile[idx] = ld_bf16(&kv_k[gidx]);
                v_tile[idx] = ld_bf16(&kv_v[gidx]);
              }
              __syncthreads();
              if (threadIdx.x < tile_len) {
                float dot = 0.0f;
                int t = threadIdx.x;
                for (int dd = 0; dd < head_dim; ++dd) {
                  dot += q[h * head_dim + dd] * k_tile[t * head_dim + dd];
                }
                dot /= sqrtf((float)head_dim);
                score_tile[t] = dot;
              }
              __syncthreads();
              for (int t = 0; t < tile_len; ++t) {
                float w = expf(score_tile[t] - max_score) / denom;
                acc += w * v_tile[t * head_dim + d];
              }
              __syncthreads();
            }
          }
          attn_out[h * head_dim + d] = acc;
        }
        __syncthreads();
      }

      for (int o = threadIdx.x; o < hidden; o += blockDim.x) {
        float acc = 0.0f;
        const __nv_bfloat16* w = wo + o;
        for (int i = 0; i < hidden; ++i) {
          acc += attn_out[i] * ld_bf16(&w[i * hidden]);
        }
        float outv = ld_bf16(&output[o]) + acc;
        output[o] = st_bf16(outv);
      }
      __syncthreads();

      float local_sum2 = 0.0f;
      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v2 = ld_bf16(&output[i]);
        local_sum2 += v2 * v2;
      }
      reduce[threadIdx.x] = local_sum2;
      __syncthreads();
      if (threadIdx.x == 0) {
        float sum2 = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
          sum2 += reduce[i];
        }
        reduce[0] = sum2 / hidden;
      }
      __syncthreads();
      float inv_rms2 = rsqrtf(reduce[0] + eps);
      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        normed[i] = ld_bf16(&output[i]) * inv_rms2 * ld_bf16(&rms_ffn[i]);
      }
      __syncthreads();

      for (int o = threadIdx.x; o < mlp_hidden; o += blockDim.x) {
        float acc_gate = 0.0f;
        float acc_up = 0.0f;
        const __nv_bfloat16* wg = wgate + o;
        const __nv_bfloat16* wu = wup + o;
        for (int i = 0; i < hidden; ++i) {
          float v0 = normed[i];
          acc_gate += v0 * ld_bf16(&wg[i * mlp_hidden]);
          acc_up += v0 * ld_bf16(&wu[i * mlp_hidden]);
        }
        mlp_gate[o] = silu(acc_gate);
        mlp_up[o] = acc_up;
      }
      __syncthreads();

      for (int o = threadIdx.x; o < hidden; o += blockDim.x) {
        float acc = 0.0f;
        const __nv_bfloat16* w = wdown + o;
        for (int i = 0; i < mlp_hidden; ++i) {
          acc += (mlp_gate[i] * mlp_up[i]) * ld_bf16(&w[i * hidden]);
        }
        float outv = ld_bf16(&output[o]) + acc;
        output[o] = st_bf16(outv);
      }
      __syncthreads();
    }
  }
}

__global__ void megakernel_naive_seq_int8kv(
    const float* inputs,
    const float* rms_attn_weight,
    const float* rms_ffn_weight,
    const float* w_qkv,
    const float* w_o,
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    int8_t* kv_k,
    int8_t* kv_v,
    float* k_scale,
    float* v_scale,
    float* outputs,
    int num_layers,
    int hidden,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    int seq_len,
    int max_seq,
    int kv_layout,
    float eps,
    float rope_theta) {
  if (blockIdx.x != 0) {
    return;
  }
  if (kv_layout != 0) {
    return;  // int8 path supports kv_layout=0 only
  }

  int kv_dim = num_kv_heads * head_dim;
  int qkv_out = hidden + 2 * kv_dim;
  int group_size = num_heads / num_kv_heads;
  int mlp_hidden = 4 * hidden;

  extern __shared__ float scratch[];
  float* normed = scratch;                         // hidden
  float* qkv = normed + hidden;                    // hidden + 2*kv_dim
  float* attn_out = qkv + qkv_out;                 // hidden
  float* mlp_gate = attn_out + hidden;             // 4*hidden
  float* mlp_up = mlp_gate + 4 * hidden;           // 4*hidden
  float* reduce = mlp_up + 4 * hidden;             // blockDim.x
  float* head_max = reduce + blockDim.x;           // num_heads
  float* head_denom = head_max + num_heads;        // num_heads
  float* head_sum = head_denom + num_heads;        // num_heads * blockDim.x
  const int tile_t = 32;
  float* k_tile = head_sum + num_heads * blockDim.x;               // tile_t * head_dim
  float* v_tile = k_tile + tile_t * head_dim;                      // tile_t * head_dim
  float* score_tile = v_tile + tile_t * head_dim;                  // tile_t

  for (int tpos = 0; tpos < seq_len; ++tpos) {
    int position = start_pos + tpos;
    int last = position;
    if (last < 0) last = 0;
    if (last >= max_seq) last = max_seq - 1;

    const float* input = inputs + tpos * hidden;
    float* output = outputs + tpos * hidden;

    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      output[i] = input[i];
    }
    __syncthreads();

    for (int layer = 0; layer < num_layers; ++layer) {
      const float* rms_attn = rms_attn_weight + layer * hidden;
      const float* rms_ffn = rms_ffn_weight + layer * hidden;
      const float* wqkv = w_qkv + layer * hidden * (hidden + 2 * kv_dim);
      const float* wo = w_o + layer * hidden * hidden;
      const float* wgate = w_gate + layer * hidden * mlp_hidden;
      const float* wup = w_up + layer * hidden * mlp_hidden;
      const float* wdown = w_down + layer * mlp_hidden * hidden;

      float local_sum = 0.0f;
      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v0 = output[i];
        local_sum += v0 * v0;
      }
      reduce[threadIdx.x] = local_sum;
      __syncthreads();
      if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
          sum += reduce[i];
        }
        reduce[0] = sum / hidden;
      }
      __syncthreads();
      float inv_rms = rsqrtf(reduce[0] + eps);
      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        normed[i] = output[i] * inv_rms * rms_attn[i];
      }
      __syncthreads();

      for (int o = threadIdx.x; o < qkv_out; o += blockDim.x) {
        float acc = 0.0f;
        const float* w = wqkv + o;
        for (int i = 0; i < hidden; ++i) {
          acc += normed[i] * w[i * qkv_out];
        }
        qkv[o] = acc;
      }
      __syncthreads();

      float* q = qkv;
      float* k = qkv + hidden;
      float* v = qkv + hidden + kv_dim;

      if (threadIdx.x == 0) {
        for (int h = 0; h < num_heads; ++h) {
          float* qh = q + h * head_dim;
          rope_apply(qh, qh, head_dim, position, rope_theta);
        }
        for (int h = 0; h < num_kv_heads; ++h) {
          float* kh = k + h * head_dim;
          rope_apply(kh, kh, head_dim, position, rope_theta);
        }
      }
      __syncthreads();

      if (position >= 0 && position < max_seq) {
        for (int h = 0; h < num_kv_heads; ++h) {
          // compute scale per head for K and V
          float maxk = 0.0f;
          float maxv = 0.0f;
          for (int d = 0; d < head_dim; ++d) {
            float kv = fabsf(k[h * head_dim + d]);
            float vv = fabsf(v[h * head_dim + d]);
            if (kv > maxk) maxk = kv;
            if (vv > maxv) maxv = vv;
          }
          float ks = maxk / 127.0f;
          float vs = maxv / 127.0f;
          k_scale[position * num_kv_heads + h] = ks;
          v_scale[position * num_kv_heads + h] = vs;
          for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            int idx = kv_index_seq_hidden(position, h, d, head_dim, kv_dim);
            int8_t kq = (int8_t)llrintf(k[h * head_dim + d] / ks);
            int8_t vq = (int8_t)llrintf(v[h * head_dim + d] / vs);
            kv_k[idx] = kq;
            kv_v[idx] = vq;
          }
        }
      }
      __syncthreads();

      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        attn_out[i] = 0.0f;
      }
      __syncthreads();

      for (int h = 0; h < num_heads; ++h) {
        int hk = h / group_size;
        float local_max = -1e30f;
        for (int t0 = 0; t0 <= last; t0 += tile_t) {
          int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
          int tile_elems = tile_len * head_dim;
          for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
            int t = idx / head_dim;
            int d = idx - t * head_dim;
            int gidx = kv_index_seq_hidden(t0 + t, hk, d, head_dim, kv_dim);
            float ks = k_scale[(t0 + t) * num_kv_heads + hk];
            k_tile[idx] = (float)kv_k[gidx] * ks;
          }
          __syncthreads();
          if (threadIdx.x < tile_len) {
            float dot = 0.0f;
            int t = threadIdx.x;
            for (int d = 0; d < head_dim; ++d) {
              dot += q[h * head_dim + d] * k_tile[t * head_dim + d];
            }
            dot /= sqrtf((float)head_dim);
            if (dot > local_max) {
              local_max = dot;
            }
          }
          __syncthreads();
        }
        head_sum[h * blockDim.x + threadIdx.x] = local_max;
        __syncthreads();
        if (threadIdx.x == 0) {
          float max_score = -1e30f;
          for (int i = 0; i < blockDim.x; ++i) {
            float v0 = head_sum[h * blockDim.x + i];
            if (v0 > max_score) {
              max_score = v0;
            }
          }
          head_max[h] = max_score;
        }
        __syncthreads();

        float local_denom = 0.0f;
        float max_score = head_max[h];
        for (int t0 = 0; t0 <= last; t0 += tile_t) {
          int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
          int tile_elems = tile_len * head_dim;
          for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
            int t = idx / head_dim;
            int d = idx - t * head_dim;
            int gidx = kv_index_seq_hidden(t0 + t, hk, d, head_dim, kv_dim);
            float ks = k_scale[(t0 + t) * num_kv_heads + hk];
            k_tile[idx] = (float)kv_k[gidx] * ks;
          }
          __syncthreads();
          if (threadIdx.x < tile_len) {
            float dot = 0.0f;
            int t = threadIdx.x;
            for (int d = 0; d < head_dim; ++d) {
              dot += q[h * head_dim + d] * k_tile[t * head_dim + d];
            }
            dot /= sqrtf((float)head_dim);
            local_denom += expf(dot - max_score);
          }
          __syncthreads();
        }
        head_sum[h * blockDim.x + threadIdx.x] = local_denom;
        __syncthreads();
        if (threadIdx.x == 0) {
          float denom = 0.0f;
          for (int i = 0; i < blockDim.x; ++i) {
            denom += head_sum[h * blockDim.x + i];
          }
          head_denom[h] = denom;
        }
        __syncthreads();
      }

      for (int h = 0; h < num_heads; ++h) {
        int hk = h / group_size;
        if (threadIdx.x < head_dim) {
          int d = threadIdx.x;
          float max_score = head_max[h];
          float denom = head_denom[h];
          float acc = 0.0f;
          for (int t0 = 0; t0 <= last; t0 += tile_t) {
            int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
            int tile_elems = tile_len * head_dim;
            for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
              int t = idx / head_dim;
              int dd = idx - t * head_dim;
              int gidx = kv_index_seq_hidden(t0 + t, hk, dd, head_dim, kv_dim);
              float ks = k_scale[(t0 + t) * num_kv_heads + hk];
              float vs = v_scale[(t0 + t) * num_kv_heads + hk];
              k_tile[idx] = (float)kv_k[gidx] * ks;
              v_tile[idx] = (float)kv_v[gidx] * vs;
            }
            __syncthreads();
            if (threadIdx.x < tile_len) {
              float dot = 0.0f;
              int t = threadIdx.x;
              for (int dd = 0; dd < head_dim; ++dd) {
                dot += q[h * head_dim + dd] * k_tile[t * head_dim + dd];
              }
              dot /= sqrtf((float)head_dim);
              score_tile[t] = dot;
            }
            __syncthreads();
            for (int t = 0; t < tile_len; ++t) {
              float w = expf(score_tile[t] - max_score) / denom;
              acc += w * v_tile[t * head_dim + d];
            }
            __syncthreads();
          }
          attn_out[h * head_dim + d] = acc;
        }
        __syncthreads();
      }

      for (int o = threadIdx.x; o < hidden; o += blockDim.x) {
        float acc = 0.0f;
        const float* w = wo + o;
        for (int i = 0; i < hidden; ++i) {
          acc += attn_out[i] * w[i * hidden];
        }
        output[o] = output[o] + acc;
      }
      __syncthreads();

      float local_sum2 = 0.0f;
      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v2 = output[i];
        local_sum2 += v2 * v2;
      }
      reduce[threadIdx.x] = local_sum2;
      __syncthreads();
      if (threadIdx.x == 0) {
        float sum2 = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
          sum2 += reduce[i];
        }
        reduce[0] = sum2 / hidden;
      }
      __syncthreads();
      float inv_rms2 = rsqrtf(reduce[0] + eps);
      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        normed[i] = output[i] * inv_rms2 * rms_ffn[i];
      }
      __syncthreads();

      for (int o = threadIdx.x; o < mlp_hidden; o += blockDim.x) {
        float acc_gate = 0.0f;
        float acc_up = 0.0f;
        const float* wg = wgate + o;
        const float* wu = wup + o;
        for (int i = 0; i < hidden; ++i) {
          float v0 = normed[i];
          acc_gate += v0 * wg[i * mlp_hidden];
          acc_up += v0 * wu[i * mlp_hidden];
        }
        mlp_gate[o] = silu(acc_gate);
        mlp_up[o] = acc_up;
      }
      __syncthreads();

      for (int o = threadIdx.x; o < hidden; o += blockDim.x) {
        float acc = 0.0f;
        const float* w = wdown + o;
        for (int i = 0; i < mlp_hidden; ++i) {
          acc += (mlp_gate[i] * mlp_up[i]) * w[i * hidden];
        }
        output[o] = output[o] + acc;
      }
      __syncthreads();
    }
  }
}

__global__ void megakernel_persistent_seq_int8kv(
    const float* inputs,
    const float* rms_attn_weight,
    const float* rms_ffn_weight,
    const float* w_qkv,
    const float* w_o,
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    int8_t* kv_k,
    int8_t* kv_v,
    float* k_scale,
    float* v_scale,
    float* outputs,
    float* q_buf,
    float* attn_buf,
    int* sync_flags,
    int num_layers,
    int hidden,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    int seq_len,
    int max_seq,
    int kv_layout,
    float eps,
    float rope_theta,
    int total_blocks) {
  if (kv_layout != 0) {
    return;
  }

  BlockRole role = (BlockRole)(blockIdx.x % 4);
  int role_index = blockIdx.x / 4;
  int role_blocks = (total_blocks + 3) / 4;

  int kv_dim = num_kv_heads * head_dim;
  int qkv_out = hidden + 2 * kv_dim;
  int group_size = num_heads / num_kv_heads;
  int mlp_hidden = 4 * hidden;

  extern __shared__ float scratch[];
  float* normed = scratch;
  float* qkv = normed + hidden;
  float* attn_out = qkv + qkv_out;
  float* mlp_gate = attn_out + hidden;
  float* mlp_up = mlp_gate + 4 * hidden;
  float* reduce = mlp_up + 4 * hidden;
  float* head_max = reduce + blockDim.x;
  float* head_denom = head_max + num_heads;
  float* head_sum = head_denom + num_heads;
  const int tile_t = 32;
  float* k_tile = head_sum + num_heads * blockDim.x;
  float* v_tile = k_tile + tile_t * head_dim;
  float* score_tile = v_tile + tile_t * head_dim;

  for (int tpos = 0; tpos < seq_len; ++tpos) {
    int position = start_pos + tpos;
    int last = position;
    if (last < 0) last = 0;
    if (last >= max_seq) last = max_seq - 1;

    const float* input = inputs + tpos * hidden;
    float* output = outputs + tpos * hidden;
    float* q_global = q_buf + tpos * hidden;
    float* attn_global = attn_buf + tpos * hidden;

    if (role == ROLE_QKV) {
      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        output[i] = input[i];
      }
      if (threadIdx.x == 0) {
        sync_flags[2] = 1;
      }
    }
    __syncthreads();

    for (int layer = 0; layer < num_layers; ++layer) {
      const float* rms_attn = rms_attn_weight + layer * hidden;
      const float* rms_ffn = rms_ffn_weight + layer * hidden;
      const float* wqkv = w_qkv + layer * hidden * (hidden + 2 * kv_dim);
      const float* wo = w_o + layer * hidden * hidden;
      const float* wgate = w_gate + layer * hidden * mlp_hidden;
      const float* wup = w_up + layer * hidden * mlp_hidden;
      const float* wdown = w_down + layer * mlp_hidden * hidden;

      if (role == ROLE_QKV) {
        while (sync_flags[2] == 0) {
          __nanosleep(50);
        }
        if (threadIdx.x == 0) {
          sync_flags[0] = 0;
          sync_flags[1] = 0;
          sync_flags[2] = 0;
        }
        __syncthreads();

        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
          float v0 = output[i];
          local_sum += v0 * v0;
        }
        reduce[threadIdx.x] = local_sum;
        __syncthreads();
        if (threadIdx.x == 0) {
          float sum = 0.0f;
          for (int i = 0; i < blockDim.x; ++i) {
            sum += reduce[i];
          }
          reduce[0] = sum / hidden;
        }
        __syncthreads();
        float inv_rms = rsqrtf(reduce[0] + eps);
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
          normed[i] = output[i] * inv_rms * rms_attn[i];
        }
        __syncthreads();

        for (int o = role_index; o < qkv_out; o += role_blocks) {
          float acc = 0.0f;
          const float* w = wqkv + o;
          for (int i = 0; i < hidden; ++i) {
            acc += normed[i] * w[i * qkv_out];
          }
          qkv[o] = acc;
        }
        __syncthreads();

        float* q = qkv;
        float* k = qkv + hidden;
        float* v = qkv + hidden + kv_dim;

        if (threadIdx.x == 0 && role_index == 0) {
          for (int h = 0; h < num_heads; ++h) {
            float* qh = q + h * head_dim;
            rope_apply(qh, qh, head_dim, position, rope_theta);
          }
          for (int h = 0; h < num_kv_heads; ++h) {
            float* kh = k + h * head_dim;
            rope_apply(kh, kh, head_dim, position, rope_theta);
          }
        }
        __syncthreads();

        if (position >= 0 && position < max_seq) {
          for (int h = role_index; h < num_kv_heads; h += role_blocks) {
            float maxk = 0.0f;
            float maxv = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
              float kvv = fabsf(k[h * head_dim + d]);
              float vvv = fabsf(v[h * head_dim + d]);
              if (kvv > maxk) maxk = kvv;
              if (vvv > maxv) maxv = vvv;
            }
            float ks = maxk / 127.0f;
            float vs = maxv / 127.0f;
            k_scale[position * num_kv_heads + h] = ks;
            v_scale[position * num_kv_heads + h] = vs;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
              int idx = kv_index_seq_hidden(position, h, d, head_dim, kv_dim);
              int8_t kq = (int8_t)llrintf(k[h * head_dim + d] / ks);
              int8_t vq = (int8_t)llrintf(v[h * head_dim + d] / vs);
              kv_k[idx] = kq;
              kv_v[idx] = vq;
            }
          }
        }
        for (int i = role_index; i < hidden; i += role_blocks) {
          q_global[i] = q[i];
        }
        if (threadIdx.x == 0) {
          int done = atomicAdd(&sync_flags[0], 1) + 1;
          if (done == role_blocks) {
            __threadfence();
            sync_flags[0] = -1;
          }
        }
      }

      if (role == ROLE_PREFETCH) {
        // no-op
      }

      if (role == ROLE_ATTN) {
        while (sync_flags[0] != -1) {
          __nanosleep(50);
        }
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
          attn_out[i] = 0.0f;
        }
        __syncthreads();

        for (int h = role_index; h < num_heads; h += role_blocks) {
          int hk = h / group_size;
          float local_max = -1e30f;
          for (int t0 = 0; t0 <= last; t0 += tile_t) {
            int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
            int tile_elems = tile_len * head_dim;
            for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
              int t = idx / head_dim;
              int d = idx - t * head_dim;
              int gidx = kv_index_seq_hidden(t0 + t, hk, d, head_dim, kv_dim);
              float ks = k_scale[(t0 + t) * num_kv_heads + hk];
              k_tile[idx] = (float)kv_k[gidx] * ks;
            }
            __syncthreads();
            if (threadIdx.x < tile_len) {
              float dot = 0.0f;
              int t = threadIdx.x;
              for (int d = 0; d < head_dim; ++d) {
                dot += q_global[h * head_dim + d] * k_tile[t * head_dim + d];
              }
              dot /= sqrtf((float)head_dim);
              if (dot > local_max) {
                local_max = dot;
              }
            }
            __syncthreads();
          }
          head_sum[h * blockDim.x + threadIdx.x] = local_max;
          __syncthreads();
          if (threadIdx.x == 0) {
            float max_score = -1e30f;
            for (int i = 0; i < blockDim.x; ++i) {
              float v0 = head_sum[h * blockDim.x + i];
              if (v0 > max_score) {
                max_score = v0;
              }
            }
            head_max[h] = max_score;
          }
          __syncthreads();

          float local_denom = 0.0f;
          float max_score = head_max[h];
          for (int t0 = 0; t0 <= last; t0 += tile_t) {
            int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
            int tile_elems = tile_len * head_dim;
            for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
              int t = idx / head_dim;
              int d = idx - t * head_dim;
              int gidx = kv_index_seq_hidden(t0 + t, hk, d, head_dim, kv_dim);
              float ks = k_scale[(t0 + t) * num_kv_heads + hk];
              k_tile[idx] = (float)kv_k[gidx] * ks;
            }
            __syncthreads();
            if (threadIdx.x < tile_len) {
              float dot = 0.0f;
              int t = threadIdx.x;
              for (int d = 0; d < head_dim; ++d) {
                dot += q_global[h * head_dim + d] * k_tile[t * head_dim + d];
              }
              dot /= sqrtf((float)head_dim);
              local_denom += expf(dot - max_score);
            }
            __syncthreads();
          }
          head_sum[h * blockDim.x + threadIdx.x] = local_denom;
          __syncthreads();
          if (threadIdx.x == 0) {
            float denom = 0.0f;
            for (int i = 0; i < blockDim.x; ++i) {
              denom += head_sum[h * blockDim.x + i];
            }
            head_denom[h] = denom;
          }
          __syncthreads();
        }

        for (int h = role_index; h < num_heads; h += role_blocks) {
          int hk = h / group_size;
          if (threadIdx.x < head_dim) {
            int d = threadIdx.x;
            float max_score = head_max[h];
            float denom = head_denom[h];
            float acc = 0.0f;
            for (int t0 = 0; t0 <= last; t0 += tile_t) {
              int tile_len = (t0 + tile_t <= last + 1) ? tile_t : (last + 1 - t0);
              int tile_elems = tile_len * head_dim;
              for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
                int t = idx / head_dim;
                int dd = idx - t * head_dim;
                int gidx = kv_index_seq_hidden(t0 + t, hk, dd, head_dim, kv_dim);
                float ks = k_scale[(t0 + t) * num_kv_heads + hk];
                float vs = v_scale[(t0 + t) * num_kv_heads + hk];
                k_tile[idx] = (float)kv_k[gidx] * ks;
                v_tile[idx] = (float)kv_v[gidx] * vs;
              }
              __syncthreads();
              if (threadIdx.x < tile_len) {
                float dot = 0.0f;
                int t = threadIdx.x;
                for (int dd = 0; dd < head_dim; ++dd) {
                  dot += q_global[h * head_dim + dd] * k_tile[t * head_dim + dd];
                }
                dot /= sqrtf((float)head_dim);
                score_tile[t] = dot;
              }
              __syncthreads();
              for (int t = 0; t < tile_len; ++t) {
                float w = expf(score_tile[t] - max_score) / denom;
                acc += w * v_tile[t * head_dim + d];
              }
              __syncthreads();
            }
            attn_out[h * head_dim + d] = acc;
          }
          __syncthreads();
        }

        for (int h = role_index; h < num_heads; h += role_blocks) {
          for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            attn_global[h * head_dim + d] = attn_out[h * head_dim + d];
          }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          int done = atomicAdd(&sync_flags[1], 1) + 1;
          if (done == role_blocks) {
            __threadfence();
            sync_flags[1] = -1;
          }
        }
      }

      if (role == ROLE_MLP) {
        while (sync_flags[1] != -1) {
          __nanosleep(50);
        }
        for (int o = role_index; o < hidden; o += role_blocks) {
          float acc = 0.0f;
          const float* w = wo + o;
          for (int i = 0; i < hidden; ++i) {
            acc += attn_global[i] * w[i * hidden];
          }
          output[o] = output[o] + acc;
        }
        __syncthreads();

        float local_sum2 = 0.0f;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
          float v2 = output[i];
          local_sum2 += v2 * v2;
        }
        reduce[threadIdx.x] = local_sum2;
        __syncthreads();
        if (threadIdx.x == 0) {
          float sum2 = 0.0f;
          for (int i = 0; i < blockDim.x; ++i) {
            sum2 += reduce[i];
          }
          reduce[0] = sum2 / hidden;
        }
        __syncthreads();
        float inv_rms2 = rsqrtf(reduce[0] + eps);
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
          normed[i] = output[i] * inv_rms2 * rms_ffn[i];
        }
        __syncthreads();

        for (int o = role_index; o < mlp_hidden; o += role_blocks) {
          float acc_gate = 0.0f;
          float acc_up = 0.0f;
          const float* wg = wgate + o;
          const float* wu = wup + o;
          for (int i = 0; i < hidden; ++i) {
            float v0 = normed[i];
            acc_gate += v0 * wg[i * mlp_hidden];
            acc_up += v0 * wu[i * mlp_hidden];
          }
          mlp_gate[o] = silu(acc_gate);
          mlp_up[o] = acc_up;
        }
        __syncthreads();

        for (int o = role_index; o < hidden; o += role_blocks) {
          float acc = 0.0f;
          const float* w = wdown + o;
          for (int i = 0; i < mlp_hidden; ++i) {
            acc += (mlp_gate[i] * mlp_up[i]) * w[i * hidden];
          }
          output[o] = output[o] + acc;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          int done = atomicAdd(&sync_flags[2], 1) + 1;
          if (done == role_blocks) {
            __threadfence();
            sync_flags[2] = 1;
          }
        }
      }
    }
  }
}
