#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void greedy_sample(const float* logits, int vocab_size, int* out_token) {
  // Simple block reduction for max logit.
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  float best_val = -1e30f;
  int best_idx = 0;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    float v = logits[i];
    if (v > best_val) {
      best_val = v;
      best_idx = i;
    }
  }
  sdata[tid] = best_val;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid + s] > sdata[tid]) {
        sdata[tid] = sdata[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    float max_val = sdata[0];
    // Find index of max_val (second pass)
    for (int i = 0; i < vocab_size; ++i) {
      if (logits[i] == max_val) {
        out_token[0] = i;
        return;
      }
    }
    out_token[0] = 0;
  }
}
