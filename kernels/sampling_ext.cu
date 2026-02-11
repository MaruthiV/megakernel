#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

extern "C" __global__ void greedy_sample(const float* logits, int vocab_size, int* out_token);

void greedy_sample_cuda(torch::Tensor logits, torch::Tensor out_token) {
  int vocab = logits.numel();
  const int threads = 256;
  const int blocks = 1;
  greedy_sample<<<blocks, threads, threads * sizeof(float)>>>(
      logits.data_ptr<float>(), vocab, out_token.data_ptr<int>());
}
