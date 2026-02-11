#include <torch/extension.h>

void greedy_sample_cuda(torch::Tensor logits, torch::Tensor out_token);

torch::Tensor greedy_sample(torch::Tensor logits) {
  if (!logits.is_cuda()) {
    throw std::invalid_argument("greedy_sample expects CUDA logits");
  }
  auto out = torch::zeros({1}, torch::TensorOptions().device(logits.device()).dtype(torch::kInt32));
  greedy_sample_cuda(logits, out);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("greedy_sample", &greedy_sample, "Greedy sample (CUDA)");
}
