#include <torch/extension.h>

void prefill_qkv_cuda(torch::Tensor inputs, torch::Tensor w_qkv, torch::Tensor out);
void prefill_mlp_cuda(torch::Tensor normed, torch::Tensor w_gate, torch::Tensor w_up, torch::Tensor w_down, torch::Tensor out);

torch::Tensor prefill_qkv(torch::Tensor inputs, torch::Tensor w_qkv) {
  auto out = torch::zeros({inputs.size(0), w_qkv.size(1)}, torch::TensorOptions().device(inputs.device()).dtype(torch::kFloat32));
  prefill_qkv_cuda(inputs, w_qkv, out);
  return out;
}

torch::Tensor prefill_mlp(torch::Tensor normed, torch::Tensor w_gate, torch::Tensor w_up, torch::Tensor w_down) {
  auto out = torch::zeros({normed.size(0), w_down.size(1)}, torch::TensorOptions().device(normed.device()).dtype(torch::kFloat32));
  prefill_mlp_cuda(normed, w_gate, w_up, w_down, out);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prefill_qkv", &prefill_qkv, "Prefill QKV (CUDA)");
  m.def("prefill_mlp", &prefill_mlp, "Prefill MLP (CUDA)");
}
