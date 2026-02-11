#include <torch/extension.h>

void wmma_gemm_bf16_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);

torch::Tensor wmma_gemm_bf16(torch::Tensor A, torch::Tensor B) {
  if (!A.is_cuda() || !B.is_cuda()) {
    throw std::invalid_argument("wmma_gemm_bf16 expects CUDA tensors");
  }
  auto C = torch::zeros({A.size(0), B.size(1)}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));
  wmma_gemm_bf16_cuda(A, B, C);
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wmma_gemm_bf16", &wmma_gemm_bf16, "WMMA GEMM BF16 (CUDA)");
}
