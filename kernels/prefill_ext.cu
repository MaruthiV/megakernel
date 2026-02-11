#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <cuda_bf16.h>

using namespace nvcuda;

__global__ void wmma_gemm_bf16_kernel_prefill(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int M, int N, int K) {
  int tile_m = blockIdx.y;
  int tile_n = blockIdx.x;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += 16) {
    const __nv_bfloat16* A_tile = A + (tile_m * 16) * K + k0;
    const __nv_bfloat16* B_tile = B + (k0) * N + tile_n * 16;
    wmma::load_matrix_sync(a_frag, A_tile, K);
    wmma::load_matrix_sync(b_frag, B_tile, N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  float* C_tile = C + (tile_m * 16) * N + tile_n * 16;
  wmma::store_matrix_sync(C_tile, c_frag, N, wmma::mem_row_major);
}

void prefill_qkv_cuda(torch::Tensor inputs, torch::Tensor w_qkv, torch::Tensor out) {
  int M = inputs.size(0);
  int K = inputs.size(1);
  int N = w_qkv.size(1);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  dim3 block(32, 1, 1);
  wmma_gemm_bf16_kernel_prefill<<<grid, block>>>(
      reinterpret_cast<const __nv_bfloat16*>(inputs.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w_qkv.data_ptr<at::BFloat16>()),
      out.data_ptr<float>(),
      M, N, K);
}

void prefill_mlp_cuda(torch::Tensor normed, torch::Tensor w_gate, torch::Tensor w_up, torch::Tensor w_down, torch::Tensor out) {
  // Simplified: out = (silu(normed*gate) * (normed*up)) * down
  auto gate = torch::zeros({normed.size(0), w_gate.size(1)}, torch::TensorOptions().device(normed.device()).dtype(torch::kFloat32));
  auto up = torch::zeros({normed.size(0), w_up.size(1)}, torch::TensorOptions().device(normed.device()).dtype(torch::kFloat32));

  int M = normed.size(0);
  int K = normed.size(1);
  int N1 = w_gate.size(1);
  int N2 = w_up.size(1);

  dim3 grid1((N1 + 15) / 16, (M + 15) / 16);
  dim3 grid2((N2 + 15) / 16, (M + 15) / 16);
  dim3 block(32, 1, 1);
  wmma_gemm_bf16_kernel_prefill<<<grid1, block>>>(
      reinterpret_cast<const __nv_bfloat16*>(normed.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w_gate.data_ptr<at::BFloat16>()),
      gate.data_ptr<float>(),
      M, N1, K);
  wmma_gemm_bf16_kernel_prefill<<<grid2, block>>>(
      reinterpret_cast<const __nv_bfloat16*>(normed.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w_up.data_ptr<at::BFloat16>()),
      up.data_ptr<float>(),
      M, N2, K);

  // gate = silu(gate)
  gate = gate / (1.0f + torch::exp(-gate));
  auto mul = gate * up;

  int N3 = w_down.size(1);
  dim3 grid3((N3 + 15) / 16, (M + 15) / 16);
  wmma_gemm_bf16_kernel_prefill<<<grid3, block>>>(
      reinterpret_cast<const __nv_bfloat16*>(mul.to(torch::kBFloat16).data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w_down.data_ptr<at::BFloat16>()),
      out.data_ptr<float>(),
      M, N3, N1);
}
