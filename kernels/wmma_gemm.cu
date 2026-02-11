#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

using namespace nvcuda;

// Minimal WMMA GEMM: C = A*B for 16x16x16 tiles, BF16
// A: [M,K], B: [K,N], C: [M,N]
__global__ void wmma_gemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int M, int N, int K) {
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

void wmma_gemm_bf16_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  dim3 block(32, 1, 1);
  wmma_gemm_bf16<<<grid, block>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
      C.data_ptr<float>(),
      M, N, K);
}
