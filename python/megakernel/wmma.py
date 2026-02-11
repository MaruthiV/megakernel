from typing import Dict, Any

import torch

from .extension import load_extension


def wmma_gemm_bf16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    ext = load_extension()
    return ext.wmma_gemm_bf16(A, B)


def prefill_qkv_wmma(inputs: torch.Tensor, w_qkv: torch.Tensor) -> torch.Tensor:
    # inputs: [T, H], w_qkv: [H, 3H]
    return wmma_gemm_bf16(inputs, w_qkv)


def prefill_mlp_wmma(normed: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    gate = torch.nn.functional.silu(wmma_gemm_bf16(normed, w_gate))
    up = wmma_gemm_bf16(normed, w_up)
    return wmma_gemm_bf16(gate * up, w_down)
