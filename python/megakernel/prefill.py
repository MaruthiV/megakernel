import torch

from .extension import load_extension


def prefill_qkv(inputs: torch.Tensor, w_qkv: torch.Tensor) -> torch.Tensor:
    ext = load_extension()
    return ext.prefill_qkv(inputs, w_qkv)


def prefill_mlp(normed: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    ext = load_extension()
    return ext.prefill_mlp(normed, w_gate, w_up, w_down)
