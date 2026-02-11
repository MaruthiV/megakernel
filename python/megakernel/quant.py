from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class KvQuantized:
    k_q: torch.Tensor
    v_q: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor


def quantize_per_head(x: torch.Tensor, num_kv_heads: int, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: [seq, kv_dim] or [kv_heads, head_dim, seq]
    if x.dim() == 2:
        seq, kv_dim = x.shape
        xh = x.view(seq, num_kv_heads, head_dim)
        # scale per (seq, head)
        maxv = xh.abs().amax(dim=2).clamp_min(1e-6)
        scale = maxv / 127.0
        xq = torch.round(xh / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        return xq.view(seq, kv_dim), scale
    else:
        # [kv_heads, head_dim, seq]
        kv_heads, head_dim2, seq = x.shape
        xh = x.permute(2, 0, 1)  # [seq, heads, head_dim]
        maxv = xh.abs().amax(dim=2).clamp_min(1e-6)
        scale = maxv / 127.0
        xq = torch.round(xh / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        return xq.permute(1, 2, 0), scale


def dequant_per_head(xq: torch.Tensor, scale: torch.Tensor, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    if xq.dim() == 2:
        seq, kv_dim = xq.shape
        xh = xq.view(seq, num_kv_heads, head_dim).float()
        x = xh * scale.unsqueeze(-1)
        return x.view(seq, kv_dim)
    else:
        # [kv_heads, head_dim, seq]
        kv_heads, head_dim2, seq = xq.shape
        xh = xq.permute(2, 0, 1).float()
        x = xh * scale.unsqueeze(-1)
        return x.permute(1, 2, 0)
