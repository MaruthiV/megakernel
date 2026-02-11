import math
from typing import Tuple

import torch


def rope_base_qwen(x: torch.Tensor, position: int, theta: float) -> torch.Tensor:
    """
    Standard RoPE (base Qwen/LLaMA style). Applies rotary to last dim.
    """
    head_dim = x.shape[-1]
    d = torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
    inv_freq = torch.pow(torch.tensor(theta, device=x.device), -d / head_dim)
    angle = position * inv_freq
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack([out_even, out_odd], dim=-1).flatten(-2)


def rope_scaled(
    x: torch.Tensor,
    position: int,
    theta: float,
    factor: float,
) -> torch.Tensor:
    """
    Simple linear RoPE scaling: scale positions by 1/factor.
    """
    scaled_pos = position / factor
    return rope_base_qwen(x, scaled_pos, theta)


def apply_rope(
    x: torch.Tensor,
    position: int,
    theta: float,
    rope_scaling: dict | None,
) -> torch.Tensor:
    if not rope_scaling:
        return rope_base_qwen(x, position, theta)

    typ = rope_scaling.get("type", "linear")
    factor = float(rope_scaling.get("factor", 1.0))
    if typ == "linear":
        return rope_scaled(x, position, theta, factor)
    raise NotImplementedError(f"Unsupported rope_scaling type: {typ}")
