import math
from typing import Dict, Any, Tuple

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    mean_sq = x.pow(2).mean()
    return x * torch.rsqrt(mean_sq + eps) * weight


from .rope import apply_rope


def rope_apply(
    q: torch.Tensor,
    k: torch.Tensor,
    position: int,
    rope_theta: float,
    rope_scaling: dict | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_out = apply_rope(q, position, rope_theta, rope_scaling)
    k_out = apply_rope(k, position, rope_theta, rope_scaling)
    return q_out, k_out


def attention_single(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    position: int,
    head_dim: int,
    kv_layout: int,
    num_kv_heads: int,
) -> torch.Tensor:
    num_heads = q.shape[0]
    group_size = num_heads // num_kv_heads
    last = max(0, position)
    if kv_layout == 0:
        # [seq, hidden]
        k = k_cache[: last + 1].view(last + 1, num_kv_heads, head_dim)
        v = v_cache[: last + 1].view(last + 1, num_kv_heads, head_dim)
    else:
        # [heads, head_dim, seq]
        k = k_cache[:, :, : last + 1].permute(2, 0, 1)
        v = v_cache[:, :, : last + 1].permute(2, 0, 1)

    qh = q.unsqueeze(0)  # [1, heads, head_dim]
    # Expand K/V to match heads via grouping
    k_exp = k.repeat_interleave(group_size, dim=1)
    v_exp = v.repeat_interleave(group_size, dim=1)
    scores = (qh * k_exp).sum(-1) / math.sqrt(head_dim)  # [seq, heads]
    weights = torch.softmax(scores, dim=0)
    attn = (weights.unsqueeze(-1) * v_exp).sum(0)  # [heads, head_dim]
    return attn


def forward_single_token(
    x: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    kv_cache: Dict[str, torch.Tensor],
    position: int,
    num_heads: int,
    head_dim: int,
    eps: float,
    rope_theta: float,
    kv_layout: int,
    num_kv_heads: int,
    rope_scaling: dict | None,
) -> torch.Tensor:
    hidden = x.numel()
    num_layers = weights["w_qkv"].shape[0]
    out = x

    for layer in range(num_layers):
        rms_attn = weights["rms_attn"][layer]
        rms_ffn = weights["rms_ffn"][layer]
        w_qkv = weights["w_qkv"][layer]
        w_o = weights["w_o"][layer]
        w_gate = weights["w_gate"][layer]
        w_up = weights["w_up"][layer]
        w_down = weights["w_down"][layer]

        normed = rms_norm(out, rms_attn, eps)
        qkv = normed @ w_qkv  # [3*hidden]
        kv_dim = num_kv_heads * head_dim
        q = qkv[:hidden].view(num_heads, head_dim)
        k = qkv[hidden : hidden + kv_dim].view(num_kv_heads, head_dim)
        v = qkv[hidden + kv_dim :].view(num_kv_heads, head_dim)

        q, q = rope_apply(q, q, position, rope_theta, rope_scaling)
        k, k = rope_apply(k, k, position, rope_theta, rope_scaling)

        # Write KV cache
        if kv_layout == 0:
            kv_cache["k"][position].copy_(k.view(-1))
            kv_cache["v"][position].copy_(v.view(-1))
        else:
            kv_cache["k"][:, :, position].copy_(k)
            kv_cache["v"][:, :, position].copy_(v)

        attn = attention_single(q, kv_cache["k"], kv_cache["v"], position, head_dim, kv_layout, num_kv_heads)
        attn_out = attn.view(-1)

        out = out + (attn_out @ w_o)
        normed2 = rms_norm(out, rms_ffn, eps)
        gate = torch.nn.functional.silu(normed2 @ w_gate)
        up = normed2 @ w_up
        down = (gate * up) @ w_down
        out = out + down

    return out


def compare_with_kernel(
    ext,
    model_cfg: Dict[str, Any],
    weights: Dict[str, torch.Tensor],
    kv_cache: Dict[str, torch.Tensor],
    x: torch.Tensor,
    position: int,
) -> Dict[str, float]:
    hidden = int(model_cfg["model"]["hidden_size"])
    num_heads = int(model_cfg["model"]["num_heads"])
    head_dim = int(model_cfg["model"]["head_dim"])
    num_layers = int(model_cfg["model"]["num_layers"])
    num_kv_heads = int(model_cfg["model"].get("num_kv_heads", num_heads))
    eps = float(model_cfg.get("rms_eps", 1e-6))
    rope_theta = float(model_cfg.get("rope_theta", 10000.0))
    rope_scaling = model_cfg.get("rope_scaling")
    kv_layout = int(model_cfg.get("kv_layout", 0))

    # Clone kv for kernel/reference so they don't interfere.
    kv_k = kv_cache["k"].clone()
    kv_v = kv_cache["v"].clone()
    kv_k_ref = kv_cache["k"].clone()
    kv_v_ref = kv_cache["v"].clone()

    out_kernel = ext.megakernel_forward(
        x,
        weights["rms_attn"],
        weights["rms_ffn"],
        weights["w_qkv"],
        weights["w_o"],
        weights["w_gate"],
        weights["w_up"],
        weights["w_down"],
        kv_k,
        kv_v,
        int(num_layers),
        int(position),
        int(num_heads),
        int(num_kv_heads),
        int(head_dim),
        int(kv_layout),
        float(eps),
        float(rope_theta),
    )

    out_ref = forward_single_token(
        x,
        weights,
        {"k": kv_k_ref, "v": kv_v_ref},
        position,
        num_heads,
        head_dim,
        eps,
        rope_theta,
        kv_layout,
        num_kv_heads,
        rope_scaling,
    )

    diff = (out_kernel - out_ref).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }
