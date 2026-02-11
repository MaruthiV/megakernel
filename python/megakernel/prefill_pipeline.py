import math
from typing import Dict, Any, Tuple

import torch

from .rope import apply_rope
from .wmma import wmma_gemm_bf16
from .cache import KvCache, init_kv_cache, ensure_kv_capacity


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def prefill_wmma(
    inputs: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    model_cfg: Dict[str, Any],
    kv_cache: KvCache | None = None,
    start_pos: int = 0,
) -> Tuple[torch.Tensor, KvCache]:
    """
    Prefill-style forward using WMMA GEMMs for QKV/MLP and torch attention.
    inputs: [T, H] BF16
    Returns outputs and updated kv_cache.
    """
    hidden = int(model_cfg["model"]["hidden_size"])
    num_heads = int(model_cfg["model"]["num_heads"])
    num_kv_heads = int(model_cfg["model"].get("num_kv_heads", num_heads))
    head_dim = int(model_cfg["model"]["head_dim"])
    num_layers = int(model_cfg["model"]["num_layers"])
    eps = float(model_cfg.get("rms_eps", 1e-6))
    rope_theta = float(model_cfg.get("rope_theta", 10000.0))
    rope_scaling = model_cfg.get("rope_scaling")
    kv_layout = int(model_cfg.get("kv_layout", 0))

    device = inputs.device
    x = inputs.to(device=device, dtype=torch.bfloat16)

    if kv_cache is None:
        kv_cache = init_kv_cache(num_kv_heads, head_dim, start_pos + x.size(0), kv_layout, device, x.dtype)
    else:
        kv_cache = ensure_kv_capacity(kv_cache, start_pos + x.size(0), kv_layout)

    out = x
    for layer in range(num_layers):
        rms_attn = weights["rms_attn"][layer].to(device=device, dtype=x.dtype)
        rms_ffn = weights["rms_ffn"][layer].to(device=device, dtype=x.dtype)
        w_qkv = weights["w_qkv"][layer].to(device=device, dtype=x.dtype)
        w_o = weights["w_o"][layer].to(device=device, dtype=x.dtype)
        w_gate = weights["w_gate"][layer].to(device=device, dtype=x.dtype)
        w_up = weights["w_up"][layer].to(device=device, dtype=x.dtype)
        w_down = weights["w_down"][layer].to(device=device, dtype=x.dtype)

        normed = _rms_norm(out, rms_attn, eps)
        qkv = wmma_gemm_bf16(normed, w_qkv)  # [T, 3H]
        q = qkv[:, :hidden].view(-1, num_heads, head_dim)
        kv_dim = num_kv_heads * head_dim
        k = qkv[:, hidden : hidden + kv_dim].view(-1, num_kv_heads, head_dim)
        v = qkv[:, hidden + kv_dim :].view(-1, num_kv_heads, head_dim)

        # RoPE per position
        for t in range(q.size(0)):
            q[t] = apply_rope(q[t], start_pos + t, rope_theta, rope_scaling).view(num_heads, head_dim)
            k[t] = apply_rope(k[t], start_pos + t, rope_theta, rope_scaling).view(num_kv_heads, head_dim)

        # KV cache update
        if kv_layout == 0:
            kv_cache.k[start_pos : start_pos + q.size(0)] = k.reshape(q.size(0), -1)
            kv_cache.v[start_pos : start_pos + q.size(0)] = v.reshape(q.size(0), -1)
        else:
            kv_cache.k[:, :, start_pos : start_pos + q.size(0)] = k.permute(1, 2, 0)
            kv_cache.v[:, :, start_pos : start_pos + q.size(0)] = v.permute(1, 2, 0)

        # Attention (torch, batched)
        if kv_layout == 0:
            k_all = kv_cache.k[: start_pos + q.size(0)].view(-1, num_kv_heads, head_dim)
            v_all = kv_cache.v[: start_pos + q.size(0)].view(-1, num_kv_heads, head_dim)
        else:
            k_all = kv_cache.k[:, :, : start_pos + q.size(0)].permute(2, 0, 1)
            v_all = kv_cache.v[:, :, : start_pos + q.size(0)].permute(2, 0, 1)

        # Expand KV heads to match Q heads
        group_size = num_heads // num_kv_heads
        k_all = k_all.repeat_interleave(group_size, dim=1)
        v_all = v_all.repeat_interleave(group_size, dim=1)

        # scores: [T, S, H]
        scores = torch.einsum("thd,shd->tsh", q, k_all) / math.sqrt(head_dim)
        weights_attn = torch.softmax(scores, dim=1)
        attn = torch.einsum("tsh,shd->thd", weights_attn, v_all).reshape(q.size(0), -1)

        # O projection via WMMA
        out = out + wmma_gemm_bf16(attn.to(torch.bfloat16), w_o)

        # MLP via WMMA
        normed2 = _rms_norm(out, rms_ffn, eps)
        gate = torch.nn.functional.silu(wmma_gemm_bf16(normed2, w_gate))
        up = wmma_gemm_bf16(normed2, w_up)
        out = out + wmma_gemm_bf16((gate * up).to(torch.bfloat16), w_down)

    kv_cache.pos = start_pos + x.size(0)
    return out, kv_cache
