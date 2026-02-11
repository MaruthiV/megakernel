from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class KvCache:
    k: torch.Tensor
    v: torch.Tensor
    pos: int
    k_q: Optional[torch.Tensor] = None
    v_q: Optional[torch.Tensor] = None
    k_scale: Optional[torch.Tensor] = None
    v_scale: Optional[torch.Tensor] = None


def init_kv_cache(
    num_kv_heads: int,
    head_dim: int,
    max_seq: int,
    kv_layout: int,
    device: torch.device,
    dtype: torch.dtype,
    kv_quant: str | None = None,
) -> KvCache:
    if kv_layout == 0:
        if kv_quant == "int8":
            k = torch.zeros(max_seq, num_kv_heads * head_dim, device=device, dtype=torch.int8)
            v = torch.zeros(max_seq, num_kv_heads * head_dim, device=device, dtype=torch.int8)
            k_scale = torch.zeros(max_seq, num_kv_heads, device=device, dtype=torch.float32)
            v_scale = torch.zeros(max_seq, num_kv_heads, device=device, dtype=torch.float32)
            return KvCache(k=k, v=v, pos=0, k_scale=k_scale, v_scale=v_scale)
        k = torch.zeros(max_seq, num_kv_heads * head_dim, device=device, dtype=dtype)
        v = torch.zeros(max_seq, num_kv_heads * head_dim, device=device, dtype=dtype)
    else:
        if kv_quant == "int8":
            k = torch.zeros(num_kv_heads, head_dim, max_seq, device=device, dtype=torch.int8)
            v = torch.zeros(num_kv_heads, head_dim, max_seq, device=device, dtype=torch.int8)
            k_scale = torch.zeros(max_seq, num_kv_heads, device=device, dtype=torch.float32)
            v_scale = torch.zeros(max_seq, num_kv_heads, device=device, dtype=torch.float32)
            return KvCache(k=k, v=v, pos=0, k_scale=k_scale, v_scale=v_scale)
        k = torch.zeros(num_kv_heads, head_dim, max_seq, device=device, dtype=dtype)
        v = torch.zeros(num_kv_heads, head_dim, max_seq, device=device, dtype=dtype)
    return KvCache(k=k, v=v, pos=0)


def ensure_kv_capacity(cache: KvCache, needed_seq: int, kv_layout: int) -> KvCache:
    if kv_layout == 0:
        if cache.k.size(0) >= needed_seq:
            return cache
        new_k = torch.zeros(needed_seq, cache.k.size(1), device=cache.k.device, dtype=cache.k.dtype)
        new_v = torch.zeros(needed_seq, cache.v.size(1), device=cache.v.device, dtype=cache.v.dtype)
        new_k[: cache.k.size(0)].copy_(cache.k)
        new_v[: cache.v.size(0)].copy_(cache.v)
        k_scale = cache.k_scale
        v_scale = cache.v_scale
        if k_scale is not None:
            new_ks = torch.zeros(needed_seq, k_scale.size(1), device=k_scale.device, dtype=k_scale.dtype)
            new_vs = torch.zeros(needed_seq, v_scale.size(1), device=v_scale.device, dtype=v_scale.dtype)
            new_ks[: k_scale.size(0)].copy_(k_scale)
            new_vs[: v_scale.size(0)].copy_(v_scale)
            k_scale = new_ks
            v_scale = new_vs
    else:
        if cache.k.size(2) >= needed_seq:
            return cache
        new_k = torch.zeros(cache.k.size(0), cache.k.size(1), needed_seq, device=cache.k.device, dtype=cache.k.dtype)
        new_v = torch.zeros(cache.v.size(0), cache.v.size(1), needed_seq, device=cache.v.device, dtype=cache.v.dtype)
        new_k[:, :, : cache.k.size(2)].copy_(cache.k)
        new_v[:, :, : cache.v.size(2)].copy_(cache.v)
        k_scale = cache.k_scale
        v_scale = cache.v_scale
        if k_scale is not None:
            new_ks = torch.zeros(needed_seq, k_scale.size(1), device=k_scale.device, dtype=k_scale.dtype)
            new_vs = torch.zeros(needed_seq, v_scale.size(1), device=v_scale.device, dtype=v_scale.dtype)
            new_ks[: k_scale.size(0)].copy_(k_scale)
            new_vs[: v_scale.size(0)].copy_(v_scale)
            k_scale = new_ks
            v_scale = new_vs
    return KvCache(k=new_k, v=new_v, pos=cache.pos, k_scale=k_scale, v_scale=v_scale)
