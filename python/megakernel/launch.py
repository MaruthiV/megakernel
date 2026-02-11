import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import torch

from .extension import load_extension
from .cache import KvCache, init_kv_cache, ensure_kv_capacity

class MegakernelNotBuiltError(RuntimeError):
    pass


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_decode(
    model_cfg: Dict[str, Any],
    weights: Optional[Any],
    kv_cache: Optional[Any],
    input_token: int,
    position: int,
    sampling_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Single-token decode using a naive CUDA megakernel.
    """
    try:
        ext = load_extension()
    except Exception as e:  # noqa: BLE001
        raise MegakernelNotBuiltError(
            f"CUDA extension not built or failed to load: {e}"
        ) from e

    hidden = int(model_cfg["model"]["hidden_size"])
    num_heads = int(model_cfg["model"]["num_heads"])
    head_dim = int(model_cfg["model"]["head_dim"])
    num_layers = int(model_cfg["model"]["num_layers"])
    num_kv_heads = int(model_cfg["model"].get("num_kv_heads", num_heads))
    eps = float(model_cfg.get("rms_eps", 1e-6))
    rope_theta = float(model_cfg.get("rope_theta", 10000.0))
    kv_layout = int(model_cfg.get("kv_layout", 0))
    device = torch.device("cuda")
    precision = model_cfg.get("precision", "fp32")
    dtype = torch.float32 if precision in ("fp32", "float32") else torch.bfloat16
    kv_quant = model_cfg.get("kv_quant")
    nvtx = bool(model_cfg.get("nvtx", False))
    threads_per_block = int(model_cfg.get("threads_per_block", 256))

    # Placeholder input embedding vector for a single token.
    x = torch.randn(hidden, device=device, dtype=dtype)

    if weights is None:
        # Random weights for scaffolding.
        w_qkv = torch.randn(num_layers, hidden, 3 * hidden, device=device, dtype=dtype)
        w_o = torch.randn(num_layers, hidden, hidden, device=device, dtype=dtype)
        w_gate = torch.randn(num_layers, hidden, 4 * hidden, device=device, dtype=dtype)
        w_up = torch.randn(num_layers, hidden, 4 * hidden, device=device, dtype=dtype)
        w_down = torch.randn(num_layers, 4 * hidden, hidden, device=device, dtype=dtype)
        rms_attn_weight = torch.ones(num_layers, hidden, device=device, dtype=dtype)
        rms_ffn_weight = torch.ones(num_layers, hidden, device=device, dtype=dtype)
    else:
        w_qkv = weights["w_qkv"].to(device=device, dtype=dtype)
        w_o = weights["w_o"].to(device=device, dtype=dtype)
        w_gate = weights["w_gate"].to(device=device, dtype=dtype)
        w_up = weights["w_up"].to(device=device, dtype=dtype)
        w_down = weights["w_down"].to(device=device, dtype=dtype)
        rms_attn_weight = weights["rms_attn"].to(device=device, dtype=dtype)
        rms_ffn_weight = weights["rms_ffn"].to(device=device, dtype=dtype)
        if w_qkv.dim() != 3:
            raise ValueError("weights must be stacked per-layer: w_qkv shape [L, H, 3H]")

    if kv_cache is None:
        max_seq = max(1, position + 1)
        kv_dim = num_kv_heads * head_dim
        if kv_layout == 0:
            kv_k = torch.zeros(max_seq, kv_dim, device=device, dtype=dtype)
            kv_v = torch.zeros(max_seq, kv_dim, device=device, dtype=dtype)
        else:
            kv_k = torch.zeros(num_kv_heads, head_dim, max_seq, device=device, dtype=dtype)
            kv_v = torch.zeros(num_kv_heads, head_dim, max_seq, device=device, dtype=dtype)
    else:
        kv_k = kv_cache["k"].to(device=device, dtype=dtype)
        kv_v = kv_cache["v"].to(device=device, dtype=dtype)

    out = ext.megakernel_forward(
        x,
        rms_attn_weight,
        rms_ffn_weight,
        w_qkv,
        w_o,
        w_gate,
        w_up,
        w_down,
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

    return {"token": int(input_token), "position": int(position), "status": "ok", "out": out}


def run_decode_steps(
    model_cfg: Dict[str, Any],
    weights: Dict[str, Any],
    kv_cache: Optional[Dict[str, Any]],
    inputs: torch.Tensor,
    start_pos: int = 0,
) -> torch.Tensor:
    """
    Multi-step decode over a sequence of input embeddings.
    inputs: [T, hidden]
    Returns: [T, hidden]
    """
    try:
        ext = load_extension()
    except Exception as e:  # noqa: BLE001
        raise MegakernelNotBuiltError(
            f"CUDA extension not built or failed to load: {e}"
        ) from e

    hidden = int(model_cfg["model"]["hidden_size"])
    num_heads = int(model_cfg["model"]["num_heads"])
    head_dim = int(model_cfg["model"]["head_dim"])
    num_layers = int(model_cfg["model"]["num_layers"])
    num_kv_heads = int(model_cfg["model"].get("num_kv_heads", num_heads))
    eps = float(model_cfg.get("rms_eps", 1e-6))
    rope_theta = float(model_cfg.get("rope_theta", 10000.0))
    kv_layout = int(model_cfg.get("kv_layout", 0))
    precision = model_cfg.get("precision", "fp32")
    dtype = torch.float32 if precision in ("fp32", "float32") else torch.bfloat16

    device = inputs.device
    xseq = inputs.to(device=device, dtype=dtype)
    if xseq.dim() == 3:
        # Batch mode: loop over batch for correctness-first behavior.
        outs = []
        caches = []
        for b in range(xseq.size(0)):
            out_b, cache_b = run_decode_steps(model_cfg, weights, kv_cache, xseq[b], start_pos)
            outs.append(out_b)
            caches.append(cache_b)
        return torch.stack(outs, dim=0), caches
    if xseq.dim() != 2 or xseq.size(1) != hidden:
        raise ValueError("inputs must be [T, hidden] or [B, T, hidden]")

    w_qkv = weights["w_qkv"].to(device=device, dtype=dtype)
    w_o = weights["w_o"].to(device=device, dtype=dtype)
    w_gate = weights["w_gate"].to(device=device, dtype=dtype)
    w_up = weights["w_up"].to(device=device, dtype=dtype)
    w_down = weights["w_down"].to(device=device, dtype=dtype)
    rms_attn_weight = weights["rms_attn"].to(device=device, dtype=dtype)
    rms_ffn_weight = weights["rms_ffn"].to(device=device, dtype=dtype)

    if kv_cache is None:
        cache = init_kv_cache(num_kv_heads, head_dim, start_pos + xseq.size(0), kv_layout, device, dtype, kv_quant)
    else:
        cache = KvCache(
            k=kv_cache["k"].to(device=device, dtype=dtype),
            v=kv_cache["v"].to(device=device, dtype=dtype),
            pos=int(kv_cache.get("pos", start_pos)),
            k_scale=kv_cache.get("k_scale"),
            v_scale=kv_cache.get("v_scale"),
        )
    cache = ensure_kv_capacity(cache, start_pos + xseq.size(0), kv_layout)
    kv_k, kv_v = cache.k, cache.v
    if kv_quant == "int8":
        if kv_layout != 0:
            raise NotImplementedError("int8 KV supports kv_layout=0 only")
        if kv_k.dtype != torch.int8:
            kq, kscale = quantize_per_head(kv_k, num_kv_heads, head_dim)
            vq, vscale = quantize_per_head(kv_v, num_kv_heads, head_dim)
            cache.k = kq
            cache.v = vq
            cache.k_scale = kscale
            cache.v_scale = vscale
            kv_k, kv_v = cache.k, cache.v
        if cache.k_scale is None or cache.v_scale is None:
            # initialize scales if missing
            seq = kv_k.size(0)
            cache.k_scale = torch.zeros(seq, num_kv_heads, device=device, dtype=torch.float32)
            cache.v_scale = torch.zeros(seq, num_kv_heads, device=device, dtype=torch.float32)

    if model_cfg.get("persistent", False):
        num_blocks = int(model_cfg.get("persistent_blocks", 4))
        barrier_counter = torch.zeros(1, device=device, dtype=torch.int32)
        barrier_sense = torch.zeros(1, device=device, dtype=torch.int32)
        sync_flags = torch.zeros(3, device=device, dtype=torch.int32)
        q_buf = torch.zeros(xseq.size(0), hidden, device=device, dtype=dtype)
        attn_buf = torch.zeros(xseq.size(0), hidden, device=device, dtype=dtype)
        if kv_quant == "int8":
            if kv_layout != 0:
                raise NotImplementedError("int8 KV persistent supports kv_layout=0 only")
            if nvtx:
                torch.cuda.nvtx.range_push("megakernel_persistent_int8kv")
            out = ext.megakernel_forward_seq_persistent_int8kv(
                xseq,
                rms_attn_weight,
                rms_ffn_weight,
                w_qkv,
                w_o,
                w_gate,
                w_up,
                w_down,
                kv_k,
                kv_v,
                cache.k_scale,
                cache.v_scale,
                q_buf,
                attn_buf,
                barrier_counter,
                barrier_sense,
                sync_flags,
                int(num_layers),
                int(start_pos),
                int(num_heads),
                int(num_kv_heads),
                int(head_dim),
                int(kv_layout),
                float(eps),
                float(rope_theta),
                int(num_blocks),
                int(threads_per_block),
            )
            if nvtx:
                torch.cuda.nvtx.range_pop()
        else:
            if nvtx:
                torch.cuda.nvtx.range_push("megakernel_persistent_fp32")
            out = ext.megakernel_forward_seq_persistent(
                xseq,
                rms_attn_weight,
                rms_ffn_weight,
                w_qkv,
                w_o,
                w_gate,
                w_up,
                w_down,
                kv_k,
                kv_v,
                q_buf,
                attn_buf,
                barrier_counter,
                barrier_sense,
                sync_flags,
                int(num_layers),
                int(start_pos),
                int(num_heads),
                int(num_kv_heads),
                int(head_dim),
                int(kv_layout),
                float(eps),
                float(rope_theta),
                int(num_blocks),
                int(threads_per_block),
            )
            if nvtx:
                torch.cuda.nvtx.range_pop()
        cache.pos = start_pos + xseq.size(0)
        return out, {
            "k": cache.k,
            "v": cache.v,
            "pos": cache.pos,
            "k_q": cache.k_q,
            "v_q": cache.v_q,
            "k_scale": cache.k_scale,
            "v_scale": cache.v_scale,
        }

    if kv_quant == "int8":
        if nvtx:
            torch.cuda.nvtx.range_push("megakernel_seq_int8kv")
        out = ext.megakernel_forward_seq_int8kv(
            xseq,
            rms_attn_weight,
            rms_ffn_weight,
            w_qkv,
            w_o,
            w_gate,
            w_up,
            w_down,
            kv_k,
            kv_v,
            cache.k_scale,
            cache.v_scale,
            int(num_layers),
            int(start_pos),
            int(num_heads),
            int(num_kv_heads),
            int(head_dim),
            int(kv_layout),
            float(eps),
            float(rope_theta),
            int(threads_per_block),
        )
        if nvtx:
            torch.cuda.nvtx.range_pop()
    else:
        if nvtx:
            torch.cuda.nvtx.range_push("megakernel_seq_fp32")
        out = ext.megakernel_forward_seq(
            xseq,
            rms_attn_weight,
            rms_ffn_weight,
            w_qkv,
            w_o,
            w_gate,
            w_up,
            w_down,
            kv_k,
            kv_v,
            int(num_layers),
            int(start_pos),
            int(num_heads),
            int(num_kv_heads),
            int(head_dim),
            int(kv_layout),
            float(eps),
            float(rope_theta),
            int(threads_per_block),
        )
        if nvtx:
            torch.cuda.nvtx.range_pop()
    cache.pos = start_pos + xseq.size(0)
    return out, {
        "k": cache.k,
        "v": cache.v,
        "pos": cache.pos,
        "k_q": cache.k_q,
        "v_q": cache.v_q,
        "k_scale": cache.k_scale,
        "v_scale": cache.v_scale,
    }


def save_result(path: str, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
