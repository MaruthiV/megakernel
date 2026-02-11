from typing import Dict, Any, Tuple

import torch


class UnsupportedQwenConfigError(RuntimeError):
    pass


def _find_layer_prefix(state: Dict[str, torch.Tensor]) -> str:
    candidates = [
        "model.layers.{i}.",
        "model.model.layers.{i}.",
        "transformer.layers.{i}.",
        "model.decoder.layers.{i}.",
    ]
    for cand in candidates:
        key = cand.format(i=0) + "self_attn.q_proj.weight"
        if key in state:
            return cand
    raise KeyError("Could not find a supported layer prefix in state_dict")


def _get(state: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in state:
        raise KeyError(f"Missing key: {key}")
    return state[key]


def _stack_linear_transposed(state: Dict[str, torch.Tensor], key_tmpl: str, num_layers: int) -> torch.Tensor:
    layers = []
    for i in range(num_layers):
        w = _get(state, key_tmpl.format(i=i))
        layers.append(w.t().contiguous())
    return torch.stack(layers, dim=0)


def load_qwen3_weights_hf(
    model_name_or_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """
    Load Qwen3 weights from HuggingFace and map to the kernel format.

    Limitations:
    - Expects num_heads * head_dim == hidden
    - Expects num_key_value_heads == num_heads
    """
    try:
        from transformers import AutoModelForCausalLM
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("transformers is required to load HF weights") from e

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
    cfg = model.config
    state = model.state_dict()

    hidden = int(cfg.hidden_size)
    num_layers = int(cfg.num_hidden_layers)
    num_heads = getattr(cfg, "num_attention_heads", None)
    if num_heads is None:
        num_heads = getattr(cfg, "num_heads", None)
    if num_heads is None:
        raise UnsupportedQwenConfigError("Could not infer num_attention_heads from config")
    num_heads = int(num_heads)
    head_dim = int(getattr(cfg, "head_dim", hidden // num_heads))
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))

    if num_heads * head_dim != hidden:
        raise UnsupportedQwenConfigError(
            f"Unsupported: num_heads * head_dim != hidden ({num_heads}*{head_dim} != {hidden})"
        )
    if num_heads % num_kv_heads != 0:
        raise UnsupportedQwenConfigError(
            "Unsupported: num_heads must be divisible by num_kv_heads"
        )

    layer_prefix = _find_layer_prefix(state)
    def kp(name: str) -> str:
        return layer_prefix + name

    w_q = _stack_linear_transposed(state, kp("self_attn.q_proj.weight"), num_layers)
    w_k = _stack_linear_transposed(state, kp("self_attn.k_proj.weight"), num_layers)
    w_v = _stack_linear_transposed(state, kp("self_attn.v_proj.weight"), num_layers)
    # Q: [hidden, hidden], K/V: [hidden, kv_dim]
    w_qkv = torch.cat([w_q, w_k, w_v], dim=2)

    w_o = _stack_linear_transposed(state, kp("self_attn.o_proj.weight"), num_layers)
    w_gate = _stack_linear_transposed(state, kp("mlp.gate_proj.weight"), num_layers)
    w_up = _stack_linear_transposed(state, kp("mlp.up_proj.weight"), num_layers)
    w_down = _stack_linear_transposed(state, kp("mlp.down_proj.weight"), num_layers)

    rms_attn = torch.stack([
        _get(state, kp("input_layernorm.weight").format(i=i)) for i in range(num_layers)
    ], dim=0)
    rms_ffn = torch.stack([
        _get(state, kp("post_attention_layernorm.weight").format(i=i)) for i in range(num_layers)
    ], dim=0)

    model_cfg = {
        "model": {
            "name": getattr(cfg, "model_type", "qwen3"),
            "hidden_size": hidden,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "vocab_size": int(cfg.vocab_size),
        },
        "precision": "bf16" if dtype == torch.bfloat16 else "fp32",
        "rms_eps": float(getattr(cfg, "rms_norm_eps", 1e-6)),
        "rope_theta": float(getattr(cfg, "rope_theta", 10000.0)),
        "rope_scaling": getattr(cfg, "rope_scaling", None),
        "kv_layout": 0,
    }

    weights = {
        "w_qkv": w_qkv.to(device=device, dtype=dtype),
        "w_o": w_o.to(device=device, dtype=dtype),
        "w_gate": w_gate.to(device=device, dtype=dtype),
        "w_up": w_up.to(device=device, dtype=dtype),
        "w_down": w_down.to(device=device, dtype=dtype),
        "rms_attn": rms_attn.to(device=device, dtype=dtype),
        "rms_ffn": rms_ffn.to(device=device, dtype=dtype),
    }

    return model_cfg, weights
