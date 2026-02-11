import argparse
import torch

from megakernel.rope import apply_rope
from megakernel.weights import load_qwen3_weights_hf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--pos", type=int, default=0)
    p.add_argument("--head-dim", type=int, default=64)
    return p.parse_args()


def _find_rotary_emb(model):
    # Try common paths
    paths = [
        ("model", "layers", 0, "self_attn", "rotary_emb"),
        ("model", "model", "layers", 0, "self_attn", "rotary_emb"),
        ("transformer", "layers", 0, "self_attn", "rotary_emb"),
    ]
    for path in paths:
        obj = model
        ok = True
        for p in path:
            if isinstance(p, int):
                if hasattr(obj, "__getitem__"):
                    obj = obj[p]
                else:
                    ok = False
                    break
            else:
                if not hasattr(obj, p):
                    ok = False
                    break
                obj = getattr(obj, p)
        if ok:
            return obj
    return None


def _apply_rotary_hf(rotary, q, position):
    # Try to get cos/sin from the HF rotary module
    if hasattr(rotary, "forward"):
        try:
            cos, sin = rotary(q, position)
            return q * cos + rotate_half(q) * sin
        except Exception:
            pass
    if hasattr(rotary, "get_cos_sin"):
        try:
            cos, sin = rotary.get_cos_sin(position, q.device, q.dtype)
            return q * cos + rotate_half(q) * sin
        except Exception:
            pass
    raise RuntimeError("Could not apply HF rotary embedding")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def main() -> None:
    args = parse_args()
    model_cfg, _ = load_qwen3_weights_hf(args.model)

    try:
        from transformers import AutoModelForCausalLM
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("transformers is required") from e

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    rotary = _find_rotary_emb(model)
    if rotary is None:
        raise RuntimeError("Could not locate rotary embedding module in model")

    device = torch.device("cuda")
    q = torch.randn(1, 1, args.head_dim, device=device, dtype=torch.float32)
    rope_theta = float(model_cfg.get("rope_theta", 10000.0))
    rope_scaling = model_cfg.get("rope_scaling")

    ours = apply_rope(q[0, 0], args.pos, rope_theta, rope_scaling)
    ours = ours.view(1, 1, -1)

    try:
        theirs = _apply_rotary_hf(rotary, q, args.pos)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to compute HF rotary: {e}") from e

    diff = (ours - theirs).abs()
    print({"max_abs": float(diff.max().item()), "mean_abs": float(diff.mean().item())})


if __name__ == "__main__":
    main()
