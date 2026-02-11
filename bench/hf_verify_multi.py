import argparse
import torch

from megakernel import load_qwen3_weights_hf
from megakernel.extension import load_extension
from megakernel.reference import forward_single_token
from megakernel.cache import init_kv_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--pos", type=int, default=0)
    p.add_argument("--tokens", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg, weights = load_qwen3_weights_hf(args.model)
    hidden = int(model_cfg["model"]["hidden_size"])
    num_heads = int(model_cfg["model"]["num_heads"])
    num_kv_heads = int(model_cfg["model"].get("num_kv_heads", num_heads))
    head_dim = int(model_cfg["model"]["head_dim"])
    kv_layout = int(model_cfg.get("kv_layout", 0))
    eps = float(model_cfg.get("rms_eps", 1e-6))
    rope_theta = float(model_cfg.get("rope_theta", 10000.0))
    rope_scaling = model_cfg.get("rope_scaling")

    device = torch.device("cuda")
    xseq = torch.randn(args.tokens, hidden, device=device, dtype=torch.float32)

    cache = init_kv_cache(num_kv_heads, head_dim, args.pos + args.tokens, kv_layout, device, torch.float32)
    ext = load_extension()

    # Kernel path (persistent off)
    kv_k = cache.k
    kv_v = cache.v
    outs = []
    for i in range(args.tokens):
        out = ext.megakernel_forward(
            xseq[i],
            weights["rms_attn"],
            weights["rms_ffn"],
            weights["w_qkv"],
            weights["w_o"],
            weights["w_gate"],
            weights["w_up"],
            weights["w_down"],
            kv_k,
            kv_v,
            int(weights["w_qkv"].shape[0]),
            int(args.pos + i),
            int(num_heads),
            int(num_kv_heads),
            int(head_dim),
            int(kv_layout),
            float(eps),
            float(rope_theta),
        )
        outs.append(out)
    out_kernel = torch.stack(outs, dim=0)

    # Reference path
    kv_ref = {"k": cache.k.clone(), "v": cache.v.clone()}
    out_ref = []
    for i in range(args.tokens):
        out = forward_single_token(
            xseq[i],
            weights,
            kv_ref,
            args.pos + i,
            num_heads,
            head_dim,
            eps,
            rope_theta,
            kv_layout,
            num_kv_heads,
            rope_scaling,
        )
        out_ref.append(out)
    out_ref = torch.stack(out_ref, dim=0)

    diff = (out_kernel - out_ref).abs()
    print({"max_abs": float(diff.max().item()), "mean_abs": float(diff.mean().item())})


if __name__ == "__main__":
    main()
