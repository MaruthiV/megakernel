import argparse
import torch

from megakernel.extension import load_extension
from megakernel.reference import forward_single_token
from megakernel.cache import init_kv_cache
from megakernel.launch import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="configs/qwen3_0p6b_bf16.yaml")
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--pos", type=int, default=0)
    p.add_argument("--persistent-blocks", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.model_config)
    hidden = int(cfg["model"]["hidden_size"])
    num_heads = int(cfg["model"]["num_heads"])
    num_kv_heads = int(cfg["model"].get("num_kv_heads", num_heads))
    head_dim = int(cfg["model"]["head_dim"])
    num_layers = int(cfg["model"]["num_layers"])
    kv_layout = int(cfg.get("kv_layout", 0))
    eps = float(cfg.get("rms_eps", 1e-6))
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    rope_scaling = cfg.get("rope_scaling")

    device = torch.device("cuda")
    xseq = torch.randn(args.tokens, hidden, device=device, dtype=torch.float32)

    weights = {
        "w_qkv": torch.randn(num_layers, hidden, 3 * hidden, device=device),
        "w_o": torch.randn(num_layers, hidden, hidden, device=device),
        "w_gate": torch.randn(num_layers, hidden, 4 * hidden, device=device),
        "w_up": torch.randn(num_layers, hidden, 4 * hidden, device=device),
        "w_down": torch.randn(num_layers, 4 * hidden, hidden, device=device),
        "rms_attn": torch.ones(num_layers, hidden, device=device),
        "rms_ffn": torch.ones(num_layers, hidden, device=device),
    }

    cache = init_kv_cache(num_kv_heads, head_dim, args.pos + args.tokens, kv_layout, device, torch.float32)
    ext = load_extension()

    q_buf = torch.zeros(args.tokens, hidden, device=device, dtype=torch.float32)
    attn_buf = torch.zeros(args.tokens, hidden, device=device, dtype=torch.float32)
    barrier_counter = torch.zeros(1, device=device, dtype=torch.int32)
    barrier_sense = torch.zeros(1, device=device, dtype=torch.int32)
    sync_flags = torch.zeros(3, device=device, dtype=torch.int32)

    out_kernel = ext.megakernel_forward_seq_persistent(
        xseq,
        weights["rms_attn"],
        weights["rms_ffn"],
        weights["w_qkv"],
        weights["w_o"],
        weights["w_gate"],
        weights["w_up"],
        weights["w_down"],
        cache.k,
        cache.v,
        q_buf,
        attn_buf,
        barrier_counter,
        barrier_sense,
        sync_flags,
        int(num_layers),
        int(args.pos),
        int(num_heads),
        int(num_kv_heads),
        int(head_dim),
        int(kv_layout),
        float(eps),
        float(rope_theta),
        int(args.persistent_blocks),
    )

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
