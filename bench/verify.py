import argparse
import torch

from megakernel.extension import load_extension
from megakernel.launch import load_config
from megakernel.reference import compare_with_kernel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="configs/qwen3_0p6b_bf16.yaml")
    p.add_argument("--pos", type=int, default=0)
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

    device = torch.device("cuda")
    x = torch.randn(hidden, device=device, dtype=torch.float32)

    weights = {
        "w_qkv": torch.randn(num_layers, hidden, 3 * hidden, device=device),
        "w_o": torch.randn(num_layers, hidden, hidden, device=device),
        "w_gate": torch.randn(num_layers, hidden, 4 * hidden, device=device),
        "w_up": torch.randn(num_layers, hidden, 4 * hidden, device=device),
        "w_down": torch.randn(num_layers, 4 * hidden, hidden, device=device),
        "rms_attn": torch.ones(num_layers, hidden, device=device),
        "rms_ffn": torch.ones(num_layers, hidden, device=device),
    }

    max_seq = args.pos + 1
    kv_dim = num_kv_heads * head_dim
    if kv_layout == 0:
        kv = {
            "k": torch.zeros(max_seq, kv_dim, device=device),
            "v": torch.zeros(max_seq, kv_dim, device=device),
        }
    else:
        kv = {
            "k": torch.zeros(num_kv_heads, head_dim, max_seq, device=device),
            "v": torch.zeros(num_kv_heads, head_dim, max_seq, device=device),
        }

    ext = load_extension()
    stats = compare_with_kernel(ext, cfg, weights, kv, x, args.pos)
    print(stats)


if __name__ == "__main__":
    main()
