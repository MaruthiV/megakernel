import argparse
import torch

from megakernel import load_qwen3_weights_hf
from megakernel.extension import load_extension
from megakernel.reference import compare_with_kernel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--pos", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg, weights = load_qwen3_weights_hf(args.model)
    hidden = int(model_cfg["model"]["hidden_size"])
    num_heads = int(model_cfg["model"]["num_heads"])
    num_kv_heads = int(model_cfg["model"].get("num_kv_heads", num_heads))
    head_dim = int(model_cfg["model"]["head_dim"])
    kv_layout = int(model_cfg.get("kv_layout", 0))

    device = torch.device("cuda")
    x = torch.randn(hidden, device=device, dtype=torch.float32)

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
    stats = compare_with_kernel(ext, model_cfg, weights, kv, x, args.pos)
    print(stats)


if __name__ == "__main__":
    main()
