import argparse
import torch

from megakernel import prefill_wmma
from megakernel.launch import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="configs/qwen3_0p6b_bf16.yaml")
    p.add_argument("--tokens", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.model_config)
    hidden = int(cfg["model"]["hidden_size"])
    num_layers = int(cfg["model"]["num_layers"])

    x = torch.randn(args.tokens, hidden, device="cuda", dtype=torch.bfloat16)
    weights = {
        "w_qkv": torch.randn(num_layers, hidden, 3 * hidden, device="cuda", dtype=torch.bfloat16),
        "w_o": torch.randn(num_layers, hidden, hidden, device="cuda", dtype=torch.bfloat16),
        "w_gate": torch.randn(num_layers, hidden, 4 * hidden, device="cuda", dtype=torch.bfloat16),
        "w_up": torch.randn(num_layers, hidden, 4 * hidden, device="cuda", dtype=torch.bfloat16),
        "w_down": torch.randn(num_layers, 4 * hidden, hidden, device="cuda", dtype=torch.bfloat16),
        "rms_attn": torch.ones(num_layers, hidden, device="cuda", dtype=torch.bfloat16),
        "rms_ffn": torch.ones(num_layers, hidden, device="cuda", dtype=torch.bfloat16),
    }

    out, cache = prefill_wmma(x, weights, cfg, None, 0)
    print({"out_shape": list(out.shape), "cache_pos": cache.pos})


if __name__ == "__main__":
    main()
