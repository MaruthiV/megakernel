import argparse
import torch

from megakernel import prefill_qkv, prefill_mlp
from megakernel.launch import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="configs/qwen3_0p6b_bf16.yaml")
    p.add_argument("--tokens", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.model_config)
    hidden = int(cfg["model"]["hidden_size"])

    x = torch.randn(args.tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w_qkv = torch.randn(hidden, 3 * hidden, device="cuda", dtype=torch.bfloat16)
    w_gate = torch.randn(hidden, 4 * hidden, device="cuda", dtype=torch.bfloat16)
    w_up = torch.randn(hidden, 4 * hidden, device="cuda", dtype=torch.bfloat16)
    w_down = torch.randn(4 * hidden, hidden, device="cuda", dtype=torch.bfloat16)

    qkv = prefill_qkv(x, w_qkv)
    mlp = prefill_mlp(x, w_gate, w_up, w_down)

    print({"qkv_shape": list(qkv.shape), "mlp_shape": list(mlp.shape)})


if __name__ == "__main__":
    main()
