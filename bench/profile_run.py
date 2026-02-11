import argparse
import time

import torch

from megakernel.launch import load_config, run_decode_steps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="configs/qwen3_0p6b_bf16.yaml")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--pos", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.model_config)
    hidden = int(cfg["model"]["hidden_size"])

    x = torch.randn(args.tokens, hidden, device="cuda", dtype=torch.float32)
    # Warmup
    for _ in range(3):
        run_decode_steps(cfg, None, None, x, args.pos)
    torch.cuda.synchronize()

    start = time.time()
    run_decode_steps(cfg, None, None, x, args.pos)
    torch.cuda.synchronize()
    end = time.time()
    print({"ms": 1000.0 * (end - start)})


if __name__ == "__main__":
    main()
