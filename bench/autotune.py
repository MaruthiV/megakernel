import argparse
import json
from pathlib import Path

import torch

from megakernel.autotune import autotune
from megakernel.launch import load_config, run_decode_steps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="configs/qwen3_0p6b_bf16.yaml")
    p.add_argument("--out", default="bench/results/autotune.json")
    p.add_argument("--tokens", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.model_config)
    tune = autotune(cfg.get("kernel", {}))

    hidden = int(cfg["model"]["hidden_size"])
    inputs = torch.randn(args.tokens, hidden, device="cuda", dtype=torch.float32)

    # Just run once to validate configuration (no timing yet).
    _ = run_decode_steps(cfg, None, None, inputs, start_pos=0)
    torch.cuda.synchronize()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tune, f, indent=2)


if __name__ == "__main__":
    main()
