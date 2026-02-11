import argparse
import time

import torch

from megakernel.autotune import autotune
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
    tune = autotune(cfg.get("kernel", {}))

    hidden = int(cfg["model"]["hidden_size"])
    x = torch.randn(args.tokens, hidden, device="cuda", dtype=torch.float32)

    best = None
    results = []

    for blocks in tune["blocks_candidates"]:
        for threads in tune["threads_candidates"]:
            cfg["persistent"] = True
            cfg["persistent_blocks"] = blocks
            cfg["threads_per_block"] = threads
            torch.cuda.synchronize()
            start = time.time()
            _ = run_decode_steps(cfg, None, None, x, args.pos)
            torch.cuda.synchronize()
            end = time.time()
            ms = 1000.0 * (end - start)
            results.append({"blocks": blocks, "threads": threads, "ms": ms})
            if best is None or ms < best["ms"]:
                best = {"blocks": blocks, "threads": threads, "ms": ms}

    out = {
        "best": best,
        "results": results,
    }
    Path("bench/results").mkdir(parents=True, exist_ok=True)
    with open("bench/results/autotune.json", "w", encoding="utf-8") as f:
        import json
        json.dump(out, f, indent=2)
    print(out)


if __name__ == "__main__":
    main()
