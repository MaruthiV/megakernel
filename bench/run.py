import argparse
import json
import time
from pathlib import Path

import torch

from megakernel.launch import load_config, run_decode_steps, MegakernelNotBuiltError
from megakernel.sampling import greedy_sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", required=True)
    p.add_argument("--precision", default="bf16")
    p.add_argument("--pos", default="1")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--model-config", default="configs/qwen3_0p6b_bf16.yaml")
    p.add_argument("--out", default="bench/results/run.json")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--include-sampling", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.model_config)

    positions = [int(x.strip()) for x in args.pos.split(",") if x.strip()]
    results = {
        "gpu": args.gpu,
        "precision": args.precision,
        "batch": args.batch,
        "positions": positions,
        "tokens": args.tokens,
        "include_sampling": args.include_sampling,
        "status": "ok",
        "tok_per_s": {},
    }

    for pos in positions:
        start = time.time()
        try:
            hidden = int(cfg["model"]["hidden_size"])
            inputs = torch.randn(args.tokens, hidden, device="cuda", dtype=torch.float32)
            outs, _ = run_decode_steps(cfg, None, None, inputs, start_pos=pos)
            if args.include_sampling:
                # Fake logits for sampling cost measurement
                logits = torch.randn(cfg["model"]["vocab_size"], device="cuda", dtype=torch.float32)
                _ = greedy_sample(logits)
            torch.cuda.synchronize()
            end = time.time()
            # placeholder; real throughput comes from actual decode loop
            results["tok_per_s"][str(pos)] = args.tokens / max(1e-6, end - start)
        except MegakernelNotBuiltError as e:
            results["status"] = "not_built"
            results["error"] = str(e)
            results["tok_per_s"][str(pos)] = 0.0
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
