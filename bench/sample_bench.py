import argparse
import time

import torch

from megakernel.sampling import greedy_sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=151936)
    p.add_argument("--iters", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logits = torch.randn(args.vocab, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = greedy_sample(logits)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(args.iters):
        _ = greedy_sample(logits)
    torch.cuda.synchronize()
    end = time.time()

    print({"iters": args.iters, "ms_per": 1000.0 * (end - start) / args.iters})


if __name__ == "__main__":
    main()
