import argparse
import time

import torch

from megakernel.extension import load_extension


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--iters", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ext = load_extension()
    A = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(args.k, args.n, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(5):
        _ = ext.wmma_gemm_bf16(A, B)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(args.iters):
        _ = ext.wmma_gemm_bf16(A, B)
    torch.cuda.synchronize()
    end = time.time()

    print({"iters": args.iters, "ms_per": 1000.0 * (end - start) / args.iters})


if __name__ == "__main__":
    main()
