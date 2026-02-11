import argparse
import torch

from megakernel.rope import apply_rope


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--theta", type=float, default=10000.0)
    p.add_argument("--pos", type=int, default=0)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--scale-factor", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x = torch.randn(args.head_dim, device="cuda", dtype=torch.float32)
    rope_scaling = None
    if args.scale_factor != 1.0:
        rope_scaling = {"type": "linear", "factor": args.scale_factor}
    out = apply_rope(x, args.pos, args.theta, rope_scaling)
    print({"mean": float(out.mean().item()), "std": float(out.std().item())})


if __name__ == "__main__":
    main()
