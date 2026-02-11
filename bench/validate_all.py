import argparse
import subprocess
import sys


def run(cmd: list[str]) -> int:
    print({"cmd": " ".join(cmd)})
    return subprocess.call(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-hf", action="store_true")
    p.add_argument("--skip-int8", action="store_true")
    p.add_argument("--skip-persistent", action="store_true")
    p.add_argument("--tokens", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    failures = 0

    # Basic correctness
    failures += run([sys.executable, "bench/verify.py", "--pos", "0"])

    # Persistent correctness
    if not args.skip_persistent:
        failures += run([sys.executable, "bench/persistent_verify_fp32.py", "--tokens", str(args.tokens), "--pos", "0", "--persistent-blocks", "4"])
        failures += run([sys.executable, "bench/persistent_verify_kv_layout1.py", "--tokens", str(args.tokens), "--pos", "0", "--persistent-blocks", "4"])

    # int8 KV correctness (non-persistent + persistent)
    if not args.skip_int8:
        failures += run([sys.executable, "bench/persistent_verify_int8.py", "--tokens", str(args.tokens), "--pos", "0", "--persistent-blocks", "4"])

    # HF checks
    if not args.skip_hf:
        failures += run([sys.executable, "bench/hf_verify.py", "--model", "Qwen/Qwen3-0.6B", "--pos", "0"])
        failures += run([sys.executable, "bench/hf_verify_multi.py", "--model", "Qwen/Qwen3-0.6B", "--pos", "0", "--tokens", str(args.tokens)])
        failures += run([sys.executable, "bench/rope_hf_verify.py", "--model", "Qwen/Qwen3-0.6B", "--pos", "0", "--head-dim", "64"])

    if failures != 0:
        print({"status": "failed", "failures": failures})
        sys.exit(1)

    print({"status": "ok"})


if __name__ == "__main__":
    main()
