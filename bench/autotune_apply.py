import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/qwen3_0p6b_bf16.yaml")
    p.add_argument("--autotune-json", default="bench/results/autotune.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = json.loads(Path(args.autotune_json).read_text(encoding="utf-8"))
    best = data.get("best")
    if not best:
        raise RuntimeError("autotune json missing best")

    cfg["persistent"] = True
    cfg["persistent_blocks"] = int(best["blocks"])
    cfg["threads_per_block"] = int(best["threads"])

    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print({"updated": str(cfg_path), "best": best})


if __name__ == "__main__":
    main()
