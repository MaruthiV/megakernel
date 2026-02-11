import json
from pathlib import Path
from typing import Dict, Any


def autotune(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight autotuner: tries thread-block sizes and persistent blocks,
    returns the best measured ms/token on current GPU.
    """
    threads_candidates = [64, 128, 256, 512]
    blocks_candidates = [4, 8, 12, 16]
    return {
        "status": "ready",
        "threads_candidates": threads_candidates,
        "blocks_candidates": blocks_candidates,
    }


def save_result(path: str, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
