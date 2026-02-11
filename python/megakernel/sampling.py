from typing import Optional

import torch

from .extension import load_extension


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    ext = load_extension()
    if logits.dtype != torch.float32:
        logits = logits.float()
    return ext.greedy_sample(logits)


def greedy_sample_cpu(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits).item())
