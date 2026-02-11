from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load


_EXT: Optional[object] = None


def load_extension() -> object:
    global _EXT
    if _EXT is not None:
        return _EXT

    root = Path(__file__).resolve().parents[2]
    sources = [
        str(root / "kernels" / "megakernel_ext.cpp"),
        str(root / "kernels" / "megakernel_ext.cu"),
        str(root / "kernels" / "sampling_ext.cpp"),
        str(root / "kernels" / "sampling_ext.cu"),
        str(root / "kernels" / "sampling.cu"),
        str(root / "kernels" / "wmma_gemm_ext.cpp"),
        str(root / "kernels" / "wmma_gemm.cu"),
        str(root / "kernels" / "prefill_ext.cpp"),
        str(root / "kernels" / "prefill_ext.cu"),
    ]

    _EXT = load(
        name="megakernel_ext",
        sources=sources,
        extra_cuda_cflags=["--use_fast_math"],
        verbose=True,
        build_directory=str(root / ".build"),
    )
    return _EXT
