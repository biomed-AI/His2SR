from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

import torch


def ensure_directory(path: Path) -> None:
    """Create ``path`` (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def normalize_device(device: Union[int, str, torch.device]) -> torch.device:
    """Convert heterogeneous device specifications into a ``torch.device``."""
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")

    device_str = str(device).strip()
    if device_str.isdigit():
        return torch.device(f"cuda:{device_str}")
    if device_str.lower() in {"cpu", "cuda", "mps"}:
        return torch.device(device_str.lower())
    if device_str.startswith(("cuda:", "cpu:", "mps:")):
        return torch.device(device_str)
    raise ValueError(f"Unrecognised device specification: {device}")


def select_embedding_slice(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Select embedding channels according to ``mode``."""
    mode = mode.lower()
    if mode == "all":
        return x
    if mode == "sub":
        return x[..., 768:2304]
    raise ValueError(f"Unsupported embedding_part: {mode}")


def should_use_amp(requested: bool, device: torch.device) -> bool:
    """AMP can only run on CUDA devices."""
    return requested and device.type == "cuda"


@contextmanager
def autocast_if_available(enabled: bool) -> Iterator[None]:
    """Safely enter autocast when CUDA AMP is available."""
    if not enabled:
        yield
        return
    with torch.cuda.amp.autocast():
        yield
