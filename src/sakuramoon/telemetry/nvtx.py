"""Optional NVTX ranges with balanced push/pop semantics."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import torch

from sakuramoon.telemetry.metrics import TIMING_PHASES


@contextmanager
def nvtx_range(phase: str, *, enabled: bool) -> Generator[None]:
    if phase not in TIMING_PHASES:
        raise ValueError(f"unknown NVTX phase: {phase}")
    pushed = enabled and torch.cuda.is_available()
    if pushed:
        torch.cuda.nvtx.range_push(  # pyright: ignore[reportUnknownMemberType]
            f"sakuramoon:{phase}"
        )
    try:
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()


__all__ = ["nvtx_range"]
