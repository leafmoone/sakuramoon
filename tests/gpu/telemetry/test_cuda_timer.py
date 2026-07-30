from __future__ import annotations

import torch

from sakuramoon.telemetry.timers import PhaseTimer


def test_cuda_event_timer_collects_without_phase_boundary_sync() -> None:
    device = torch.device("cuda", 0)
    timer = PhaseTimer(device=device)
    left = torch.randn((256, 256), device=device)
    right = torch.randn((256, 256), device=device)

    with timer.record("dit_forward"):
        output = left @ right

    assert timer.pending_cuda_pairs == 1
    torch.cuda.synchronize(device)
    durations = timer.collect_ready()

    assert bool(torch.isfinite(output).all().item())
    assert durations["dit_forward"] > 0.0
    assert timer.pending_cuda_pairs == 0
