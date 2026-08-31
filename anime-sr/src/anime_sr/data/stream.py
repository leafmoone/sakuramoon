"""Deterministic SR training stream: the slot formula + window oracle.

The trainer's §11.5 stream is a pure function of (step, rank, bs, world,
n): slot = :func:`latent_sample_index`, sample = ``slot_map[slot]``.

The streaming (tar-direct) data plane pins/leases shards AHEAD of the
trainer through a window driver: it needs the SET OF SHARDS a step
interval touches across all DDP ranks. This module is the single source
of truth for the formula on BOTH sides (trainer and window driver), so
the demand oracle and the consumer can never drift apart.
"""

from __future__ import annotations

from collections.abc import Callable


def latent_sample_index(step: int, rank: int, i: int, bs: int, world: int, n: int) -> int:
    """M2-style deterministic stream slot (plan §11.5): the global slot
    ``step * bs * world + rank * bs + i`` wrapped by the set size. A resume
    at step ``s0`` reproduces the same stream position at step ``s``."""
    return (step * (bs * world) + rank * bs + i) % n


def window_shards(
    start_step: int,
    end_step: int,
    *,
    bs: int,
    world: int,
    n: int,
    slot_map: Callable[[int], int],
    shards: list[str],
) -> list[str]:
    """Shard names touched by steps ``[start_step, end_step)`` across ALL
    ranks, in first-appearance order (deduplicated).

    ``slot_map`` is the trainer's SlotMap (slot -> dataset index);
    ``shards`` maps dataset index -> pin-dir shard name (same order as
    ``ds.samples``). Pure function — the window driver and any verifier
    compute identical windows."""
    seen: dict[str, None] = {}
    for s in range(start_step, end_step):
        for r in range(world):
            for i in range(bs):
                j = slot_map[latent_sample_index(s, r, i, bs, world, n)]
                shard = shards[j]
                if shard not in seen:
                    seen[shard] = None
    return list(seen)


__all__ = ["latent_sample_index", "window_shards"]
