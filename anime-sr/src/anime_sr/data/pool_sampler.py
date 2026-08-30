"""P1 pool sampler (M4-prep work order, 2026-08-29; M4-1024 semantics
frozen 2026-08-31): a DIVERSITY-FIRST FULL-SET DETERMINISTIC PERMUTATION of
the eligible samples — instead of the legacy straight read of the
index/store order.

Formal contract (all pure functions of the inputs — resume-safe, DDP-safe):

* per cycle of ``n`` slots, every eligible sample appears EXACTLY once
  (the cycle order is a permutation of the dataset indices; no
  oversampling, no downsampling, no repetition within a cycle);
* the LONG-TERM pool composition therefore equals the data's NATURAL
  composition (priority / regular / aux as labeled in the index).  For the
  M4-1024 production set that is ~19 / 60 / 21 and is ACCEPTED as a data
  statistic — M4-1024 does NOT force an 80/10/10 quota and does NOT
  re-label the index to chase one;
* blocks are permuted within themselves (seeded by pool + cycle), then the
  concatenated blocks get one final global permutation (seeded by cycle),
  so consecutive slots never straight-read one pool or the index order;
* ``SlotMap[slot]`` = ``perm(cycle)[pos]`` with ``cycle = slot // n`` and
  ``pos = slot % n`` — a pure function of the slot. DDP ranks never
  collide because ``latent_sample_index`` assigns each rank a disjoint set
  of global slots; a resume to the same slot reproduces the same sample
  from a FRESH map (no hidden state beyond the cycle cache, which is a
  pure function of the cycle).

``[sampling] core/regular/aux_fraction`` and ``[filter]
aux_max_fraction`` are LEGACY QUOTA KNOBS and are INACTIVE for M4-1024:
``pool_counts`` always resolves to the natural pool membership (each pool's
base allocation is ``min(members, floor(n*target))`` and the deficit
redistribution exactly exhausts the spare members, since the pools
partition the dataset) — so the emitted stream is bit-identical with any
quota values.  They are kept for schema/back-compat only; do not read them
as achieved shares.

``enabled = false`` (or no pool membership) degenerates to the legacy
``order[slot % n]`` read — bit-for-bit the old stream.
"""

from __future__ import annotations

import math
import random

from anime_sr.config.schema import Config

# fixed pool order: blocks are concatenated in this order; a core shortfall
# redistributes to regular then aux (the minor pools first).
POOLS = ("priority", "regular", "aux")
_POOL_KEYS = ("priority", "regular", "aux")


def _seeded_permutation(items: list[int], seed_str: str) -> list[int]:
    """Deterministic Fisher-Yates of ``items``; seed is a stable string
    (blake2b-derived int, not the platform-``hash()`` builtin)."""
    import hashlib

    seed = int.from_bytes(
        hashlib.blake2b(seed_str.encode("utf-8"), digest_size=8).digest(), "little"
    )
    out = list(items)
    random.Random(seed).shuffle(out)
    return out


def pool_counts(
    n: int,
    sizes: dict[str, int],
    cfg: Config,
) -> dict[str, int]:
    """Exact per-pool slot counts for one cycle of ``n`` slots.

    The counts sum to exactly ``n`` and never exceed a pool's member count;
    each pool's base allocation is ``min(members, floor(n * target))``
    (targets normalized over their sum, aux additionally capped by
    ``cfg.filter.aux_max_fraction``), and the remaining slots flow to pools
    with spare members in the fixed order priority -> regular -> aux.
    Because the pools partition the dataset, the spare capacity exactly
    equals the deficit: the result ALWAYS equals the natural pool
    membership (the legacy target fractions are an inactive no-op — see the
    module docstring). Deterministic and total-preserving: every cycle is a
    permutation of all n samples (no repetition within a cycle)."""
    s = cfg.sampling
    total = s.core_fraction + s.regular_fraction + s.aux_fraction
    targets = {
        "priority": s.core_fraction / total,
        "regular": s.regular_fraction / total,
        "aux": min(s.aux_fraction / total, cfg.filter.aux_max_fraction),
    }
    counts = {
        p: min(sizes[p], math.floor(n * targets[p])) for p in _POOL_KEYS
    }
    deficit = n - sum(counts.values())
    for pool in ("priority", "regular", "aux"):
        if deficit <= 0:
            break
        room = sizes[pool] - counts[pool]
        take = min(deficit, room)
        counts[pool] += take
        deficit -= take
    if deficit > 0:
        raise RuntimeError(
            f"pool sampler: pools cover {sum(sizes.values())} samples but "
            f"the dataset has {n}; pool membership is inconsistent"
        )
    if any(v < 0 for v in counts.values()):
        raise RuntimeError(f"pool sampler: negative allocation {counts}")
    return counts


def build_cycle_order(
    members: dict[str, list[int]],
    n: int,
    cfg: Config,
    cycle: int,
    salt: str = "",
) -> list[int]:
    """The dataset-index order for one full cycle (a permutation of all
    ``n`` indices): per-pool seeded permutations of the target counts,
    then one global seeded permutation of the concatenated blocks."""
    sizes = {p: len(members.get(p, ())) for p in _POOL_KEYS}
    counts = pool_counts(n, sizes, cfg)
    blocks: list[int] = []
    for pool in POOLS:  # fixed order: priority, regular, aux
        ms = sorted(members.get(pool, ()))
        take = counts[pool]
        if take:
            blocks.extend(_seeded_permutation(ms, f"pool|{pool}|{cycle}|{n}|{salt}")[:take])
    assert len(blocks) == n, f"cycle order covers {len(blocks)}/{n} samples"
    return _seeded_permutation(blocks, f"mix|{cycle}|{n}|{salt}")


class SlotMap:
    """slot -> dataset index.

    Enabled: the P1 pool stream (per-cycle permutation, pool-mixed).
    Disabled / no membership: the legacy ``legacy_order[slot % n]`` read."""

    def __init__(
        self,
        n: int,
        members: dict[str, list[int]] | None,
        cfg: Config,
        legacy_order: list[int],
        salt: str = "",
    ) -> None:
        if n <= 0:
            raise ValueError("SlotMap: n must be > 0")
        self.n = n
        self.cfg = cfg
        self.salt = salt
        self.legacy_order = list(legacy_order)
        if members is None:
            self.enabled = False
            self.members: dict[str, list[int]] = {}
        else:
            self.enabled = bool(cfg.sampling.enabled)
            self.members = {p: sorted(members.get(p, ())) for p in _POOL_KEYS}
            if sum(len(v) for v in self.members.values()) != n:
                raise RuntimeError(
                    f"SlotMap: pools cover {sum(len(v) for v in self.members.values())} "
                    f"of {n} samples — every eligible sample must be in exactly one pool"
                )
        self._cache: dict[int, list[int]] = {}

    def __getitem__(self, slot: int) -> int:
        if not self.enabled:
            return self.legacy_order[slot % self.n]
        cycle = slot // self.n
        order = self._cycle_order(cycle)
        return order[slot % self.n]

    def _cycle_order(self, cycle: int) -> list[int]:
        order = self._cache.get(cycle)
        if order is None:
            order = build_cycle_order(self.members, self.n, self.cfg, cycle, self.salt)
            self._cache[cycle] = order
            # cycles advance monotonically in training; keep the cache tiny
            while len(self._cache) > 4:
                self._cache.pop(min(self._cache))
        return order

    def pool_report(self) -> dict[str, int]:
        """Effective per-cycle pool sizes (for startup telemetry + ckpt
        provenance)."""
        if not self.enabled:
            return {"legacy": self.n}
        return pool_counts(self.n, {p: len(v) for p, v in self.members.items()}, self.cfg)
