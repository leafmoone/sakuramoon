"""P1 pool sampler (M4-prep 2026-08-29): config-driven, deterministic,
resume-safe pool-mixed train stream.

Contract under test:
  * each cycle order is a permutation of ALL eligible samples (every
    sample exactly once per cycle);
  * pool allocation arithmetic: base = min(members, floor(n * target))
    (aux additionally capped by [filter] aux_max_fraction), the deficit
    flows priority -> regular -> aux; when the pools PARTITION the
    dataset (the M4-1024 production case) this always resolves to the
    NATURAL pool composition, so the legacy target fractions are an
    inactive no-op there (08-31 M4 resolution) — the config still drives
    the allocation in the non-partition synthetic cases below;
  * deterministic: same (pools, cycle, cfg) -> same order; different cycle
    -> different order; a fresh SlotMap after a "resume" reproduces the
    exact sample stream;
  * disabled -> the legacy straight-read order[slot % n], bit-for-bit;
  * DDP: global slots from the latent_sample_index arithmetic are disjoint
    across ranks, so no rank double-serves a global slot;
  * the cycle order is NOT the straight index read (the mix is applied).
"""

from __future__ import annotations

import pytest
from anime_sr.config.schema import Config, SamplingSpec
from anime_sr.data.pool_sampler import SlotMap, build_cycle_order, pool_counts
from anime_sr.train.latent_flow import latent_sample_index

N = 1000


def _members(prio: int, reg: int, aux: int) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    i = 0
    for pool, k in (("priority", prio), ("regular", reg), ("aux", aux)):
        if k:
            out[pool] = list(range(i, i + k))
            i += k
    assert i == prio + reg + aux
    return out


def _cfg(**sampling) -> Config:
    base = Config()
    spec = SamplingSpec(**sampling)
    return base.model_copy(update={"sampling": spec})


def test_cycle_order_is_permutation() -> None:
    members = _members(800, 100, 100)
    for cycle in (0, 1, 7):
        order = build_cycle_order(members, N, _cfg(), cycle)
        assert sorted(order) == list(range(N)), f"cycle {cycle} not a permutation"


def test_deterministic_and_cycle_sensitive() -> None:
    members = _members(800, 100, 100)
    cfg = _cfg()
    a0 = build_cycle_order(members, N, cfg, 0)
    a1 = build_cycle_order(members, N, cfg, 1)
    b0 = build_cycle_order(members, N, cfg, 0)
    assert a0 == b0  # same cycle, fresh call -> identical
    assert a0 != a1  # different cycle -> different order


def test_not_straight_read() -> None:
    members = _members(800, 100, 100)
    for cycle in (0, 1):
        order = build_cycle_order(members, N, _cfg(), cycle)
        assert order != list(range(N)), "cycle order must not be the index straight read"


def test_pool_fractions_follow_config() -> None:
    members = _members(800, 100, 100)
    cfg = _cfg(core_fraction=0.8, regular_fraction=0.1, aux_fraction=0.1)
    counts = pool_counts(N, {p: len(v) for p, v in members.items()}, cfg)
    assert counts == {"priority": 800, "regular": 100, "aux": 100}
    # and the ACTUAL composition of the cycle order matches the counts
    order = build_cycle_order(members, N, cfg, 3)
    seen: dict[str, int] = {}
    pool_of = {
        **{i: "priority" for i in members["priority"]},
        **{i: "regular" for i in members["regular"]},
        **{i: "aux" for i in members["aux"]},
    }
    for i in order:
        seen[pool_of[i]] = seen.get(pool_of[i], 0) + 1
    assert seen == counts


def test_aux_hard_capped() -> None:
    """A 50% aux target is clamped to [filter] aux_max_fraction (0.20)."""
    members = _members(500, 300, 200)
    cfg = _cfg(core_fraction=0.3, regular_fraction=0.2, aux_fraction=0.5)
    counts = pool_counts(N, {p: len(v) for p, v in members.items()}, cfg)
    assert counts["aux"] == 200  # min(target 500, members 200, cap 200)
    assert counts["priority"] + counts["regular"] + counts["aux"] == N


def test_small_pool_redistribution() -> None:
    """A priority pool smaller than its target: the excess flows to
    regular then aux — total stays n, still a permutation."""
    members = _members(100, 450, 450)  # core target 800, only 100 available
    cfg = _cfg(core_fraction=0.8, regular_fraction=0.1, aux_fraction=0.1)
    counts = pool_counts(N, {p: len(v) for p, v in members.items()}, cfg)
    assert counts["priority"] == 100
    assert counts["priority"] + counts["regular"] + counts["aux"] == N
    order = build_cycle_order(members, N, cfg, 0)
    assert sorted(order) == list(range(N))


def test_config_drives_composition() -> None:
    """The config fractions decide the composition in the NON-PARTITION
    synthetic cases (pool sizes do not sum to n — never the production
    case, where the result is the natural composition instead):
    * no pool saturation -> composition = the exact floor(n * target);
    * partially saturated core -> targets still respected for the pools
      that have room, and a different config changes the composition;
    * regular=0 is a first-class config."""
    # 1) no saturation: exact targets
    sizes = {"priority": 1000, "regular": 1000, "aux": 1000}
    c1 = pool_counts(1200, sizes, _cfg(core_fraction=0.8, regular_fraction=0.1, aux_fraction=0.1))
    assert c1 == {"priority": 960, "regular": 120, "aux": 120}
    # note: an aux target above [filter] aux_max_fraction (0.20) is
    # clamped to 0.20, and any freed slots flow to the core anchor
    c2 = pool_counts(1200, sizes, _cfg(core_fraction=0.5, regular_fraction=0.25, aux_fraction=0.25))
    assert c2 == {"priority": 660, "regular": 300, "aux": 240}
    # 2) core pool saturated (500 members < the 80%/50% targets of 1200):
    #    the config still steers regular vs aux
    sizes_sat = {"priority": 500, "regular": 1000, "aux": 1000}
    a = pool_counts(1200, sizes_sat, _cfg(core_fraction=0.8, regular_fraction=0.1, aux_fraction=0.1))
    b = pool_counts(1200, sizes_sat, _cfg(core_fraction=0.5, regular_fraction=0.25, aux_fraction=0.25))
    assert a == {"priority": 500, "regular": 580, "aux": 120}
    assert b == {"priority": 500, "regular": 460, "aux": 240}
    # 3) regular=0 first-class: regular base 0, aux at its target
    sizes3 = {"priority": 800, "regular": 0, "aux": 200}
    c3 = pool_counts(1000, sizes3, _cfg(core_fraction=0.8, regular_fraction=0.0, aux_fraction=0.2))
    assert c3 == {"priority": 800, "regular": 0, "aux": 200}


def test_slot_map_disabled_is_legacy_order() -> None:
    legacy = list(range(N - 1, -1, -1))  # a non-trivial legacy order
    members = _members(800, 100, 100)
    cfg_off = _cfg(enabled=False)
    sm = SlotMap(N, members, cfg_off, legacy)
    assert not sm.enabled
    for slot in range(0, 3 * N, 7):
        assert sm[slot] == legacy[slot % N]


def test_slot_map_resume_reproducible() -> None:
    """Two FRESH maps (a resume rebuilds everything from config) produce
    the identical sample stream at the same global slots."""
    members = _members(800, 100, 100)
    cfg = _cfg()
    sm1 = SlotMap(N, members, cfg, list(range(N)))
    sm2 = SlotMap(N, members, cfg, list(range(N)))
    slots = [0, 1, 2 * N + 5, 3 * N + 999, 4 * N + 7]
    assert [sm1[s] for s in slots] == [sm2[s] for s in slots]
    # and consecutive slots are distinct samples (no immediate repeats)
    seq = [sm1[s] for s in range(4 * N + 7, 4 * N + 7 + 50)]
    assert len(set(seq)) == 50


def test_ddp_global_slots_disjoint() -> None:
    """The latent_sample_index slot arithmetic gives each rank a disjoint
    set of global slots per step (bs=4, world=2, 5 steps): no double-serve."""
    bs, world, steps = 4, 2, 5
    slots: list[int] = []
    for step in range(steps):
        for rank in range(world):
            slots += [latent_sample_index(step, rank, i, bs, world, N) for i in range(bs)]
    assert len(slots) == len(set(slots)), "global slots must be unique across ranks"
    # and each slot maps to exactly one sample (the stream contract)
    members = _members(800, 100, 100)
    sm = SlotMap(N, members, _cfg(), list(range(N)))
    assert len({sm[s] for s in slots}) == len(slots)


def test_slot_map_members_must_cover_n() -> None:
    members = _members(700, 100, 100)  # only 900 of 1000
    with pytest.raises(RuntimeError, match="exactly one pool"):
        SlotMap(N, members, _cfg(), list(range(N)))


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="sum to > 0"):
        c = _cfg(core_fraction=0.0, regular_fraction=0.0, aux_fraction=0.0)
        c.validate_all()
