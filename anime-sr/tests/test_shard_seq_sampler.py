"""shard_seq stream (2026-09-01 SR_v2 venue decision): per-cycle
PERMUTATION OF SHARDS with intra-shard sequential streaming (tars are
shuffled, samples inside a tar are streamed in index order — no
per-sample sampling).

Contract under test:
  * every cycle is a FULL-SET permutation (each sample exactly once);
  * within a cycle, one shard's samples appear in ascending row order —
    intra-shard streaming;
  * deterministic + resume-safe: a fresh map reproduces the stream at the
    same slot; different cycles get different shard orders;
  * DDP: disjoint global slots -> disjoint samples;
  * window boundedness: a 300-step window (bs 8 x world 2 = 16/step)
    touches only ~window_samples/shard_size shards (+boundaries) — the
    property that makes the tar-direct pin window feasible (the pool
    sampler's full-set permutation touches ~every shard instead);
  * enabled=false takes precedence over strategy (legacy straight read);
  * construction is fail-closed (missing blocks / coverage mismatch).
"""

from __future__ import annotations

import pytest
from anime_sr.config.schema import Config, SamplingSpec
from anime_sr.data.pool_sampler import SlotMap
from anime_sr.data.stream import latent_sample_index, window_shards


def _cfg(enabled: bool = True, strategy: str = "shard_seq") -> Config:
    base = Config()
    spec = SamplingSpec(enabled=enabled, strategy=strategy)
    return base.model_copy(update={"sampling": spec})


def _shard_blocks(sizes: list[int], interleaved: bool = False) -> list[tuple[str, list[int]]]:
    """(shard name, ascending row indices) in first-appearance order."""
    n = sum(sizes)
    blocks: list[tuple[str, list[int]]] = []
    for k, size in enumerate(sizes):
        if interleaved:
            rows = list(range(k, n, len(sizes)))[:size]
        else:
            start = sum(sizes[:k])
            rows = list(range(start, start + size))
        blocks.append((f"shard-{k:04d}", rows))
    covered = sum(len(rows) for _, rows in blocks)
    assert covered == n
    return blocks


def _shard_of_row(blocks: list[tuple[str, list[int]]]) -> list[str]:
    names: list[str] = []
    for name, rows in blocks:
        for row in rows:
            names.append(name)
    return names


N_SMALL = 1000
SIZES = [400, 250, 300, 50]  # covers N_SMALL, non-uniform incl. a small shard


def _small() -> tuple[int, list[tuple[str, list[int]]]]:
    blocks = _shard_blocks(SIZES)
    return sum(SIZES), blocks


def test_cycle_is_full_permutation() -> None:
    n, blocks = _small()
    sm = SlotMap(n, None, _cfg(), list(range(n)), salt="seed", shard_blocks=blocks)
    for cycle in (0, 1, 7):
        seq = [sm[cycle * n + pos] for pos in range(n)]
        assert sorted(seq) == list(range(n)), f"cycle {cycle} not a full-set permutation"


def test_intra_shard_sequential() -> None:
    """Within a cycle, each shard's subsequence is its ascending row list."""
    n, blocks = _small()
    sm = SlotMap(n, None, _cfg(), list(range(n)), salt="seed", shard_blocks=blocks)
    for cycle in (0, 3):
        per_shard: dict[str, list[int]] = {name: [] for name, _ in blocks}
        for pos in range(n):
            idx = sm[cycle * n + pos]
            for name, rows in blocks:
                if idx in rows:
                    per_shard[name].append(idx)
                    break
        for name, rows in blocks:
            assert per_shard[name] == rows, f"shard {name} not streamed in row order"


def test_deterministic_and_resume_reproducible() -> None:
    n, blocks = _small()
    sm1 = SlotMap(n, None, _cfg(), list(range(n)), salt="seed", shard_blocks=blocks)
    sm2 = SlotMap(n, None, _cfg(), list(range(n)), salt="seed", shard_blocks=blocks)
    slots = [0, 5, 999, 2 * n + 3, 3 * n + 700, 4 * n + 1]
    assert [sm1[s] for s in slots] == [sm2[s] for s in slots]
    # salt changes the stream (different corpus seed)
    sm3 = SlotMap(n, None, _cfg(), list(range(n)), salt="other", shard_blocks=blocks)
    assert [sm1[s] for s in slots] != [sm3[s] for s in slots]


def test_different_cycles_different_shard_order() -> None:
    n, blocks = _small()
    sm = SlotMap(n, None, _cfg(), list(range(n)), salt="seed", shard_blocks=blocks)
    names = _shard_of_row(blocks)
    order: list[list[str]] = []
    for cycle in (0, 1):
        seen: list[str] = []
        for pos in range(n):
            shard = names[sm[cycle * n + pos]]
            if shard not in seen:
                seen.append(shard)
        order.append(seen)
    assert order[0] != order[1], "cycle shard orders must differ"
    for o in order:
        assert sorted(o) == sorted(names), "every shard appears exactly once per cycle"


def test_ddp_disjoint_slots_disjoint_samples() -> None:
    n, blocks = _small()
    sm = SlotMap(n, None, _cfg(), list(range(n)), salt="seed", shard_blocks=blocks)
    bs, world, steps = 4, 2, 5
    slots = [
        latent_sample_index(step, rank, i, bs, world, n)
        for step in range(steps)
        for rank in range(world)
        for i in range(bs)
    ]
    assert len(slots) == len(set(slots))
    assert len({sm[s] for s in slots}) == len(slots)


def test_window_is_bounded_to_few_shards() -> None:
    """The streaming-feasibility property: 300 steps x 16 samples = 4800
    samples over 2000-sample shards -> at most ceil(4800/2000)+1 shards,
    not the whole corpus (the pool sampler's failure mode)."""
    n_shards, shard_size = 100, 2000
    sizes = [shard_size] * n_shards
    blocks = _shard_blocks(sizes)
    n = sum(sizes)
    sm = SlotMap(n, None, _cfg(), list(range(n)), salt="seed", shard_blocks=blocks)
    names = _shard_of_row(blocks)
    bs, world = 8, 2
    for start in (0, 37, 500):
        window = window_shards(
            start, start + 300, bs=bs, world=world, n=n, slot_map=sm, shards=names
        )
        expected_span = (300 * bs * world) / shard_size
        assert len(window) <= int(expected_span) + 2, (
            f"window at step {start} touches {len(window)}/{n_shards} shards"
        )
        assert len(window) << n_shards


def test_disabled_precedes_strategy() -> None:
    n, blocks = _small()
    legacy = list(range(n - 1, -1, -1))
    sm = SlotMap(n, None, _cfg(enabled=False), legacy, salt="seed", shard_blocks=blocks)
    assert not sm.enabled
    for slot in range(0, 3 * n, 97):
        assert sm[slot] == legacy[slot % n]


def test_shard_seq_requires_blocks() -> None:
    n, _ = _small()
    with pytest.raises(RuntimeError, match="requires shard_blocks"):
        SlotMap(n, None, _cfg(), list(range(n)), shard_blocks=None)


def test_shard_seq_coverage_must_match_n() -> None:
    n, _ = _small()
    with pytest.raises(RuntimeError, match="shard blocks cover"):
        SlotMap(n, None, _cfg(), list(range(n)), shard_blocks=[("s", list(range(n - 1)))])


def test_pool_strategy_is_default() -> None:
    """Back-compat: an unset strategy resolves to the M4-1024-frozen pool
    sampler, so existing configs keep their bit-exact stream."""
    spec = SamplingSpec()
    assert spec.strategy == "pool"
