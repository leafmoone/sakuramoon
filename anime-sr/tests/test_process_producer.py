"""P0-1 integration tests: process-pool producer (real multiprocessing Pool).

Covers the production transport the thread backend never exercises:

* ``apply_async`` single-tuple payload binding (``_pp_fetch(args)`` —
  a flattened ``(slot, step)`` payload would TypeError in the worker and
  lose the batch);
* the ``AsyncResult`` wrappers (``_fut_done``/``_fut_result`` must use
  ``.ready()``/``.get(timeout)`` — an AsyncResult has no ``.done()``/
  ``.result()``);
* deterministic ``(slot, step)`` fetch identity (bit-exact resubmit);
* ``Pool.close()`` + ``join()`` lifecycle;
* worker death -> membership-diff detection -> exactly-once resubmit with
  bit-identical results (P1-WORKER-RECOVERY r2);
* crash-loop guard (RuntimeError after ``_PP_MAX_WORKER_CRASHES`` deaths).

Fork start method only (the production contract): skipped where the fork
start method is unavailable (e.g. Windows); the production gate runs these
on the Linux DTK host.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.pool import AsyncResult as _MPAsyncResult
from typing import Any, cast

import pytest
import torch
from anime_sr.config.schema import Config
from anime_sr.data.degradation import degrade_hr as _real_degrade_hr
from anime_sr.data.pipeline import _EXPOSURE_PER_CYCLE
from anime_sr.train import latent_flow as lf

FORK = "fork" in multiprocessing.get_all_start_methods()
requires_fork = pytest.mark.skipif(
    not FORK, reason="fork start method unavailable (production gate: Linux)"
)

_CFG = Config()

# fork-safety baseline (production contract, latent_flow P1 ⑤ + P1-WEDGE-FIX):
# the parent process must be single-threaded (intra-op=1) before ANY pool
# fork — a forked child inheriting an active OpenMP pool deadlocks on its
# first parallel CPU op. Pin it process-wide; _pp_worker_init keeps workers
# at OMP_NUM_THREADS (1 here).
torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")


# ---------------------------------------------------------------------------
# stub dataset: the minimum SRDataset surface _pp_fetch consumes
# ---------------------------------------------------------------------------


class _StubMeta:
    def __init__(self, sample_id: str) -> None:
        self.sample_id = sample_id


class _StubDS:
    """n samples, deterministic HR crops; crop pinned to (0, 0) (store mode)."""

    def __init__(self, n: int, size: int = 64) -> None:
        g = torch.Generator().manual_seed(1234)
        self._hr = (torch.rand(n, 3, size, size, generator=g) * 2.0 - 1.0)
        self.samples = [_StubMeta(f"s{i:04d}") for i in range(n)]
        self.global_seed = 42

    def decode_hr_timed(self, meta: _StubMeta):
        i = self.samples.index(meta)
        # unbatched [3, H, W] — the real SRDataset.decode_hr contract
        return self._hr[i], {"shard": 0.001, "decode": 0.001}

    def crop(self, meta: _StubMeta, data_cycle: int, exposure_index: int):
        return 0, 0  # pinned (0,0) — matches the pre-encoded z_hr


def _install_ctx(n: int, bucket_hr: int = 64) -> _StubDS:
    # production contract (latent_flow P1 ⑤ + P1-WEDGE-FIX): the parent
    # pins torch intra-op to 1 BEFORE any fork; a forked child that
    # inherits an active OpenMP pool deadlocks on its first parallel CPU op.
    # The workers re-tune from OMP_NUM_THREADS (kept at 1 here so the
    # small test pool stays single-threaded end to end).
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    ds = _StubDS(n)
    # P1 pool sampler: members=None -> disabled -> the legacy identity
    # stream (order[slot % n] with order=range(n)), i.e. the pre-sampler
    # contract these tests pin down.
    lf._PRODUCER_CTX = {
        "ds": ds,
        "order": list(range(n)),
        "n": n,
        "slot_map": lf.SlotMap(n, None, _CFG, list(range(n))),
        "cfg": _CFG,
        "store": None,  # on-fly style: the hr crop is returned to the caller
        "global_seed": ds.global_seed,
        "bucket_hr": bucket_hr,
        "exposure_per_cycle": _EXPOSURE_PER_CYCLE,
    }
    return ds


def _identity(args: tuple) -> tuple:
    return args


# ---------------------------------------------------------------------------
# future wrappers
# ---------------------------------------------------------------------------


@requires_fork
def test_future_wrappers_process_backend() -> None:
    """_fut_done/_fut_result on a real AsyncResult (.ready()/.get path)."""
    pp = lf._make_process_pool(2)
    try:
        f = pp.apply_async(_identity, ((7, "slot-step"),))
        assert isinstance(f, _MPAsyncResult)
        deadline = time.time() + 15
        while not lf._fut_done(f):
            assert time.time() < deadline, "task did not complete in 15 s"
            time.sleep(0.05)
        assert lf._fut_result(f, timeout=5) == (7, "slot-step")
    finally:
        pp.close()
        pp.join()


def test_future_wrappers_thread_backend() -> None:
    """Same wrappers on a concurrent.futures.Future (regression guard)."""
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="t") as tp:
        f = tp.submit(_identity, (3,))
        assert not isinstance(f, _MPAsyncResult)
        deadline = time.time() + 15
        while not lf._fut_done(f):
            assert time.time() < deadline
            time.sleep(0.01)
        assert lf._fut_result(f, timeout=5) == (3,)


# ---------------------------------------------------------------------------
# payload binding + determinism
# ---------------------------------------------------------------------------


@requires_fork
def test_pp_fetch_deterministic_slot_step() -> None:
    """(slot, step) binding through the single-tuple payload; bit-exact
    identity across resubmits; slot % n sample mapping."""
    ds = _install_ctx(n=4, bucket_hr=64)
    pp = lf._make_process_pool(2)
    try:
        # the production packing: apply_async(_pp_fetch, ((slot, step),))
        f1 = pp.apply_async(lf._pp_fetch, ((3, 7),))
        f2 = pp.apply_async(lf._pp_fetch, ((3, 7),))
        r1 = lf._fut_result(f1, timeout=60)
        r2 = lf._fut_result(f2, timeout=60)
        hr1, lq1, z1, meta1, _st1 = r1
        hr2, lq2, z2, meta2, _st2 = r2
        # store=None -> hr returned, z_hr None
        assert z1 is None and z2 is None
        assert meta1.sample_id == "s0003"
        # resubmit of the same (slot, step) is bit-exact (deterministic)
        assert torch.equal(hr1, hr2)
        assert torch.equal(lq1, lq2)
        assert meta1.sample_id == meta2.sample_id
        # slot % n sample mapping: slot 5 with n=4 -> index 1
        f3 = pp.apply_async(lf._pp_fetch, ((5, 0),))
        hr3, _, _, meta3, _ = lf._fut_result(f3, timeout=60)
        assert meta3.sample_id == "s0001"
        assert torch.equal(hr3, ds._hr[1])
    finally:
        pp.close()
        pp.join()


@requires_fork
def test_pool_close_join_drains() -> None:
    """close()+join() terminates every worker (no leaked processes)."""
    _install_ctx(n=4)
    pp = lf._make_process_pool(4)
    futs = [pp.apply_async(lf._pp_fetch, ((s, 0),)) for s in range(4)]
    for f in futs:
        lf._fut_result(f, timeout=60)
    pp.close()
    pp.join()
    living = [
        w for w in cast(Any, pp)._pool if w is not None and w.is_alive()
    ]  # Pool impl detail (typeshed-undeclared)
    assert living == []


# ---------------------------------------------------------------------------
# worker death recovery (P1-WORKER-RECOVERY r2)
# ---------------------------------------------------------------------------


@requires_fork
def test_worker_death_resubmit_exactly_once() -> None:
    """Kill one worker mid-task: membership diff detects it, the lost
    (slot, step) is resubmitted, and the batch completes bit-identical to
    the control with exactly-once consumption."""

    def _slow_degrade(hr, cfg, **kw):
        time.sleep(3.0)  # widen the in-flight window so the kill lands
        return _real_degrade_hr(hr, cfg, **kw)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(lf, "degrade_hr", _slow_degrade)
    pp: lf._MPPool | None = None  # bound for the finally-teardown
    try:
        _install_ctx(n=8)
        pp = lf._make_process_pool(4)
        ready: deque[list[Future | _MPAsyncResult]] = deque()
        inflight: dict[Future | _MPAsyncResult, tuple[int, int]] = {}
        for step in (0, 1):
            batch: list[Future | _MPAsyncResult] = [
                pp.apply_async(lf._pp_fetch, ((s, step),)) for s in range(4)
            ]
            for f, task in zip(batch, ((s, step) for s in range(4))):
                inflight[f] = task
            ready.append(batch)
        time.sleep(1.0)  # let every worker start its task
        # kill one live worker mid-task (os.kill: DTK torch's ForkProcess
        # wrapper lacks Process.send_signal)
        victims = [
            w for w in cast(Any, pp)._pool if w is not None and w.is_alive()
        ]  # Pool impl detail (typeshed-undeclared)
        assert victims, "no live workers found to kill"
        os.kill(victims[0].pid, getattr(signal, "SIGKILL", 9))
        crash_state = [0]
        seen: set = set()
        seen.update(cast(Any, pp)._pool)
        batch = ready[0]
        # the production collect loop: poll + recover until the batch is done
        deadline = time.time() + 90
        while not all(lf._fut_done(f) for f in batch):
            assert time.time() < deadline, "batch never completed after death"
            lf._pp_recover_lost_tasks(
                pp, batch, ready, inflight, crash_state, seen,
                lf._pp_fetch, lambda t: t,
            )
            time.sleep(0.2)
        results = [lf._fut_result(f, timeout=60) for f in batch]
        for f in batch:
            inflight.pop(f, None)
        # drain the second batch (its lost task, if any, was resubmitted too)
        batch2 = ready[1]
        deadline = time.time() + 90
        while not all(lf._fut_done(f) for f in batch2):
            assert time.time() < deadline, "batch2 never completed after death"
            lf._pp_recover_lost_tasks(
                pp, batch2, ready, inflight, crash_state, seen,
                lf._pp_fetch, lambda t: t,
            )
            time.sleep(0.2)
        results2 = [lf._fut_result(f, timeout=60) for f in batch2]
        assert crash_state[0] == 1, f"expected exactly 1 death, saw {crash_state[0]}"
        # every completed slot maps to the right sample
        for (s, step), r in zip(((s, 0) for s in range(4)), results):
            assert r[3].sample_id == f"s{s:04d}"
        # bit-exact deterministic resubmit: a fresh task with the same
        # (slot, step) recomputes the identical (hr, lq)
        check = [pp.apply_async(lf._pp_fetch, ((s, 0),)) for s in range(4)]
        for f, r in zip(check, results):
            r_again = lf._fut_result(f, timeout=60)
            assert torch.equal(r[0], r_again[0])
            assert torch.equal(r[1], r_again[1])
            assert r[3].sample_id == r_again[3].sample_id
        for f in check:
            inflight.pop(f, None)
        assert len(results) + len(results2) == 8  # every task consumed exactly once
    finally:
        monkeypatch.undo()
        try:
            if pp is not None:
                pp.terminate()  # test teardown: kill any slow/respawned workers
                pp.join()
        except Exception:  # noqa: S110, BLE001 - teardown must never mask the test result
            pass


@requires_fork
def test_crash_loop_guard_aborts() -> None:
    """A body that crashes on every attempt: recovery resubmits, deaths
    accumulate to _PP_MAX_WORKER_CRASHES, then the run aborts loudly
    (RuntimeError) instead of crash-looping forever."""

    def _crash_degrade(hr, cfg, **kw):
        os._exit(1)  # simulate a worker segfault: dies with no result

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(lf, "degrade_hr", _crash_degrade)
    pp: lf._MPPool | None = None  # bound for the finally-teardown
    try:
        _install_ctx(n=2)
        pp = lf._make_process_pool(2)
        batch: list[Future | _MPAsyncResult] = [
            pp.apply_async(lf._pp_fetch, ((s, 0),)) for s in range(2)
        ]
        inflight: dict[Future | _MPAsyncResult, tuple[int, int]] = {
            f: (s, 0) for f, s in zip(batch, range(2))
        }
        crash_state = [0]
        seen: set = set()
        seen.update(cast(Any, pp)._pool)  # Pool impl detail (typeshed-undeclared)
        deadline = time.time() + 120
        with pytest.raises(RuntimeError, match="crash-loop"):
            while not all(lf._fut_done(f) for f in batch):
                assert time.time() < deadline, "crash-loop guard never fired"
                lf._pp_recover_lost_tasks(
                    pp, batch, deque(), inflight, crash_state, seen,
                    lf._pp_fetch, lambda t: t,
                )
                time.sleep(0.2)
        assert crash_state[0] >= lf._PP_MAX_WORKER_CRASHES
    finally:
        monkeypatch.undo()
        try:
            if pp is not None:
                pp.terminate()  # test teardown: kill any slow/respawned workers
                pp.join()
        except Exception:  # noqa: S110, BLE001 - teardown must never mask the test result
            pass
