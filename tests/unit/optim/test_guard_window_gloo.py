"""Persistent CPU/Gloo guard-window integration: real constructors, steps and load.

CPU AdamW and stateful pass-through SR replace CUDA-only builder collaborators;
this is NOT native TorchAO/SR, CUDA/NCCL, production checkpoint or capsule testing.
Healthy BF16 and rescued FP32 NS use the real CPU kernels. Only fault kernels
inject NaNs; owner, counters, verdicts, collectives and publication are real.
Windows alone substitutes a local fsync sink for the POSIX-only durable sink.
All synthetic diagnostics live under pytest tmp_path; capsules are suppressed.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
import socket
import time
import traceback
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import pytest
import torch
import torch.distributed as dist
from torch import nn

import sakuramoon.optim.fp32_rescue as rescue
import sakuramoon.optim.guarded_canonical as guarded
import sakuramoon.train.guard_window_stream as stream
from sakuramoon.cli.guard_window_merge import merge_records, read_records
from sakuramoon.optim.cmuon import (
    CMuonChunkSpec,
    CMuonConfig,
    CMuonRouting,
    resolve_ns_map,
)
from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
from sakuramoon.optim.groups import ParameterAudit, ParameterSpec

_FIXTURE = (
    ("dit.blocks.slot_01.attention.q_proj.weight", 1, "attention_q"),
    ("dit.blocks.slot_00.mlp.in_proj.weight", 2, "ffn_in"),
    ("dit.conditioner.shared_block_projection.weight", 6, "adaln_shared"),
)
_OWNERS = {
    1: [0] * 9,
    2: [1, 1, 0, 1, 0, 1, 0, 1, 0],
    4: [3, 1, 2, 1, 2, 3, 0, 1, 2],
}
_COUNTS: dict[int, list[int]] = {1: [9], 2: [4, 5], 4: [1, 3, 3, 2]}
_ROLES = {role: count for _, count, role in _FIXTURE}
_FQNS = {name: count for name, count, _ in _FIXTURE}
_O_FIELDS = (
    "bf16_attempts",
    "bf16_safety_failures",
    "fp32_attempts",
    "fp32_rescues",
    "fp32_rescue_failures",
)
_T = TypeVar("_T")


class _CPUSR:
    """Test-only SR call/state adapter: no rounding or CUDA RNG initialization."""

    def __init__(self) -> None:
        self.calls = 0

    def run_step(self, fn: Callable[[], _T]) -> _T:
        result = fn()
        self.calls += 1
        return result

    def state_dict(self) -> dict[str, int]:
        return {"cpu_sr_calls": self.calls}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.calls = state["cpu_sr_calls"]


class _CPUAdamW:
    """Constructor-compatible CPU adapter, not a mock optimizer step."""

    def __init__(self, specs: tuple[ParameterSpec, ...], sr: _CPUSR) -> None:
        self.optimizer = torch.optim.AdamW(
            [s.parameter for s in specs], lr=0.01, weight_decay=0.0, foreach=False
        )
        self.audit, self.sr_rng = ParameterAudit(specs), sr

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.sr_rng.load_state_dict(state["sr_rng"])


class _WindowsFsyncSink:
    """Windows-only IO adapter; real sealing/no-clobber stays untouched."""

    def __init__(self, path: Path, fsync_every_records: int = 1) -> None:
        assert fsync_every_records == 1
        self.path = path

    def write(self, record: object) -> None:
        import json

        with self.path.open("ab") as handle:
            handle.write((json.dumps(record, allow_nan=False) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())

    def close(self) -> None:
        pass  # write already fsyncs and closes before real atomic publication


def _no_capsule(*_args: object, **_kwargs: object) -> None:
    """Test-only serializer no-op: no forensic tensor bytes or project hashes."""


def _make(root: Path, rank: int, world: int, *, base: bool = False) -> Any:
    specs: list[CMuonChunkSpec] = []
    audit: list[ParameterSpec] = []
    for name, count, role in _FIXTURE:
        parameter = nn.Parameter(torch.zeros((2 * count, 2), dtype=torch.bfloat16))
        semantics = tuple(f"synthetic_{i}" for i in range(count))
        specs.append(CMuonChunkSpec(name, parameter, 0.0, count, 0, semantics, role))
        audit.append(ParameterSpec(name, parameter, "matrix_decay", 0.0))
    scalar = nn.Parameter(torch.ones(1))
    adam_specs = (ParameterSpec("synthetic.bias", scalar, "sensitive_no_decay", 0.0),)
    routing = CMuonRouting(
        ParameterAudit(tuple(audit) + adam_specs), tuple(specs), adam_specs
    )
    sr = _CPUSR()
    # Collaborators satisfy the runtime protocol, not CUDA-specific nominal types.
    cfg = CMuonConfig(
        lr=0.01,
        momentum=0.5,
        momentum_dtype="bfloat16",
        ns_steps_by_role=resolve_ns_map(None, 4),
    )
    guard_cfg = guarded.GuardedCanonicalGuardConfig(0.5, 0.9, 1e-6, 1e-10, 10, True)
    kwargs: dict[str, Any] = {
        "routing": routing,
        "cfg": cfg,
        "guard_cfg": guard_cfg,
        "adamw": _CPUAdamW(adam_specs, sr),
        "sr_rng": sr,
        "rank": rank,
        "world_size": world,
        "guard_bootstrap_refs": {s.name: 0.1 for s in specs},
    }
    if base:
        return guarded.HybridCMuonGuardedCanonical(**kwargs)
    return rescue.HybridCMuonCanonicalNS4FP32Rescue(
        **kwargs,
        legacy_forensic_dir=str(root / "legacy"),
        emergency_capsule_root=str(root / "unused-capsules"),
        hard_fail_artifact_root=str(root / "unused-mirror"),
        run_id="synthetic-gloo",
    )


def _recorder(
    root: Path,
    opt: Any,
    rank: int,
    world: int,
    session: str,
    *,
    resumed: bool = False,
    cadence: int = 1,
    published: Callable[[], str | None] | None = None,
) -> stream.GuardWindowRecorder:
    return stream.GuardWindowRecorder(
        artifact_dir=root,
        identity=stream.GuardWindowIdentity(
            run_id="synthetic-gloo",
            session_id=session,
            rank=rank,
            world_size=world,
            resume_source_checkpoint="synthetic-cp" if resumed else "fresh-start",
            resume_successful_updates=2 if resumed else 0,
            launch_kind="user_resume" if resumed else "fresh_bootstrap",
        ),
        optimizer=opt,
        window_updates=cadence,
        last_published_checkpoint=published,
    )


def _observe(recorder: stream.GuardWindowRecorder, update: int) -> None:
    observation: Any = SimpleNamespace(
        loop=SimpleNamespace(
            update=SimpleNamespace(state=SimpleNamespace(successful_updates=update))
        )
    )
    recorder.maybe_record(observation)
    assert not recorder.disabled


def _step(opt: Any, rank: int, world: int, mode: str, *, low: bool = False) -> None:
    """Spy only at kernel boundaries; never replace ownership/count/verdict logic."""
    expected: dict[float, tuple[str, int]] = {}
    keys: list[tuple[str, int]] = []
    index = 0
    for spec in opt.routing.cmuon_specs:
        grad = torch.empty_like(spec.parameter)
        for chunk in range(spec.chunk_count):
            grad.narrow(0, 2 * chunk, 2).fill_((index + 1) * (1e-5 if low else 0.25))
            keys.append((spec.name, chunk))
            index += 1
        spec.parameter.grad = grad
        # Predict input signatures solely to identify actual kernel calls.
        buf = opt._momenta[spec.parameter].clone()
        buf.lerp_(grad, 1 - opt.cfg.momentum)
        signal = grad.lerp(buf, opt.cfg.momentum)
        for chunk in range(spec.chunk_count):
            signature = float(signal[2 * chunk, 0])
            assert signature not in expected
            expected[signature] = (spec.name, chunk)
    for spec in opt.routing.adamw_specs:
        spec.parameter.grad = torch.full_like(spec.parameter, 0.3)
    owned = [
        key for key in keys if guarded.stable_owner(*key, world) == rank and not low
    ]
    bf_calls: list[tuple[str, int]] = []
    fp_calls: list[tuple[str, int]] = []
    real_bf, real_fp = rescue.cmuon_zeroth_power_bf16, rescue.cmuon_zeroth_power_fp32

    def kernel(
        tensor: torch.Tensor,
        steps: int,
        coefficients: tuple[float, float, float],
        eps: float,
        *,
        fp: bool = False,
    ) -> torch.Tensor:
        key = expected[float(tensor.flatten()[0])]
        assert guarded.stable_owner(*key, world) == rank
        (fp_calls if fp else bf_calls).append(key)
        if (not fp and mode != "healthy") or (fp and mode == "failure"):
            return torch.full_like(tensor, float("nan"))  # explicit fault injection
        return (real_fp if fp else real_bf)(tensor, steps, coefficients, eps)

    def fp_kernel(
        tensor: torch.Tensor,
        steps: int,
        coefficients: tuple[float, float, float],
        eps: float,
    ) -> torch.Tensor:
        return kernel(tensor, steps, coefficients, eps, fp=True)

    before = [s.parameter.detach().clone() for s in opt.audit.specs]
    observations, sr_calls = opt.observations, opt.sr_rng.calls
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(guarded, "cmuon_zeroth_power", kernel)
        patch.setattr(rescue, "cmuon_zeroth_power_bf16", kernel)
        patch.setattr(rescue, "cmuon_zeroth_power_fp32", fp_kernel)
        if mode == "failure":
            with pytest.raises(CMuonSafetyError):
                opt.step()
            assert opt.observations == observations and opt.sr_rng.calls == sr_calls
            assert all(
                torch.equal(s.parameter, p) for s, p in zip(opt.audit.specs, before)
            )
        else:
            opt.step()
            assert (
                opt.observations == observations + 1
                and opt.sr_rng.calls == sr_calls + 1
            )
            scalar = opt.routing.adamw_specs[0].parameter
            assert float(opt.optimizer.state[scalar]["step"]) == sr_calls + 1
            assert not torch.equal(scalar, before[-1])
            assert all(
                torch.equal(s.parameter, p) == low
                for s, p in zip(opt.routing.cmuon_specs, before[:-1])
            )
    assert bf_calls == owned
    assert fp_calls == (owned if mode != "healthy" else [])
    for spec in opt.audit.specs:
        copies = [torch.empty_like(spec.parameter) for _ in range(world)]
        dist.all_gather(copies, spec.parameter.detach())  # pyright: ignore[reportUnknownMemberType]
        assert all(torch.equal(copies[0], value) for value in copies)


def _check_owner(opt: Any, expected: tuple[int, ...]) -> None:
    assert tuple(getattr(opt, name) for name in _O_FIELDS) == expected


def _exercise(root: Path, rank: int, world: int) -> dict[str, Any]:
    owners = [
        guarded.stable_owner(name, chunk, world)
        for name, count, _ in _FIXTURE
        for chunk in range(count)
    ]
    assert owners == _OWNERS[world]
    count = _COUNTS[world][rank]
    assert owners.count(rank) == count
    roles = {
        role: sum(guarded.stable_owner(name, c, world) == rank for c in range(n))
        for name, n, role in _FIXTURE
    }
    skipped = _make(root, rank, world, base=True)
    recorder = _recorder(root, skipped, rank, world, "skip")
    for update in (1, 2):
        _step(skipped, rank, world, "healthy", low=True)
        _observe(recorder, update)
        assert skipped.skip_total == 9 * update and skipped.observations == update
        assert skipped.skip_by_role == {role: n * update for role, n in _ROLES.items()}
        assert skipped.skip_by_fqn == {name: n * update for name, n in _FQNS.items()}
    recorder.close("clean")
    healthy = _make(root, rank, world)
    _step(healthy, rank, world, "healthy")  # actual BF16 CPU NS, no injections
    _check_owner(healthy, (count, 0, 0, 0, 0))
    assert healthy.rescue_by_role == {}

    opt = _make(root, rank, world)
    assert opt.rescue_by_role == {}  # first rescue role appears AFTER baseline
    published: list[str | None] = [None]
    recorder = _recorder(
        root, opt, rank, world, "rescue", published=lambda: published[0]
    )
    for update in (1, 2):
        _step(opt, rank, world, "rescue")  # injected BF16 NaN -> REAL safe FP32 NS
        _observe(recorder, update)
        _check_owner(opt, (count * update,) * 4 + (0,))
        assert opt.rescue_by_role == {
            role: n * update for role, n in roles.items() if n
        }
        published[0] = "synthetic-new-publication"  # metadata, NOT resume
    recorder.close("clean")

    state: list[Any] = [copy.deepcopy(opt.state_dict()) if rank == 0 else None]
    dist.broadcast_object_list(state, src=0)  # same rank0 OUTER state into all ranks
    restored = _make(root, rank, world)
    for field, value, message in (
        ("world_size", world + 1, "world_size mismatch"),
        ("owner_mapping_version", "invalid-mapping", "owner mapping version mismatch"),
    ):
        bad = copy.deepcopy(state[0])
        bad["guard"][field] = value
        with pytest.raises(CMuonSafetyError, match=message):
            restored.load_state_dict(bad)
    restored.load_state_dict(copy.deepcopy(state[0]))
    baseline = 2 * _COUNTS[world][0]
    _check_owner(restored, (baseline,) * 4 + (0,))
    assert restored.observations == 2 and restored.sr_rng.calls == 2
    recorder = _recorder(root, restored, rank, world, "restored", resumed=True)
    _step(restored, rank, world, "rescue")
    _observe(recorder, 3)
    recorder.close("clean")
    _check_owner(restored, (baseline + count,) * 4 + (0,))

    failed = _make(root, rank, world)
    recorder = _recorder(root, failed, rank, world, "failure", cadence=2)
    _step(failed, rank, world, "rescue")
    _observe(recorder, 1)
    _step(failed, rank, world, "failure")  # failed candidates; NO success callback
    _check_owner(failed, (2 * count, 2 * count, 2 * count, count, count))
    assert failed.observations == 1 and failed.sr_rng.calls == 1
    recorder.close("raised")  # tail uses last success, not failed attempt counters
    return {
        "rank": rank,
        "owned": count,
        "baseline": baseline,
        "restored": restored.bf16_attempts,
        "skip": skipped.skip_total,
        "observations": skipped.observations,
    }


def _worker(rank: int, world: int, port: int, root: str, results: Any) -> None:
    """Module-level/picklable spawn entry; send scalars/errors, never tensor IPC."""
    try:
        torch.set_num_threads(1)
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("SAKURAMOON_FORENSIC_DIR", str(Path(root) / "base-legacy"))
            patch.setattr(
                rescue.HybridCMuonCanonicalNS4FP32Rescue,
                "_publish_minimal_hardfail_capsule",
                _no_capsule,
            )
            if os.name == "nt":
                patch.setattr(stream, "DurableJsonlSink", _WindowsFsyncSink)
            assert dist.is_gloo_available(), "Gloo required, not silently skipped"
            dist.init_process_group(
                "gloo",
                init_method=f"tcp://127.0.0.1:{port}",
                rank=rank,
                world_size=world,
                timeout=timedelta(seconds=30),
            )
            results.put(_exercise(Path(root), rank, world))
    except Exception:  # noqa: BLE001 - propagate child traceback to parent assertion
        results.put({"rank": rank, "error": traceback.format_exc()})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn(world: int, root: Path) -> list[dict[str, Any]]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    results: Any = context.Queue()
    children = [
        context.Process(target=_worker, args=(r, world, port, str(root), results))
        for r in range(world)
    ]
    deadline = time.monotonic() + 120
    try:
        for child in children:
            child.start()
        # Drain before join: no queue/pipe backpressure deadlock.
        rows: list[dict[str, Any]] = [
            results.get(timeout=max(0.1, deadline - time.monotonic())) for _ in children
        ]
        for child in children:
            child.join(timeout=max(0.1, deadline - time.monotonic()))
        assert all(child.exitcode == 0 for child in children), [
            c.exitcode for c in children
        ]
        assert not [row for row in rows if "error" in row], rows
        return sorted(rows, key=lambda row: row["rank"])
    finally:
        for child in children:
            if child.pid is not None:
                if child.is_alive():
                    child.terminate()
                child.join(timeout=5)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=5)
                child.close()
        results.close()
        results.join_thread()


@pytest.mark.parametrize("world", [1, 2, 4], ids=["ws1", "ws2", "ws4"])
def test_guard_window_real_cpu_gloo(
    tmp_path: Path, world: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    rows = _spawn(world, tmp_path)
    assert [row["owned"] for row in rows] == _COUNTS[world]
    assert [row["skip"] for row in rows] == [18] * world
    assert [row["observations"] for row in rows] == [2] * world
    assert [row["baseline"] for row in rows] == [2 * _COUNTS[world][0]] * world
    restored_counts = {1: [27], 2: [12, 13], 4: [3, 5, 5, 4]}
    assert [row["restored"] for row in rows] == restored_counts[world]
    for session, start, end in (("skip", 1, 2), ("rescue", 1, 2), ("restored", 3, 3)):
        records: list[Any] = read_records(
            [tmp_path / "guard_window" / f"session-{session}"]
        )
        assert len(records) == world * (end - start + 1)
        merged: Any = merge_records(records, start_update=start, end_update=end)
        assert merged["integrity"]["coverage"] == "complete"
        assert merged["integrity"]["owner_partition_verified"] is True
        for window in merged["windows"]:
            delta = window["delta"]
            assert window["complete"] is True and window["run_outcome"] is None
            assert delta["observations"] == 1  # replicated, never W
            assert delta["skip_total"] == (9 if session == "skip" else 0)
            assert delta["skip_by_role"] == {
                r: n if session == "skip" else 0 for r, n in _ROLES.items()
            }
            assert delta["skip_by_fqn"] == {
                n: c if session == "skip" else 0 for n, c in _FQNS.items()
            }
            if session != "skip":
                assert tuple(delta[f] for f in _O_FIELDS) == (9, 9, 9, 9, 0)
                assert delta["rescue_by_role"] == _ROLES
        if session == "rescue":
            for record in records:
                launch = record["launch"]
                assert launch["resume_source_checkpoint"] == "fresh-start"
                assert launch["launch_kind"] == "fresh_bootstrap"
                expected = (
                    None if record["start_update"] == 1 else "synthetic-new-publication"
                )
                assert launch["last_published_checkpoint"] == expected
                if record["start_update"] == 1:
                    assert record["baseline"]["counts"][
                        "rescue_by_role"
                    ] == dict.fromkeys(_ROLES, 0)
        if session == "restored":
            for record in records:
                assert (
                    record["baseline"]["counts"]["bf16_attempts"] == rows[0]["baseline"]
                )
                assert record["baseline"]["source"] == "restored_rank0"
                assert record["absolute_history"]["bf16_attempts"] == "rank0_owner_only"
                assert record["launch"]["launch_kind"] == "user_resume"
            assert sum(r["absolute"]["bf16_attempts"] for r in records) != 9
    failed: list[Any] = read_records([tmp_path / "guard_window" / "session-failure"])
    assert len(failed) == world
    for record in failed:
        count = _COUNTS[world][int(record["rank"])]
        assert record["complete"] is False and record["run_outcome"] == "raised"
        assert record["end_update"] == record["absolute"]["observations"] == 1
        assert tuple(record["absolute"][f] for f in _O_FIELDS) == (count,) * 4 + (0,)
    with pytest.raises(ValueError, match="clean final tail"):
        merge_records(failed, start_update=1, end_update=1)
    if world > 1:
        assert max(row["owned"] for row in rows) != 9  # max cannot replace owner sum
        assert sum(row["skip"] for row in rows) != 18  # sum cannot replace R parity
