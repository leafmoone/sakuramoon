"""Core unit tests for the per-rank guard window stream (Phase 2B/2D ws1/ws2/ws4).

Runs on CPU with a lightweight fake of the guarded optimizer: the fake
bypasses the real constructor (no torch tensors needed) but keeps the real
``HybridCMuonGuardedCanonical`` type so the isinstance gate is exercised,
and carries the real public int/map counter surface the stream reads.

Failure semantics under test (root ruling): ANY snapshot/IO/publication
fault latches the session DISABLED for the whole run — no per-window
recovery that could falsely imply continuous comparable coverage.  The
on-disk sequence simply shows the gap (explicit incomplete, never
backfilled).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Self, cast

import pytest

from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
from sakuramoon.optim.fp32_rescue import HybridCMuonCanonicalNS4FP32Rescue
from sakuramoon.optim.guarded_canonical import (
    OWNER_MAPPING_VERSION,
    HybridCMuonGuardedCanonical,
    stable_owner,
)
from sakuramoon.train.guard_window_stream import (
    GuardWindowIdentity,
    GuardWindowRecorder,
)


def _spec(
    name: str,
    chunk_count: int,
    roles: tuple[str, ...],
    role: str,
    chunk_dim: int = 0,
    shape: tuple[int, ...] = (8, 4),
) -> Any:
    return types.SimpleNamespace(
        name=name,
        parameter=types.SimpleNamespace(shape=shape),
        chunk_count=chunk_count,
        chunk_dim=chunk_dim,
        roles=roles,
        role=role,
    )


def _make_optimizer(
    specs: list[Any],
    **counters: Any,
) -> HybridCMuonGuardedCanonical:
    """Bypass the real constructor; keep the real type for the isinstance gate.

    The attribute surface is set through an ``Any`` view: the fake carries
    exactly the public int/map counter names the stream reads, nothing else.
    Pass ``rescue=True`` to keep the REAL rescue subclass type (the variant
    is bound from the type, never from which fields happen to exist).
    """
    cls = (
        HybridCMuonCanonicalNS4FP32Rescue
        if counters.pop("rescue", False)
        else HybridCMuonGuardedCanonical
    )
    opt: Any = object.__new__(cls)
    opt.routing = types.SimpleNamespace(cmuon_specs=list(specs))
    opt.observations = counters.get("observations", 0)
    opt.skip_total = counters.get("skip_total", 0)
    opt.skip_by_role = dict(counters.get("skip_by_role", {}))
    opt.skip_by_fqn = dict(counters.get("skip_by_fqn", {}))
    for key in (
        "bf16_attempts",
        "bf16_safety_failures",
        "fp32_attempts",
        "fp32_rescues",
        "fp32_rescue_failures",
        "rescue_by_role",
    ):
        if key in counters:
            setattr(opt, key, counters[key])
    return cast("HybridCMuonGuardedCanonical", opt)


def _identity(
    session_id: str = "sess-1",
    rank: int = 0,
    world_size: int = 2,
    resume_source: str = "fresh-start",
    resume_updates: int = 0,
    launch_kind: str | None = None,
) -> GuardWindowIdentity:
    if launch_kind is None:
        # unit-level default: a non-"fresh-start" source stands in for a
        # user resume; production always passes launch_kind EXPLICITLY.
        launch_kind = (
            "user_resume" if resume_source != "fresh-start" else "fresh_bootstrap"
        )
    return GuardWindowIdentity(
        run_id="run-1",
        session_id=session_id,
        rank=rank,
        world_size=world_size,
        resume_source_checkpoint=resume_source,
        resume_successful_updates=resume_updates,
        launch_kind=launch_kind,
    )


def _observation(update_no: int) -> Any:
    return types.SimpleNamespace(
        loop=types.SimpleNamespace(
            update=types.SimpleNamespace(
                state=types.SimpleNamespace(successful_updates=update_no)
            )
        )
    )


def _recorder(
    tmp_path: Path,
    optimizer: Any,
    window_updates: int = 2,
    identity: GuardWindowIdentity | None = None,
    last_published: Any = None,
) -> GuardWindowRecorder:
    return GuardWindowRecorder(
        artifact_dir=tmp_path,
        identity=identity or _identity(),
        optimizer=optimizer,
        window_updates=window_updates,
        last_published_checkpoint=last_published,
    )


def _sealing_recorder(
    tmp_path: Path,
    optimizer: Any,
    monkeypatch: pytest.MonkeyPatch,
    window_updates: int = 2,
    identity: GuardWindowIdentity | None = None,
    last_published: Any = None,
) -> GuardWindowRecorder:
    """Build a recorder with the durable-sink dependency stubbed on non-POSIX.

    ``DurableJsonlSink`` uses POSIX-only open flags; the stream's sealing
    logic is OS-independent, so Windows runs test with a minimal local sink
    that keeps the same write->close contract inside ``prepare``.
    """
    import os

    if not hasattr(os, "O_CLOEXEC"):
        import json as _json

        class _LocalSink:
            """POSIX-free stand-in: same write->close contract inside prepare."""

            def __init__(self, path: Path, fsync_every_records: int = 1) -> None:
                self._path = path

            def write(self, record: object) -> None:
                with open(self._path, "ab") as fh:
                    fh.write((_json.dumps(record) + "\n").encode())

            def close(self) -> None:
                pass

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *_args: object) -> None:
                self.close()

        monkeypatch.setattr(
            "sakuramoon.train.guard_window_stream.DurableJsonlSink", _LocalSink
        )
    return _recorder(
        tmp_path,
        optimizer,
        window_updates=window_updates,
        identity=identity,
        last_published=last_published,
    )


def _windows(rank_dir: Path) -> list[Path]:
    return sorted(rank_dir.glob("guard-window-u*.jsonl"))


def _read(record_file: Path) -> dict[str, Any]:
    lines = record_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])  # type: ignore[return-value]


ADALN_SPECS = [
    # legal for the real routing validator: chunk_dim extent 24 divides
    # evenly by chunk_count 6 (never loosen production validation to fit a
    # fixture).
    _spec(
        "model.blocks.0.adaln_shared",
        chunk_count=6,
        roles=("gamma_in", "shift_in", "scale_in", "gate_in", "gamma_mod", "shift_mod"),
        role="adaln_shared",
        chunk_dim=0,
        shape=(24, 4),
    ),
    _spec(
        "model.blocks.0.attn.qkv",
        chunk_count=2,
        roles=("qkv_a", "qkv_b"),
        role="attn_2d",
        shape=(24, 4),
    ),
]

PLAIN_SPECS = [
    _spec("model.lin_a", chunk_count=2, roles=("a", "b"), role="linear"),
]


class TestSessionComponent:
    @pytest.mark.parametrize(
        "bad",
        ["", "a/b", "a\\b", "..", "a..b", "x" * 121],
    )
    def test_illegal_session_ids_are_rejected(self, tmp_path: Path, bad: str) -> None:
        opt = _make_optimizer(PLAIN_SPECS)
        with pytest.raises(CMuonSafetyError, match="session_id"):
            _recorder(tmp_path, opt, identity=_identity(session_id=bad))

    def test_legal_session_id_accepted(self, tmp_path: Path) -> None:
        opt = _make_optimizer(PLAIN_SPECS)
        recorder = _recorder(
            tmp_path, opt, identity=_identity(session_id="r2-x86-2024")
        )
        assert not recorder.disabled

    def test_non_guarded_optimizer_is_a_config_error(self, tmp_path: Path) -> None:
        class NotGuarded:  # not a HybridCMuonGuardedCanonical
            pass

        with pytest.raises(CMuonSafetyError, match="guarded HybridCMuon"):
            _recorder(tmp_path, NotGuarded())


class TestClaim:
    def test_second_claim_same_session_rank_is_refused(self, tmp_path: Path) -> None:
        opt = _make_optimizer(PLAIN_SPECS)
        first = _recorder(tmp_path, opt, identity=_identity())
        assert not first.disabled
        # A second recorder for the same session+rank (simulating reuse) must
        # be disabled; it writes nothing and its hooks are no-ops.
        second = _recorder(tmp_path, _make_optimizer(PLAIN_SPECS), identity=_identity())
        assert second.disabled
        second.maybe_record(_observation(1))
        second.close("clean")
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        assert _windows(rank_dir) == []

    def test_same_session_different_rank_allowed(self, tmp_path: Path) -> None:
        opt0 = _make_optimizer(PLAIN_SPECS)
        opt1 = _make_optimizer(PLAIN_SPECS)
        r0 = _recorder(tmp_path, opt0, identity=_identity(rank=0))
        r1 = _recorder(tmp_path, opt1, identity=_identity(rank=1))
        assert not r0.disabled
        assert not r1.disabled

    def test_malformed_baseline_at_construction_disables_not_raises(
        self, tmp_path: Path
    ) -> None:
        opt = _make_optimizer(PLAIN_SPECS, observations=-5)
        recorder = _recorder(tmp_path, opt)
        assert recorder.disabled
        recorder.maybe_record(_observation(1))
        recorder.close("clean")
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        assert _windows(rank_dir) == []


class TestWindowLifecycle:
    def test_fresh_full_window_seals_one_valid_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opt = _make_optimizer(PLAIN_SPECS)
        recorder = _sealing_recorder(tmp_path, opt, monkeypatch, window_updates=2)
        recorder.maybe_record(_observation(1))
        opt.observations = 1
        opt.skip_by_role = {"linear": 2}
        recorder.maybe_record(_observation(2))
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        files = _windows(rank_dir)
        assert [f.name for f in files] == ["guard-window-u1-2.jsonl"]
        record = _read(files[0])
        assert record["schema_version"] == 1
        assert record["run_id"] == "run-1"
        assert record["session_id"] == "sess-1"
        assert record["rank"] == 0
        assert record["world_size"] == 2
        assert record["start_update"] == 1
        assert record["end_update"] == 2
        assert record["complete"] is True
        assert record["run_outcome"] is None
        assert record["window_id"] == "sess-1:u1-2:cfresh-start"
        # routing: full spec structure incl. shape, semantic roles AND role
        assert record["routing"] == [
            {
                "fqn": "model.lin_a",
                "shape": [8, 4],
                "chunk_count": 2,
                "chunk_dim": 0,
                "roles": ["a", "b"],
                "role": "linear",
            }
        ]
        # owner partition recomputed with stable_owner
        expected_partition: dict[str, list[str]] = {"0": [], "1": []}
        for chunk in range(2):
            owner = stable_owner("model.lin_a", chunk, 2)
            expected_partition[str(owner)].append(f"model.lin_a#{chunk}")
        assert record["owner_partition"] == {
            k: sorted(v) for k, v in expected_partition.items()
        }
        # launch identity
        launch = record["launch"]
        assert launch["resume_source_checkpoint"] == "fresh-start"
        assert launch["resume_successful_updates"] == 0
        assert launch["owner_mapping_version"] == OWNER_MAPPING_VERSION
        assert launch["last_published_checkpoint"] is None
        assert launch["launch_kind"] == "fresh_bootstrap"
        assert launch["optimizer"] == "hybrid_cmuon_guarded_canonical"
        # baseline source + counters
        assert record["baseline"]["source"] == "fresh"
        assert record["absolute"]["observations"] == 1
        assert record["delta"]["observations"] == 1
        assert record["delta_status"]["observations"] == "ok"
        # sparse known role normalized to zero in baseline, delta from 0
        assert record["baseline"]["counts"]["skip_by_role"] == {"linear": 0}
        assert record["delta"]["skip_by_role"] == {"linear": 2}
        # absolute_history: rank provenance for R fields
        assert record["absolute_history"]["observations"] == "rank_provenance"
        assert record["absolute_history"]["skip_by_fqn"] == "rank_provenance"

    def test_no_file_before_window_boundary(self, tmp_path: Path) -> None:
        opt = _make_optimizer(PLAIN_SPECS)
        recorder = _recorder(tmp_path, opt, window_updates=3)
        recorder.maybe_record(_observation(1))
        recorder.maybe_record(_observation(2))
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        assert _windows(rank_dir) == []

    def test_adaln_routing_preserves_semantic_and_canonical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opt = _make_optimizer(ADALN_SPECS)
        recorder = _sealing_recorder(tmp_path, opt, monkeypatch, window_updates=1)
        opt.observations = 1
        recorder.maybe_record(_observation(1))
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        record = _read(_windows(rank_dir)[0])
        routing: dict[str, Any] = {entry["fqn"]: entry for entry in record["routing"]}
        entry = routing["model.blocks.0.adaln_shared"]
        assert entry["shape"] == [24, 4]
        assert entry["chunk_count"] == 6
        assert entry["roles"] == [
            "gamma_in",
            "shift_in",
            "scale_in",
            "gate_in",
            "gamma_mod",
            "shift_mod",
        ]
        assert entry["role"] == "adaln_shared"
        assert entry["chunk_dim"] == 0
        # sparse role map domain is the CANONICAL role only
        assert record["baseline"]["counts"]["skip_by_role"] == {
            "adaln_shared": 0,
            "attn_2d": 0,
        }


class TestResumeIdentity:
    def test_resumed_stream_carries_rank0_provenance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the rescue counter surface belongs to the REAL rescue subclass;
        # the variant label is bound from the type, not from field presence
        opt = _make_optimizer(
            PLAIN_SPECS,
            rescue=True,
            bf16_attempts=10,
            bf16_safety_failures=1,
            fp32_attempts=3,
            fp32_rescues=2,
            fp32_rescue_failures=0,
            rescue_by_role={"linear": 2},
        )
        identity = _identity(resume_source="checkpoint-step-1000", resume_updates=1000)
        recorder = _sealing_recorder(
            tmp_path, opt, monkeypatch, window_updates=1, identity=identity
        )
        opt.observations = 1
        recorder.maybe_record(_observation(1001))
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        record = _read(_windows(rank_dir)[0])
        assert record["start_update"] == 1001
        assert record["end_update"] == 1001
        assert record["complete"] is True
        assert record["baseline"]["source"] == "restored_rank0"
        assert record["launch"]["resume_source_checkpoint"] == "checkpoint-step-1000"
        assert record["launch"]["resume_successful_updates"] == 1000
        assert record["launch"]["launch_kind"] == "user_resume"
        assert record["launch"]["optimizer"] == "hybrid_cmuon_canonical_ns4_fp32_rescue"
        # rescue absolute values are rank-0-owned after a real restore
        assert record["absolute"]["fp32_rescues"] == 2
        assert record["delta"]["fp32_rescues"] == 0
        assert record["absolute_history"]["fp32_rescues"] == "rank0_owner_only"
        assert record["absolute_history"]["rescue_by_role"] == "rank0_owner_only"
        assert record["absolute_history"]["observations"] == "rank_provenance"

    def test_last_published_checkpoint_reflected_at_seal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opt = _make_optimizer(PLAIN_SPECS)
        published: dict[str, str | None] = {"last": None}
        recorder = _sealing_recorder(
            tmp_path,
            opt,
            monkeypatch,
            window_updates=1,
            last_published=lambda: published["last"],
        )
        opt.observations = 1
        recorder.maybe_record(_observation(1))
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        record = _read(_windows(rank_dir)[0])
        assert record["launch"]["last_published_checkpoint"] is None
        # a second window after a periodic checkpoint save sees the new name
        published["last"] = "checkpoint-step-1001"
        opt.observations = 2
        recorder.maybe_record(_observation(2))
        record2 = _read(_windows(rank_dir)[1])
        assert record2["launch"]["last_published_checkpoint"] == "checkpoint-step-1001"


class TestTail:
    def test_clean_tail_writes_incomplete_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opt = _make_optimizer(PLAIN_SPECS)
        recorder = _sealing_recorder(tmp_path, opt, monkeypatch, window_updates=3)
        opt.observations = 5
        recorder.maybe_record(_observation(1))
        recorder.maybe_record(_observation(2))
        recorder.close("clean")
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        files = _windows(rank_dir)
        assert [f.name for f in files] == ["guard-window-u1-2.jsonl"]
        record = _read(files[0])
        assert record["complete"] is False
        assert record["run_outcome"] == "clean"
        assert record["absolute"]["observations"] == 5

    def test_no_tail_when_nothing_unsealed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opt = _make_optimizer(PLAIN_SPECS)
        recorder = _sealing_recorder(tmp_path, opt, monkeypatch, window_updates=2)
        opt.observations = 2
        recorder.maybe_record(_observation(1))
        recorder.maybe_record(_observation(2))  # full window sealed
        recorder.close("clean")
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        files = _windows(rank_dir)
        assert [f.name for f in files] == ["guard-window-u1-2.jsonl"]
        assert _read(files[0])["run_outcome"] is None

    def test_raised_tail(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        opt = _make_optimizer(PLAIN_SPECS)
        recorder = _sealing_recorder(tmp_path, opt, monkeypatch, window_updates=3)
        opt.observations = 1
        recorder.maybe_record(_observation(1))
        recorder.close("raised")
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        record = _read(_windows(rank_dir)[0])
        assert record["run_outcome"] == "raised"
        assert record["complete"] is False

    def test_invalid_outcome_normalized_to_unknown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opt = _make_optimizer(PLAIN_SPECS)
        recorder = _sealing_recorder(tmp_path, opt, monkeypatch, window_updates=3)
        opt.observations = 1
        recorder.maybe_record(_observation(1))
        recorder.close("definitely-not-an-outcome")
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        record = _read(_windows(rank_dir)[0])
        assert record["run_outcome"] == "unknown"


class TestFailureDisables:
    """Root ruling: any fault latches the session DISABLED (no recovery)."""

    def test_illegal_role_key_disables_session(self, tmp_path: Path) -> None:
        opt = _make_optimizer(PLAIN_SPECS)
        recorder = _recorder(tmp_path, opt, window_updates=2)
        opt.observations = 1
        opt.skip_by_role = {"bogus_role": 1}  # illegal key
        recorder.maybe_record(_observation(1))  # must NOT raise
        assert recorder.disabled
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        assert _windows(rank_dir) == []
        # subsequent updates and close write nothing (no per-window recovery)
        opt.skip_by_role = {"linear": 1}
        opt.observations = 2
        recorder.maybe_record(_observation(2))
        recorder.maybe_record(_observation(3))
        recorder.close("clean")
        assert _windows(rank_dir) == []

    def test_publish_failure_disables_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sakuramoon.train.guard_window_stream as stream

        opt = _make_optimizer(PLAIN_SPECS)
        recorder = _sealing_recorder(tmp_path, opt, monkeypatch, window_updates=1)
        calls = {"n": 0}

        def boom(target: Path, temp_dir: Path, prepare: Any) -> Any:
            calls["n"] += 1
            raise OSError(28, "disk full")

        monkeypatch.setattr(stream, "publish_atomically", boom)  # type: ignore[arg-type]
        opt.observations = 1
        recorder.maybe_record(_observation(1))  # seal fails -> disabled
        assert recorder.disabled
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        assert _windows(rank_dir) == []
        recorder.maybe_record(_observation(2))  # no-op after the latch
        recorder.maybe_record(_observation(3))
        recorder.close("clean")
        assert _windows(rank_dir) == []
        assert calls["n"] == 1  # no further publish attempts

    @pytest.mark.skipif(sys.platform != "linux", reason="real POSIX sink")
    def test_write_fault_releases_fd_preserves_primary_and_latches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import os

        import sakuramoon.train.guard_window_stream as stream
        from sakuramoon.telemetry.metrics import DurableJsonlSink

        opt = _make_optimizer(PLAIN_SPECS)
        recorder = _recorder(tmp_path, opt, window_updates=1)
        original = OSError(5, "injected write failure")
        descriptors: list[int] = []
        errors: list[BaseException | None] = []
        writes: list[int] = []

        def fail_write(sink: DurableJsonlSink, record: object) -> None:
            descriptors.append(sink._fd)  # pyright: ignore[reportPrivateUsage]
            writes.append(1)
            raise original

        def warning(message: str, error: BaseException | None = None) -> None:
            errors.append(error)

        monkeypatch.setattr(DurableJsonlSink, "write", fail_write)
        monkeypatch.setattr(stream, "_stderr", warning)
        recorder.maybe_record(_observation(1))
        assert recorder.disabled
        assert errors == [original]
        assert len(descriptors) == 1
        with pytest.raises(OSError) as closed:
            os.fstat(descriptors[0])
        assert closed.value.errno == 9  # EBADF
        recorder.maybe_record(_observation(2))
        recorder.close("raised")
        assert writes == [1]
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        assert _windows(rank_dir) == []
        assert list((rank_dir / ".tmp").iterdir()) == []

    def test_duplicate_window_target_disables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sakuramoon.train.guard_window_stream as stream
        from sakuramoon.telemetry.guard_publish import (
            PublicationResult,
            PublicationStatus,
        )

        opt = _make_optimizer(PLAIN_SPECS)
        recorder = _sealing_recorder(tmp_path, opt, monkeypatch, window_updates=1)

        def exists(target: Path, temp_dir: Path, prepare: Any) -> Any:
            return PublicationResult(
                status=PublicationStatus.TARGET_EXISTS,
                target=target,
                temp=None,
                warnings=(),
            )

        monkeypatch.setattr(stream, "publish_atomically", exists)  # type: ignore[arg-type]
        opt.observations = 1
        recorder.maybe_record(_observation(1))
        assert recorder.disabled
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        assert _windows(rank_dir) == []
        recorder.close("clean")
        assert _windows(rank_dir) == []

    def test_mid_window_rollback_disables_even_if_it_recovers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Root's MIDWINDOW_ROLLBACK counterexample.

        cadence 3, baseline skip 0: (u1, skip=5) -> (u2, skip=2 rollback)
        -> (u3, skip=6 recovery).  The rollback at u2 latches the session;
        the later recovery must NOT make the window complete, and no
        end-baseline delta may mask the middle.
        """
        opt = _make_optimizer(PLAIN_SPECS)
        recorder = _sealing_recorder(tmp_path, opt, monkeypatch, window_updates=3)
        opt.observations = 1
        opt.skip_by_role = {"linear": 5}
        recorder.maybe_record(_observation(1))
        assert not recorder.disabled
        opt.skip_by_role = {"linear": 2}  # mid-window rollback
        recorder.maybe_record(_observation(2))
        assert recorder.disabled
        # recovery cannot restore completeness
        opt.skip_by_role = {"linear": 6}
        recorder.maybe_record(_observation(3))
        recorder.close("clean")
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        assert _windows(rank_dir) == []

    def test_role_and_fqn_map_key_decrease_disables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A per-key map decrease (other keys still rising) also disables."""
        opt = _make_optimizer(PLAIN_SPECS)
        recorder = _sealing_recorder(tmp_path, opt, monkeypatch, window_updates=3)
        opt.observations = 1
        opt.skip_by_role = {"linear": 4}
        opt.skip_by_fqn = {"model.lin_a": 4}
        recorder.maybe_record(_observation(1))
        opt.skip_by_role = {"linear": 5}  # role key rises...
        opt.skip_by_fqn = {"model.lin_a": 1}  # ...but the FQN key falls
        recorder.maybe_record(_observation(2))
        assert recorder.disabled
        recorder.close("clean")
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        assert _windows(rank_dir) == []

    def test_update_duplicate_rewind_skip_disable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Duplicate, rewind, or skipped update numbers disable the session
        (no silent defensive return that later keeps sealing)."""
        for bad in ("duplicate", "rewind", "skip"):
            tmp = tmp_path / bad
            opt = _make_optimizer(PLAIN_SPECS)
            rec = _sealing_recorder(tmp, opt, monkeypatch, window_updates=3)
            opt.observations = 1
            if bad == "duplicate":
                rec.maybe_record(_observation(1))
                rec.maybe_record(_observation(1))
            elif bad == "rewind":
                rec.maybe_record(_observation(2))  # first observed != start
            else:
                rec.maybe_record(_observation(1))
                rec.maybe_record(_observation(3))  # 2 skipped
            assert rec.disabled, bad
            rec.close("clean")
            rank_dir = tmp / "guard_window" / "session-sess-1" / "rank-0"
            assert _windows(rank_dir) == [], bad


class TestVariantBinding:
    def test_base_type_with_o_attributes_is_not_labeled_rescue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Type binding: a base-class fake carrying O fields stays base.

        A fixture may never attach rescue fields to a base instance and
        claim the rescue variant; production validation is not loosened.
        """
        opt = _make_optimizer(
            PLAIN_SPECS,
            bf16_attempts=1,
            bf16_safety_failures=0,
            fp32_attempts=0,
            fp32_rescues=0,
            fp32_rescue_failures=0,
            rescue_by_role={},
        )
        recorder = _sealing_recorder(tmp_path, opt, monkeypatch, window_updates=1)
        opt.observations = 1
        recorder.maybe_record(_observation(1))
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        record = _read(_windows(rank_dir)[0])
        assert record["launch"]["optimizer"] == "hybrid_cmuon_guarded_canonical"
        # base variant: rescue fields are absent, never faked as zeros
        assert "fp32_rescues" not in record["absolute"]
        assert "fp32_rescues" not in record["delta"]


class TestSnapshotSurface:
    def test_snapshot_never_calls_state_dict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opt = _make_optimizer(PLAIN_SPECS)

        def forbidden():
            raise AssertionError("state_dict must never be called by the stream")

        opt.state_dict = forbidden  # type: ignore[method-assign]
        recorder = _sealing_recorder(tmp_path, opt, monkeypatch, window_updates=1)
        opt.observations = 1
        recorder.maybe_record(_observation(1))
        recorder.close("clean")
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        assert len(_windows(rank_dir)) == 1

    def test_negative_counter_disables_not_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opt = _make_optimizer(PLAIN_SPECS)
        recorder = _sealing_recorder(tmp_path, opt, monkeypatch, window_updates=1)
        opt.observations = -1
        recorder.maybe_record(_observation(1))  # must not raise
        assert recorder.disabled
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        assert _windows(rank_dir) == []

    def test_non_dict_map_disables_not_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opt = _make_optimizer(PLAIN_SPECS)
        recorder = _sealing_recorder(tmp_path, opt, monkeypatch, window_updates=1)
        opt.skip_by_role = ["not", "a", "dict"]  # type: ignore[assignment]
        recorder.maybe_record(_observation(1))  # must not raise
        assert recorder.disabled
        rank_dir = tmp_path / "guard_window" / "session-sess-1" / "rank-0"
        assert _windows(rank_dir) == []
