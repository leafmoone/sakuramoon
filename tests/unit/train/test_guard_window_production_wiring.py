"""Unit tests for the production-side guard window construction/close guards.

Covers the fail-closed / disable split of the production wiring helpers:
- ``_construct_guard_window``: config-level errors propagate; every other
  fault (including a broken stderr) disables the stream WITHOUT aborting
  or masking the training launch;
- ``_close_guard_window``: a plain Exception from the tail close is
  swallowed on BOTH the clean and the in-flight-original-exception paths,
  never replacing the original exception identity; a BaseException is not
  caught (existing interpreter/group rules, no new group framework).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import pytest

from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
from sakuramoon.train.guard_window_stream import GuardWindowRecorder
from sakuramoon.train.production import (
    # Wiring tests target the real private helpers (intentional; no
    # production-side Any relaxation to satisfy the tests):
    _build_guard_window,  # pyright: ignore[reportPrivateUsage]
    _close_guard_window,  # pyright: ignore[reportPrivateUsage]
    _construct_guard_window,  # pyright: ignore[reportPrivateUsage]
)


def _raise_oserror(**kwargs: object) -> NoReturn:
    raise OSError(5, "access denied")


def _raise_config_error(**kwargs: object) -> NoReturn:
    raise CMuonSafetyError("unsupported optimizer")


class _BrokenStderr:
    """A stderr whose write/flush always fail (simulated torn pipe)."""

    def write(self, text: str) -> int:
        raise OSError(32, "broken pipe (simulated)")

    def flush(self) -> None:
        raise OSError(32, "broken pipe (simulated)")


class _CloseOSErrorRecorder:
    """Recorder stub whose tail close raises a plain OSError."""

    def __init__(self) -> None:
        self.closed_with: list[str] = []

    def close(self, outcome: str) -> None:
        self.closed_with.append(outcome)
        raise OSError(5, "tail close IO failure (simulated)")


class _CloseKeyboardInterruptRecorder:
    def close(self, outcome: str) -> None:
        raise KeyboardInterrupt


class TestConstructGuard:
    def test_ctor_oserror_with_broken_stderr_disables_not_aborts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ctor OSError + broken stderr -> recorder None, training intact."""
        monkeypatch.setattr(sys, "stderr", _BrokenStderr())
        result = _construct_guard_window(
            build=cast("Any", _raise_oserror), artifact_dir=tmp_path
        )
        assert result is None

    def test_ctor_config_error_propagates_even_with_broken_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "stderr", _BrokenStderr())
        with pytest.raises(CMuonSafetyError, match="unsupported optimizer"):
            _construct_guard_window(
                build=cast("Any", _raise_config_error), artifact_dir=tmp_path
            )

    def test_success_path_returns_recorder_unchanged(self, tmp_path: Path) -> None:
        sentinel = object()

        def build(**kwargs: object) -> GuardWindowRecorder:
            return cast("GuardWindowRecorder", sentinel)

        result = _construct_guard_window(build=build, artifact_dir=tmp_path)
        assert result is sentinel

    def test_plain_oserror_alone_also_disables(self, tmp_path: Path) -> None:
        def boom(**kwargs: object) -> NoReturn:
            raise OSError(28, "disk full")

        assert (
            _construct_guard_window(build=cast("Any", boom), artifact_dir=tmp_path)
            is None
        )


class TestCloseGuard:
    def test_clean_path_close_oserror_does_not_fail_training(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _CloseOSErrorRecorder()
        monkeypatch.setattr(sys, "stderr", _BrokenStderr())
        # must not raise on the clean path
        _close_guard_window(cast("Any", recorder), "clean")
        assert recorder.closed_with == ["clean"]

    def test_inflight_original_exception_identity_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _CloseOSErrorRecorder()
        monkeypatch.setattr(sys, "stderr", _BrokenStderr())
        original = ValueError("training failure (primary)")
        with pytest.raises(ValueError) as excinfo:
            try:
                raise original
            finally:
                _close_guard_window(cast("Any", recorder), "raised")
        assert excinfo.value is original

    def test_none_recorder_is_a_noop(self) -> None:
        _close_guard_window(None, "clean")

    def test_keyboard_interrupt_from_close_is_not_caught(self) -> None:
        recorder = _CloseKeyboardInterruptRecorder()
        with pytest.raises(KeyboardInterrupt):
            _close_guard_window(cast("Any", recorder), "clean")

    def test_broken_stderr_does_not_mask_clean_outcome(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # even with a healthy close the guard must not raise
        class _Ok:
            def close(self, outcome: str) -> None:
                pass

        monkeypatch.setattr(sys, "stderr", _BrokenStderr())
        _close_guard_window(cast("Any", _Ok()), "clean")

    def test_exception_group_identity_preserved_when_close_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A runtime stream-close ExceptionGroup must survive the guarded
        close: the original group object is the one that propagates."""
        recorder = _CloseOSErrorRecorder()
        monkeypatch.setattr(sys, "stderr", _BrokenStderr())
        members = [ValueError("primary"), OSError(9, "close fault")]
        original = BaseExceptionGroup("training failure and close failure", members)
        with pytest.raises(BaseExceptionGroup) as excinfo:
            try:
                raise original
            finally:
                _close_guard_window(cast("Any", recorder), "raised")
        assert excinfo.value is original

    def test_system_exit_identity_preserved_when_close_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _CloseOSErrorRecorder()
        monkeypatch.setattr(sys, "stderr", _BrokenStderr())
        original = SystemExit(3)
        with pytest.raises(SystemExit) as excinfo:
            try:
                raise original
            finally:
                _close_guard_window(cast("Any", recorder), "raised")
        assert excinfo.value is original


class TestBootstrapSource:
    """Production fresh start is bootstrap save+restore: the identity must
    carry the EXPLICIT launch_kind and the real bootstrap checkpoint name,
    never a hand-waved "fresh-start" inferred from object presence."""

    def test_build_uses_explicit_launch_kind_and_bootstrap_name(
        self, tmp_path: Path
    ) -> None:
        from sakuramoon.optim.guarded_canonical import HybridCMuonGuardedCanonical

        class _FakeSpec:
            name = "model.lin_a"
            parameter = SimpleNamespace(shape=(8, 4))
            chunk_count = 2
            chunk_dim = 0
            roles = ("a", "b")
            role = "linear"

        class _FakeOptimizer(HybridCMuonGuardedCanonical):
            def __init__(self) -> None:
                self.routing = SimpleNamespace(cmuon_specs=[_FakeSpec()])
                self.observations = 0
                self.skip_total = 0
                self.skip_by_role: dict[str, int] = {}
                self.skip_by_fqn: dict[str, int] = {}

        recorder = _build_guard_window(
            artifact_root=tmp_path,
            run_id="run-1",
            session_id="sess-b",
            window_updates=2,
            optimizer=_FakeOptimizer(),
            rank=0,
            world_size=2,
            restored=cast("Any", _fake_restored("ckpt_0_bootstrap", 0)),
            verified_checkpoints=[],
            launch_kind="fresh_bootstrap",
        )
        identity = recorder._identity  # type: ignore[attr-defined]
        assert identity.launch_kind == "fresh_bootstrap"
        assert identity.resume_source_checkpoint == "ckpt_0_bootstrap"
        assert identity.resume_successful_updates == 0
        recorder.close("clean")

    def test_build_resume_launch_kind_is_explicit(self, tmp_path: Path) -> None:
        from sakuramoon.optim.guarded_canonical import HybridCMuonGuardedCanonical

        class _FakeSpec2:
            name = "model.lin_a"
            parameter = SimpleNamespace(shape=(8, 4))
            chunk_count = 2
            chunk_dim = 0
            roles = ("a", "b")
            role = "linear"

        class _FakeOptimizer2(HybridCMuonGuardedCanonical):
            def __init__(self) -> None:
                self.routing = SimpleNamespace(cmuon_specs=[_FakeSpec2()])
                self.observations = 5
                self.skip_total = 0
                self.skip_by_role: dict[str, int] = {}
                self.skip_by_fqn: dict[str, int] = {}

        recorder = _build_guard_window(
            artifact_root=tmp_path,
            run_id="run-1",
            session_id="sess-r",
            window_updates=2,
            optimizer=_FakeOptimizer2(),
            rank=0,
            world_size=2,
            restored=cast("Any", _fake_restored("checkpoint-step-1000", 1000)),
            verified_checkpoints=[],
            launch_kind="user_resume",
        )
        identity = recorder._identity  # type: ignore[attr-defined]
        assert identity.launch_kind == "user_resume"
        assert identity.resume_source_checkpoint == "checkpoint-step-1000"
        assert identity.resume_successful_updates == 1000
        recorder.close("clean")


def _fake_restored(name: str, successful_updates: int):
    return SimpleNamespace(
        path=Path(name),
        state=SimpleNamespace(
            trainer=SimpleNamespace(successful_updates=successful_updates)
        ),
    )
