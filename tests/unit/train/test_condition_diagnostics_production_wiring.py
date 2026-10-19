"""Optional diagnostics follow Guard Window's outcome-preserving guards."""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import pytest

from sakuramoon.train import production
from sakuramoon.train.condition_diagnostics_stream import ConditionDiagnosticsRecorder
from sakuramoon.train.production import (
    _close_condition_diagnostics,  # pyright: ignore[reportPrivateUsage]
    _construct_condition_diagnostics,  # pyright: ignore[reportPrivateUsage]
)


class _BrokenStderr:
    def write(self, text: str) -> int:
        raise OSError(32, "broken pipe")

    def flush(self) -> None:
        raise OSError(32, "broken pipe")


class _BrokenClose:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls = 0

    def close(self) -> None:
        self.calls += 1
        raise self.error


def test_construct_io_fault_disables_with_broken_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build(**kwargs: object) -> NoReturn:
        raise OSError(28, "disk full")

    monkeypatch.setattr(sys, "stderr", _BrokenStderr())
    assert _construct_condition_diagnostics(build=build) is None


@pytest.mark.parametrize("error", [ValueError("bad config"), RuntimeError("code bug")])
def test_construct_non_io_error_propagates(error: Exception) -> None:
    def build(**kwargs: object) -> NoReturn:
        raise error

    with pytest.raises(type(error)) as caught:
        _construct_condition_diagnostics(build=build)
    assert caught.value is error


def test_construct_returns_same_recorder() -> None:
    sentinel = object()

    def build(**kwargs: object) -> ConditionDiagnosticsRecorder:
        return cast(ConditionDiagnosticsRecorder, sentinel)

    assert _construct_condition_diagnostics(build=build) is sentinel


@pytest.mark.parametrize("broken_stderr", [False, True])
@pytest.mark.parametrize("close_error", [OSError(5, "close I/O"), RuntimeError("close bug")])
def test_close_never_changes_clean_result(
    monkeypatch: pytest.MonkeyPatch, broken_stderr: bool, close_error: Exception,
) -> None:
    recorder = _BrokenClose(close_error)
    if broken_stderr:
        monkeypatch.setattr(sys, "stderr", _BrokenStderr())

    def train() -> object:
        try:
            return result
        finally:
            _close_condition_diagnostics(cast(Any, recorder))

    result = object()
    assert train() is result
    assert recorder.calls == 1


@pytest.mark.parametrize("primary", [
    ValueError("loss failure"),
    OSError(5, "main-path I/O"),
    RuntimeError("optimizer failure"),
    ExceptionGroup("training group", [ValueError("model"), RuntimeError("loss")]),
    SystemExit(3),
    KeyboardInterrupt(),
])
def test_close_preserves_original_training_exception_object(
    monkeypatch: pytest.MonkeyPatch, primary: BaseException,
) -> None:
    recorder = _BrokenClose(OSError(5, "diagnostic close fault"))
    monkeypatch.setattr(sys, "stderr", _BrokenStderr())
    with pytest.raises(type(primary)) as caught:
        try:
            raise primary
        finally:
            _close_condition_diagnostics(cast(Any, recorder))
    assert caught.value is primary
    assert recorder.calls == 1


def test_close_baseexception_is_not_swallowed() -> None:
    error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        _close_condition_diagnostics(cast(Any, _BrokenClose(error)))
    assert caught.value is error


def test_close_none_is_noop() -> None:
    _close_condition_diagnostics(None)


def test_real_production_try_cleans_condition_recorder_on_guard_ctor_failure() -> None:
    # Execute production's ACTUAL try/finally, failing before model/runtime
    # entry.  No copied lifecycle and no production/model assets required.
    tree = ast.parse(inspect.getsource(production))
    candidates = [
        node for node in ast.walk(tree) if isinstance(node, ast.Try)
        and any(
            isinstance(statement, ast.If)
            and any(
                isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "_construct_guard_window"
                for call in ast.walk(statement)
            )
            for statement in node.body
        )
    ]
    # An enclosing production try also contains the call transitively;
    # select the innermost resource-owning try rather than copying it.
    assert candidates
    lifecycle = min(candidates, key=lambda node: len(list(ast.walk(node))))
    recorder = _BrokenClose(OSError(5, "condition close fault"))
    original = RuntimeError("guard construction failed")

    def fail_guard(**kwargs: object) -> NoReturn:
        raise original

    scope: dict[str, Any] = {
        "gw_config": SimpleNamespace(enabled=True, session_id="test", window_updates=2),
        "in_calibration": False,
        "_construct_guard_window": fail_guard,
        "_build_guard_window": object(),
        "artifact_root": Path("unused"),
        "config": SimpleNamespace(run=SimpleNamespace(run_id="test")),
        "optimizer": object(), "rank": 0, "world_size": 1,
        "restored": None, "verified_checkpoints": [], "resume": None,
        "guard_window_recorder": None, "guard_window_run_outcome": "unknown",
        "condition_diagnostics_recorder": recorder,
        "_close_guard_window": production._close_guard_window,  # pyright: ignore[reportPrivateUsage]
        "_close_condition_diagnostics": _close_condition_diagnostics,
    }
    fragment = ast.fix_missing_locations(ast.Module(body=[lifecycle], type_ignores=[]))
    with pytest.raises(RuntimeError) as caught:
        exec(compile(fragment, "<production lifecycle>", "exec"), scope)  # noqa: S102 - trusted repository AST only
    assert caught.value is original
    assert recorder.calls == 1
    assert scope["guard_window_run_outcome"] == "raised"
