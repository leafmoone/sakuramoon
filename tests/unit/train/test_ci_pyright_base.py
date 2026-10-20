"""Disposable git-graph regression tests for CI pyright base selection.

Each test builds a throwaway bare origin + working clone under ``tmp_path``
and drives :func:`select_pyright_base.select_base` with explicit event
fields, so no GitHub environment is required and no user git config is
touched.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "dev-tools" / "select_pyright_base.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("select_pyright_base", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


spb = cast(Any, _load_module())


class _Graph:
    """Minimal disposable origin/work graph with an explicit git env."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.env = {
            **os.environ,
            "HOME": str(root),
            "GIT_CONFIG_GLOBAL": str(root / "gitconfig"),
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        }
        (root / "gitconfig").write_text("", encoding="utf-8")
        self.origin = root / "origin.git"
        self.origin.mkdir()
        self.run(self.origin, "init", "--bare")
        self.work = root / "work"
        self.run(self.origin, "clone", str(self.origin), str(self.work))
        self.run(self.work, "config", "user.email", "ci@example.invalid")
        self.run(self.work, "config", "user.name", "ci-test")

    def run(self, repo: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
            env=self.env,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"git {' '.join(args)} failed: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        return result.stdout.strip()

    def rev(self, ref: str) -> str:
        return self.run(self.work, "rev-parse", ref)

    def branch(self, name: str, start_point: str | None) -> None:
        if start_point is None:
            self.run(self.work, "checkout", "-b", name)
        else:
            self.run(self.work, "checkout", "-b", name, start_point)

    def commit(self, path: str, content: str) -> str:
        target = self.work / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self.run(self.work, "add", "--", path)
        self.run(self.work, "commit", "-m", path)
        return self.rev("HEAD")

    def push(self, ref: str) -> None:
        self.run(self.work, "push", "origin", ref)

    def select(
        self,
        event: str,
        ref_name: str,
        base_ref: str,
        before: str,
    ) -> tuple[str, str]:
        return spb.select_base(self.work, event, ref_name, base_ref, before, "dev")


@pytest.fixture()
def graph(tmp_path: Path) -> _Graph:
    return _Graph(tmp_path)


def test_pull_request_uses_actual_target_merge_base(graph: _Graph) -> None:
    graph.branch("dev", None)
    base_tip = graph.commit("base.py", "a = 1\n")
    dev_tip = graph.commit("dev.py", "b = 2\n")
    graph.push("dev")
    graph.branch("feature", base_tip)
    head = graph.commit("feature.py", "c = 3\n")
    selected, source = graph.select("pull_request", "feature", "dev", "")
    assert source == "pr-target-merge-base"
    assert selected == base_tip
    assert selected != dev_tip
    assert selected != head


def test_pull_request_non_default_target(graph: _Graph) -> None:
    graph.branch("dev", None)
    graph.commit("dev.py", "a = 1\n")
    graph.push("dev")
    graph.branch("release", "HEAD")
    target_tip = graph.commit("release.py", "b = 2\n")
    graph.push("release")
    graph.branch("pr", target_tip)
    head = graph.commit("pr.py", "c = 3\n")
    selected, source = graph.select("pull_request", "pr", "release", "")
    assert source == "pr-target-merge-base"
    assert selected == target_tip
    assert selected != head


def test_feature_push_uses_dev_merge_base(graph: _Graph) -> None:
    graph.branch("dev", None)
    base_tip = graph.commit("base.py", "a = 1\n")
    graph.commit("dev.py", "b = 2\n")
    graph.push("dev")
    graph.branch("feature", base_tip)
    graph.commit("feature.py", "c = 3\n")
    selected, source = graph.select("push", "feature", "", "")
    assert source == "dev-merge-base"
    assert selected == base_tip


def test_dev_push_single_commit(graph: _Graph) -> None:
    graph.branch("dev", None)
    before = graph.commit("one.py", "a = 1\n")
    head = graph.commit("two.py", "b = 2\n")
    graph.push("dev")
    selected, source = graph.select("push", "dev", "", before)
    assert source == "push-before"
    assert selected == before
    assert selected != head


def test_dev_push_merge_covers_all_merged_commits(graph: _Graph) -> None:
    graph.branch("dev", None)
    before = graph.commit("base.py", "a = 1\n")
    graph.branch("feature", before)
    graph.commit("m1.py", "b = 2\n")
    graph.commit("m2.py", "c = 3\n")
    graph.run(graph.work, "checkout", "dev")
    graph.run(graph.work, "merge", "--no-ff", "feature", "-m", "merge")
    merge = graph.rev("HEAD")
    graph.push("dev")
    selected, source = graph.select("push", "dev", "", before)
    assert source == "push-before"
    assert selected == before
    diff = graph.run(
        graph.work, "diff", "--name-only", selected, merge, "--", "*.py"
    ).splitlines()
    assert sorted(diff) == ["m1.py", "m2.py"]


def test_dev_push_multi_commit(graph: _Graph) -> None:
    graph.branch("dev", None)
    before = graph.commit("one.py", "a = 1\n")
    graph.commit("two.py", "b = 2\n")
    head = graph.commit("three.py", "c = 3\n")
    graph.push("dev")
    selected, source = graph.select("push", "dev", "", before)
    assert source == "push-before"
    assert selected == before
    diff = graph.run(
        graph.work, "diff", "--name-only", selected, head, "--", "*.py"
    ).splitlines()
    assert sorted(diff) == ["three.py", "two.py"]


def test_python_no_change_selects_base_and_yields_empty_list(graph: _Graph) -> None:
    graph.branch("dev", None)
    before = graph.commit("one.py", "a = 1\n")
    head = graph.commit("notes.md", "# docs only\n")
    graph.push("dev")
    selected, source = graph.select("push", "dev", "", before)
    assert source == "push-before"
    assert selected == before
    diff = graph.run(
        graph.work, "diff", "--name-only", selected, head, "--", "*.py"
    )
    assert diff == ""


@pytest.mark.parametrize(
    ("event", "ref_name", "base_ref", "before_expr", "match"),
    [
        ("push", "dev", "", "0" * 40, "zero SHA"),
        ("push", "dev", "", "f" * 40, "cat-file"),
        ("push", "dev", "", "", "without GITHUB_BEFORE"),
        ("push", "dev", "", "HEAD", "equals HEAD"),
        ("push", "dev", "", "side", "not an ancestor"),
        ("push", "", "", "HEAD", "without GITHUB_REF_NAME"),
        ("workflow_dispatch", "dev", "", "", "unknown GITHUB_EVENT_NAME"),
        ("pull_request", "feature", "", "", "without GITHUB_BASE_REF"),
    ],
    ids=[
        "zero-sha",
        "unknown-sha",
        "missing-sha",
        "head-self",
        "non-ancestor",
        "push-without-branch",
        "unknown-event",
        "pr-without-target",
    ],
)
def test_fail_closed(
    graph: _Graph,
    event: str,
    ref_name: str,
    base_ref: str,
    before_expr: str,
    match: str,
) -> None:
    graph.branch("dev", None)
    tip = graph.commit("one.py", "a = 1\n")
    graph.push("dev")
    before = (
        tip if before_expr == "side" else
        graph.rev("HEAD") if before_expr == "HEAD" else before_expr
    )
    if before_expr == "side":
        graph.branch("side", tip)
        graph.commit("side.py", "s = 1\n")
        before = graph.rev("side")
        graph.run(graph.work, "checkout", "dev")
    with pytest.raises(spb.BaseSelectionError, match=match):
        graph.select(event, ref_name, base_ref, before)


def _cli_run(graph: _Graph, extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {
        "GITHUB_EVENT_NAME": "",
        "GITHUB_REF_NAME": "",
        "GITHUB_BASE_REF": "",
        "GITHUB_BEFORE": "",
        "GITHUB_EVENT_PATH": "",
        **graph.env,
        **extra,
    }
    return subprocess.run(
        [sys.executable, str(_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(graph.work),
        env=env,
    )


def test_cli_dev_push_reads_before_from_event_path(graph: _Graph, tmp_path: Path) -> None:
    graph.branch("dev", None)
    before = graph.commit("one.py", "a = 1\n")
    graph.commit("two.py", "b = 2\n")
    graph.push("dev")
    event_file = tmp_path / "push.json"
    event_file.write_text(json.dumps({"before": before}), encoding="utf-8")
    result = _cli_run(
        graph,
        {
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF_NAME": "dev",
            "GITHUB_EVENT_PATH": str(event_file),
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == before
    assert "source=push-before" in result.stderr


def test_cli_env_before_wins_over_event_path(graph: _Graph, tmp_path: Path) -> None:
    graph.branch("dev", None)
    first = graph.commit("one.py", "a = 1\n")
    second = graph.commit("two.py", "b = 2\n")
    graph.commit("three.py", "c = 3\n")
    graph.push("dev")
    event_file = tmp_path / "push.json"
    event_file.write_text(json.dumps({"before": first}), encoding="utf-8")
    result = _cli_run(
        graph,
        {
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF_NAME": "dev",
            "GITHUB_BEFORE": second,
            "GITHUB_EVENT_PATH": str(event_file),
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == second


def test_cli_dev_push_without_before_fails_closed(graph: _Graph) -> None:
    graph.branch("dev", None)
    graph.commit("one.py", "a = 1\n")
    graph.commit("two.py", "b = 2\n")
    graph.push("dev")
    result = _cli_run(
        graph,
        {"GITHUB_EVENT_NAME": "push", "GITHUB_REF_NAME": "dev"},
    )
    assert result.returncode == 1
    assert result.stdout.strip() == ""
    assert "without GITHUB_BEFORE" in result.stderr
