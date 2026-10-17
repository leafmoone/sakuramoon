"""Tests for the structural config inheritance checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

SCRIPT = Path(__file__).parents[3] / "scripts" / "check_configs.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_configs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = cast(Any, _load_module())


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_resolves_nested_extends(tmp_path: Path) -> None:
    _write(tmp_path / "base.toml", "[run]\nrun_id = 'base'\n")
    _write(tmp_path / "child.toml", "extends = ['base.toml']\n")
    _write(tmp_path / "grand.toml", "extends = ['child.toml']\n")

    report = checker.check(tmp_path)

    assert report["ok"] is True
    assert report["files"] == 3
    assert report["missing"] == []
    assert report["cycles"] == []


def test_missing_parent_fails_unless_declared_external(tmp_path: Path) -> None:
    _write(tmp_path / "child.toml", "extends = ['absent.toml']\n")

    report = checker.check(tmp_path)

    assert report["ok"] is False
    assert report["missing"][0]["entry"] == "absent.toml"

    _write(
        tmp_path / "external-parents.toml",
        "[external]\n\"absent.toml\" = 'deployment-only'\n",
    )
    declared = checker.check(tmp_path)

    assert declared["ok"] is True
    assert declared["missing"] == []
    assert declared["external"][0]["status"] == "external"


def test_unquoted_external_key_is_reported(tmp_path: Path) -> None:
    _write(tmp_path / "child.toml", "extends = ['absent.toml']\n")
    _write(
        tmp_path / "external-parents.toml",
        "[external]\nabsent.toml = 'deployment-only'\n",
    )

    report = checker.check(tmp_path)

    assert report["ok"] is False
    assert "quoted" in report["errors"][0]["error"]


def test_detects_cycles(tmp_path: Path) -> None:
    _write(tmp_path / "a.toml", "extends = ['b.toml']\n")
    _write(tmp_path / "b.toml", "extends = ['a.toml']\n")

    report = checker.check(tmp_path)

    assert report["ok"] is False
    assert report["cycles"]
    assert "a.toml" in report["cycles"][0]


def test_reports_duplicate_and_invalid_extends(tmp_path: Path) -> None:
    _write(tmp_path / "base.toml", "")
    _write(tmp_path / "dupe.toml", "extends = ['base.toml', 'base.toml']\n")
    _write(tmp_path / "bad.toml", "extends = 'base.toml'\n")

    report = checker.check(tmp_path)

    assert report["ok"] is False
    messages = [item["error"] for item in report["errors"]]
    assert any("duplicate" in message for message in messages)
    assert any("array of relative" in message for message in messages)


def test_reports_invalid_toml(tmp_path: Path) -> None:
    _write(tmp_path / "broken.toml", "extends = [\n")

    report = checker.check(tmp_path)

    assert report["ok"] is False
    assert report["errors"][0]["path"].endswith("broken.toml")


def test_external_keys_are_root_relative(tmp_path: Path) -> None:
    _write(tmp_path / "sub" / "child.toml", "extends = ['base.toml']\n")
    _write(
        tmp_path / "external-parents.toml",
        "[external]\n\"base.toml\" = 'root-only'\n",
    )

    root_key = checker.check(tmp_path)
    assert root_key["ok"] is False
    assert root_key["missing"][0]["target"] == "sub/base.toml"

    _write(
        tmp_path / "external-parents.toml",
        "[external]\n\"sub/base.toml\" = 'deployment-only'\n",
    )
    nested_key = checker.check(tmp_path)
    assert nested_key["ok"] is True
    assert nested_key["external"][0]["target"] == "sub/base.toml"


def test_absolute_extends_is_invalid(tmp_path: Path) -> None:
    absolute = (tmp_path / "outside.toml").as_posix()
    _write(tmp_path / "child.toml", f"extends = ['{absolute}']\n")

    report = checker.check(tmp_path)

    assert report["ok"] is False
    assert report["invalid"][0]["entry"] == absolute
    assert any("absolute" in item["error"] for item in report["errors"])


def test_repository_configs_resolve(tmp_path: Path) -> None:
    """The checked-in config tree must pass once external parents are declared."""

    repository_config = Path(__file__).parents[3] / "config"
    if not repository_config.is_dir():
        pytest.skip("config directory is unavailable")

    report = checker.check(repository_config)

    assert report["ok"] is True, report["missing"] + report["errors"]
    assert any(
        item["entry"] == "train_g1_cmuon_production_camera_p100.toml"
        for item in report["external"]
    )
