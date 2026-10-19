"""Contract tests for config-root closure and exact external parent declarations."""

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


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")


def _guard_reads(
    monkeypatch: pytest.MonkeyPatch, root: Path, *, opaque: Path | None = None
) -> None:
    """Assert the checker never opens outside or declared-external TOML."""
    original = Path.read_text

    def read_local(
        path: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        canonical = path.resolve()
        assert canonical.is_relative_to(root.resolve()), f"outside read: {path}"
        assert canonical != opaque, f"external read: {path}"
        return original(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", read_local)


def test_resolves_nested_extends(tmp_path: Path) -> None:
    _write(tmp_path / "base.toml", "[run]\nrun_id = 'base'\n")
    _write(tmp_path / "child.toml", "extends = ['base.toml']\n")
    _write(tmp_path / "grand.toml", "extends = ['child.toml']\n")

    report = checker.check(tmp_path)

    assert report["ok"] is True
    assert report["files"] == 3
    assert report["missing"] == []
    assert report["cycles"] == []


def test_parent_traversal_within_root_is_allowed(tmp_path: Path) -> None:
    """Like runtime loading, relative entries are based on the including file."""
    _write(tmp_path / "base.toml", "")
    _write(tmp_path / "sub" / "child.toml", "extends = ['../base.toml']\n")

    report = checker.check(tmp_path)

    assert report["ok"] is True
    assert report["references"] == [
        {
            "source": "sub/child.toml",
            "entry": "../base.toml",
            "target": "base.toml",
            "status": "resolved",
        }
    ]


@pytest.mark.parametrize("exists", [False, True])
def test_undeclared_traversal_outside_root_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exists: bool
) -> None:
    root = tmp_path / "config"
    outside = tmp_path / "outside.toml"
    _write(root / "child.toml", "extends = ['../outside.toml']\n")
    if exists:
        _write(outside, "not valid TOML [\n")
    _guard_reads(monkeypatch, root)

    report = checker.check(root)

    assert report["ok"] is False
    assert report["invalid"] == [
        {
            "source": "child.toml",
            "entry": "../outside.toml",
            "target": outside.as_posix(),
            "status": "invalid",
        }
    ]
    assert report["missing"] == []
    assert report["external"] == []
    assert "escapes the config root" in report["errors"][0]["error"]


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


@pytest.mark.parametrize("exists", [False, True])
def test_explicit_outside_parent_is_an_opaque_external_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exists: bool
) -> None:
    """A root-relative manifest key with '..' grants one canonical target."""
    root = tmp_path / "config"
    outside = tmp_path / "outside.toml"
    _write(root / "sub" / "child.toml", "extends = ['../../outside.toml']\n")
    _write(
        root / "external-parents.toml",
        "[external]\n'../outside.toml' = 'deployment-only'\n",
    )
    if exists:
        _write(outside, "not valid TOML [\n")
    _guard_reads(monkeypatch, root)

    report = checker.check(root)

    assert report["ok"] is True
    assert report["files"] == 1
    assert report["errors"] == []
    assert report["external"][0]["target"] == outside.as_posix()
    assert report["external"][0]["source"] == "sub/child.toml"
    assert report["cycles"] == []


def test_existing_local_external_target_is_not_read_or_traversed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """External declarations take precedence over existence, also inside root."""
    parent = tmp_path / "parent.toml"
    _write(tmp_path / "child.toml", "extends = ['parent.toml']\n")
    _write(parent, "extends = ['child.toml']\n")
    _write(
        tmp_path / "external-parents.toml",
        "[external]\n'parent.toml' = 'opaque boundary'\n",
    )
    _guard_reads(monkeypatch, tmp_path, opaque=parent)

    report = checker.check(tmp_path)

    assert report["ok"] is True
    assert report["files"] == 1
    assert report["external"][0]["target"] == "parent.toml"
    assert report["cycles"] == []


@pytest.mark.parametrize("key", ["../outside/parent.toml", "../outside/*.toml"])
def test_declaration_does_not_authorize_siblings_or_expand_globs(
    tmp_path: Path, key: str
) -> None:
    root = tmp_path / "config"
    _write(root / "child.toml", "extends = ['../outside/other.toml']\n")
    _write(tmp_path / "outside" / "other.toml", "")
    _write(
        root / "external-parents.toml", f"[external]\n'{key}' = 'one literal target'\n"
    )

    report = checker.check(root)

    assert report["ok"] is False
    assert report["invalid"][0]["entry"] == "../outside/other.toml"
    assert report["external"] == []


def test_unquoted_external_key_is_reported(tmp_path: Path) -> None:
    _write(tmp_path / "child.toml", "extends = ['absent.toml']\n")
    _write(
        tmp_path / "external-parents.toml",
        "[external]\nabsent.toml = 'deployment-only'\n",
    )

    report = checker.check(tmp_path)

    assert report["ok"] is False
    assert "quoted" in report["errors"][0]["error"]


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ("[external\n", ""),
        ("external = []\n", "must be a table"),
        ("[external]\n'parent.toml' = 3\n", "must be strings"),
        ("[external]\n'../outside' = 'directory'\n", "individual TOML files"),
        ("[external]\n'' = 'root'\n", "individual TOML files"),
    ],
)
def test_invalid_external_manifest_is_reported(
    tmp_path: Path, manifest: str, message: str
) -> None:
    _write(tmp_path / "external-parents.toml", manifest)

    report = checker.check(tmp_path)

    assert report["ok"] is False
    assert report["files"] == 0
    assert report["errors"][0]["path"].endswith("external-parents.toml")
    assert message in report["errors"][0]["error"]


def test_external_manifest_rejects_absolute_keys(tmp_path: Path) -> None:
    absolute = (tmp_path / "parent.toml").as_posix()
    _write(
        tmp_path / "external-parents.toml", f"[external]\n'{absolute}' = 'absolute'\n"
    )

    report = checker.check(tmp_path)

    assert report["ok"] is False
    assert "root-relative" in report["errors"][0]["error"]


def test_external_manifest_rejects_existing_directory_target(tmp_path: Path) -> None:
    root = tmp_path / "config"
    (tmp_path / "outside.toml").mkdir()
    _write(
        root / "external-parents.toml",
        "[external]\n'../outside.toml' = 'not a file'\n",
    )

    report = checker.check(root)

    assert report["ok"] is False
    assert "individual TOML files" in report["errors"][0]["error"]


def test_detects_cycles(tmp_path: Path) -> None:
    _write(tmp_path / "a.toml", "extends = ['b.toml']\n")
    _write(tmp_path / "b.toml", "extends = ['a.toml']\n")

    report = checker.check(tmp_path)

    assert report["ok"] is False
    assert report["cycles"]
    assert "a.toml" in report["cycles"][0]


@pytest.mark.parametrize("kind", ["file", "directory", "directory-dotdot"])
def test_symlink_escape_is_invalid_without_reading_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    root = tmp_path / "config"
    outside = tmp_path / "outside"
    _write(outside / "parent.toml", "not valid TOML [\n")
    root.mkdir()
    if kind == "file":
        _symlink(root / "alias.toml", outside / "parent.toml")
        entry = "alias.toml"
    elif kind == "directory":
        _symlink(root / "alias", outside, directory=True)
        entry = "alias/parent.toml"
    else:
        (outside / "sub").mkdir()
        _symlink(root / "alias", outside / "sub", directory=True)
        entry = "alias/../parent.toml"
    _write(root / "child.toml", f"extends = ['{entry}']\n")
    _guard_reads(monkeypatch, root)

    report = checker.check(root)

    assert report["ok"] is False
    assert report["invalid"][0]["entry"] == entry
    assert report["invalid"][0]["target"] == (outside / "parent.toml").as_posix()
    assert report["missing"] == []
    assert all("escapes the config root" in item["error"] for item in report["errors"])


@pytest.mark.parametrize("directory", [False, True])
def test_explicit_symlink_external_target_is_opaque(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory: bool
) -> None:
    root = tmp_path / "config"
    outside = tmp_path / "outside"
    _write(outside / "parent.toml", "not valid TOML [\n")
    root.mkdir()
    if directory:
        _symlink(root / "alias", outside, directory=True)
        key = "alias/parent.toml"
    else:
        _symlink(root / "alias.toml", outside / "parent.toml")
        key = "alias.toml"
    _write(root / "child.toml", f"extends = ['{key}', '../outside/parent.toml']\n")
    _write(root / "external-parents.toml", f"[external]\n'{key}' = 'exact target'\n")
    _guard_reads(monkeypatch, root)

    report = checker.check(root)

    assert report["ok"] is True
    assert report["files"] == 1
    assert len(report["external"]) == 2
    assert all(
        item["target"] == (outside / "parent.toml").as_posix()
        for item in report["external"]
    )


def test_source_alias_resolves_parents_from_canonical_location(tmp_path: Path) -> None:
    _write(tmp_path / "nested" / "source.toml", "extends = ['base.toml']\n")
    _write(tmp_path / "nested" / "base.toml", "")
    _symlink(tmp_path / "alias.toml", tmp_path / "nested" / "source.toml")

    report = checker.check(tmp_path)

    assert report["ok"] is True
    assert report["files"] == 2
    assert report["references"][0]["source"] == "nested/source.toml"
    assert report["references"][0]["target"] == "nested/base.toml"


def test_detects_cycle_through_source_alias(tmp_path: Path) -> None:
    _write(tmp_path / "nested" / "source.toml", "extends = ['../alias.toml']\n")
    _symlink(tmp_path / "alias.toml", tmp_path / "nested" / "source.toml")

    report = checker.check(tmp_path)

    assert report["ok"] is False
    assert report["files"] == 1
    assert report["missing"] == []
    assert report["cycles"] == [["nested/source.toml", "nested/source.toml"]]


def test_symlink_resolution_loop_is_reported(tmp_path: Path) -> None:
    _write(tmp_path / "child.toml", "extends = ['loop.toml']\n")
    _symlink(tmp_path / "loop.toml", tmp_path / "loop.toml")

    report = checker.check(tmp_path)

    assert report["ok"] is False
    assert report["invalid"][0]["entry"] == "loop.toml"
    assert any("cannot resolve" in item["error"] for item in report["errors"])


def test_manifest_symlink_cannot_escape_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "config"
    root.mkdir()
    outside = tmp_path / "external.toml"
    _write(outside, "[external]\n'../parent.toml' = 'outside manifest'\n")
    _symlink(root / "external-parents.toml", outside)
    _guard_reads(monkeypatch, root)

    report = checker.check(root)

    assert report["ok"] is False
    assert "manifest escapes" in report["errors"][0]["error"]


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("hidden_parent", ["../outside.toml", "child.toml"])
def test_manifest_cannot_hide_inheritance_edges(
    tmp_path: Path, alias: bool, hidden_parent: str,
) -> None:
    root = tmp_path / "config"
    manifest = root / "external-parents.toml"
    _write(manifest, f"extends = ['{hidden_parent}']\n[external]\n")
    entry = "external-parents.toml"
    if alias:
        entry = "manifest-alias.toml"
        _symlink(root / entry, manifest)
    _write(root / "child.toml", f"extends = ['{entry}']\n")
    report = checker.check(root)
    assert report["ok"] is False
    assert report["invalid"][0]["target"] == "external-parents.toml"
    assert any("external manifest" in error["error"] for error in report["errors"])


def test_inheritance_cannot_hide_in_non_toml_leaf(tmp_path: Path) -> None:
    _write(tmp_path / "child.toml", "extends = ['hidden.txt']\n")
    _write(tmp_path / "hidden.txt", "extends = ['../outside.toml']\n")
    report = checker.check(tmp_path)
    assert report["ok"] is False
    assert report["invalid"][0]["target"] == "hidden.txt"
    assert any("not a checked TOML" in error["error"] for error in report["errors"])


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
