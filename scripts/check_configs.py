"""Structural checks for the TOML config inheritance graph.

Unlike the trusted-path runtime loader, this checker only traverses resolved
paths inside the config root.  It does not validate business schema, secrets or
runtime assets; those belong to separate checks and need deployment inputs.

``external-parents.toml`` uses ``[external]`` entries mapping exact TOML paths
to descriptive strings.  Keys are relative to the config root (not an including
file); explicit ``..`` components may name outside parents.  Each key authorizes
only its resolved target, never a directory or sibling.  Existing root-relative
declarations remain valid.  Declared targets are opaque ``external`` leaves,
whether present or missing: their TOML is never read and their edges are not
traversed.  The manifest itself must resolve inside the config root and is
not an inheritance node unless explicitly declared external.

Local source and target identities are canonical, matching runtime inheritance
through symlink aliases.  Relative ``..`` and symlinks are allowed only when the
resolved target stays inside the root or is explicitly declared external.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

EXTERNAL_MANIFEST = "external-parents.toml"
SCHEMA_VERSION = 1


@dataclass
class ConfigFile:
    path: str
    extends: list[str] = field(default_factory=list[str])
    errors: list[str] = field(default_factory=list[str])


@dataclass
class Reference:
    source: str
    entry: str
    target: str
    status: str  # resolved / external / missing / invalid


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _load_external(path: Path, root: Path) -> dict[Path, str]:
    """Load exact canonical targets; declarations never authorize subtrees."""
    if not path.resolve().is_relative_to(root):
        raise ValueError("external-parents.toml: manifest escapes the config root")
    if not path.exists():
        return {}
    if not path.is_file():
        raise ValueError("external-parents.toml: manifest must be a file")
    payload = cast(dict[str, object], tomllib.loads(path.read_text(encoding="utf-8")))
    external = payload.get("external", {})
    if not isinstance(external, dict):
        raise TypeError("external-parents.toml: [external] must be a table")
    result: dict[Path, str] = {}
    for raw_key, value in cast(dict[object, object], external).items():
        if type(raw_key) is not str:
            raise TypeError("external-parents.toml: keys must be strings")
        if type(value) is not str:
            raise ValueError(
                "external-parents.toml: values must be strings and keys must be quoted; "
                f"{raw_key!r} may have parsed as a nested table"
            )
        key = raw_key
        candidate = Path(key)
        if candidate.anchor:
            raise ValueError(
                f"external-parents.toml: keys must be root-relative; got {key!r}"
            )
        target = (root / candidate).resolve()
        if (
            candidate.suffix != ".toml"
            or target.suffix != ".toml"
            or (target.exists() and not target.is_file())
        ):
            raise ValueError(
                "external-parents.toml: keys must name individual TOML files; "
                f"got {key!r}"
            )
        result[target] = value
    return result


def _read_config(path: Path) -> ConfigFile:
    record = ConfigFile(path=path.as_posix())
    try:
        payload = cast(
            dict[str, object],
            tomllib.loads(path.read_text(encoding="utf-8")),
        )
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        record.errors.append(f"invalid TOML: {type(exc).__name__}")
        return record
    except OSError as exc:
        record.errors.append(f"cannot read config: {type(exc).__name__}")
        return record
    raw = payload.get("extends", [])
    if not isinstance(raw, list):
        record.errors.append("extends must be an array of relative TOML paths")
        return record
    items = cast(list[object], raw)
    if any(type(item) is not str for item in items):
        record.errors.append("extends must be an array of relative TOML paths")
        return record
    extends = cast(list[str], items)
    if len(extends) != len(set(extends)):
        record.errors.append("extends contains duplicate paths")
    record.extends = extends
    return record


def check(config_root: Path) -> dict[str, Any]:
    """Check local canonical files and edges without reading external TOML."""
    if not config_root.is_dir():
        raise ValueError(f"config root is not a directory: {config_root}")
    root = config_root.resolve()
    manifest_path = root / EXTERNAL_MANIFEST
    try:
        external = _load_external(manifest_path, root)
    except (
        OSError,
        RuntimeError,
        UnicodeDecodeError,
        tomllib.TOMLDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "config_root": str(root),
            "files": 0,
            "references": [],
            "external": [],
            "missing": [],
            "invalid": [],
            "cycles": [],
            "errors": [{"path": str(manifest_path), "error": str(exc)}],
            "ok": False,
        }

    manifest_target = manifest_path.resolve()
    records_by_path: dict[Path, ConfigFile] = {}
    path_errors: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.toml")):
        if path == manifest_path:
            continue
        try:
            canonical = path.resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            path_errors.append(
                {
                    "path": str(path),
                    "error": f"cannot resolve config: {type(exc).__name__}",
                }
            )
            continue
        if canonical in external or canonical == manifest_target:
            continue
        if not canonical.is_relative_to(root):
            path_errors.append(
                {"path": str(path), "error": "config path escapes the config root"}
            )
            continue
        if canonical.is_file() and canonical not in records_by_path:
            records_by_path[canonical] = _read_config(canonical)
    records = list(records_by_path.values())
    by_path = {_relative(Path(record.path), root): record for record in records}

    references: list[Reference] = []
    edges: dict[str, set[str]] = {name: set() for name in by_path}
    for name, record in by_path.items():
        source = Path(record.path)
        for entry in record.extends:
            if Path(entry).anchor:
                references.append(Reference(name, entry, entry, "invalid"))
                record.errors.append(f"extends entry is absolute or anchored: {entry}")
                continue
            try:
                target = (source.parent / entry).resolve()
            except (OSError, RuntimeError, ValueError) as exc:
                references.append(Reference(name, entry, entry, "invalid"))
                record.errors.append(
                    f"cannot resolve extends entry: {entry} ({type(exc).__name__})"
                )
                continue
            target_name = _relative(target, root)
            if target in external:
                references.append(Reference(name, entry, target_name, "external"))
            elif not target.is_relative_to(root):
                references.append(Reference(name, entry, target_name, "invalid"))
                record.errors.append(f"extends target escapes the config root: {entry}")
            elif target == manifest_target:
                references.append(Reference(name, entry, target_name, "invalid"))
                record.errors.append("extends cannot target the external manifest")
            elif target.is_file():
                if target not in records_by_path:
                    # A resolved edge must have a checked graph node, not a
                    # hidden leaf (e.g. a file outside the *.toml inventory).
                    references.append(Reference(name, entry, target_name, "invalid"))
                    record.errors.append(f"extends target is not a checked TOML file: {entry}")
                    continue
                references.append(Reference(name, entry, target_name, "resolved"))
                edges[name].add(target_name)
            else:
                references.append(Reference(name, entry, target_name, "missing"))

    cycles: list[list[str]] = []

    def visit(node: str, stack: list[str], done: set[str]) -> None:
        if node in stack:
            cycles.append(stack[stack.index(node) :] + [node])
            return
        if node in done:
            return
        stack.append(node)
        for child in sorted(edges.get(node, ())):
            visit(child, stack, done)
        stack.pop()
        done.add(node)

    done: set[str] = set()
    for name in sorted(edges):
        visit(name, [], done)

    errors = path_errors + [
        {"path": record.path, "error": error}
        for record in records
        for error in record.errors
    ]
    missing = [asdict(ref) for ref in references if ref.status == "missing"]
    invalid = [asdict(ref) for ref in references if ref.status == "invalid"]
    external_refs = [asdict(ref) for ref in references if ref.status == "external"]
    ok = not errors and not missing and not invalid and not cycles
    return {
        "schema_version": SCHEMA_VERSION,
        "config_root": str(root),
        "files": len(records),
        "references": [asdict(ref) for ref in references],
        "external": external_refs,
        "missing": missing,
        "invalid": invalid,
        "cycles": cycles,
        "errors": errors,
        "notes": [
            (
                "Only resolved targets inside the config root are traversed; "
                "cycles through declared external or missing parents are not visible. "
                "The files count excludes external targets and deduplicates aliases."
            )
        ],
        "ok": ok,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = check(args.config_root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out is None:
        print(rendered)
    else:
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
