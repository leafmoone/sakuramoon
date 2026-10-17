"""Structural checks for the TOML config inheritance graph.

This checker answers a narrower question than ``load_config``: can every
``extends`` edge be resolved inside this checkout, and is the graph acyclic?
It deliberately does not validate business schema, secrets or runtime assets;
those belong to separate checks and need deployment inputs.

Parents that are intentionally kept outside the repository are declared in
``external-parents.toml``.  A declared external parent is reported as
``external`` and does not fail the check; an undeclared missing parent does.
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


def _load_external(path: Path, root: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    payload = cast(
        dict[str, object], tomllib.loads(path.read_text(encoding="utf-8"))
    )
    external = payload.get("external", {})
    if not isinstance(external, dict):
        raise TypeError("external-parents.toml: [external] must be a table")
    result: dict[str, str] = {}
    for raw_key, value in cast(dict[object, object], external).items():
        if type(raw_key) is not str:
            raise TypeError("external-parents.toml: keys must be strings")
        if type(value) is not str:
            raise ValueError(
                "external-parents.toml: keys must be quoted; "
                f"{raw_key!r} parsed as a nested table"
            )
        key = raw_key
        candidate = Path(key)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(
                "external-parents.toml: keys must be root-relative without '..'; "
                f"got {key!r}"
            )
        try:
            normalized = (root / candidate).resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                "external-parents.toml: key escapes the config root; "
                f"got {key!r}"
            ) from exc
        result[normalized] = value
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


def _target(source: Path, entry: str) -> Path:
    return (source.parent / entry).resolve()


def check(config_root: Path) -> dict[str, Any]:
    if not config_root.is_dir():
        raise ValueError(f"config root is not a directory: {config_root}")
    root = config_root.resolve()
    manifest_path = root / EXTERNAL_MANIFEST
    try:
        external = _load_external(manifest_path, root)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, TypeError, ValueError) as exc:
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
    files = sorted(
        path
        for path in root.rglob("*.toml")
        if path != manifest_path and path.is_file()
    )
    records = [_read_config(path) for path in files]
    by_path = {_relative(Path(record.path), root): record for record in records}

    references: list[Reference] = []
    edges: dict[str, set[str]] = {name: set() for name in by_path}
    for name, record in by_path.items():
        source = Path(record.path)
        for entry in record.extends:
            if Path(entry).is_absolute():
                references.append(Reference(name, entry, entry, "invalid"))
                record.errors.append(f"extends entry is absolute: {entry}")
                continue
            target = _target(source, entry)
            target_name = _relative(target, root)
            if target.is_file():
                references.append(Reference(name, entry, target_name, "resolved"))
                edges[name].add(target_name)
            elif target_name in external:
                references.append(Reference(name, entry, target_name, "external"))
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

    errors = [
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
        "files": len(files),
        "references": [asdict(ref) for ref in references],
        "external": external_refs,
        "missing": missing,
        "invalid": invalid,
        "cycles": cycles,
        "errors": errors,
        "notes": [
            (
                "Only extends edges resolvable inside this checkout are traversed; "
                "cycles through external or missing parents are not visible."
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
