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
from typing import Any

EXTERNAL_MANIFEST = "external-parents.toml"
SCHEMA_VERSION = 1


@dataclass
class ConfigFile:
    path: str
    extends: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


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


def _load_external(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    external = payload.get("external", {})
    if type(external) is not dict:
        raise ValueError("external-parents.toml: [external] must be a table")
    result: dict[str, str] = {}
    for key, value in external.items():
        if type(value) is not str:
            raise ValueError(
                "external-parents.toml: keys must be quoted; "
                f"{key!r} parsed as a nested table"
            )
        result[key] = value
    return result


def _read_config(path: Path) -> ConfigFile:
    record = ConfigFile(path=path.as_posix())
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        record.errors.append(f"invalid TOML: {type(exc).__name__}")
        return record
    raw = payload.get("extends", [])
    if type(raw) is not list or any(type(item) is not str for item in raw):
        record.errors.append("extends must be an array of relative TOML paths")
        return record
    if len(raw) != len(set(raw)):
        record.errors.append("extends contains duplicate paths")
    record.extends = list(raw)
    return record


def _target(source: Path, entry: str) -> Path:
    return (source.parent / entry).resolve()


def check(config_root: Path) -> dict[str, Any]:
    if not config_root.is_dir():
        raise ValueError(f"config root is not a directory: {config_root}")
    root = config_root.resolve()
    manifest_path = root / EXTERNAL_MANIFEST
    try:
        external = _load_external(manifest_path)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
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
    files = sorted(path for path in root.rglob("*.toml") if path != manifest_path)
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
            elif entry in external or target_name in external:
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
