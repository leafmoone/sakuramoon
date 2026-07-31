"""Validate SakuraMoon requirement traceability without reading ignored assets."""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

import tomli_w
from markdown_it import MarkdownIt
from markdown_it.token import Token

REGISTRY_PATH = Path("docs/model-architecture/progress/traceability.toml")
MAPPING_FIELDS = (
    "config_keys",
    "modules",
    "reference_modules",
    "tests",
    "benchmarks",
    "artifacts",
)
HARDWARE_LEVELS = {"CPU": 0, "1GPU": 1, "4GPU": 2}
STATUSES = {"planned", "blocked", "implemented", "verified", "superseded", "alias"}
TOP_LEVEL_KEYS = {
    "schema_version",
    "registry_revision",
    "archive_manifest",
    "roadmap",
    "sources",
    "profiles",
    "blockers",
    "inventory",
    "changes",
    "requirements",
}
SOURCE_KEYS = {
    "path",
    "kind",
    "sha256",
    "initial_sha256",
    "revision",
    "include_top_headings",
    "excluded_top_headings",
}
PROFILE_KEYS = {
    "name",
    "owner_tasks",
    *MAPPING_FIELDS,
    "hardware",
    "not_applicable_fields",
    "not_applicable_reason",
}
BLOCKER_KEYS = {"id", "kind", "description"}
INVENTORY_KEYS = {
    "module_root",
    "config_root",
    "ignored_module_paths",
    "ignored_config_paths",
    "no_modules_reason",
    "no_configs_reason",
}
CHANGE_KEYS = {
    "source_path",
    "revision",
    "previous_sha256",
    "new_sha256",
    "changed_at",
    "summary",
}
REQUIREMENT_KEYS = {
    "id",
    "kind",
    "status",
    "profile",
    "source_path",
    "heading_path",
    "node_kind",
    "source_fingerprint",
    "source_occurrence",
    "blocked_by",
    "alias_of",
    "superseded_by",
    "implementation_commit_ref",
    "implementation_paths",
    "evidence_hardware",
    "evidence_artifacts",
    "ai_review",
    "infra_review",
    *MAPPING_FIELDS,
    "not_applicable_fields",
    "not_applicable_reason",
}
SKIP_TEXT_PREFIXES = (
    "来源：",
    "关联执行清单：",
    "现行方案：",
    "会话证据：",
)
CANONICAL_SOURCES: dict[str, dict[str, object]] = {
    "docs/model-architecture/current/confirmed-decisions.md": {
        "kind": "confirmed",
        "initial_sha256": "899d93cd6c65271faf4f76581349d3e5bcc3faccef7753885780e98118035bbb",
        "include_top_headings": tuple(f"{value}." for value in range(15)),
        "excluded_top_headings": ("15.",),
    },
    "docs/model-architecture/current/open-items.md": {
        "kind": "open_items",
        "initial_sha256": "fa62704d2f4193388ab121dfd93b1aa2f08f49be18c2906d51da8c81179187c6",
        "include_top_headings": tuple(f"{value}." for value in range(11)),
        "excluded_top_headings": ("11.",),
    },
    "docs/model-architecture/current/observability-and-evaluation.md": {
        "kind": "observability",
        "initial_sha256": "9a8b4fd39c5e5bf76a343f1b02e3507abddc27e98e34946a0fd0f1e98be64201",
        "include_top_headings": ("可观测性与评估补充决定",),
        "excluded_top_headings": (),
    },
}
TRUSTED_ARCHIVE_MANIFEST_SHA256 = (
    "8080e7d8e02345c5b6487b34de5d666630f524ddb9eca4c22e21ccedbffbee04"
)
TRUSTED_REQUIREMENT_BINDING_SHA256S = frozenset(
    {
        "999eff1fece89b69eba0497d60bdf8adc358d0f7f2a5243c1e17a591a233bb6b",
        "626d7348cff00d6a89282bebf274b3c8e25ed36e1fd662ceed7020ebdd9470f2",
    }
)
FORWARD_IDENTITY_LOCK_REVISION = 62
FORWARD_IDENTITY_FIELDS = (
    "source_path",
    "heading_path",
    "node_kind",
    "source_fingerprint",
    "source_occurrence",
)
PACKAGE_REVIEW_TASKS = {
    "R002": "FOUNDATION",
    "D001": "FOUNDATION",
    "C001": "FOUNDATION",
    "A001": "FOUNDATION",
    **{f"D{serial:03d}": "DATA" for serial in range(10, 16)},
    **{f"T{serial:03d}": "ENCODERS" for serial in range(20, 25)},
    **{f"M{serial:03d}": "DENSE" for serial in range(30, 34)},
    **{f"T{serial:03d}": "TRAINING_UTILITIES" for serial in range(51, 54)},
}
INDEPENDENT_REVIEW_TASKS = {
    "K001",
    "T040",
    "T041",
    "T042",
    "T043",
    "T050",
    "T054",
}
BASELINE_REQUIREMENT_MAXIMA = {
    "ARCH": 2,
    "C01": 5,
    "C02": 7,
    "C03": 13,
    "C04": 9,
    "C05": 11,
    "C06": 6,
    "C07": 6,
    "C08": 6,
    "C10": 9,
    "C11": 7,
    "C12": 11,
    "DEC": 1,
    "DOC": 5,
    "OBS": 12,
    "OPEN": 99,
    "SUP": 10,
}
@dataclasses.dataclass(frozen=True)
class SourceNode:
    path: str
    heading_path: tuple[str, ...]
    kind: str
    text: str
    fingerprint: str
    occurrence: int

    @property
    def locator(self) -> tuple[str, tuple[str, ...], str, int]:
        return (self.path, self.heading_path, self.fingerprint, self.occurrence)


@dataclasses.dataclass
class VerificationReport:
    errors: list[str]
    requirement_count: int = 0
    source_node_count: int = 0
    archive_file_count: int = 0
    module_count: int = 0
    config_key_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "counts": {
                "requirements": self.requirement_count,
                "source_nodes": self.source_node_count,
                "archive_files": self.archive_file_count,
                "production_modules": self.module_count,
                "runtime_config_keys": self.config_key_count,
            },
        }


def normalize_text(value: str) -> str:
    return " ".join(value.replace("\u00a0", " ").split())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def node_fingerprint(kind: str, headings: Sequence[str], text: str) -> str:
    payload = "\0".join((kind, " > ".join(headings), normalize_text(text)))
    return sha256_bytes(payload.encode("utf-8"))


def _unwrap_notion_containers(text: str) -> str:
    callout = re.compile(r"(?ms)^[ \t]*<callout\b[^>]*>\n?(.*?)\n?[ \t]*</callout>[ \t]*$")

    def unwrap_callout(match: re.Match[str]) -> str:
        lines = match.group(1).splitlines()
        return "\n".join(line.removeprefix("\t") for line in lines)

    text = callout.sub(unwrap_callout, text)
    container = re.compile(
        r"(?m)^[ \t]*</?(?:page|ancestor-path|parent-page|properties|content)"
        r"\b[^>]*>[ \t]*\n?"
    )
    return container.sub("", text)


def _matches_heading(top: str, includes: Sequence[str], excludes: Sequence[str]) -> bool:
    if any(top.startswith(prefix) for prefix in excludes):
        return False
    return any(top.startswith(prefix) for prefix in includes)


def _should_include_text(text: str) -> bool:
    if not text or text.startswith(SKIP_TEXT_PREFIXES):
        return False
    return "<mention-page" not in text and not text.startswith("Here is the result of")


def _clean_node_text(text: str) -> str:
    for marker in ("\n来源：", "\n来源:", "\n关联执行清单：", "\n现行方案："):
        text = text.split(marker, 1)[0]
    return normalize_text(text)


def extract_source_nodes(root: Path, source: Mapping[str, Any]) -> list[SourceNode]:
    relative = source["path"]
    text = _unwrap_notion_containers((root / relative).read_text(encoding="utf-8"))
    tokens: list[Token] = MarkdownIt("commonmark").parse(text)
    headings: list[str] = []
    raw_nodes: list[tuple[tuple[str, ...], str, str]] = []
    item_stack: list[list[str]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type == "heading_open" and i + 1 < len(tokens):
            # Notion exports may leave HTML/list containers open. A top-level
            # heading is still an unconditional boundary between requirements.
            item_stack.clear()
            level = int(token.tag[1:])
            heading = normalize_text(tokens[i + 1].content)
            headings[level - 1 :] = [heading]
            i += 3
            continue
        if token.type == "list_item_open":
            item_stack.append([])
        elif token.type == "inline" and item_stack:
            item_stack[-1].append(token.content)
        elif token.type == "list_item_close":
            if item_stack and headings:
                parts = item_stack.pop()
                raw_nodes.append(
                    (tuple(headings), "list_item", _clean_node_text(" ".join(parts)))
                )
        elif token.type == "paragraph_open" and not item_stack and i + 1 < len(tokens):
            inline = tokens[i + 1]
            if inline.type == "inline" and headings:
                raw_nodes.append(
                    (tuple(headings), "paragraph", _clean_node_text(inline.content))
                )
        elif token.type in {"fence", "code_block"} and headings:
            raw_nodes.append(
                (tuple(headings), "code_block", normalize_text(token.content))
            )
        i += 1

    includes = source["include_top_headings"]
    excludes = source["excluded_top_headings"]
    occurrence_counter: Counter[tuple[tuple[str, ...], str]] = Counter()
    nodes: list[SourceNode] = []
    for heading_path, kind, node_text in raw_nodes:
        if not _matches_heading(heading_path[0], includes, excludes):
            continue
        if not _should_include_text(node_text):
            continue
        fingerprint = node_fingerprint(kind, heading_path, node_text)
        occurrence_key = (heading_path, fingerprint)
        occurrence = occurrence_counter[occurrence_key]
        occurrence_counter[occurrence_key] += 1
        nodes.append(
            SourceNode(relative, heading_path, kind, node_text, fingerprint, occurrence)
        )
    return nodes


def _type_name(value: Any) -> str:
    return type(value).__name__


def _strict_keys(
    item: Mapping[str, Any], expected: set[str], context: str, errors: list[str]
) -> None:
    unknown = sorted(set(item) - expected)
    missing = sorted(expected - set(item))
    if unknown:
        errors.append(f"{context}: unknown keys: {', '.join(unknown)}")
    if missing:
        errors.append(f"{context}: missing keys: {', '.join(missing)}")


def _expect_type(
    item: Mapping[str, Any], key: str, expected: type, context: str, errors: list[str]
) -> bool:
    if key not in item:
        return False
    if type(item[key]) is not expected:
        errors.append(
            f"{context}.{key}: expected {expected.__name__}, got {_type_name(item[key])}"
        )
        return False
    return True


def _expect_str_list(
    item: Mapping[str, Any], key: str, context: str, errors: list[str]
) -> bool:
    if not _expect_type(item, key, list, context, errors):
        return False
    values = item[key]
    if any(type(value) is not str for value in values):
        errors.append(f"{context}.{key}: expected list[str]")
        return False
    return True


def _table_items(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list:
        return []
    return [
        cast(dict[str, Any], item)
        for item in cast(list[object], value)
        if type(item) is dict
    ]


def _safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _validate_repo_path(root: Path, value: str, context: str, errors: list[str]) -> None:
    plain = value.split("::", 1)[0]
    wildcard_at = min([plain.find(char) for char in "*?[" if char in plain] or [len(plain)])
    prefix = plain[:wildcard_at].rstrip("/")
    if not _safe_relative(prefix or plain):
        errors.append(f"{context}: path must be repository-relative without traversal: {value}")
        return
    current = root
    for part in PurePosixPath(prefix).parts:
        current /= part
        if current.is_symlink():
            errors.append(f"{context}: symlink path component is forbidden: {value}")
            return


def _validate_schema(data: Mapping[str, Any], errors: list[str]) -> None:
    _strict_keys(data, TOP_LEVEL_KEYS, "registry", errors)
    for key in ("schema_version", "registry_revision"):
        _expect_type(data, key, int, "registry", errors)
    for key in ("archive_manifest", "roadmap"):
        _expect_type(data, key, str, "registry", errors)
    sections = {
        "sources": (SOURCE_KEYS, "source"),
        "profiles": (PROFILE_KEYS, "profile"),
        "blockers": (BLOCKER_KEYS, "blocker"),
        "changes": (CHANGE_KEYS, "change"),
        "requirements": (REQUIREMENT_KEYS, "requirement"),
    }
    for section, (keys, label) in sections.items():
        if not _expect_type(data, section, list, "registry", errors):
            continue
        for index, item in enumerate(cast(list[object], data[section])):
            context = f"{label}[{index}]"
            if type(item) is not dict:
                errors.append(f"{context}: expected table, got {_type_name(item)}")
                continue
            _strict_keys(cast(dict[str, Any], item), keys, context, errors)
    if _expect_type(data, "inventory", dict, "registry", errors):
        inventory_table = cast(dict[str, Any], data["inventory"])
        _strict_keys(inventory_table, INVENTORY_KEYS, "inventory", errors)

    for index, source in enumerate(_table_items(data.get("sources"))):
        context = f"source[{index}]"
        for key in ("path", "kind", "sha256", "initial_sha256"):
            _expect_type(source, key, str, context, errors)
        _expect_type(source, "revision", int, context, errors)
        _expect_str_list(source, "include_top_headings", context, errors)
        _expect_str_list(source, "excluded_top_headings", context, errors)
    for index, profile in enumerate(_table_items(data.get("profiles"))):
        context = f"profile[{index}]"
        for key in ("name", "hardware", "not_applicable_reason"):
            _expect_type(profile, key, str, context, errors)
        for key in ("owner_tasks", *MAPPING_FIELDS, "not_applicable_fields"):
            _expect_str_list(profile, key, context, errors)
    for index, blocker in enumerate(_table_items(data.get("blockers"))):
        for key in BLOCKER_KEYS:
            _expect_type(blocker, key, str, f"blocker[{index}]", errors)
    for index, change in enumerate(_table_items(data.get("changes"))):
        context = f"change[{index}]"
        for key in CHANGE_KEYS - {"revision"}:
            _expect_type(change, key, str, context, errors)
        _expect_type(change, "revision", int, context, errors)
    for index, requirement in enumerate(_table_items(data.get("requirements"))):
        context = f"requirement[{index}]"
        for key in REQUIREMENT_KEYS - {
            "source_occurrence",
            "heading_path",
            "blocked_by",
            "implementation_paths",
            "evidence_artifacts",
            "not_applicable_fields",
            *MAPPING_FIELDS,
        }:
            _expect_type(requirement, key, str, context, errors)
        _expect_type(requirement, "source_occurrence", int, context, errors)
        for key in (
            "heading_path",
            "blocked_by",
            "implementation_paths",
            "evidence_artifacts",
            "not_applicable_fields",
            *MAPPING_FIELDS,
        ):
            _expect_str_list(requirement, key, context, errors)
    inventory = data.get("inventory")
    if isinstance(inventory, dict):
        inventory = cast(dict[str, Any], inventory)
        for key in ("module_root", "config_root", "no_modules_reason", "no_configs_reason"):
            _expect_type(inventory, key, str, "inventory", errors)
        for key in ("ignored_module_paths", "ignored_config_paths"):
            _expect_str_list(inventory, key, "inventory", errors)


def _detect_graph_cycles(edges: Mapping[str, str], label: str, errors: list[str]) -> None:
    for origin in edges:
        seen: set[str] = set()
        current = origin
        while current in edges:
            if current in seen:
                errors.append(f"{label} cycle contains {current}")
                break
            seen.add(current)
            current = edges[current]


def _requirement_locator(requirement: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        requirement["source_path"],
        tuple(requirement["heading_path"]),
        requirement["node_kind"],
        requirement["source_fingerprint"],
        requirement["source_occurrence"],
    )


def _baseline_requirement_ids() -> set[str]:
    return {
        f"{prefix}-{serial:03d}"
        for prefix, maximum in BASELINE_REQUIREMENT_MAXIMA.items()
        for serial in range(1, maximum + 1)
    }


def _requirement_bindings_sha256(
    requirements: Sequence[Mapping[str, Any]],
) -> str:
    baseline_ids = _baseline_requirement_ids()
    fields = (
        "id",
        "source_path",
        "heading_path",
        "node_kind",
        "source_fingerprint",
        "source_occurrence",
    )
    bindings = [
        {field: requirement[field] for field in fields}
        for requirement in sorted(requirements, key=lambda item: cast(str, item["id"]))
        if requirement["id"] in baseline_ids
    ]
    payload = json.dumps(
        bindings,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256_bytes(payload)


def _validate_bootstrap_bindings(
    requirements: Sequence[Mapping[str, Any]], errors: list[str]
) -> None:
    baseline_ids = _baseline_requirement_ids()
    requirement_by_id = {
        cast(str, requirement["id"]): requirement for requirement in requirements
    }
    missing = sorted(baseline_ids - set(requirement_by_id))
    if missing:
        errors.append(f"bootstrap requirement IDs were removed: {missing}")
        return
    digest = _requirement_bindings_sha256(requirements)
    if digest not in TRUSTED_REQUIREMENT_BINDING_SHA256S:
        errors.append(
            "bootstrap requirement bindings do not match a trusted locator anchor"
        )


def _validate_registry_history(
    snapshots: Sequence[Mapping[str, Any]], errors: list[str]
) -> None:
    if not snapshots:
        return
    _validate_bootstrap_bindings(
        cast(Sequence[Mapping[str, Any]], snapshots[0]["requirements"]), errors
    )
    issued_ids: set[str] = set()
    maximum_by_prefix: dict[str, int] = {}
    locator_owner: dict[tuple[object, ...], str] = {}
    previous: Mapping[str, Any] | None = None
    for snapshot_index, snapshot in enumerate(snapshots):
        requirements = cast(Sequence[Mapping[str, Any]], snapshot["requirements"])
        current_by_id = {
            cast(str, requirement["id"]): requirement for requirement in requirements
        }
        if previous is not None:
            previous_requirements = cast(
                Sequence[Mapping[str, Any]], previous["requirements"]
            )
            previous_by_id = {
                cast(str, requirement["id"]): requirement
                for requirement in previous_requirements
            }
            missing = sorted(set(previous_by_id) - set(current_by_id))
            if missing:
                errors.append(
                    f"registry history snapshot {snapshot_index}: stable requirement IDs were removed: {missing}"
                )
            if previous["registry_revision"] >= FORWARD_IDENTITY_LOCK_REVISION:
                previous_ids = list(previous_by_id)
                retained_ids = [
                    req_id for req_id in current_by_id if req_id in previous_by_id
                ]
                if retained_ids != previous_ids:
                    errors.append(
                        f"registry history snapshot {snapshot_index}: stable requirement IDs were reordered"
                    )
                for req_id in set(previous_by_id) & set(current_by_id):
                    changed_fields = [
                        field
                        for field in FORWARD_IDENTITY_FIELDS
                        if previous_by_id[req_id][field] != current_by_id[req_id][field]
                    ]
                    if changed_fields:
                        errors.append(
                            f"{req_id}: historical requirement identity was rewritten: {changed_fields}"
                        )
            if snapshot["registry_revision"] != previous["registry_revision"] + 1:
                errors.append(
                    f"registry history snapshot {snapshot_index}: registry_revision must increment by exactly one"
                )
        for req_id, requirement in current_by_id.items():
            locator = _requirement_locator(requirement)
            owner = locator_owner.get(locator)
            if owner is not None and owner != req_id:
                errors.append(
                    f"{req_id}: source locator was historically bound to stable ID {owner}"
                )
            locator_owner[locator] = req_id
            prefix, separator, serial_text = req_id.rpartition("-")
            if not separator or not serial_text.isdigit():
                continue
            serial = int(serial_text)
            if req_id not in issued_ids:
                previous_maximum = maximum_by_prefix.get(prefix, 0)
                if previous is not None and prefix in maximum_by_prefix and serial <= previous_maximum:
                    errors.append(
                        f"{req_id}: new requirement ID must use an unused serial above {prefix}-{previous_maximum:03d}"
                    )
                maximum_by_prefix[prefix] = max(previous_maximum, serial)
                issued_ids.add(req_id)
        previous = snapshot


def _load_registry_history(
    root: Path,
    registry_path: Path,
    current: Mapping[str, Any],
    errors: list[str],
) -> list[Mapping[str, Any]]:
    git_marker = root / ".git"
    if not git_marker.exists():
        _validate_bootstrap_bindings(
            cast(Sequence[Mapping[str, Any]], current["requirements"]), errors
        )
        return []

    def run_git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    top_level = run_git("rev-parse", "--show-toplevel")
    if top_level.returncode != 0 or Path(top_level.stdout.strip()).resolve() != root.resolve():
        errors.append("cannot establish repository root for requirement ID history")
        return []
    history = run_git(
        "log", "--format=%H", "--reverse", "--follow", "--", registry_path.as_posix()
    )
    if history.returncode != 0:
        errors.append("cannot read requirement registry Git history")
        return []
    snapshots: list[Mapping[str, Any]] = []
    for commit in history.stdout.splitlines():
        result = run_git("show", f"{commit}:{registry_path.as_posix()}")
        if result.returncode != 0:
            errors.append(f"cannot read requirement registry at commit {commit}")
            return []
        try:
            snapshot = tomllib.loads(result.stdout)
        except tomllib.TOMLDecodeError as exc:
            errors.append(f"invalid historical requirement registry at {commit}: {exc}")
            return []
        snapshots.append(snapshot)
    if not snapshots or current != snapshots[-1]:
        snapshots.append(current)
    return snapshots


def _validate_archive(root: Path, manifest_value: str, errors: list[str]) -> int:
    error_count = len(errors)
    _validate_repo_path(root, manifest_value, "archive_manifest", errors)
    if len(errors) != error_count:
        return 0
    manifest = root / manifest_value
    if not manifest.is_file():
        errors.append(f"archive manifest does not exist: {manifest_value}")
        return 0
    if sha256_file(manifest) != TRUSTED_ARCHIVE_MANIFEST_SHA256:
        errors.append("archive manifest does not match the trusted bootstrap anchor")
        return 0
    expected: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            errors.append(f"{manifest_value}:{line_number}: invalid checksum line")
            continue
        digest, relative = match.groups()
        if relative.startswith("archive/"):
            expected[relative] = digest
    base = manifest.parent
    archive_entries = list((base / "archive").rglob("*"))
    for path in archive_entries:
        if path.is_symlink():
            errors.append(
                f"archive symlink is forbidden: {path.relative_to(base).as_posix()}"
            )
    actual_paths = {
        path.relative_to(base).as_posix()
        for path in archive_entries
        if path.is_file() and not path.is_symlink()
    }
    if actual_paths != set(expected):
        missing = sorted(actual_paths - set(expected))
        stale = sorted(set(expected) - actual_paths)
        if missing:
            errors.append(f"archive files missing from checksum manifest: {missing}")
        if stale:
            errors.append(f"archive checksum entries missing files: {stale}")
    for relative, digest in expected.items():
        path = base / relative
        before = len(errors)
        _validate_repo_path(base, relative, f"archive checksum {relative}", errors)
        if len(errors) != before:
            continue
        if path.is_file() and sha256_file(path) != digest:
            errors.append(f"archive checksum mismatch: {relative}")
    return len(expected)


def _iter_config_leaves(prefix: str, value: object) -> Iterable[str]:
    if isinstance(value, dict):
        table = cast(dict[str, object], value)
        if not value:
            yield prefix
        for key, child in table.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            yield from _iter_config_leaves(child_prefix, child)
    else:
        yield prefix


def _matches_any(value: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def _required_review_paths(task_id: str) -> tuple[str, str] | None:
    package = PACKAGE_REVIEW_TASKS.get(task_id)
    if package is not None:
        root = f"docs/model-architecture/reviews/{package}"
        return (f"{root}/ai_review.md", f"{root}/infra_review.md")
    if task_id in INDEPENDENT_REVIEW_TASKS or task_id.startswith("S"):
        root = f"docs/model-architecture/reviews/{task_id}"
        return (f"{root}/ai_review.md", f"{root}/infra_review.md")
    return None


def _validate_mapping_dimensions(
    root: Path,
    item: Mapping[str, Any],
    context: str,
    errors: list[str],
) -> None:
    not_applicable = set(item["not_applicable_fields"])
    unknown_na = sorted(not_applicable - set(MAPPING_FIELDS))
    if unknown_na:
        errors.append(f"{context}.not_applicable_fields invalid: {unknown_na}")
    for field in MAPPING_FIELDS:
        values = item[field]
        if values and field in not_applicable:
            errors.append(f"{context}.{field} is mapped and cannot be not-applicable")
        if not values and field not in not_applicable:
            errors.append(f"{context}.{field} is empty without explicit not-applicable")
        if field != "config_keys":
            for value in values:
                _validate_repo_path(root, value, f"{context}.{field}", errors)
    if not_applicable and not item["not_applicable_reason"]:
        errors.append(f"{context}.not_applicable_reason is required")


def _validate_inventory(
    root: Path,
    inventory: Mapping[str, Any],
    requirements: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> tuple[int, int]:
    context = "inventory"
    for key in ("module_root", "config_root", "no_modules_reason", "no_configs_reason"):
        _expect_type(inventory, key, str, context, errors)
    for key in ("ignored_module_paths", "ignored_config_paths"):
        _expect_str_list(inventory, key, context, errors)
    if errors and not all(key in inventory for key in INVENTORY_KEYS):
        return 0, 0
    module_root = inventory["module_root"]
    config_root = inventory["config_root"]
    _validate_repo_path(root, module_root, "inventory.module_root", errors)
    _validate_repo_path(root, config_root, "inventory.config_root", errors)
    blanket_module_pattern = f"{module_root.rstrip('/')}/**"
    module_patterns = [
        pattern
        for item in (*requirements, *profiles)
        for pattern in item["modules"]
        if pattern != blanket_module_pattern
    ]
    config_patterns = [
        pattern for requirement in requirements for pattern in requirement["config_keys"]
    ]
    ignored_modules = inventory["ignored_module_paths"]
    ignored_configs = inventory["ignored_config_paths"]
    if ignored_modules:
        errors.append("inventory.ignored_module_paths must remain empty")
    if ignored_configs:
        errors.append("inventory.ignored_config_paths must remain empty")

    module_paths: list[str] = []
    module_base = root / module_root
    if module_base.exists():
        for path in module_base.rglob("*.py"):
            relative = path.relative_to(root).as_posix()
            before = len(errors)
            _validate_repo_path(root, relative, "production module", errors)
            if len(errors) != before:
                continue
            if not _matches_any(relative, ignored_modules):
                module_paths.append(relative)
    if module_paths and inventory["no_modules_reason"]:
        errors.append("inventory.no_modules_reason must be cleared when production modules exist")
    if not module_paths and not inventory["no_modules_reason"]:
        errors.append("inventory.no_modules_reason is required while no production modules exist")
    for path in module_paths:
        if not _matches_any(path, module_patterns):
            errors.append(f"production module has no reverse requirement mapping: {path}")

    config_keys: list[str] = []
    config_base = root / config_root
    if config_base.exists():
        for path in config_base.rglob("*.toml"):
            relative = path.relative_to(root).as_posix()
            before = len(errors)
            _validate_repo_path(root, relative, "runtime config", errors)
            if len(errors) != before:
                continue
            if _matches_any(relative, ignored_configs):
                continue
            try:
                payload = tomllib.loads(path.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError) as exc:
                errors.append(f"runtime config cannot be parsed: {relative}: {exc}")
                continue
            for key in _iter_config_leaves("", payload):
                config_keys.append(f"{relative}::{key}")
                if not (_matches_any(key, config_patterns) or _matches_any(f"{relative}::{key}", config_patterns)):
                    errors.append(f"runtime config key has no reverse requirement mapping: {relative}::{key}")
    if config_keys and inventory["no_configs_reason"]:
        errors.append("inventory.no_configs_reason must be cleared when runtime configs exist")
    if not config_keys and not inventory["no_configs_reason"]:
        errors.append("inventory.no_configs_reason is required while no runtime configs exist")
    return len(module_paths), len(config_keys)


def _validate_local_links(root: Path, source_paths: Iterable[str], errors: list[str]) -> None:
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for source in source_paths:
        path = root / source
        for target in pattern.findall(path.read_text(encoding="utf-8")):
            clean = target.strip().strip("<>").split("#", 1)[0]
            if not clean or re.match(r"^[a-z]+://", clean):
                continue
            candidate = (path.parent / clean).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                errors.append(f"local link escapes repository: {source} -> {target}")
                continue
            if not candidate.exists():
                errors.append(f"broken local link: {source} -> {target}")


def verify(root: Path, registry_path: Path = REGISTRY_PATH) -> VerificationReport:
    errors: list[str] = []
    _validate_repo_path(root, registry_path.as_posix(), "registry path", errors)
    if errors:
        return VerificationReport(errors)
    full_registry = root / registry_path
    try:
        data = tomllib.loads(full_registry.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return VerificationReport([f"cannot read registry {registry_path}: {exc}"])
    _validate_schema(data, errors)
    if errors:
        return VerificationReport(errors)

    if data["schema_version"] != 1:
        errors.append(f"unsupported schema_version: {data['schema_version']}")
    if data["registry_revision"] < 1:
        errors.append("registry_revision must be >= 1")

    expected_paths = {
        "archive_manifest": "docs/model-architecture/SHA256SUMS",
        "roadmap": "docs/model-architecture/progress/IMPLEMENTATION_ROADMAP.md",
    }
    for key, expected in expected_paths.items():
        if data[key] != expected:
            errors.append(f"registry.{key} must be {expected}")
    inventory = data["inventory"]
    if inventory["module_root"] != "src/sakuramoon":
        errors.append("inventory.module_root must be src/sakuramoon")
    if inventory["config_root"] != "config":
        errors.append("inventory.config_root must be config")
    if errors:
        return VerificationReport(errors)

    for key in ("archive_manifest", "roadmap"):
        _validate_repo_path(root, data[key], f"registry.{key}", errors)
        if not (root / data[key]).is_file():
            errors.append(f"registry.{key} does not exist: {data[key]}")

    sources = cast(list[dict[str, Any]], data["sources"])
    profiles = cast(list[dict[str, Any]], data["profiles"])
    blockers = cast(list[dict[str, Any]], data["blockers"])
    requirements = cast(list[dict[str, Any]], data["requirements"])
    source_paths = [source["path"] for source in sources]
    if set(source_paths) != set(CANONICAL_SOURCES) or len(source_paths) != len(
        CANONICAL_SOURCES
    ):
        errors.append("registry sources must exactly match the canonical source set")
    for source in sources:
        canonical = CANONICAL_SOURCES.get(source["path"])
        if canonical is None:
            continue
        for key in (
            "kind",
            "initial_sha256",
            "include_top_headings",
            "excluded_top_headings",
        ):
            actual: object = source[key]
            expected = canonical[key]
            if isinstance(actual, list):
                actual = tuple(cast(list[object], actual))
            if actual != expected:
                errors.append(f"{source['path']}: canonical {key} scope was changed")
    if not profiles:
        errors.append("registry profiles must not be empty")
    if not requirements:
        errors.append("registry requirements must not be empty")
    if errors:
        return VerificationReport(errors)
    source_by_path: dict[str, Mapping[str, Any]] = {}
    actual_nodes: dict[tuple[str, tuple[str, ...], str, int], SourceNode] = {}
    for index, source in enumerate(sources):
        context = f"source[{index}]"
        for key in ("path", "kind", "sha256", "initial_sha256"):
            _expect_type(source, key, str, context, errors)
        _expect_type(source, "revision", int, context, errors)
        _expect_str_list(source, "include_top_headings", context, errors)
        _expect_str_list(source, "excluded_top_headings", context, errors)
        path_value = source["path"]
        if path_value in source_by_path:
            errors.append(f"duplicate source path: {path_value}")
            continue
        source_by_path[path_value] = source
        if not path_value.startswith("docs/model-architecture/current/"):
            errors.append(f"{context}.path must remain under current documentation")
            continue
        before = len(errors)
        _validate_repo_path(root, path_value, context, errors)
        if len(errors) != before:
            continue
        source_path = root / path_value
        if not source_path.is_file():
            errors.append(f"source does not exist: {path_value}")
            continue
        actual_sha = sha256_file(source_path)
        for digest_key in ("sha256", "initial_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", source[digest_key]):
                errors.append(f"{context}.{digest_key} must be SHA-256")
        if actual_sha != source["sha256"]:
            errors.append(
                f"source SHA drift requires registry nodes and changelog update: {path_value}"
            )
        try:
            for node in extract_source_nodes(root, source):
                if node.locator in actual_nodes:
                    errors.append(f"duplicate extracted source locator: {node.locator}")
                actual_nodes[node.locator] = node
        except (OSError, ValueError) as exc:
            errors.append(f"cannot parse source {path_value}: {exc}")

    profile_by_name: dict[str, Mapping[str, Any]] = {}
    known_tasks = set(
        re.findall(
            r"^### ([A-Z][0-9]{3})：",
            (root / data["roadmap"]).read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    )
    for index, profile in enumerate(profiles):
        context = f"profile[{index}]"
        for key in ("name", "hardware", "not_applicable_reason"):
            _expect_type(profile, key, str, context, errors)
        for key in ("owner_tasks", *MAPPING_FIELDS, "not_applicable_fields"):
            _expect_str_list(profile, key, context, errors)
        name = profile["name"]
        if name in profile_by_name:
            errors.append(f"duplicate profile name: {name}")
        profile_by_name[name] = profile
        if not profile["owner_tasks"]:
            errors.append(f"{context}.owner_tasks must not be empty")
        unknown_tasks = sorted(set(profile["owner_tasks"]) - known_tasks)
        if unknown_tasks:
            errors.append(f"{context}.owner_tasks unknown roadmap IDs: {unknown_tasks}")
        if profile["hardware"] not in HARDWARE_LEVELS:
            errors.append(f"{context}.hardware invalid: {profile['hardware']}")
        _validate_mapping_dimensions(root, profile, context, errors)

    blocker_ids: set[str] = set()
    for index, blocker in enumerate(blockers):
        context = f"blocker[{index}]"
        for key in BLOCKER_KEYS:
            _expect_type(blocker, key, str, context, errors)
        if blocker["id"] in blocker_ids:
            errors.append(f"duplicate blocker ID: {blocker['id']}")
        blocker_ids.add(blocker["id"])

    requirement_by_id: dict[str, Mapping[str, Any]] = {}
    registry_locators: dict[tuple[str, tuple[str, ...], str, int], str] = {}
    exact_requirement_matches: set[str] = set()
    exact_source_matches: set[tuple[str, tuple[str, ...], str, int]] = set()
    alias_edges: dict[str, str] = {}
    superseded_edges: dict[str, str] = {}
    for index, requirement in enumerate(requirements):
        context = f"requirement[{index}]"
        for key in (
            "id",
            "kind",
            "status",
            "profile",
            "source_path",
            "node_kind",
            "source_fingerprint",
            "alias_of",
            "superseded_by",
            "implementation_commit_ref",
            "evidence_hardware",
            "ai_review",
            "infra_review",
        ):
            _expect_type(requirement, key, str, context, errors)
        for key in ("heading_path", "blocked_by", "implementation_paths"):
            _expect_str_list(requirement, key, context, errors)
        _expect_type(requirement, "source_occurrence", int, context, errors)
        req_id = requirement["id"]
        if not re.fullmatch(r"[A-Z][A-Z0-9-]*-[0-9]{3}", req_id):
            errors.append(f"{context}.id is not a stable requirement ID: {req_id}")
        if req_id in requirement_by_id:
            errors.append(f"duplicate requirement ID: {req_id}")
        requirement_by_id[req_id] = requirement
        if requirement["status"] not in STATUSES:
            errors.append(f"{context}.status invalid: {requirement['status']}")
        if requirement["profile"] not in profile_by_name:
            errors.append(f"{context}.profile unknown: {requirement['profile']}")
        _validate_mapping_dimensions(root, requirement, context, errors)
        if requirement["source_path"] not in source_by_path:
            errors.append(f"{context}.source_path unknown: {requirement['source_path']}")
        if not re.fullmatch(r"[0-9a-f]{64}", requirement["source_fingerprint"]):
            errors.append(f"{context}.source_fingerprint must be SHA-256")
        locator = (
            requirement["source_path"],
            tuple(requirement["heading_path"]),
            requirement["source_fingerprint"],
            requirement["source_occurrence"],
        )
        actual_node = actual_nodes.get(locator)
        if actual_node and requirement["node_kind"] != actual_node.kind:
            errors.append(
                f"{req_id}: node_kind {requirement['node_kind']} does not match {actual_node.kind}"
            )
        elif actual_node:
            exact_requirement_matches.add(req_id)
            exact_source_matches.add(locator)
        if locator in registry_locators:
            errors.append(
                f"duplicate primary source mapping: {req_id} and {registry_locators[locator]}"
            )
        registry_locators[locator] = req_id
        if requirement["status"] == "blocked":
            if not requirement["blocked_by"]:
                errors.append(f"{req_id}: blocked requirement has no blocker")
            missing_blockers = sorted(set(requirement["blocked_by"]) - blocker_ids)
            if missing_blockers:
                errors.append(f"{req_id}: unknown blockers: {missing_blockers}")
        elif requirement["blocked_by"]:
            errors.append(f"{req_id}: only blocked requirements may set blocked_by")
        if requirement["status"] == "alias":
            if requirement["kind"] != "alias" or not requirement["alias_of"]:
                errors.append(f"{req_id}: alias requires kind=alias and alias_of")
            else:
                alias_edges[req_id] = requirement["alias_of"]
        elif requirement["alias_of"]:
            errors.append(f"{req_id}: only alias requirements may set alias_of")
        if requirement["status"] == "superseded":
            if not requirement["superseded_by"]:
                errors.append(f"{req_id}: superseded requirement has no replacement")
            else:
                superseded_edges[req_id] = requirement["superseded_by"]
        elif requirement["superseded_by"]:
            errors.append(f"{req_id}: only superseded requirements may set superseded_by")
        implementation_task: str | None = None
        if requirement["status"] in {"implemented", "verified"}:
            if not requirement["implementation_commit_ref"]:
                errors.append(f"{req_id}: implemented requirement lacks commit reference")
            else:
                commit_ref = requirement["implementation_commit_ref"]
                task_match = re.fullmatch(r"task:([A-Z][0-9]{3})", commit_ref)
                profile = profile_by_name.get(requirement["profile"])
                if task_match:
                    implementation_task = task_match.group(1)
                    if profile and task_match.group(1) not in profile["owner_tasks"]:
                        errors.append(f"{req_id}: task commit reference is not an owner task")
                elif not re.fullmatch(r"[0-9a-f]{40}", commit_ref):
                    errors.append(f"{req_id}: implementation commit reference is invalid")
            if not requirement["implementation_paths"]:
                errors.append(f"{req_id}: implemented requirement lacks implementation paths")
            for value in requirement["implementation_paths"]:
                _validate_repo_path(root, value, f"{req_id}.implementation_paths", errors)
                if not (root / value).exists():
                    errors.append(f"{req_id}: implementation path does not exist: {value}")
        elif requirement["implementation_commit_ref"] or requirement["implementation_paths"]:
            errors.append(f"{req_id}: only implemented/verified may set implementation evidence")
        if requirement["status"] == "verified":
            profile = profile_by_name.get(requirement["profile"])
            if not requirement["ai_review"] or not requirement["infra_review"]:
                errors.append(f"{req_id}: verified requirement lacks AI/Infra review")
            for review_kind in ("ai_review", "infra_review"):
                review = requirement[review_kind]
                if not review:
                    continue
                _validate_repo_path(root, review, f"{req_id}.{review_kind}", errors)
                review_path = root / review
                if not review.startswith("docs/model-architecture/reviews/"):
                    errors.append(f"{req_id}: review must be under the review evidence tree")
                if review_path.is_symlink() or not review_path.is_file():
                    errors.append(f"{req_id}: review evidence does not exist: {review}")
            if requirement["ai_review"] == requirement["infra_review"]:
                errors.append(f"{req_id}: AI and Infra reviews must be independent files")
            if implementation_task is not None:
                required_reviews = _required_review_paths(implementation_task)
                actual_reviews = (
                    requirement["ai_review"],
                    requirement["infra_review"],
                )
                if required_reviews is not None and actual_reviews != required_reviews:
                    errors.append(
                        f"{req_id}: task {implementation_task} review scope must be {required_reviews}"
                    )
            elif profile is not None:
                allowed_reviews = {
                    paths
                    for owner_task in profile["owner_tasks"]
                    if (paths := _required_review_paths(owner_task)) is not None
                }
                actual_reviews = (
                    requirement["ai_review"],
                    requirement["infra_review"],
                )
                if allowed_reviews and actual_reviews not in allowed_reviews:
                    errors.append(
                        f"{req_id}: commit-SHA review scope must match one declared owner task policy"
                    )
            if not requirement["evidence_artifacts"]:
                errors.append(f"{req_id}: verified requirement lacks evidence artifacts")
            for artifact in requirement["evidence_artifacts"]:
                _validate_repo_path(root, artifact, f"{req_id}.evidence_artifacts", errors)
                artifact_path = root / artifact
                if artifact_path.is_symlink() or not artifact_path.is_file():
                    errors.append(f"{req_id}: evidence artifact does not exist: {artifact}")
                if not _matches_any(artifact, requirement["artifacts"]):
                    errors.append(f"{req_id}: evidence artifact is outside its artifact mapping")
            if requirement["evidence_hardware"] not in HARDWARE_LEVELS:
                errors.append(f"{req_id}: verified requirement has invalid evidence_hardware")
            elif profile and HARDWARE_LEVELS[requirement["evidence_hardware"]] < HARDWARE_LEVELS[profile["hardware"]]:
                errors.append(
                    f"{req_id}: {requirement['evidence_hardware']} evidence cannot close {profile['hardware']} requirement"
                )
        elif (
            requirement["evidence_hardware"]
            or requirement["evidence_artifacts"]
            or requirement["ai_review"]
            or requirement["infra_review"]
        ):
            errors.append(f"{req_id}: only verified requirements may set verification evidence")

    for label, edges in (("alias", alias_edges), ("supersession", superseded_edges)):
        for origin, target in edges.items():
            if target not in requirement_by_id:
                errors.append(f"{origin}: dangling {label} target: {target}")
            if origin == target:
                errors.append(f"{origin}: {label} target cannot self-reference")
        _detect_graph_cycles(edges, label, errors)
        for origin, target in edges.items():
            target_requirement = requirement_by_id.get(target)
            if target_requirement and target_requirement["status"] in {"alias", "superseded"}:
                errors.append(f"{origin}: {label} target is not a terminal requirement")

    unmatched_source: dict[
        tuple[str, tuple[str, ...], str], list[SourceNode]
    ] = defaultdict(list)
    for locator, node in actual_nodes.items():
        if locator not in exact_source_matches:
            unmatched_source[(node.path, node.heading_path, node.kind)].append(node)
    unmatched_registry: dict[
        tuple[str, tuple[str, ...], str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for requirement in requirements:
        if requirement["id"] not in exact_requirement_matches:
            key = (
                requirement["source_path"],
                tuple(requirement["heading_path"]),
                requirement["node_kind"],
            )
            unmatched_registry[key].append(requirement)
    for key in unmatched_source.keys() | unmatched_registry.keys():
        source_nodes = unmatched_source[key]
        registry_nodes = unmatched_registry[key]
        matched_count = min(len(source_nodes), len(registry_nodes))
        for node in source_nodes[matched_count:]:
            errors.append(
                f"unregistered normative node: {node.path} :: {' > '.join(node.heading_path)} :: {node.text[:100]}"
            )
        for requirement in registry_nodes[matched_count:]:
            errors.append(
                f"{requirement['id']}: source structural slot is missing under its historical source/heading/node-kind"
            )

    history = _load_registry_history(root, registry_path, data, errors)
    if history:
        _validate_registry_history(history, errors)

    changes_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, change in enumerate(data["changes"]):
        context = f"change[{index}]"
        for key in CHANGE_KEYS - {"revision"}:
            _expect_type(change, key, str, context, errors)
        _expect_type(change, "revision", int, context, errors)
        if change["source_path"] not in source_by_path:
            errors.append(f"{context}.source_path is unknown: {change['source_path']}")
        for digest_key in ("previous_sha256", "new_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", change[digest_key]):
                errors.append(f"{context}.{digest_key} must be SHA-256")
        changes_by_source[change["source_path"]].append(change)
    for source in sources:
        changes = sorted(changes_by_source[source["path"]], key=lambda item: item["revision"])
        if source["revision"] < 1:
            errors.append(f"{source['path']}: revision must be >= 1")
        if len(changes) != source["revision"] - 1:
            errors.append(
                f"{source['path']}: revision {source['revision']} requires {source['revision'] - 1} changelog entries"
            )
        previous = source["initial_sha256"]
        for expected_revision, change in enumerate(changes, 2):
            if change["revision"] != expected_revision:
                errors.append(f"{source['path']}: changelog revision sequence is invalid")
            if change["source_path"] != source["path"]:
                errors.append(f"change source mismatch: {change['source_path']}")
            if change["previous_sha256"] != previous:
                errors.append(f"{source['path']}: changelog hash chain is broken")
            if not change["summary"] or not change["changed_at"]:
                errors.append(f"{source['path']}: changelog entry lacks summary/date")
            previous = change["new_sha256"]
        if not changes and source["sha256"] != source["initial_sha256"]:
            errors.append(f"{source['path']}: revision 1 SHA must equal initial_sha256")
        if changes and changes[-1]["new_sha256"] != source["sha256"]:
            errors.append(f"{source['path']}: latest changelog hash does not match source")

    archive_count = _validate_archive(root, data["archive_manifest"], errors)
    module_count, config_count = _validate_inventory(
        root, data["inventory"], requirements, profiles, errors
    )
    documentation = [
        path.relative_to(root).as_posix()
        for path in (root / "docs/model-architecture").rglob("*.md")
        if "archive" not in path.relative_to(root / "docs/model-architecture").parts
        and not path.is_symlink()
    ]
    _validate_local_links(root, documentation, errors)
    return VerificationReport(
        errors=errors,
        requirement_count=len(requirements),
        source_node_count=len(actual_nodes),
        archive_file_count=archive_count,
        module_count=module_count,
        config_key_count=config_count,
    )


def _profile(
    name: str,
    owner_tasks: list[str],
    *,
    config_keys: list[str],
    modules: list[str],
    tests: list[str],
    artifacts: list[str],
    hardware: str,
    benchmarks: list[str] | None = None,
    reference_modules: list[str] | None = None,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "name": name,
        "owner_tasks": owner_tasks,
        "config_keys": config_keys,
        "modules": modules,
        "reference_modules": reference_modules or [],
        "tests": tests,
        "benchmarks": benchmarks or [],
        "artifacts": artifacts,
        "hardware": hardware,
    }
    values["not_applicable_fields"] = [field for field in MAPPING_FIELDS if not values[field]]
    values["not_applicable_reason"] = (
        "This mapping dimension does not apply to this requirement profile."
        if values["not_applicable_fields"]
        else ""
    )
    return values


def _bootstrap_profiles() -> list[dict[str, Any]]:
    return [
        _profile("governance", ["D001", "G001"], config_keys=[], modules=["tools/verify_traceability.py"], tests=["tests/unit/docs/**"], artifacts=["docs/model-architecture/reviews/D001/**", "docs/model-architecture/reviews/G001/**"], hardware="CPU"),
        _profile("constraints", ["S000", "T050"], config_keys=["stage.*", "failure.*"], modules=["src/sakuramoon/train/**", "src/sakuramoon/cli/**"], tests=["tests/**"], benchmarks=["benchmarks/**"], artifacts=["artifacts/**"], hardware="4GPU"),
        _profile("architecture", ["M032"], config_keys=["model.*"], modules=["src/sakuramoon/model/**"], tests=["tests/unit/model/**"], benchmarks=["benchmarks/model/**"], artifacts=["artifacts/model/**"], hardware="1GPU"),
        _profile("vae", ["A001", "D013", "T020"], config_keys=["assets.vae.*", "data.image.*"], modules=["src/sakuramoon/assets/**", "src/sakuramoon/data/image.py", "src/sakuramoon/encoders/vae.py", "src/sakuramoon/eval/vae_reconstruction.py"], tests=["tests/**/vae*", "tests/**/image*"], benchmarks=["benchmarks/vae/**"], artifacts=["artifacts/vae/**"], hardware="1GPU"),
        _profile("caption_qwen", ["A001", "D014", "T021"], config_keys=["assets.qwen.*", "caption.*", "model.text_encoder.*"], modules=["src/sakuramoon/data/caption.py", "src/sakuramoon/data/tokenize.py", "src/sakuramoon/encoders/qwen.py"], tests=["tests/**/caption*", "tests/**/qwen*"], benchmarks=["benchmarks/qwen/**"], artifacts=["artifacts/qwen/**"], hardware="1GPU"),
        _profile("text_style", ["T022", "T023"], config_keys=["model.text_adapter.*", "model.style.*"], modules=["src/sakuramoon/conditioning/text.py", "src/sakuramoon/conditioning/style.py"], tests=["tests/**/text*", "tests/**/style*"], benchmarks=["benchmarks/conditioning/**"], artifacts=["artifacts/conditioning/**"], hardware="1GPU"),
        _profile("packing_rope", ["T024", "M031", "K001"], config_keys=["model.packing.*", "model.rope.*", "kernels.*"], modules=["src/sakuramoon/model/packing.py", "src/sakuramoon/model/rope.py", "src/sakuramoon/model/attention.py"], tests=["tests/**/packing*", "tests/**/rope*", "tests/**/attention*"], benchmarks=["benchmarks/attention/**"], artifacts=["artifacts/kernels/**"], hardware="1GPU"),
        _profile("dit", ["M031", "M032", "K001"], config_keys=["model.dit.*", "kernels.*"], modules=["src/sakuramoon/model/**"], tests=["tests/unit/model/**", "tests/integration/model/**"], benchmarks=["benchmarks/model/**"], artifacts=["artifacts/model/**", "artifacts/kernels/**"], hardware="1GPU"),
        _profile("conditioning", ["M030", "M032"], config_keys=["model.conditioning.*", "model.output_head.*"], modules=["src/sakuramoon/model/conditioning.py", "src/sakuramoon/model/head.py"], tests=["tests/**/conditioning*", "tests/**/head*"], benchmarks=["benchmarks/model/**"], artifacts=["artifacts/model/**"], hardware="1GPU"),
        _profile("objective", ["M033"], config_keys=["objective.*", "evaluation.sampling.*"], modules=["src/sakuramoon/objective/**", "src/sakuramoon/sampling/**"], tests=["tests/**/objective*", "tests/**/sampling*"], benchmarks=["benchmarks/sampling/**"], artifacts=["artifacts/objective/**"], hardware="1GPU"),
        _profile("curriculum_growth", ["C002", "D013", "T043", "S000", "S001", "S002", "S003", "S004"], config_keys=["stage.*", "growth.*", "data.image.*"], modules=["src/sakuramoon/train/stage.py", "src/sakuramoon/model/growth.py", "src/sakuramoon/checkpoint/migrate.py"], tests=["tests/**/stage*", "tests/**/growth*"], benchmarks=["benchmarks/stage/**"], artifacts=["artifacts/stages/**"], hardware="4GPU"),
        _profile("data", ["D010", "D011", "D012", "D013", "D014", "D015"], config_keys=["data.*", "paths.cache*"], modules=["src/sakuramoon/data/**"], tests=["tests/**/data*", "tests/**/cache*"], benchmarks=["benchmarks/data/**"], artifacts=["artifacts/data/**"], hardware="1GPU"),
        _profile("training_system", ["R002", "T040", "T041", "T042", "T043", "T050", "T054", "S000", "S001", "S002", "S003"], config_keys=["optimizer.*", "scheduler.*", "gradient.*", "distributed.*", "checkpoint.*", "failure.*"], modules=["src/sakuramoon/optim/**", "src/sakuramoon/distributed/**", "src/sakuramoon/checkpoint/**", "src/sakuramoon/train/**"], tests=["tests/**/optim*", "tests/**/distributed*", "tests/**/checkpoint*", "tests/fault_injection/**"], benchmarks=["benchmarks/training/**"], artifacts=["artifacts/training/**", "checkpoints/**"], hardware="4GPU"),
        _profile("observability", ["C001", "C002", "T051", "T052", "T053"], config_keys=["logging.*", "wandb.*", "timing.*", "profiling.*", "evaluation.*"], modules=["src/sakuramoon/telemetry/**", "src/sakuramoon/eval/**"], tests=["tests/**/telemetry*", "tests/**/eval*"], benchmarks=["benchmarks/**"], artifacts=["artifacts/metrics/**", "artifacts/evaluation/**", "artifacts/profiles/**"], hardware="4GPU"),
        _profile("dropout_decision", ["C002", "D014"], config_keys=["caption.dropout.*"], modules=["src/sakuramoon/data/caption.py"], tests=["tests/**/caption*"], artifacts=["artifacts/data/caption_dry_run*"], hardware="CPU"),
        _profile("alias", ["D001"], config_keys=[], modules=[], tests=["tests/unit/docs/**"], artifacts=["docs/model-architecture/progress/traceability.toml"], hardware="CPU"),
        _profile("post512", ["S004"], config_keys=["stage.h1.*", "stage.h2.*"], modules=["src/sakuramoon/train/stage.py"], tests=["tests/**/stage*"], benchmarks=["benchmarks/stage/**"], artifacts=["artifacts/stages/**"], hardware="4GPU"),
    ]


def _top_number(heading: str) -> int | None:
    match = re.match(r"([0-9]+)\.", heading)
    return int(match.group(1)) if match else None


def _find_requirement_by_text(
    requirements: Sequence[dict[str, Any]], nodes: Mapping[str, SourceNode], needle: str
) -> str:
    for requirement in requirements:
        node = nodes.get(requirement["id"])
        if node and needle in node.text:
            return requirement["id"]
    raise ValueError(f"cannot find canonical requirement containing: {needle}")


def bootstrap_registry(root: Path) -> dict[str, Any]:
    source_specs: list[dict[str, Any]] = []
    for path, canonical in CANONICAL_SOURCES.items():
        source_specs.append(
            {
                "path": path,
                "kind": canonical["kind"],
                "revision": 1,
                "include_top_headings": list(
                    cast(Sequence[str], canonical["include_top_headings"])
                ),
                "excluded_top_headings": list(
                    cast(Sequence[str], canonical["excluded_top_headings"])
                ),
            }
        )
    for source in source_specs:
        source_path = cast(str, source["path"])
        digest = sha256_file(root / source_path)
        source["sha256"] = digest
        canonical = CANONICAL_SOURCES[source_path]
        source["initial_sha256"] = canonical["initial_sha256"]

    profiles = _bootstrap_profiles()
    profile_by_name = {profile["name"]: profile for profile in profiles}
    requirements: list[dict[str, Any]] = []
    node_by_id: dict[str, SourceNode] = {}
    counters: Counter[str] = Counter()
    confirmed_map = {
        0: ("DOC", "governance"),
        1: ("C01", "constraints"),
        2: ("ARCH", "architecture"),
        3: ("C02", "vae"),
        4: ("C03", "caption_qwen"),
        5: ("C04", "text_style"),
        6: ("C05", "packing_rope"),
        7: ("C06", "dit"),
        8: ("C07", "conditioning"),
        9: ("C08", "objective"),
        10: ("C10", "curriculum_growth"),
        11: ("C11", "data"),
        12: ("C12", "training_system"),
        13: ("DEC", "dropout_decision"),
        14: ("SUP", "governance"),
    }

    def add_requirement(
        node: SourceNode,
        prefix: str,
        profile: str,
        *,
        kind: str = "requirement",
        status: str = "planned",
        blocked_by: list[str] | None = None,
        alias_of: str = "",
        superseded_by: str = "",
    ) -> dict[str, Any]:
        counters[prefix] += 1
        req_id = f"{prefix}-{counters[prefix]:03d}"
        mapping = profile_by_name[profile]
        requirement = {
            "id": req_id,
            "kind": kind,
            "status": status,
            "profile": profile,
            "source_path": node.path,
            "heading_path": list(node.heading_path),
            "node_kind": node.kind,
            "source_fingerprint": node.fingerprint,
            "source_occurrence": node.occurrence,
            "blocked_by": blocked_by or [],
            "alias_of": alias_of,
            "superseded_by": superseded_by,
            "implementation_commit_ref": "",
            "implementation_paths": [],
            "evidence_hardware": "",
            "evidence_artifacts": [],
            "ai_review": "",
            "infra_review": "",
            **{field: list(mapping[field]) for field in MAPPING_FIELDS},
            "not_applicable_fields": list(mapping["not_applicable_fields"]),
            "not_applicable_reason": mapping["not_applicable_reason"],
        }
        requirements.append(requirement)
        node_by_id[req_id] = node
        return requirement

    confirmed_nodes = extract_source_nodes(root, source_specs[0])
    for node in confirmed_nodes:
        section = _top_number(node.heading_path[0])
        if section not in confirmed_map:
            raise ValueError(f"unmapped confirmed heading: {node.heading_path[0]}")
        prefix, profile = confirmed_map[section]
        if section == 13:
            add_requirement(node, prefix, profile, status="blocked", blocked_by=["DECISION-DROPOUT-VALUES"])
        elif section == 14:
            add_requirement(node, prefix, profile, kind="supersession")
        else:
            add_requirement(node, prefix, profile)

    canonical_needles = [
        "Microsoft 官方 Mage-VAE",
        "ModelScope",
        "主文本 caption 类别骨架固定",
        "Artist 只进入 style 分支",
        "text_condition_max=512",
        "7 层做 gated softmax mixing",
        "三类输入分别加 learned",
        "hidden_size=2560",
        "condition_hidden",
        "网络输出 `x_pred`",
        "首版阶段顺序固定",
        "S0 使用单卡原生模型",
    ]
    canonical_targets = [
        _find_requirement_by_text(requirements, node_by_id, needle)
        for needle in canonical_needles
    ]
    archive_policy_target = _find_requirement_by_text(
        requirements, node_by_id, "历史决定中自动取值"
    )
    condition_target = _find_requirement_by_text(
        requirements, node_by_id, "text_condition_max=512"
    )

    open_profile = {
        0: "governance",
        1: "dropout_decision",
        2: "alias",
        3: "data",
        4: "dit",
        5: "training_system",
        6: "training_system",
        7: "curriculum_growth",
        8: "observability",
        9: "post512",
        10: "governance",
    }
    closed_index = 0
    for node in extract_source_nodes(root, source_specs[1]):
        section = _top_number(node.heading_path[0])
        if section is None or section not in open_profile:
            raise ValueError(f"unmapped open-items heading: {node.heading_path[0]}")
        if section == 1:
            add_requirement(node, "OPEN", open_profile[section], kind="open_item", status="blocked", blocked_by=["DECISION-DROPOUT-VALUES"])
        elif section == 2:
            target = canonical_targets[closed_index]
            closed_index += 1
            add_requirement(node, "OPEN", "alias", kind="alias", status="alias", alias_of=target)
        elif section == 10 and node.text.startswith("[x]"):
            add_requirement(node, "OPEN", "alias", kind="alias", status="alias", alias_of=condition_target)
        elif section == 10:
            add_requirement(node, "OPEN", "governance", kind="historical_cleanup", status="superseded", superseded_by=archive_policy_target)
        else:
            add_requirement(node, "OPEN", open_profile[section], kind="open_item")

    for node in extract_source_nodes(root, source_specs[2]):
        add_requirement(node, "OBS", "observability")

    return {
        "schema_version": 1,
        "registry_revision": 1,
        "archive_manifest": "docs/model-architecture/SHA256SUMS",
        "roadmap": "docs/model-architecture/progress/IMPLEMENTATION_ROADMAP.md",
        "sources": source_specs,
        "profiles": profiles,
        "blockers": [
            {
                "id": "DECISION-DROPOUT-VALUES",
                "kind": "user_decision",
                "description": "All dropout probabilities except all_condition=0.10 remain explicitly undecided.",
            }
        ],
        "inventory": {
            "module_root": "src/sakuramoon",
            "config_root": "config",
            "ignored_module_paths": [],
            "ignored_config_paths": [],
            "no_modules_reason": "No production source tree exists at D001; the checker will require reverse mappings as soon as one is added.",
            "no_configs_reason": "No runtime config TOML exists at D001; the checker will require reverse mappings as soon as one is added.",
        },
        "changes": [],
        "requirements": requirements,
    }


def _write_bootstrap(root: Path, registry_path: Path) -> None:
    destination = root / registry_path
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing registry: {registry_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(tomli_w.dumps(bootstrap_registry(root)), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.bootstrap:
        try:
            _write_bootstrap(root, args.registry)
        except (OSError, ValueError) as exc:
            print(f"bootstrap failed: {exc}", file=sys.stderr)
            return 2
    report = verify(root, args.registry)
    if args.format == "json":
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    elif report.ok:
        counts = report.as_dict()["counts"]
        print(
            "traceability verification passed: "
            + ", ".join(f"{key}={value}" for key, value in counts.items())
        )
    else:
        for error in report.errors:
            print(f"ERROR: {error}", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
