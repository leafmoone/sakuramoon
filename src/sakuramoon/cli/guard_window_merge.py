"""Offline, fail-closed merge of one guard-window session.

Run with ``python -m sakuramoon.cli.guard_window_merge --input DIR ...
--output merged.json --start-update N --end-update M``. Bounds are inclusive
and mandatory: completeness covers only that requested interval, not a session.
Directories contribute formal *.jsonl files, never claims or .tmp contents.

``--legacy FILE ...`` is an explicit, opt-in partial mode (no extra allow flag).
Legacy rows retain their original payload and unknown identity/delta; they never
supply missing modern ranks or improve coverage. Mixed output is marked partial.
An optional ``--routing-manifest`` accepts the expected routing list as JSON;
without it the real writer's embedded manifest is the provenance anchor, NOT
independent proof of the model's routing. Owner recomputation is mandatory.

Only the owner check lazily imports the actual optimizer's stable_owner. No
training, checkpoint, tensor or runtime interfaces are used. Publication uses
an fsynced, closed temp and the shared no-clobber helper; directory-entry crash
durability is not promised. Errors intentionally propagate with tracebacks.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

_R_FIELDS = ("observations", "skip_total", "skip_by_role", "skip_by_fqn")
_O_FIELDS = (
    "bf16_attempts",
    "bf16_safety_failures",
    "fp32_attempts",
    "fp32_rescues",
    "fp32_rescue_failures",
    "rescue_by_role",
)
Count = int | dict[str, int]
Counts = dict[str, Count]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")  # noqa: TRY004 - invalid JSON contract
    raw = cast("dict[object, object]", value)
    _require(all(isinstance(k, str) for k in raw), f"{label}: non-string key")
    return cast("dict[str, object]", raw)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")  # noqa: TRY004 - invalid JSON contract
    return cast("list[object]", value)


def _int(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}, not {value!r}")
    return value


def _str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _routing(value: object) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    names: set[str] = set()
    for item in _list(value, "routing"):
        spec = _object(item, "routing spec")
        name = _str(spec.get("fqn"), "routing fqn")
        _require(name not in names, f"duplicate routing fqn: {name}")
        names.add(name)
        count = _int(spec.get("chunk_count"), "chunk_count", 1)
        dim = _int(spec.get("chunk_dim"), "chunk_dim")
        shape = [_int(n, "shape extent", 1) for n in _list(spec.get("shape"), "shape")]
        _require(dim < len(shape), "chunk_dim outside shape")
        _require(shape[dim] % count == 0, "shape not divisible by chunk_count")
        roles = [_str(r, "semantic role") for r in _list(spec.get("roles"), "roles")]
        _require(len(roles) == count, "semantic roles must cover every chunk")
        result.append(
            {
                "fqn": name,
                "shape": shape,
                "chunk_count": count,
                "chunk_dim": dim,
                "roles": roles,
                "role": _str(spec.get("role"), "canonical role"),
            }
        )
    _require(bool(result), "routing must contain at least one CMuon spec")
    return sorted(result, key=lambda spec: str(spec["fqn"]))


def _partition(routing: list[dict[str, object]], world: int) -> dict[str, list[str]]:
    # Use the actual implementation, never a second copy of the hash algorithm.
    from sakuramoon.optim.guarded_canonical import stable_owner

    result: dict[str, list[str]] = {str(rank): [] for rank in range(world)}
    for spec in routing:
        name = cast("str", spec["fqn"])
        for chunk in range(cast("int", spec["chunk_count"])):
            result[str(stable_owner(name, chunk, world))].append(f"{name}#{chunk}")
    return {rank: sorted(chunks) for rank, chunks in result.items()}


def _counts(
    value: object, fields: tuple[str, ...], roles: set[str], fqns: set[str]
) -> Counts:
    raw = _object(value, "counts")
    _require(set(raw) == set(fields), "missing or unexpected counter field")
    result: Counts = {}
    for field in fields:
        if field in ("skip_by_role", "rescue_by_role", "skip_by_fqn"):
            domain = fqns if field == "skip_by_fqn" else roles
            sparse = _object(raw[field], field)
            _require(set(sparse) <= domain, f"{field}: illegal canonical role/FQN key")
            result[field] = {
                key: _int(sparse.get(key, 0), f"{field}.{key}")
                for key in sorted(domain)
            }
        else:
            result[field] = _int(raw[field], field)
    return result


def _difference(end: Counts, baseline: Counts) -> Counts:
    result: Counts = {}
    for field, value in end.items():
        base = baseline[field]
        if isinstance(value, dict):
            assert isinstance(base, dict)
            result[field] = {
                key: _int(n - base[key], f"negative delta {field}.{key}")
                for key, n in value.items()
            }
        else:
            assert isinstance(base, int)
            result[field] = _int(value - base, f"negative delta {field}")
    return result


@dataclass
class _Record:
    raw: dict[str, object]
    identity: dict[str, object]
    rank: int
    start: int
    end: int
    baseline: Counts
    absolute: Counts
    delta: Counts


def _record(value: object, expected_routing: list[dict[str, object]] | None) -> _Record:
    raw = _object(value, "record")
    _require(
        _int(raw.get("schema_version"), "schema_version") == 1,
        "unsupported schema_version",
    )
    run = _str(raw.get("run_id"), "run_id")
    session = _str(raw.get("session_id"), "session_id")
    world = _int(raw.get("world_size"), "world_size", 1)
    rank = _int(raw.get("rank"), "rank")
    _require(rank < world, "rank outside world_size")
    start = _int(raw.get("start_update"), "start_update")
    end = _int(raw.get("end_update"), "end_update")
    _require(start <= end, "start_update exceeds end_update")
    launch = _object(raw.get("launch"), "launch")
    kind = _str(launch.get("launch_kind"), "launch.launch_kind")
    _require(kind in ("fresh_bootstrap", "user_resume"), "invalid launch_kind")
    variant = _str(launch.get("optimizer"), "launch.optimizer")
    _require(
        variant
        in (
            "hybrid_cmuon_guarded_canonical",
            "hybrid_cmuon_canonical_ns4_fp32_rescue",
        ),
        "unsupported guard optimizer variant",
    )
    resume = _str(launch.get("resume_source_checkpoint"), "resume_source_checkpoint")
    resumed_at = _int(
        launch.get("resume_successful_updates"), "resume_successful_updates"
    )
    _require(start > resumed_at, "window precedes resume boundary")
    _require(
        launch.get("owner_mapping_version") == "fnv1a64-v1",
        "unsupported owner_mapping_version",
    )
    _require(
        "last_published_checkpoint" in launch,
        "missing last_published_checkpoint metadata",
    )
    checkpoint = launch["last_published_checkpoint"]
    _require(
        checkpoint is None or isinstance(checkpoint, str), "invalid checkpoint metadata"
    )
    _require(
        raw.get("window_id") == f"{session}:u{start}-{end}:c{resume}",
        "window_id mismatch",
    )
    _require(type(raw.get("complete")) is bool, "complete must be bool")
    _require(
        "run_outcome" in raw
        and raw["run_outcome"] in (None, "clean", "raised", "unknown"),
        "invalid run_outcome",
    )
    _require(raw.get("delta_coverage") == "complete", "unknown delta coverage")
    routing = _routing(raw.get("routing"))
    if expected_routing is not None:
        _require(routing == expected_routing, "routing differs from expected manifest")
    partition = _object(raw.get("owner_partition"), "owner_partition")
    expected = _partition(routing, world)
    _require(set(partition) == set(expected), "owner_partition world mismatch")
    for owner, chunks in expected.items():
        supplied = [
            _str(c, "owner chunk") for c in _list(partition[owner], "owner chunks")
        ]
        _require(
            sorted(supplied) == chunks,
            f"owner_partition mismatch for rank {owner}: duplicate, missing or wrong owner chunk",
        )
    roles = {cast("str", spec["role"]) for spec in routing}
    fqns = {cast("str", spec["fqn"]) for spec in routing}
    absolute_raw = _object(raw.get("absolute"), "absolute")
    has_owner = variant == "hybrid_cmuon_canonical_ns4_fp32_rescue"
    fields = _R_FIELDS + (_O_FIELDS if has_owner else ())
    baseline_raw = _object(raw.get("baseline"), "baseline")
    source = "fresh" if resume == "fresh-start" else "restored_rank0"
    _require(
        baseline_raw.get("source") == source,
        "baseline source disagrees with resume identity",
    )
    baseline = _counts(baseline_raw.get("counts"), fields, roles, fqns)
    absolute = _counts(absolute_raw, fields, roles, fqns)
    delta = _counts(raw.get("delta"), fields, roles, fqns)
    status = _object(raw.get("delta_status"), "delta_status")
    _require(
        set(status) == set(fields) and all(v == "ok" for v in status.values()),
        "delta_status must be ok for every field",
    )
    _require(
        delta == _difference(absolute, baseline),
        "forged delta: does not match endpoints",
    )
    _require(
        delta["observations"] == end - start + 1,
        "observations delta does not match the successful-update interval",
    )
    for field in ("skip_by_role", "skip_by_fqn"):
        _require(
            sum(cast("dict[str, int]", delta[field]).values()) == delta["skip_total"],
            f"{field} total differs from skip_total delta",
        )
    if has_owner:
        bf_attempts = cast("int", delta["bf16_attempts"])
        bf_failures = cast("int", delta["bf16_safety_failures"])
        fp_attempts = cast("int", delta["fp32_attempts"])
        rescues = cast("int", delta["fp32_rescues"])
        failures = cast("int", delta["fp32_rescue_failures"])
        _require(
            bf_failures == fp_attempts <= bf_attempts,
            "BF16/FP32 attempt deltas are inconsistent",
        )
        _require(
            rescues + failures <= fp_attempts,
            "FP32 candidate verdicts exceed attempts",
        )
        _require(
            sum(cast("dict[str, int]", delta["rescue_by_role"]).values()) == rescues,
            "rescue_by_role total differs from fp32_rescues delta",
        )
        owned = set(expected[str(rank)])
        if not owned:
            _require(
                all(
                    delta[field] == 0
                    for field in _O_FIELDS
                    if field != "rescue_by_role"
                ),
                "owner-local increment on a rank with no owned chunk",
            )
        owned_roles = {
            cast("str", spec["role"])
            for spec in routing
            if any(
                f"{spec['fqn']}#{chunk}" in owned
                for chunk in range(cast("int", spec["chunk_count"]))
            )
        }
        rescue_roles = cast("dict[str, int]", delta["rescue_by_role"])
        _require(
            all(
                value == 0 or role in owned_roles
                for role, value in rescue_roles.items()
            ),
            "rescue role increment outside this rank's owner partition",
        )
    history = _object(raw.get("absolute_history"), "absolute_history")
    expected_history = {field: "rank_provenance" for field in _R_FIELDS}
    if has_owner:
        expected_history.update(
            {
                field: "local" if source == "fresh" else "rank0_owner_only"
                for field in _O_FIELDS
            }
        )
    _require(history == expected_history, "invalid absolute historical provenance")
    identity: dict[str, object] = {
        "run_id": run,
        "session_id": session,
        "optimizer": variant,
        "launch_kind": kind,
        "world_size": world,
        "resume_source_checkpoint": resume,
        "resume_successful_updates": resumed_at,
        "owner_mapping_version": "fnv1a64-v1",
        "routing": routing,
        "owner_partition": expected,
        "counter_fields": list(fields),
    }
    return _Record(raw, identity, rank, start, end, baseline, absolute, delta)


def _merge_delta(records: list[_Record]) -> Counts:
    first = records[0]
    for record in records[1:]:
        for field in _R_FIELDS:
            _require(
                all(
                    getattr(record, attr)[field] == getattr(first, attr)[field]
                    for attr in ("baseline", "absolute", "delta")
                ),
                f"replicated counter mismatch: {field}",
            )
    result: Counts = {field: first.delta[field] for field in _R_FIELDS}
    for field in _O_FIELDS:
        if field not in first.delta:
            continue
        value = first.delta[field]
        if isinstance(value, dict):
            result[field] = {
                key: sum(
                    cast("dict[str, int]", record.delta[field])[key]
                    for record in records
                )
                for key in value
            }
        else:
            result[field] = sum(cast("int", record.delta[field]) for record in records)
    return result


def merge_records(
    records: Sequence[object],
    *,
    start_update: int,
    end_update: int,
    legacy_records: Sequence[object] = (),
    routing_manifest: object | None = None,
) -> dict[str, object]:
    """Validate all supplied records; merge one exact, inclusive interval.

    No input is silently filtered to the interval. Modern gaps, abort tails and
    unknown counters raise ValueError. Legacy-only is explicitly partial; mixed
    legacy cannot repair modern gaps. No global historical absolute is emitted.
    """
    _int(start_update, "requested start_update")
    _int(end_update, "requested end_update")
    _require(start_update <= end_update, "requested start_update exceeds end_update")
    expected = None if routing_manifest is None else _routing(routing_manifest)
    parsed = [_record(row, expected) for row in records]
    legacy: list[dict[str, object]] = []
    for value in legacy_records:
        row = _object(value, "legacy record")
        _require(
            not (set(row) & {"schema_version", "run_id", "session_id", "window_id"}),
            "modern identity is not allowed in explicit legacy mode",
        )
        legacy.append(
            {
                "record": row,
                "coverage": "partial",
                "owner_coverage": "rank0_owner_only",
                "replicated_coverage": "unverified_rank0",
                "identity": None,
                "delta": None,
            }
        )
    _require(bool(parsed or legacy), "no input records")
    windows: list[dict[str, object]] = []
    identity = parsed[0].identity if parsed else None
    groups: dict[tuple[int, int], dict[int, _Record]] = {}
    for record in parsed:
        _require(record.identity == identity, "session identity/routing/world mismatch")
        group = groups.setdefault((record.start, record.end), {})
        _require(
            record.rank not in group,
            "duplicate rank/window (even identical payloads are forbidden)",
        )
        group[record.rank] = record
    previous: list[_Record] | None = None
    cursor = start_update
    cadence: int | None = None
    for index, ((start, end), ranks) in enumerate(sorted(groups.items())):
        assert identity is not None
        world = cast("int", identity["world_size"])
        _require(set(ranks) == set(range(world)), "window missing/extra rank")
        rows = [ranks[rank] for rank in range(world)]
        first = rows[0].raw
        _require(
            all(
                (row.raw["complete"], row.raw["run_outcome"])
                == (first["complete"], first["run_outcome"])
                for row in rows
            ),
            "rank window metadata mismatch",
        )
        _require(
            start == cursor and end <= end_update,
            "window gap/overlap or requested bounds mismatch",
        )
        length = end - start + 1
        if first["complete"]:
            _require(
                first["run_outcome"] in (None, "clean"),
                "aborted/unknown window cannot be complete",
            )
            cadence = length if cadence is None else cadence
            _require(length == cadence, "full-window cadence mismatch")
        else:
            _require(
                first["run_outcome"] == "clean" and index == len(groups) - 1,
                "incomplete window requires a clean final tail",
            )
            _require(
                cadence is None or length < cadence,
                "tail must be shorter than full cadence",
            )
        if index < len(groups) - 1:
            _require(
                first["run_outcome"] is None, "records follow a terminal run_outcome"
            )
        if previous is not None:
            _require(
                all(
                    old.absolute == new.baseline
                    for old, new in zip(previous, rows, strict=True)
                ),
                "adjacent endpoint chain mismatch",
            )
        merged = _merge_delta(rows)
        windows.append(
            {
                "window_id": first["window_id"],
                "start_update": start,
                "end_update": end,
                "complete": first["complete"],
                "run_outcome": first["run_outcome"],
                "delta_coverage": "complete",
                "delta": merged,
                "per_rank": [
                    {
                        "rank": row.rank,
                        "baseline": {
                            "source": _object(row.raw["baseline"], "baseline")[
                                "source"
                            ],
                            "counts": row.baseline,
                        },
                        "absolute": row.absolute,
                        "absolute_history": row.raw["absolute_history"],
                        "launch": row.raw["launch"],
                    }
                    for row in rows
                ],
            }
        )
        previous = rows
        cursor = end + 1
    if parsed:
        _require(cursor == end_update + 1, "missing requested tail")
    return {
        "schema": {
            "name": "guard_window_merge",
            "version": 1,
            "input_schema_version": 1,
            "replicated_fields": list(_R_FIELDS),
            "owner_local_fields": list(_O_FIELDS),
        },
        "integrity": {
            "coverage": "partial" if legacy else "complete",
            "modern_coverage": "complete" if parsed else "unknown",
            "requested_interval": {
                "start_update": start_update,
                "end_update": end_update,
            },
            "identity": identity,
            "owner_partition_verified": bool(parsed),
            "routing_anchor": (
                "external_manifest"
                if expected is not None
                else "embedded_writer_manifest"
            )
            if parsed
            else None,
            "absolute_history": "per_rank_only",
            "session_completion": "not_asserted",
            "run_outcome_scope": "observed_training_loop",
        },
        "windows": windows,
        "legacy": legacy,
    }


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite JSON value: {value}")


def _loads(text: str) -> object:
    return cast(
        "object",
        json.loads(text, object_pairs_hook=_unique_object, parse_constant=_nonfinite),
    )


def read_records(paths: Sequence[Path], *, directories: bool = True) -> list[object]:
    """Read explicit JSONL files or recursively expand formal directory files."""
    result: list[object] = []
    for path in paths:
        if path.is_dir():
            _require(directories, "legacy inputs must be explicit files")
            files = sorted(
                p
                for p in path.rglob("*.jsonl")
                if not any(
                    part.startswith(".") or ".tmp" in part
                    for part in p.relative_to(path).parts
                )
            )
        else:
            _require(
                not any(".tmp" in part or part == ".claimed" for part in path.parts),
                "not a formal input file",
            )
            files = [path]
        for file in files:
            with file.open(encoding="utf-8") as stream:
                for line_no, line in enumerate(stream, 1):
                    if line.strip():
                        try:
                            result.append(_loads(line))
                        except ValueError as error:
                            raise ValueError(f"{file}:{line_no}: {error}") from error
    return result


def write_output(target: Path, document: dict[str, object]) -> None:
    """Serialize before publication; existing targets raise without modification."""
    from sakuramoon.telemetry.guard_publish import (
        PublicationStatus,
        publish_atomically,
    )

    payload = (
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)

    def prepare(temp: Path) -> None:
        with temp.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    result = publish_atomically(target, target.parent / ".tmp", prepare)
    if result.status is PublicationStatus.TARGET_EXISTS:
        raise FileExistsError(f"refusing to overwrite existing output: {target}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, default=[])
    parser.add_argument("--legacy", nargs="+", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-update", type=int, required=True)
    parser.add_argument("--end-update", type=int, required=True)
    parser.add_argument("--routing-manifest", type=Path)
    args = parser.parse_args(argv)
    inputs = cast("list[Path]", args.input)
    legacy = cast("list[Path]", args.legacy)
    if not inputs and not legacy:
        parser.error("at least one --input or --legacy is required")
    manifest_path = cast("Path | None", args.routing_manifest)
    manifest = (
        None
        if manifest_path is None
        else _loads(manifest_path.read_text(encoding="utf-8"))
    )
    document = merge_records(
        read_records(inputs),
        start_update=args.start_update,
        end_update=args.end_update,
        legacy_records=read_records(legacy, directories=False),
        routing_manifest=manifest,
    )
    write_output(args.output, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
