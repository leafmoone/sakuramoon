#!/usr/bin/env python
"""Audit a training metrics JSONL window without modifying it.

The audit separates three questions that are easy to conflate:

* which update numbers are present in the file;
* which telemetry rows are missing inside a requested window;
* which rows are exact duplicates or disagree with another row for the same
  update.

A missing telemetry row is reported as a telemetry gap.  It is not evidence
that the trainer never executed that update: the source run may have stopped
at a save boundary, and the target run may resume at the next update.

The tool is intentionally read-only and standard-library only.  It does not
hash the input (the project keeps no project-level artifact hashes); it
records size, modification time, line counts and update coverage instead.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MISSING_ROW_NOTE = (
    "Missing telemetry rows are gaps in this file, not proof that the "
    "corresponding updates were never executed."
)


@dataclass(frozen=True)
class WindowStatistics:
    """Descriptive statistics for one update window."""

    rows: int
    loss_mean: float | None
    loss_median: float | None
    pre_clip_median: float | None
    pre_clip_max: float | None
    clip_fraction_mean: float | None
    clipped_updates: int
    nonfinite_total: int


def _open_text(path: Path) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_lines(path: Path) -> Iterator[tuple[int, str]]:
    with _open_text(path) as handle:
        yield from enumerate(handle, 1)


def _update_of(row: dict[str, Any]) -> int | None:
    value = row.get("successful_update")
    if type(value) is int:
        return value
    return None


def _number(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if type(value) is int or type(value) is float:
        return float(value)
    return None


def _loss_of(row: dict[str, Any], field: str) -> float | None:
    value = _number(row, field)
    if value is not None:
        return value
    return _number(row, "total_loss")


def _summarize(rows: Iterable[dict[str, Any]], loss_field: str) -> WindowStatistics:
    rows = list(rows)
    losses = [value for row in rows if (value := _loss_of(row, loss_field)) is not None]
    gradients = [
        value
        for row in rows
        if (value := _number(row, "pre_clip_grad_norm")) is not None
    ]
    clip_fractions = [
        value
        for row in rows
        if (value := _number(row, "clip_fraction")) is not None
    ]
    return WindowStatistics(
        rows=len(rows),
        loss_mean=round(statistics.fmean(losses), 6) if losses else None,
        loss_median=round(statistics.median(losses), 6) if losses else None,
        pre_clip_median=(
            round(statistics.median(gradients), 6) if gradients else None
        ),
        pre_clip_max=round(max(gradients), 6) if gradients else None,
        clip_fraction_mean=(
            round(statistics.fmean(clip_fractions), 6) if clip_fractions else None
        ),
        clipped_updates=sum(1 for value in clip_fractions if value > 0.0),
        nonfinite_total=sum(
            int(value)
            for row in rows
            if (value := _number(row, "nonfinite_count")) is not None
        ),
    )


def audit(
    path: Path,
    *,
    start: int | None,
    end: int | None,
    loss_field: str,
) -> dict[str, Any]:
    """Return an audit report for ``path``.

    ``start`` and ``end`` are inclusive update bounds.  When omitted, the
    window is the full update range present in the file.
    """

    parsed: list[tuple[int, str, dict[str, Any]]] = []
    malformed: list[dict[str, Any]] = []
    for line_number, raw in _iter_lines(path):
        text = raw.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            malformed.append(
                {"line": line_number, "error": f"{type(exc).__name__}: {exc.msg}"}
            )
            continue
        if type(row) is not dict:
            malformed.append({"line": line_number, "error": "row is not an object"})
            continue
        parsed.append((line_number, text, row))

    updates = [
        update
        for _, _, row in parsed
        if (update := _update_of(row)) is not None
    ]
    present = set(updates)
    first_update = min(updates) if updates else None
    last_update = max(updates) if updates else None
    window_start = start if start is not None else first_update
    window_end = end if end is not None else last_update

    selected: list[dict[str, Any]] = []
    exact_duplicates = 0
    if window_start is not None and window_end is not None:
        if window_start > window_end:
            raise ValueError("start must not be greater than end")
        seen_text: set[str] = set()
        by_update: dict[int, list[str]] = {}
        for _, text, row in parsed:
            update = _update_of(row)
            if update is None or not window_start <= update <= window_end:
                continue
            selected.append(row)
            if text in seen_text:
                exact_duplicates += 1
            seen_text.add(text)
            by_update.setdefault(update, []).append(text)
        conflicting = sorted(
            update for update, texts in by_update.items() if len(set(texts)) > 1
        )
        expected = window_end - window_start + 1
        distinct_updates = len(by_update)
        missing = sorted(
            update for update in range(window_start, window_end + 1)
            if update not in present
        )
    else:
        conflicting = []
        expected = 0
        distinct_updates = 0
        missing = []

    schema_versions = sorted(
        {
            int(value)
            for _, _, row in parsed
            if type(value := row.get("schema_version")) is int
        }
    )
    stat = path.stat()
    return {
        "schema_version": SCHEMA_VERSION,
        "audited_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "file": {
            "path": str(path),
            "bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime)
            .astimezone()
            .isoformat(timespec="seconds"),
            "rows": len(parsed),
            "malformed_rows": len(malformed),
            "first_update": first_update,
            "last_update": last_update,
            "schema_versions": schema_versions,
        },
        "window": {
            "start": window_start,
            "end": window_end,
            "expected_updates": expected,
            "selected_rows": len(selected),
            "distinct_updates": distinct_updates,
            "coverage": (
                round(distinct_updates / expected, 6) if expected else None
            ),
            "missing_updates": missing,
            "exact_duplicate_rows": exact_duplicates,
            "conflicting_updates": conflicting,
        },
        "statistics": asdict(_summarize(selected, loss_field)),
        "malformed": malformed[:50],
        "notes": [MISSING_ROW_NOTE],
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="metrics.jsonl or .jsonl.gz")
    parser.add_argument("--start", type=int, default=None, help="inclusive update")
    parser.add_argument("--end", type=int, default=None, help="inclusive update")
    parser.add_argument("--loss-field", default="main_loss")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit(
        args.metrics,
        start=args.start,
        end=args.end,
        loss_field=args.loss_field,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out is None:
        print(rendered)
    else:
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
