#!/usr/bin/env python
"""Audit a training metrics JSONL window without modifying it.

The audit separates three questions that are easy to conflate:

* which update numbers are present in the file;
* which telemetry rows are missing inside a requested window;
* which rows are byte-identical, which repeat the same payload, and which
  disagree with another row for the same update.

A missing telemetry row is a telemetry gap, not evidence that the trainer
never executed that update: the source run may have stopped at a save
boundary, and the target run may resume at the next update.

Duplicate detection ignores ``recorded_at_unix_ns``.  The producer writes a
fresh timestamp into every row, so a rewritten row is never byte-identical
even when its payload is unchanged.

A conflicting update (two rows with different payloads for one update) is
excluded from statistics entirely instead of resolving to whichever row
happened to be read last; the statistics describe conflict-free updates
only, and ``window.conflicting_update_count`` identifies them.

Schema 4 adds ``file.status``: ``ok``, ``malformed_rows``, or
``container_error`` (in that priority order, with container errors highest).
This file-wide status is independent of the existing ``window.status``:
malformed JSON syntax and non-object JSON values cannot be assigned to an
update window.  ``file.malformed_rows`` counts all such nonblank lines read;
``malformed`` retains only the first 50 line/error details.  Valid surrounding
rows still contribute to the report.  JSON objects without an integer
``successful_update`` remain separately counted in ``file.rows_without_update``;
they are not malformed and do not themselves cause a nonzero exit.

The tool is read-only and standard-library only.  It does not persist hashes
of the input; the transient digests below exist only to separate the three
duplicate buckets.

Exit status: 0 when the audit completed without conflicts or malformed rows;
1 when the input container failed mid-read (the partial report is still
emitted); 2 for invalid arguments or inverted window bounds; 3 when the report
could not be written to ``--out``; 4 when the audited window contains conflicting
updates (those updates are excluded from statistics); 5 when malformed rows
make the audit incomplete.  Reports are emitted to stdout or ``--out`` before
returning 1, 4, or 5, with precedence 1 over 4 over 5.  Argument and output-write
failures retain their existing precedence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import statistics
import sys
import zlib
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO, cast

SCHEMA_VERSION = 4
VOLATILE_KEYS = frozenset({"recorded_at_unix_ns"})
MAX_MALFORMED = 50
MAX_MISSING_RANGES = 1000
EXIT_OK = 0
EXIT_CONTAINER_ERROR = 1
EXIT_USAGE = 2
EXIT_WRITE_ERROR = 3
EXIT_CONFLICTS = 4
EXIT_MALFORMED = 5


MISSING_ROW_NOTE = (
    "Missing telemetry rows are gaps in this file, not proof that the "
    "corresponding updates were never executed."
)
NONFINITE_NOTE = (
    "The current producer writes nonfinite_count as a constant 0; a zero "
    "total does not by itself prove that every update was finite."
)
CONFLICT_NOTE = (
    "Updates whose rows conflict are excluded from statistics; see "
    "window.conflicting_updates and statistics.conflicting_updates_excluded."
)


class WindowBoundsError(ValueError):
    """The requested or inferred window has start greater than end."""


@dataclass(frozen=True)
class WindowStatistics:
    """Descriptive statistics for one update window."""

    rows: int
    source_rows: int
    loss_field: str
    loss_field_used: str | None
    loss_fallback: bool
    loss_missing_rows: int
    loss_mean: float | None
    loss_median: float | None
    pre_clip_median: float | None
    pre_clip_max: float | None
    clip_fraction_mean: float | None
    clipped_updates: int
    nonfinite_total: int
    invalid_numeric_values: int
    conflicting_updates_excluded: int


@dataclass
class _UpdateValues:
    loss: float | None = None
    total_loss: float | None = None
    pre_clip: float | None = None
    clip_fraction: float | None = None
    nonfinite: int | None = None


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return cast(TextIO, gzip.open(path, "rt", encoding="utf-8"))
    return path.open("r", encoding="utf-8")


def _iter_lines(path: Path) -> Iterator[tuple[int, str]]:
    with _open_text(path) as handle:
        yield from enumerate(handle, 1)


def _update_of(row: dict[str, object]) -> int | None:
    value = row.get("successful_update")
    return value if type(value) is int else None


def _number(
    row: dict[str, object],
    key: str,
    invalid: list[int],
) -> float | None:
    value = row.get(key)
    if type(value) is int or type(value) is float:
        number = float(value)
        if not math.isfinite(number):
            invalid[0] += 1
            return None
        return number
    return None


def _count(
    row: dict[str, object],
    key: str,
    invalid: list[int],
) -> int | None:
    value = row.get(key)
    if type(value) is int:
        return value
    if type(value) is float and not math.isfinite(value):
        invalid[0] += 1
    return None


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def _canonical_payload(row: dict[str, object]) -> str:
    payload = {key: value for key, value in row.items() if key not in VOLATILE_KEYS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _missing_ranges(
    start: int,
    end: int,
    updates_present: Sequence[int],
) -> tuple[list[list[int]], bool, int]:
    """Compact the gaps of sorted in-window updates into [first, last] runs.

    ``updates_present`` must already be sorted and restricted to the window.
    The scan walks the present updates rather than the window, so a 10**12
    wide window with a few rows stays O(len(updates_present)); the missing
    count is exact even when the range list is truncated.
    """
    ranges: list[list[int]] = []
    cursor = start
    # A sentinel closes the final gap without copying the input sequence.
    for update in itertools.chain(updates_present, (end + 1,)):
        if update > cursor:
            if len(ranges) == MAX_MISSING_RANGES:
                return ranges, True, (end - start + 1) - len(updates_present)
            ranges.append([cursor, update - 1])
        cursor = update + 1
    return ranges, False, (end - start + 1) - len(updates_present)


def _choose_loss(
    values: dict[int, _UpdateValues],
    updates: list[int],
    field: str,
) -> tuple[str | None, bool, list[float], int]:
    requested = [values[update].loss for update in updates]
    if any(value is not None for value in requested):
        chosen = requested
        used: str | None = field
        fallback = False
    else:
        totals = [values[update].total_loss for update in updates]
        if any(value is not None for value in totals):
            chosen = totals
            used = "total_loss"
            fallback = True
        else:
            chosen = []
            used = None
            fallback = False
    losses: list[float] = []
    for value in chosen:
        if value is not None:
            losses.append(value)
    requested_missing = sum(1 for value in requested if value is None)
    return used, fallback, losses, requested_missing


def _summarize(
    values: dict[int, _UpdateValues],
    updates: list[int],
    *,
    field: str,
    source_rows: int,
    invalid: int,
    conflicting_excluded: int,
) -> WindowStatistics:
    used, fallback, losses, missing = _choose_loss(values, updates, field)
    gradients: list[float] = []
    clips: list[float] = []
    nonfinite = 0
    for update in updates:
        item = values[update]
        if item.pre_clip is not None:
            gradients.append(item.pre_clip)
        if item.clip_fraction is not None:
            clips.append(item.clip_fraction)
        if item.nonfinite is not None:
            nonfinite += item.nonfinite
    return WindowStatistics(
        rows=len(updates),
        source_rows=source_rows,
        loss_field=field,
        loss_field_used=used,
        loss_fallback=fallback,
        loss_missing_rows=missing,
        loss_mean=round(statistics.fmean(losses), 6) if losses else None,
        loss_median=round(statistics.median(losses), 6) if losses else None,
        pre_clip_median=(round(statistics.median(gradients), 6) if gradients else None),
        pre_clip_max=round(max(gradients), 6) if gradients else None,
        clip_fraction_mean=(round(statistics.fmean(clips), 6) if clips else None),
        clipped_updates=sum(1 for value in clips if value > 0.0),
        nonfinite_total=nonfinite,
        invalid_numeric_values=invalid,
        conflicting_updates_excluded=conflicting_excluded,
    )


def audit(
    path: Path,
    *,
    start: int | None,
    end: int | None,
    loss_field: str,
) -> dict[str, Any]:
    """Return an audit report for ``path``.

    ``start`` and ``end`` are inclusive update bounds.  When only one bound is
    given, the other is inferred from the updates present in the file.  An
    empty file with no usable update rows reports ``window.status=no_update_rows``
    rather than a zero-length, apparently complete window.  Check ``file.status``
    separately for malformed input or container errors, even for a narrow window;
    valid rows and all existing counts remain available in an incomplete report.
    """

    present: set[int] = set()
    values: dict[int, _UpdateValues] = {}
    payload_digests: dict[int, set[bytes]] = {}
    line_counts: Counter[int] = Counter()
    line_digests: dict[int, set[bytes]] = {}

    malformed: list[dict[str, Any]] = []
    malformed_total = 0
    schema_versions: set[int] = set()
    total_lines = 0
    blank_lines = 0
    rows = 0
    rows_without_update = 0
    invalid = [0]
    container_error: str | None = None

    try:
        for line_number, raw in _iter_lines(path):
            total_lines += 1
            text = raw.strip()
            if not text:
                blank_lines += 1
                continue
            try:
                loaded: object = json.loads(text)
            except json.JSONDecodeError as exc:
                malformed_total += 1
                if len(malformed) < MAX_MALFORMED:
                    malformed.append(
                        {
                            "line": line_number,
                            "error": f"{type(exc).__name__}: {exc.msg}",
                        }
                    )
                continue
            if type(loaded) is not dict:
                malformed_total += 1
                if len(malformed) < MAX_MALFORMED:
                    malformed.append(
                        {"line": line_number, "error": "row is not an object"}
                    )
                continue
            row = cast(dict[str, object], loaded)
            rows += 1
            update = _update_of(row)
            if update is None:
                rows_without_update += 1
                continue
            present.add(update)
            schema_version = _count(row, "schema_version", invalid)
            if schema_version is not None:
                schema_versions.add(schema_version)
            payload = _canonical_payload(row)
            payload_digests.setdefault(update, set()).add(_digest(payload))
            line_counts[update] += 1
            line_digests.setdefault(update, set()).add(_digest(text))
            values[update] = _UpdateValues(
                loss=_number(row, loss_field, invalid),
                total_loss=_number(row, "total_loss", invalid),
                pre_clip=_number(row, "pre_clip_grad_norm", invalid),
                clip_fraction=_number(row, "clip_fraction", invalid),
                nonfinite=_count(row, "nonfinite_count", invalid),
            )
    except (EOFError, OSError, UnicodeDecodeError, zlib.error) as exc:
        container_error = f"{type(exc).__name__}: {exc}"

    first_update = min(present) if present else None
    last_update = max(present) if present else None
    if not present:
        window_status = "no_update_rows"
        window_start = start
        window_end = end
        expected: int | None = None
        missing_ranges: list[list[int]] = []
        missing_truncated = False
        missing_count = 0
        selected: list[int] = []
        clean_selected: list[int] = []
        clean_rows = 0
        conflicting: list[int] = []
        same_payload_extra = 0
        byte_identical_extra = 0
        selected_rows = 0
    else:
        window_status = "ok"
        window_start = start if start is not None else first_update
        window_end = end if end is not None else last_update
        if window_start is None or window_end is None:
            raise ValueError("window bounds could not be resolved")
        if window_start > window_end:
            raise WindowBoundsError(
                f"window start {window_start} is greater than window end "
                f"{window_end}; file updates span {first_update}..{last_update}"
            )
        selected = sorted(
            update for update in present if window_start <= update <= window_end
        )
        expected = window_end - window_start + 1
        missing_ranges, missing_truncated, missing_count = _missing_ranges(
            window_start, window_end, selected
        )
        conflicting = [
            update for update in selected if len(payload_digests[update]) > 1
        ]
        if conflicting:
            window_status = "conflicting_updates"
        conflicting_set = set(conflicting)
        # Conflicting updates are kept out of the statistics so no
        # last-read-wins value is ever presented as the update's value.
        clean_selected = [
            update for update in selected if update not in conflicting_set
        ]
        clean_rows = sum(line_counts[update] for update in clean_selected)
        same_payload_extra = sum(
            line_counts[update] - len(payload_digests[update]) for update in selected
        )
        byte_identical_extra = sum(
            line_counts[update] - len(line_digests[update]) for update in selected
        )
        selected_rows = sum(line_counts[update] for update in selected)

    statistics_payload = asdict(
        _summarize(
            values,
            clean_selected,
            field=loss_field,
            source_rows=clean_rows,
            invalid=invalid[0],
            conflicting_excluded=len(conflicting),
        )
    )
    try:
        stat = path.stat()
    except OSError as exc:
        # A stream may fail because its backing path disappeared.  Metadata
        # lookup must not throw away the rows and original read error.
        stat = None
        detail = f"{type(exc).__name__}: {exc}"
        container_error = (
            f"{container_error}; metadata: {detail}" if container_error else detail
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "audited_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "file": {
            "status": (
                "container_error"
                if container_error is not None
                else "malformed_rows"
                if malformed_total
                else "ok"
            ),
            "path": str(path),
            "bytes": stat.st_size if stat is not None else None,
            "modified": (
                datetime.fromtimestamp(stat.st_mtime)
                .astimezone()
                .isoformat(timespec="seconds")
                if stat is not None
                else None
            ),
            "total_lines": total_lines,
            "blank_lines": blank_lines,
            "rows": rows,
            "malformed_rows": malformed_total,
            "rows_without_update": rows_without_update,
            "first_update": first_update,
            "last_update": last_update,
            "schema_versions": sorted(schema_versions),
            "container_error": container_error,
        },
        "window": {
            "status": window_status,
            "start": window_start,
            "end": window_end,
            "expected_updates": expected,
            "selected_rows": selected_rows,
            "distinct_updates": len(selected),
            "coverage": (round(len(selected) / expected, 6) if expected else None),
            "missing_count": missing_count,
            "missing_ranges": missing_ranges,
            "missing_ranges_truncated": missing_truncated,
            "same_payload_extra_rows": same_payload_extra,
            "byte_identical_extra_rows": byte_identical_extra,
            "conflicting_updates": conflicting,
            "conflicting_update_count": len(conflicting),
        },
        "statistics": statistics_payload,
        "malformed": malformed,
        "notes": [MISSING_ROW_NOTE, NONFINITE_NOTE, CONFLICT_NOTE],
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="metrics.jsonl or .jsonl.gz")
    parser.add_argument("--start", type=int, default=None, help="inclusive update")
    parser.add_argument("--end", type=int, default=None, help="inclusive update")
    parser.add_argument("--loss-field", default="main_loss")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.metrics.is_file():
        parser.error(f"metrics input does not exist or is not a file: {args.metrics}")
    if args.start is not None and args.end is not None and args.start > args.end:
        parser.error(f"--start {args.start} must not be greater than --end {args.end}")
    if args.out is not None and args.out.resolve() == args.metrics.resolve():
        parser.error("--out must not point at the audited input")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the audit CLI and return the exit status documented above."""
    args = _parse_args(argv)
    try:
        report = audit(
            args.metrics,
            start=args.start,
            end=args.end,
            loss_field=args.loss_field,
        )
    except WindowBoundsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out is None:
        print(rendered)
    else:
        try:
            args.out.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            print(
                f"error: cannot write --out {args.out}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return EXIT_WRITE_ERROR
    container_error = report["file"]["container_error"]
    if container_error is not None:
        print(
            "error: input container failed mid-read; partial report emitted: "
            f"{container_error}",
            file=sys.stderr,
        )
        return EXIT_CONTAINER_ERROR
    conflicting_count = report["window"]["conflicting_update_count"]
    if conflicting_count:
        print(
            f"error: {conflicting_count} update(s) have conflicting rows and "
            "are excluded from statistics: window.conflicting_updates",
            file=sys.stderr,
        )
        return EXIT_CONFLICTS
    malformed_count = report["file"]["malformed_rows"]
    if malformed_count:
        print(
            f"error: {malformed_count} malformed row(s); audit incomplete; "
            "report emitted: file.malformed_rows and malformed",
            file=sys.stderr,
        )
        return EXIT_MALFORMED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
