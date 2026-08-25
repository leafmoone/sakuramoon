#!/usr/bin/env python3
"""Concept (artist/character) frequency stats from the dan_5_9 metadata DuckDB.

Reads the precomputed tag counts (table ``tag_stats``) from the
ModelScope ``leafmoone/dan_5_9_metadata`` DuckDB file and reports the
frequency distribution per concept type (``artist`` / ``character``).
The database is opened read-only; no training state is touched.

Usage:
    python scripts/concept_frequency_stats.py --db /tmp/dan59/dan_5_9.db
    python scripts/concept_frequency_stats.py --db ... --top 50 --out stats.json

Percentiles are reported in *descending-rank* terms: ``p10`` is the count
of the concept at the 10th percentile from the top (i.e. the 90th
ascending percentile), and ``p50`` is the median.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BUCKETS = (10, 20, 50, 100, 500, 1000, 5000, 10000)
COVERAGE_TOPS = (20, 50, 100, 500, 1000, 5000)
PERCENTILES = (10, 25, 50, 75, 90, 95, 99)
CONCEPT_TYPES = ("artist", "character")


def type_stats(con, concept_type: str, top: int) -> dict:
    """Frequency distribution for one concept type (descending-rank order)."""
    counts = [
        row[0]
        for row in con.execute(
            "SELECT count FROM tag_stats WHERE type = ? ORDER BY count DESC",
            (concept_type,),
        ).fetchall()
    ]
    total = sum(counts)
    n = len(counts)
    stats: dict = {"type": concept_type, "n_concepts": n, "total_occurrences": total}
    if total:
        stats["mean_occurrences"] = round(total / n, 2)
    for p in PERCENTILES:
        stats[f"p{p}"] = counts[min(n - 1, round(p / 100 * (n - 1)))]
    for threshold in BUCKETS:
        above = sum(1 for c in counts if c >= threshold)
        stats[f"gte_{threshold}"] = above
        stats[f"gte_{threshold}_share"] = round(above / n, 4) if n else 0.0
    for topn in COVERAGE_TOPS:
        stats[f"top{topn}_coverage"] = (
            round(sum(counts[:topn]) / total, 4) if total else 0.0
        )
    tops = con.execute(
        "SELECT tag, count FROM tag_stats WHERE type = ? ORDER BY count DESC LIMIT ?",
        (concept_type, top),
    ).fetchall()
    stats[f"top{top}"] = [{"tag": tag, "count": count} for tag, count in tops]
    return stats


def run(db_path: Path, top: int) -> dict:
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        n_images = con.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
        n_tags = con.execute("SELECT COUNT(*) FROM tag_stats").fetchone()[0]
        return {
            "source": {
                "db": str(db_path),
                "n_images": n_images,
                "n_tags": n_tags,
            },
            "percentile_note": (
                "percentiles are descending-rank: p10 = count at 10th percentile "
                "from the top, p50 = median"
            ),
            "artist": type_stats(con, "artist", top),
            "character": type_stats(con, "character", top),
        }
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("/tmp/dan59/dan_5_9.db"),
        help="path to the dan_5_9 metadata DuckDB file",
    )
    parser.add_argument("--top", type=int, default=30, help="top-N list size")
    parser.add_argument("--out", type=Path, default=None, help="write JSON here")
    args = parser.parse_args()

    report = run(args.db, args.top)
    payload = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
