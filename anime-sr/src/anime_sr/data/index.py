"""M1 data index: shard scan, SR eligibility, validation split (plan §10).

Scans the danbooru-v2 webdataset shards (tar: ``*.webp`` + paired ``*.json``
meta) and builds the frozen index artifacts (docs/data-contract.md §3):

- ``sr-eligibility-v1``  one row per image (parquet when pyarrow is
  available, else JSONL with the same schema; the writer used is recorded
  in the filter report)
- ``shard-summary-v1``   per-shard counts / sizes / bytes
- ``filter-report-v1.json``  exclusion funnel + human-review sample (35%)
- ``sr-validation-v1.json``  deterministic train/validation split

Validation split is structural-zero-overlap: an image id is *validation*
iff ``blake2b(id) % 10000 < validation_permille`` (§16.1 / §10.5).
No separate audit layer is kept (repo data-service discipline).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tarfile
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from anime_sr.config.schema import Config


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetaRecord:
    """One danbooru-v2 image (the paired .json meta)."""

    sample_id: str
    shard: str
    rel_path: str  # webp member path inside the shard
    width: int
    height: int
    nsfw: str
    year: int
    quality: str  # "priority" | "aux" | "regular" (filter pools, §10.4)
    tags_general: tuple[str, ...] = field(default_factory=tuple)
    # danbooru-v2 meta fields (schema_version 1, plan §10.3 rows):
    quality_tier: str = ""  # "masterpiece".."worst" ("" when absent)
    anime_completeness: str = ""  # "polished" | "rough" | "monochrome"
    anime_classification: str = ""  # "illustration" | "comic" | "bangumi" | "3d" | "not_painting"
    ai_corrupted: bool = False  # the ai_image_corrupted meta key is present

    @property
    def sample_hash10k(self) -> int:
        """blake2b(sample_id) % 10000 — deterministic split key."""
        d = hashlib.blake2b(self.sample_id.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(d, "little") % 10_000


@dataclass
class ShardSummary:
    shard: str
    n_images: int = 0
    n_sfw: int = 0
    n_unresolved: int = 0  # json members dropped (null dims / malformed)
    bytes_total: int = 0
    min_size: int = 0
    max_size: int = 0
    scan_seconds: float = 0.0


@dataclass
class Eligibility:
    """Per-image SR eligibility (reason codes are stable tokens, §10.3)."""

    eligible_train: bool
    eligible_buckets: tuple[int, ...]  # HR sizes (512/768/1024) a full crop fits
    reasons: tuple[str, ...]  # exclusion / restriction codes (empty when fully eligible)


# ---------------------------------------------------------------------------
# scanning
# ---------------------------------------------------------------------------
def _parse_meta(raw: bytes) -> MetaRecord | None:
    """Parse one danbooru-v2 meta json; None when structurally invalid or
    dimensionally unresolved (``image.width``/``height`` can be JSON null)."""
    try:
        m = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(m, dict):
        return None
    img = m.get("image")
    if not isinstance(img, dict):
        img = {}
    # ``.get("width", 0)`` default only fires when the key is *absent*; danbooru
    # records with unresolved dimensions carry explicit JSON null → use `or 0`.
    w = img.get("width") or 0
    h = img.get("height") or 0
    try:
        w, h = int(w), int(h)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0 or not m.get("id"):
        return None
    source = m.get("source") or {}
    release = source.get("release") or ""
    year = 0
    if "_" in release and release.split("_", 1)[1].isdigit():
        year = int(release.split("_", 1)[1])
    tags = m.get("tags") or {}
    general = tuple(tags.get("general") or ())
    # danbooru-v2 meta fields (§10.3): absent in old records → "" defaults;
    # ai_image_corrupted is present only when flagged (value "corrupted").
    tier = m.get("quality")
    comp = m.get("anime_completeness")
    cls = m.get("anime_classification")
    return MetaRecord(
        sample_id=str(m["id"]),
        shard="",  # filled by the caller
        rel_path="",  # filled by the caller
        width=w,
        height=h,
        nsfw=str(m.get("nsfw") or ""),
        year=year,
        quality="",  # filled by the caller (needs filter config)
        tags_general=general,
        quality_tier=tier if isinstance(tier, str) else "",
        anime_completeness=comp if isinstance(comp, str) else "",
        anime_classification=cls if isinstance(cls, str) else "",
        ai_corrupted="ai_image_corrupted" in m,
    )


def scan_shard(shard_path: str | Path, progress: bool = True) -> tuple[list[MetaRecord], ShardSummary]:
    """Stream one shard tar and return (meta records, shard summary).

    Only the paired ``*.json`` members are parsed; the webp members are
    counted (bytes) but never decoded. Corrupt tars fail loudly (repo
    rule: no silent sample skipping).
    """
    path = Path(shard_path)
    t0 = time.monotonic()
    records: list[MetaRecord] = []
    summary = ShardSummary(shard=path.name)
    with tarfile.open(path, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            summary.bytes_total += member.size
            if member.name.endswith(".json") and member.name.endswith(".tar.json") is False:
                base = member.name.rsplit(".", 1)[0]
                if base.endswith(".json"):
                    base = base.rsplit(".", 1)[0]
                f = tf.extractfile(member)
                if f is None:
                    continue
                rec = _parse_meta(f.read())
                if rec is None:
                    summary.n_unresolved += 1
                    continue
                rec = dataclasses.replace(rec, shard=path.name, rel_path=base + ".webp")
                summary.n_images += 1
                if rec.nsfw == "sfw":
                    summary.n_sfw += 1
                summary.min_size = min(summary.min_size, min(rec.width, rec.height)) or min(rec.width, rec.height)
                summary.max_size = max(summary.max_size, max(rec.width, rec.height))
                records.append(rec)
    summary.scan_seconds = time.monotonic() - t0
    if progress:
        print(f"[index] {path.name}: {summary.n_images} images ({summary.n_sfw} sfw) in {summary.scan_seconds:.1f}s")
    return records, summary


def classify_quality(rec: MetaRecord, cfg: Config) -> MetaRecord:
    """§10.4 pools from the danbooru meta fields (not tag heuristics):

    - aux:      completeness in ``aux_completeness`` (monochrome/rough)
                 or classification in ``aux_classification`` (3d)
    - priority: completeness in ``priority_completeness`` (polished) AND
                 classification in ``priority_classification``
                 (illustration/bangumi/comic) AND danbooru quality tier in
                 ``priority_quality`` (masterpiece/best/great)
    - regular:  everything else
    """
    f = cfg.filter
    if rec.anime_completeness in f.aux_completeness or rec.anime_classification in f.aux_classification:
        quality = "aux"
    elif (
        rec.anime_completeness in f.priority_completeness
        and rec.anime_classification in f.priority_classification
        and rec.quality_tier in f.priority_quality
    ):
        quality = "priority"
    else:
        quality = "regular"
    return dataclasses.replace(rec, quality=quality)


# ---------------------------------------------------------------------------
# webp pixel-size recovery (the danbooru-v2 meta dimensions describe the
# original post, NOT the shipped webp, which is often a resized variant)
# ---------------------------------------------------------------------------
def webp_header_size(header: bytes) -> tuple[int, int] | None:
    """Read (width, height) from the first 64 bytes of a RIFF/WebP file.

    Only the pixel dimensions are needed, so no codec decode: the RIFF size
    field is the container length, not the picture size. Covers the VP8
    (lossy), VP8L (lossless) and VP8X (extended) chunks. ``None`` for
    non-WebP or headers too short to contain the dimension fields.
    """
    if len(header) < 25 or header[0:4] != b"RIFF" or header[8:12] != b"WEBP":
        return None
    chunk = header[12:16]
    if chunk == b"VP8 ":
        # chunk data @20: 3-byte frame tag, 3-byte start code, then 14-bit LE w/h
        if len(header) < 30:
            return None
        w = int.from_bytes(header[26:28], "little") & 0x3FFF
        h = int.from_bytes(header[28:30], "little") & 0x3FFF
    elif chunk == b"VP8L":
        # chunk data @20: 1-byte signature 0x2F, then 32-bit field LE @21:
        # bits [17:4] = width-1, bits [31:18] = height-1
        v = int.from_bytes(header[21:25], "little")
        w = ((v >> 4) & 0x3FFF) + 1
        h = ((v >> 18) & 0x3FFF) + 1
    elif chunk == b"VP8X":
        # chunk data @20: 4-byte flags, then 24-bit LE (width-1), (height-1)
        if len(header) < 30:
            return None
        w = int.from_bytes(header[24:27], "little") + 1
        h = int.from_bytes(header[27:30], "little") + 1
    else:
        return None
    return (w, h) if w > 0 and h > 0 else None


def collect_webp_sizes(webp_root: str | Path) -> dict[str, tuple[int, int]]:
    """Map sample_id -> actual pixel size, read from extracted webp headers.

    Layout: ``<webp_root>/<shard>.tar/<id>.webp`` (the extract contract).
    ``.part`` files (still writing) are skipped; unreadable headers are
    dropped (the caller's coverage check then fails loudly).
    """
    root = Path(webp_root)
    out: dict[str, tuple[int, int]] = {}
    n_bad = 0
    for shard_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(shard_dir.iterdir()):
            if f.suffix != ".webp" or f.name.endswith(".part"):
                continue
            with f.open("rb") as fh:
                size = webp_header_size(fh.read(64))
            if size is None:
                n_bad += 1
                continue
            out[f.stem] = size
    if n_bad:
        print(f"[sizes] WARN: {n_bad} webp headers unreadable (dropped)")
    return out


# ---------------------------------------------------------------------------
# eligibility
# ---------------------------------------------------------------------------
def _min_dim(rec: MetaRecord) -> int:
    return min(rec.width, rec.height)


def eligible_buckets(rec: MetaRecord, cfg: Config) -> tuple[int, ...]:
    """HR bucket sizes whose full square crop fits in the image (100% retention)."""
    sizes = [lq * cfg.buckets.hr_multiplier for lq in cfg.buckets.lq_sizes]
    m = _min_dim(rec)
    return tuple(s for s in sizes if m >= s)


def evaluate_eligibility(rec: MetaRecord, cfg: Config) -> Eligibility:
    """Apply the §10.4 funnel to one classified record."""
    reasons: list[str] = []
    if rec.nsfw != "sfw":
        reasons.append(f"nsfw:{rec.nsfw}")
    hard_tags = set(cfg.filter.hard_exclude) - {"nsfw"}
    hit = sorted(set(rec.tags_general) & hard_tags)
    if hit:
        reasons.append("hard-tag:" + ",".join(hit))
    if rec.ai_corrupted:
        reasons.append("ai_image_corrupted")
    if rec.anime_classification in cfg.filter.hard_classifications:
        reasons.append("hard-class:" + rec.anime_classification)
    buckets = eligible_buckets(rec, cfg)
    if not buckets:
        reasons.append("size:small")
    return Eligibility(
        eligible_train=not reasons,
        eligible_buckets=buckets,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------
def is_validation(rec: MetaRecord, cfg: Config) -> bool:
    return rec.sample_hash10k < cfg.validation.validation_permille


def split_index(records: list[MetaRecord], cfg: Config) -> tuple[list[MetaRecord], list[MetaRecord]]:
    """(train, validation) with structural zero overlap (§16.1)."""
    train = [r for r in records if not is_validation(r, cfg)]
    val = [r for r in records if is_validation(r, cfg)]
    overlap = {r.sample_id for r in train} & {r.sample_id for r in val}
    if cfg.validation.zero_overlap and overlap:
        raise AssertionError(f"train/validation overlap: {len(overlap)} ids")
    return train, val


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------
def _record_row(rec: MetaRecord, elig: Eligibility, cfg: Config) -> dict:
    return {
        "sample_id": rec.sample_id,
        "shard": rec.shard,
        "rel_path": rec.rel_path,
        "width": rec.width,
        "height": rec.height,
        "nsfw": rec.nsfw,
        "year": rec.year,
        # §10.3 row fields: danbooru quality tier + filter pool + anime fields
        "quality": rec.quality_tier,
        "sampling_pool": rec.quality,
        "anime_completeness": rec.anime_completeness,
        "anime_classification": rec.anime_classification,
        "ai_corrupted": rec.ai_corrupted,
        # §10.3: clean-score column, lazy stage (P1 ③) → null until computed
        "clean_score": None,
        "eligible_train": elig.eligible_train,
        "eligible_buckets": list(elig.eligible_buckets),
        "reasons": list(elig.reasons),
        "is_validation": is_validation(rec, cfg),
    }


def _write_table(rows: list[dict], out_dir: Path, base_name: str) -> tuple[str, str]:
    """Write rows as parquet when pyarrow is importable, else JSONL.

    Returns (format, path). The plan names the artifacts ``.parquet``; on
    hosts without pyarrow the JSONL twin keeps the same base name so the
    index consumer can read either.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        path = out_dir / f"{base_name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return "jsonl", str(path)
    table = pa.Table.from_pylist(rows)
    path = out_dir / f"{base_name}.parquet"
    pq.write_table(table, path)
    return "parquet", str(path)


def build_index(
    shard_paths: Sequence[str | Path],
    cfg: Config,
    out_dir: str | Path,
    size_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> dict:
    """Scan shards, evaluate eligibility, write all four index artifacts.

    ``size_overrides`` maps sample_id -> (width, height) and replaces the
    meta dimensions before eligibility is evaluated (the danbooru-v2 meta
    records the original post size, not the shipped webp). When given,
    records without an override are counted in ``n_size_missing`` so a
    caller can detect incomplete coverage.

    Returns a summary dict (also embedded in filter-report-v1.json).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_records: list[MetaRecord] = []
    summaries: list[ShardSummary] = []
    for sp in shard_paths:
        records, summary = scan_shard(sp)
        all_records.extend(records)
        summaries.append(summary)
    if size_overrides is not None:
        n_missing = 0
        corrected = []
        for rec in all_records:
            size = size_overrides.get(rec.sample_id)
            if size is None:
                n_missing += 1
                corrected.append(rec)
            elif (size[0], size[1]) != (rec.width, rec.height):
                corrected.append(dataclasses.replace(rec, width=size[0], height=size[1]))
            else:
                corrected.append(rec)
        all_records = corrected
    else:
        n_missing = 0
        corrected = all_records
    classified = [classify_quality(r, cfg) for r in corrected]

    n_total = len(classified)
    n_sfw = sum(1 for r in classified if r.nsfw == "sfw")
    elig = [evaluate_eligibility(r, cfg) for r in classified]
    n_eligible = sum(1 for e in elig if e.eligible_train)
    n_by_quality = {q: sum(1 for r in classified if r.quality == q) for q in ("priority", "regular", "aux")}

    rows = [_record_row(r, e, cfg) for r, e in zip(classified, elig)]
    fmt1, p_elig = _write_table(rows, out, "sr-eligibility-v1")
    shard_rows = [dataclasses.asdict(s) for s in summaries]
    fmt2, p_shard = _write_table(shard_rows, out, "shard-summary-v1")

    train, val = split_index(classified, cfg)
    val_path = out / "sr-validation-v1.json"
    val_doc = {
        "version": 1,
        "rule": "blake2b(sample_id)%10000 < validation_permille",
        "validation_permille": cfg.validation.validation_permille,
        "n_train": len(train),
        "n_validation": len(val),
        "validation_ids": sorted(r.sample_id for r in val),
        "zero_overlap": not ({r.sample_id for r in train} & {r.sample_id for r in val}),
    }
    val_path.write_text(json.dumps(val_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 35% human-review sample of hard-excluded ids (quality-gate, §10.4):
    # deterministic first-N by sorted id, no interactive step (report only).
    excluded_ids = sorted(r.sample_id for r, e in zip(classified, elig) if not e.eligible_train)
    n_review = round(len(excluded_ids) * cfg.filter.human_review_fraction)
    reason_counts: Counter[str] = Counter()
    for e in elig:
        reason_counts.update(e.reasons)
    report = {
        "version": 1,
        "n_shards": len(summaries),
        "n_images": n_total,
        "n_unresolved": sum(s.n_unresolved for s in summaries),
        "n_size_corrected": (n_total - n_missing) if size_overrides is not None else None,
        "n_size_missing": n_missing if size_overrides is not None else None,
        "n_sfw": n_sfw,
        "n_eligible_train": n_eligible,
        "n_by_quality": n_by_quality,
        "n_by_reason": dict(reason_counts),
        "aux_fraction": (n_by_quality["aux"] / n_total) if n_total else 0.0,
        "aux_cap": cfg.filter.aux_max_fraction,
        "aux_capped": (n_by_quality["aux"] / n_total) > cfg.filter.aux_max_fraction if n_total else False,
        "n_hard_excluded": len(excluded_ids),
        "human_review_sample": excluded_ids[:n_review],
        "writer": {"sr-eligibility-v1": fmt1, "shard-summary-v1": fmt2},
        "paths": {"eligibility": p_elig, "shard_summary": p_shard, "validation": str(val_path)},
    }
    (out / "filter-report-v1.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    size_note = f", sizes-corrected {n_total - n_missing}/{n_total}" if size_overrides is not None else ""
    print(f"[index] {n_total} images: {n_eligible} train-eligible, {len(val)} validation, {len(excluded_ids)} excluded, {sum(s.n_unresolved for s in summaries)} unresolved{size_note}")
    return report


def iter_index(index_path: str | Path) -> Iterator[dict]:
    """Read the eligibility table back (parquet or JSONL, auto-detect)."""
    p = Path(index_path)
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq

        yield from pq.read_table(p).to_pylist()
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
