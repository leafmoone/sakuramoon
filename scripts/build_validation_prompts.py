#!/usr/bin/env python3
"""Build a validation prompt manifest (schema 4) from a validation cohort.

The 512 production run switched its FID/IS/KID/CMMD evaluation from the
frozen v2 cohort (``s0-validation-50k-v1``, samples overlap the v3 training
corpus) to the v3 holdout (``s0-validation-512-v3``).  That holdout ships
shard tars with sidecars but no prompt manifest, so this tool derives one.

Construction (deterministic):

* candidates are every image member (``.jpg``/``.jpeg``/``.png``/``.webp``)
  of every ``*.tar`` under ``<cohort>/shards``, in the same global order the
  evaluator's real-feature reader uses (sorted archive path, tar order),
  each paired with its sibling ``<stem>.json`` sidecar;
* each sidecar is parsed by the *production* metadata parser
  (``parse_modelscope_caption_fields``) and turned into a caption plan by the
  *production* plan builder (``build_caption_plan``) with **all dropout
  probabilities zero** and the run's ``condition_mode`` read from the resolved
  config: the prompt-case schema rejects plans carrying dropout hits or a
  globally dropped condition, so the manifest carries clean surfaces (this is
  also what the legacy s0-validation-50k-v1 manifest does).  Per-branch NL
  availability still comes from the sidecar, so tag-only corpora stay
  tag-only; the ~10% of samples whose global-condition roll drops the plan are
  skipped because a prompt case cannot express an unconditional rollout;
* candidates whose plan is invalid for a prompt case are skipped (globally
  dropped conditions, empty content, rejected sidecars, missing sidecars);
* the first ``--count`` valid cases of a seeded shuffle (--seed) are written.

Case fields mirror the legacy manifest: ``prompt_id`` = ``validation-<sha256>``
of the cohort-relative sample id, ``prompt`` = the exact rendered text
(``caption_plan_prompt_text``), empty legacy ``conditions``, a deterministic
per-sample latent ``seed``, and the source image dimensions rounded to
multiples of 16 (the evaluator overrides both with the run resolution).

Output is canonical JSON (sorted keys, compact separators, trailing newline)
and is validated by a round trip through ``PromptManifest.from_canonical_bytes``
before it is written atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tarfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionError,
    NlDropoutProbabilities,
    build_caption_plan,
)
from sakuramoon.data.pipeline import PipelineSampleRejected
from sakuramoon.data.production import (
    ProductionDataError,
    parse_modelscope_caption_fields,
)
from sakuramoon.eval.spec import PromptCase, PromptManifest, caption_plan_prompt_text

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
SCHEMA_VERSION = 4

# Prompt cases must be dropout-free: ``PromptCase`` rejects any plan whose
# ``dropout_hits`` are set or whose condition was globally dropped.  The
# global-condition roll in ``build_caption_plan`` uses the module constant
# (0.10) regardless of the probabilities passed in, so those samples are
# filtered out below instead of being written as unconditional cases.
ZERO_PROBABILITIES = CaptionDropoutProbabilities(
    condition_route=0.0,
    condition_only=0.0,
    tag=0.0,
    candidate_source=0.0,
    nl=NlDropoutProbabilities(
        long_names=0.0,
        long_no_names=0.0,
        short_vibes=0.0,
        nl2=0.0,
        nl3=0.0,
    ),
)


@dataclass(frozen=True, slots=True)
class Candidate:
    archive: Path
    member: str
    sample_id: str
    relative: str


def _load_resolved(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _caption_settings(resolved: Mapping[str, Any]) -> tuple[str, dict[str, float]]:
    """Return the run's condition mode and its (reference-only) dropout values."""

    caption = resolved["caption"]
    dropout = caption["dropout"]
    nl = dropout["nl"]
    reference = {
        "condition_route": float(dropout["condition_route"]),
        "condition_only": float(dropout["condition_only"]),
        "tag": float(dropout["tag"]),
        "candidate_source": float(dropout["candidate_source"]),
        "nl.long_names": float(nl["long_names"]),
        "nl.long_no_names": float(nl["long_no_names"]),
        "nl.short_vibes": float(nl["short_vibes"]),
        "nl.nl2": float(nl["nl2"]),
        "nl.nl3": float(nl["nl3"]),
    }
    return caption["condition_mode"], reference


def _domain_seed(sample_id: str, domain: str) -> int:
    digest = hashlib.sha256(f"{domain}|{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _enumerate_candidates(shard_root: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    for archive in sorted(shard_root.rglob("*.tar")):
        with tarfile.open(archive, "r:*") as handle:
            for member in handle:
                if not member.isfile():
                    continue
                if Path(member.name).suffix.casefold() not in IMAGE_SUFFIXES:
                    continue
                relative = archive.relative_to(shard_root).as_posix()
                candidates.append(
                    Candidate(
                        archive=archive,
                        member=member.name,
                        sample_id=f"{relative}::{member.name}",
                        relative=relative,
                    )
                )
    return candidates


def _rounded(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("image dimension must be a positive integer")
    return max(16, (int(value) + 8) // 16 * 16)


def _sidecar_payloads(
    archive: Path, members: Sequence[str]
) -> dict[str, bytes]:
    """Load the sidecar payloads of ``members`` (all images of one archive)."""

    wanted = {f"{Path(name).with_suffix('')}.json" for name in members}
    payloads: dict[str, bytes] = {}
    with tarfile.open(archive, "r:*") as handle:
        for member in handle:
            if member.name not in wanted:
                continue
            extracted = handle.extractfile(member)
            if extracted is None:
                continue
            payloads[member.name] = extracted.read()
            if len(payloads) == len(wanted):
                break
    return payloads


class _Skipped(Exception):
    """A candidate that cannot become a prompt case, with a stable reason."""


def _case_for(
    candidate: Candidate, payload: bytes, *, condition_mode: str
) -> PromptCase:
    record = json.loads(payload)
    if type(record) is not dict:
        raise ValueError("sidecar payload must be an object")
    fields = parse_modelscope_caption_fields(cast(Mapping[str, object], record))
    plan_seed = _domain_seed(candidate.sample_id, "plan")
    plan = build_caption_plan(
        fields,
        ZERO_PROBABILITIES,
        condition_mode=cast(Any, condition_mode),
        seed=plan_seed,
    )
    if plan.all_condition_dropped:
        raise _Skipped("all-condition-dropped")
    if not (plan.tags or plan.condition is not None or plan.nl_text is not None):
        raise _Skipped("empty-plan")
    image = record.get("image")
    if type(image) is not dict:
        raise ValueError("sidecar image block must be an object")
    height = _rounded(image.get("height"))
    width = _rounded(image.get("width"))
    digest = hashlib.sha256(candidate.sample_id.encode("utf-8")).hexdigest()[:32]
    return PromptCase(
        prompt_id=f"validation-{digest}",
        prompt=caption_plan_prompt_text(plan),
        conditions=(),
        seed=_domain_seed(candidate.sample_id, "latent"),
        height=height,
        width=width,
        caption_plan=plan,
    )


def _skip_reason(candidate: Candidate, payloads: Mapping[str, bytes]) -> str | None:
    sidecar = f"{Path(candidate.member).with_suffix('')}.json"
    payload = payloads.get(sidecar)
    if payload is None:
        return "missing-sidecar"
    try:
        record = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "invalid-json"
    if type(record) is not dict:
        return "invalid-json"
    tags = record.get("tags")
    if type(tags) is dict:
        has_tag = any(
            isinstance(value, list) and value
            for value in cast(Mapping[str, object], tags).values()
        )
        if not has_tag:
            return "tagless"
    return None


def build_manifest(
    *,
    shard_root: Path,
    count: int,
    seed: int,
    condition_mode: str,
    progress: bool = True,
) -> tuple[PromptManifest, dict[str, object]]:
    candidates = _enumerate_candidates(shard_root)
    if len(candidates) < count:
        raise SystemExit(
            f"cohort has {len(candidates)} images, fewer than requested {count}"
        )
    members_by_archive: dict[Path, list[str]] = {}
    for candidate in candidates:
        members_by_archive.setdefault(candidate.archive, []).append(candidate.member)
    order = list(range(len(candidates)))
    random.Random(seed).shuffle(order)

    payload_cache: dict[Path, dict[str, bytes]] = {}

    def payloads_for(archive: Path) -> dict[str, bytes]:
        cached = payload_cache.get(archive)
        if cached is None:
            cached = _sidecar_payloads(archive, members_by_archive[archive])
            payload_cache[archive] = cached
        return cached

    cases: list[PromptCase] = []
    skipped: dict[str, int] = {}
    seen_sources: dict[str, int] = {}
    branches: dict[str, int] = {}
    conditions: dict[str, int] = {}
    for index in order:
        if len(cases) >= count:
            break
        candidate = candidates[index]
        payloads = payloads_for(candidate.archive)
        reason = _skip_reason(candidate, payloads)
        if reason is not None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        payload = payloads[f"{Path(candidate.member).with_suffix('')}.json"]
        try:
            case = _case_for(
                candidate,
                payload,
                condition_mode=condition_mode,
            )
        except _Skipped as error:
            skipped[str(error)] = skipped.get(str(error), 0) + 1
            continue
        except (ProductionDataError, PipelineSampleRejected, CaptionError, ValueError) as error:
            key = f"rejected:{type(error).__name__}"
            skipped[key] = skipped.get(key, 0) + 1
            continue
        cases.append(case)
        source = candidate.relative.split("/", 1)[0]
        seen_sources[source] = seen_sources.get(source, 0) + 1
        plan = case.caption_plan
        assert plan is not None
        branch = plan.selected_nl or "none"
        branches[branch] = branches.get(branch, 0) + 1
        route = (
            "none"
            if plan.condition is None
            else f"{plan.condition.source}/{plan.condition.role}"
        )
        conditions[route] = conditions.get(route, 0) + 1
    if len(cases) != count:
        raise SystemExit(
            f"only {len(cases)} valid prompt cases were available for count={count}"
        )
    manifest = PromptManifest(tuple(cases))
    report: dict[str, object] = {
        "cases": len(cases),
        "candidates": len(candidates),
        "skipped": skipped,
        "sources": seen_sources,
        "nl_branches": branches,
        "condition_routes": conditions,
        "seed": seed,
        "prompt_length_min": min(len(case.prompt) for case in cases),
        "prompt_length_max": max(len(case.prompt) for case in cases),
    }
    if progress:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), file=sys.stderr)
    return manifest, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="deployment root")
    parser.add_argument(
        "--cohort",
        type=Path,
        required=True,
        help="cohort directory relative to --root (contains shards/ and selection)",
    )
    parser.add_argument(
        "--resolved",
        type=Path,
        default=Path("runs/g1/resolved.toml"),
        help="resolved config (relative to --root) providing [caption] settings",
    )
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--out", type=Path, default=None, help="manifest path")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    cohort = (root / args.cohort).resolve()
    shard_root = cohort / "shards"
    if not shard_root.is_dir():
        raise SystemExit(f"shard root is missing: {shard_root}")
    resolved = _load_resolved((root / args.resolved).resolve())
    condition_mode, run_dropout = _caption_settings(resolved)

    manifest, report = build_manifest(
        shard_root=shard_root,
        count=args.count,
        seed=args.seed,
        condition_mode=condition_mode,
    )

    payload = manifest.canonical_bytes()
    reloaded = PromptManifest.from_canonical_bytes(payload)
    if reloaded.canonical_bytes() != payload:
        raise SystemExit("manifest round-trip validation failed")

    out = (args.out or (cohort / "validation-prompts.json")).resolve()
    if out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing manifest: {out} (use --force)")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)

    report.update(
        {
            "output": str(out),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "schema_version": SCHEMA_VERSION,
            "condition_mode": condition_mode,
            "plan_dropout": "zero (prompt-case schema rejects dropout hits)",
            "run_caption_dropout_reference": run_dropout,
            "resolved_config": str((root / args.resolved).resolve()),
        }
    )
    report_path = (
        args.report.resolve() if args.report is not None else out.with_suffix(".build-report.json")
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[prompts] wrote {out} ({len(payload)} bytes)")
    print(f"[prompts] sha256 {report['sha256']}")
    print(f"[prompts] report {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
