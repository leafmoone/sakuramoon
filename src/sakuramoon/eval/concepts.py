"""Standalone concept-conditioning benchmark suite.

The suite contrasts, for every concept in the draw manifest, three images
produced from one shared noise stream per concept:

* **canonical** -- generated with the concept's own tag text,
* **null**      -- the identical noise with the condition fully dropped,
* **swap**      -- the partner concept's tag text (nearest |count delta|
  within the same type/tier band),

against the concept's three Danbooru reference posts.  All comparisons use
a single sign convention: a positive margin means the model ranked the
intended concept above the alternative.

* ``margin_null = ref_sim(canonical) - ref_sim(null)``
* ``margin_swap = ref_sim(canonical) - ref_sim(swap)``

plus reference similarity (mean cosine to the concept's own references) and
self-retrieval: the generated canonical image queries the full reference
pool and the rank of the best own reference is reported (ties do not
demote).  This module is pure -- manifest validation, prompt-case
construction, and metric math on CLIP feature tensors -- so the CLI and
the unit tests share one implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from sakuramoon.eval.spec import PromptCase

SUITE_SCHEMA_VERSION = 1
_REF_COUNT = 3
_SAFE_CONCEPT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,15}$")
_CONCEPT_TYPES = frozenset({"artist", "character"})
_TIERS = frozenset({"high", "mid", "tail"})
_META_STATUSES = frozenset({"matched", "approx", "unresolved"})
_TOP_LEVEL_FIELDS = frozenset({"schema_version", "seed", "concepts"})
_CONCEPT_FIELDS = frozenset(
    {
        "id",
        "type",
        "tier",
        "stratum",
        "tag",
        "count",
        "meta_tag",
        "actual_count",
        "meta_status",
        "swap",
        "swap_count",
        "swap_delta",
        "refs",
        "replaced_from",
    }
)
_REF_FIELDS = frozenset({"id", "fav", "aesthetics"})
_GROUP_TYPES = ("artist", "character")
_GROUP_TIERS = ("high", "mid", "tail")


class ConceptSuiteError(ValueError):
    """The concept manifest or feature tensors violated the suite contract."""


@dataclass(frozen=True, slots=True)
class ConceptRef:
    post_id: int
    fav: int
    aesthetics: str | None

    @property
    def post_key(self) -> str:
        return str(self.post_id)


@dataclass(frozen=True, slots=True)
class ConceptSpec:
    id: str
    type: str
    tier: str
    stratum: int
    tag: str
    count: int
    meta_tag: str
    actual_count: int
    meta_status: str
    swap: str
    swap_count: int
    swap_delta: int
    refs: tuple[ConceptRef, ...]
    replaced_from: str | None

    @property
    def ref_post_ids(self) -> tuple[int, ...]:
        return tuple(ref.post_id for ref in self.refs)


@dataclass(frozen=True, slots=True)
class ConceptManifest:
    seed: int
    concepts: tuple[ConceptSpec, ...]

    @classmethod
    def from_json(cls, path: Path) -> ConceptManifest:
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ConceptSuiteError(
                f"concept manifest cannot be read: {path}"
            ) from error
        return cls.from_bytes(payload)

    @classmethod
    def from_bytes(cls, payload: bytes) -> ConceptManifest:
        if type(payload) is not bytes:
            raise TypeError("concept manifest payload must be bytes")
        try:
            parsed: object = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ConceptSuiteError("concept manifest must be valid JSON") from None
        if type(parsed) is not dict:
            raise ConceptSuiteError("concept manifest root must be an object")
        document = cast(dict[str, object], parsed)
        if frozenset(document) != _TOP_LEVEL_FIELDS:
            raise ConceptSuiteError("concept manifest top-level fields are invalid")
        if document["schema_version"] != SUITE_SCHEMA_VERSION:
            raise ConceptSuiteError("concept manifest schema version is invalid")
        seed = document["seed"]
        if type(seed) is not int or seed < 0:
            raise ConceptSuiteError("concept manifest seed must be a nonnegative int")
        raw_concepts = document["concepts"]
        if type(raw_concepts) is not list or not raw_concepts:
            raise ConceptSuiteError("concept manifest concepts must be a nonempty array")
        concepts = tuple(
            _parse_concept(item, index)
            for index, item in enumerate(cast(list[object], raw_concepts))
        )
        identifiers = tuple(concept.id for concept in concepts)
        if len(set(identifiers)) != len(identifiers):
            raise ConceptSuiteError("concept IDs must be unique")
        return cls(seed=seed, concepts=concepts)


def _require_str(value: object, field: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value.strip()):
        raise ConceptSuiteError(f"{field} is invalid")
    if value != value.strip():
        raise ConceptSuiteError(f"{field} must not have surrounding whitespace")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConceptSuiteError(f"{field} must be an int >= {minimum}")
    return value


def _parse_refs(value: object, concept_id: str) -> tuple[ConceptRef, ...]:
    if type(value) is not list:
        raise ConceptSuiteError(f"concept {concept_id} references must be an array")
    refs_list = cast(list[object], value)
    if len(refs_list) != _REF_COUNT:
        raise ConceptSuiteError(
            f"concept {concept_id} must have exactly {_REF_COUNT} references"
        )
    refs: list[ConceptRef] = []
    seen: set[int] = set()
    for item in refs_list:
        if type(item) is not dict:
            raise ConceptSuiteError(f"concept {concept_id} reference is invalid")
        record = cast(dict[str, object], item)
        if frozenset(record) != _REF_FIELDS:
            raise ConceptSuiteError(f"concept {concept_id} reference fields are invalid")
        post_id = _require_int(record["id"], f"concept {concept_id} reference id", minimum=1)
        if post_id in seen:
            raise ConceptSuiteError(
                f"concept {concept_id} references must be distinct posts"
            )
        seen.add(post_id)
        fav = _require_int(record["fav"], f"concept {concept_id} reference fav")
        aesthetics = record["aesthetics"]
        if aesthetics is not None and type(aesthetics) is not str:
            raise ConceptSuiteError(f"concept {concept_id} reference aesthetics is invalid")
        refs.append(ConceptRef(post_id=post_id, fav=fav, aesthetics=aesthetics))
    return tuple(refs)


def _parse_concept(value: object, index: int) -> ConceptSpec:
    if type(value) is not dict:
        raise ConceptSuiteError(f"concept {index} must be an object")
    record = cast(dict[str, object], value)
    if frozenset(record) != _CONCEPT_FIELDS:
        raise ConceptSuiteError(f"concept {index} fields are invalid")
    concept_id = _require_str(record["id"], f"concept {index} id")
    if _SAFE_CONCEPT_ID.fullmatch(concept_id) is None:
        raise ConceptSuiteError(f"concept id is invalid: {concept_id}")
    concept_type = _require_str(record["type"], f"concept {concept_id} type")
    if concept_type not in _CONCEPT_TYPES:
        raise ConceptSuiteError(f"concept {concept_id} type is invalid")
    tier = _require_str(record["tier"], f"concept {concept_id} tier")
    if tier not in _TIERS:
        raise ConceptSuiteError(f"concept {concept_id} tier is invalid")
    stratum = _require_int(record["stratum"], f"concept {concept_id} stratum")
    tag = _require_str(record["tag"], f"concept {concept_id} tag")
    if "think" in tag:
        raise ConceptSuiteError(f"concept {concept_id} tag is invalid")
    count = _require_int(record["count"], f"concept {concept_id} count")
    meta_tag = _require_str(record["meta_tag"], f"concept {concept_id} meta_tag")
    actual_count = _require_int(
        record["actual_count"], f"concept {concept_id} actual_count"
    )
    meta_status = _require_str(
        record["meta_status"], f"concept {concept_id} meta_status"
    )
    if meta_status not in _META_STATUSES:
        raise ConceptSuiteError(f"concept {concept_id} meta_status is invalid")
    swap = _require_str(record["swap"], f"concept {concept_id} swap")
    if swap == tag:
        raise ConceptSuiteError(f"concept {concept_id} cannot swap with itself")
    swap_count = _require_int(record["swap_count"], f"concept {concept_id} swap_count")
    swap_delta = _require_int(record["swap_delta"], f"concept {concept_id} swap_delta")
    if swap_delta != abs(count - swap_count):
        raise ConceptSuiteError(
            f"concept {concept_id} swap_delta must equal |count - swap_count|"
        )
    replaced_from = record["replaced_from"]
    if replaced_from is not None:
        replaced_from = _require_str(
            replaced_from, f"concept {concept_id} replaced_from"
        )
        if replaced_from == concept_id:
            raise ConceptSuiteError(
                f"concept {concept_id} cannot replace itself"
            )
    return ConceptSpec(
        id=concept_id,
        type=concept_type,
        tier=tier,
        stratum=stratum,
        tag=tag,
        count=count,
        meta_tag=meta_tag,
        actual_count=actual_count,
        meta_status=meta_status,
        swap=swap,
        swap_count=swap_count,
        swap_delta=swap_delta,
        refs=_parse_refs(record["refs"], concept_id),
        replaced_from=replaced_from,
    )


def _case_seed(seed: int, concept_id: str, stream: str) -> int:
    digest = hashlib.sha256(f"{seed}:{concept_id}:{stream}".encode("ascii")).hexdigest()
    return int(digest[:15], 16)


def canonical_prompt_cases(
    manifest: ConceptManifest, *, height: int, width: int
) -> tuple[PromptCase, ...]:
    """One prompt case per concept, generated with the concept's own tag."""

    return tuple(
        PromptCase(
            prompt_id=f"{concept.id}.canonical",
            prompt=concept.tag,
            conditions=(),
            seed=_case_seed(manifest.seed, concept.id, "canonical"),
            height=height,
            width=width,
        )
        for concept in manifest.concepts
    )


def swap_prompt_cases(
    manifest: ConceptManifest, *, height: int, width: int
) -> tuple[PromptCase, ...]:
    """One prompt case per concept, generated with the partner's tag.

    Swap cases reuse the canonical noise stream of the same concept, so a
    swap image differs from its canonical image only in the conditioning
    text.
    """

    return tuple(
        PromptCase(
            prompt_id=f"{concept.id}.swap",
            prompt=concept.swap,
            conditions=(),
            seed=_case_seed(manifest.seed, concept.id, "canonical"),
            height=height,
            width=width,
        )
        for concept in manifest.concepts
    )


@dataclass(frozen=True, slots=True)
class ConceptMetrics:
    concept_id: str
    type: str
    tier: str
    tag: str
    swap: str
    ref_sim_canonical: float
    ref_sim_null: float
    ref_sim_swap: float
    margin_null: float
    margin_swap: float
    retrieval_rank: int
    hit1: bool
    hit3: bool


@dataclass(frozen=True, slots=True)
class GroupAggregate:
    group: str
    n_concepts: int
    mean_margin_null: float
    median_margin_null: float
    fraction_margin_null_positive: float
    mean_margin_swap: float
    median_margin_swap: float
    fraction_margin_swap_positive: float
    mean_ref_sim_canonical: float
    mean_retrieval_rank: float
    hit1_rate: float
    hit3_rate: float


def _require_features(name: str, tensor: torch.Tensor, rows: int) -> torch.Tensor:
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    values = tensor.cpu().contiguous()
    if values.ndim != 2 or values.shape[0] != rows:
        raise ConceptSuiteError(
            f"{name} must have shape [{rows}, D], got {tuple(values.shape)}"
        )
    if not bool(torch.isfinite(values).all().item()):
        raise ConceptSuiteError(f"{name} contains nonfinite values")
    norms = torch.linalg.vector_norm(values, dim=1)
    if not bool(
        torch.allclose(norms, torch.ones_like(norms), atol=1e-3, rtol=1e-3)
    ):
        raise ConceptSuiteError(f"{name} rows must be L2-normalized")
    return values


def compute_concept_metrics(
    *,
    manifest: ConceptManifest,
    clip_canonical: torch.Tensor,
    clip_null: torch.Tensor,
    clip_swap: torch.Tensor,
    clip_refs: torch.Tensor,
) -> tuple[ConceptMetrics, ...]:
    """Per-concept margins, reference similarity, and self-retrieval rank.

    Feature rows must be L2-normalized (the CLIP pipeline guarantees it),
    so cosine similarity is a row dot product.  ``clip_refs`` holds the
    references of concept ``i`` in rows ``3i..3i+2``.  A reference with a
    similarity exactly tied to the best own reference does not demote the
    rank.
    """

    count = len(manifest.concepts)
    canon = _require_features("clip_canonical", clip_canonical, count)
    null = _require_features("clip_null", clip_null, count)
    swap = _require_features("clip_swap", clip_swap, count)
    refs = _require_features("clip_refs", clip_refs, count * _REF_COUNT)

    similarity = canon @ refs.T
    null_similarity = null @ refs.T
    swap_similarity = swap @ refs.T

    metrics: list[ConceptMetrics] = []
    for index, concept in enumerate(manifest.concepts):
        own = slice(index * _REF_COUNT, (index + 1) * _REF_COUNT)
        other = torch.ones(refs.shape[0], dtype=torch.bool)
        other[own] = False
        canon_row = similarity[index]
        best_own = canon_row[own].max()
        rank = 1 + int((canon_row[other] > best_own).sum().item())
        ref_sim_canonical = float(canon_row[own].mean())
        ref_sim_null = float(null_similarity[index, own].mean())
        ref_sim_swap = float(swap_similarity[index, own].mean())
        metrics.append(
            ConceptMetrics(
                concept_id=concept.id,
                type=concept.type,
                tier=concept.tier,
                tag=concept.tag,
                swap=concept.swap,
                ref_sim_canonical=ref_sim_canonical,
                ref_sim_null=ref_sim_null,
                ref_sim_swap=ref_sim_swap,
                margin_null=ref_sim_canonical - ref_sim_null,
                margin_swap=ref_sim_canonical - ref_sim_swap,
                retrieval_rank=rank,
                hit1=rank == 1,
                hit3=rank <= _REF_COUNT,
            )
        )
    return tuple(metrics)


def _round(value: float) -> float:
    if not math.isfinite(value):
        raise ConceptSuiteError(f"aggregate metric is not finite: {value}")
    return round(value, 6)


def aggregate_metrics(
    metrics: tuple[ConceptMetrics, ...],
) -> tuple[GroupAggregate, ...]:
    """Overall and per (type, tier) aggregates in a fixed group order."""

    groups: list[tuple[str, tuple[ConceptMetrics, ...]]] = [("overall", metrics)]
    for concept_type in _GROUP_TYPES:
        for tier in _GROUP_TIERS:
            cell = tuple(
                item
                for item in metrics
                if item.type == concept_type and item.tier == tier
            )
            if cell:
                groups.append((f"{concept_type}.{tier}", cell))
    result: list[GroupAggregate] = []
    for name, members in groups:
        total = len(members)
        result.append(
            GroupAggregate(
                group=name,
                n_concepts=total,
                mean_margin_null=sum(item.margin_null for item in members) / total,
                median_margin_null=float(
                    statistics.median(item.margin_null for item in members)
                ),
                fraction_margin_null_positive=(
                    sum(1 for item in members if item.margin_null > 0) / total
                ),
                mean_margin_swap=sum(item.margin_swap for item in members) / total,
                median_margin_swap=float(
                    statistics.median(item.margin_swap for item in members)
                ),
                fraction_margin_swap_positive=(
                    sum(1 for item in members if item.margin_swap > 0) / total
                ),
                mean_ref_sim_canonical=(
                    sum(item.ref_sim_canonical for item in members) / total
                ),
                mean_retrieval_rank=sum(
                    item.retrieval_rank for item in members
                )
                / total,
                hit1_rate=sum(1 for item in members if item.hit1) / total,
                hit3_rate=sum(1 for item in members if item.hit3) / total,
            )
        )
    return tuple(result)


def _group_document(aggregate: GroupAggregate) -> dict[str, object]:
    return {
        "n_concepts": aggregate.n_concepts,
        "mean_margin_null": _round(aggregate.mean_margin_null),
        "median_margin_null": _round(aggregate.median_margin_null),
        "fraction_margin_null_positive": _round(
            aggregate.fraction_margin_null_positive
        ),
        "mean_margin_swap": _round(aggregate.mean_margin_swap),
        "median_margin_swap": _round(aggregate.median_margin_swap),
        "fraction_margin_swap_positive": _round(
            aggregate.fraction_margin_swap_positive
        ),
        "mean_ref_sim_canonical": _round(aggregate.mean_ref_sim_canonical),
        "mean_retrieval_rank": _round(aggregate.mean_retrieval_rank),
        "hit1_rate": _round(aggregate.hit1_rate),
        "hit3_rate": _round(aggregate.hit3_rate),
    }


def suite_report_document(
    *,
    manifest: ConceptManifest,
    metrics: tuple[ConceptMetrics, ...],
    aggregates: tuple[GroupAggregate, ...],
    provenance: dict[str, object],
) -> dict[str, object]:
    """Machine-readable report document (TOML-safe nested dicts)."""

    if len(metrics) != len(manifest.concepts):
        raise ConceptSuiteError("report metric count differs from the manifest")
    suite: dict[str, object] = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "n_concepts": len(metrics),
        "seed": manifest.seed,
    }
    suite.update(provenance)
    return {
        "suite": suite,
        "aggregate": {agg.group: _group_document(agg) for agg in aggregates},
        "concepts": [
            {
                "id": item.concept_id,
                "type": item.type,
                "tier": item.tier,
                "tag": item.tag,
                "swap": item.swap,
                "ref_sim_canonical": _round(item.ref_sim_canonical),
                "ref_sim_null": _round(item.ref_sim_null),
                "ref_sim_swap": _round(item.ref_sim_swap),
                "margin_null": _round(item.margin_null),
                "margin_swap": _round(item.margin_swap),
                "retrieval_rank": item.retrieval_rank,
                "hit1": item.hit1,
                "hit3": item.hit3,
            }
            for item in metrics
        ],
    }


def render_suite_markdown(
    *,
    metrics: tuple[ConceptMetrics, ...],
    aggregates: tuple[GroupAggregate, ...],
    suite: dict[str, object],
    weakest: int = 10,
) -> str:
    """Compact human-facing report (group table + weakest swap margins)."""

    lines: list[str] = []
    provenance = " ".join(f"{key}={value}" for key, value in suite.items())
    lines.append(f"# concept-suite {provenance}")
    lines.append("")
    lines.append(
        "| group | n | m_null μ | m_null med | %>0 | m_swap μ | %>0 | "
        "refsim | rank μ | hit@1 | hit@3 |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for aggregate in aggregates:
        lines.append(
            f"| {aggregate.group} | {aggregate.n_concepts} "
            f"| {aggregate.mean_margin_null:.4f} "
            f"| {aggregate.median_margin_null:.4f} "
            f"| {aggregate.fraction_margin_null_positive * 100:.1f} "
            f"| {aggregate.mean_margin_swap:.4f} "
            f"| {aggregate.fraction_margin_swap_positive * 100:.1f} "
            f"| {aggregate.mean_ref_sim_canonical:.4f} "
            f"| {aggregate.mean_retrieval_rank:.2f} "
            f"| {aggregate.hit1_rate * 100:.1f} "
            f"| {aggregate.hit3_rate * 100:.1f} |"
        )
    ranked = sorted(metrics, key=lambda item: item.margin_swap)
    shown = ranked[:weakest]
    if shown:
        lines.append("")
        lines.append(f"## 最弱 {len(shown)} 个 margin_swap")
        lines.append("")
        lines.append("| id | tag | swap | m_swap | rank |")
        lines.append("|---|---|---|---|---|")
        for item in shown:
            lines.append(
                f"| {item.concept_id} | {item.tag} | {item.swap} "
                f"| {item.margin_swap:.4f} | {item.retrieval_rank} |"
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "SUITE_SCHEMA_VERSION",
    "ConceptManifest",
    "ConceptMetrics",
    "ConceptRef",
    "ConceptSpec",
    "ConceptSuiteError",
    "GroupAggregate",
    "aggregate_metrics",
    "canonical_prompt_cases",
    "compute_concept_metrics",
    "render_suite_markdown",
    "suite_report_document",
    "swap_prompt_cases",
]
