"""Standalone concept-conditioning benchmark suite (dual prompt pathway).

For every concept in the draw manifest the suite generates five images
sharing one canonical noise stream per concept, in two prompt pathways:

* **condition pathway** (the condition branch is tested as a dedicated
  input):
  - **condition-canonical** -- the concept's own tag as a structured
    condition (artist -> artist_text/style, character -> character_text/
    identity), main text empty;
  - **condition-swap**      -- the partner tag as the structured condition;
* **text pathway** (the tag rides the main caption text):
  - **text-canonical**      -- the concept's own tag as a structured main
    tag (source artist/character by concept type), condition empty;
  - **text-swap**           -- the partner tag as the structured main tag;
* **shared null** -- the identical canonical-stream noise with the
  condition fully dropped, generated once and reused by both pathways.

Both pathways are scored independently with the same metric math
(:func:`compute_concept_metrics`), so the reports and the flattened
telemetry namespace every metric under ``condition/`` or ``text/``.  A
positive margin means the model ranked the intended concept above the
alternative against the concept's three Danbooru reference posts.

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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from sakuramoon.data.caption import (
    CaptionPlan,
    CaptionTag,
    ConditionRequest,
    ConditionRole,
    ConditionSource,
    Tag,
    TagSource,
    empty_caption_dropout_hits,
)
from sakuramoon.eval.spec import PromptCase, caption_plan_prompt_text

SUITE_SCHEMA_VERSION = 2
PROTOCOL = "dual-path-v1"
# The manifest contract is its own version (v1 is unchanged by the dual-path
# protocol); the report document schema is versioned separately.
MANIFEST_SCHEMA_VERSION = 1
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
# The concept entry point explicitly expresses the condition routing by
# concept type; the suite never guesses a role from the tag text.
_CONCEPT_CONDITION_ROUTES: Mapping[str, tuple[ConditionSource, ConditionRole]] = {
    "artist": ("artist_text", "style"),
    "character": ("character_text", "identity"),
}
# Text pathway: the structured main-tag source mirrors the concept type.
_CONCEPT_MAIN_TAG_SOURCES: Mapping[str, TagSource] = {
    "artist": "artist",
    "character": "character",
}


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
    stratum: int | None
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
        if document["schema_version"] != MANIFEST_SCHEMA_VERSION:
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
    stratum_raw = record["stratum"]
    stratum = None if stratum_raw is None else _require_int(
        stratum_raw, f"concept {concept_id} stratum"
    )
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


def _concept_caption_plan(
    concept_id: str, display: str, canonical: str, concept_type: str
) -> CaptionPlan:
    """Structured condition plan for one concept tag.

    The tag is an explicit tag input: it becomes a structured condition tag
    at this construction boundary (surrounding whitespace stripped, empty
    tags rejected), and the existing serializer renders its display text.
    The canonical identity comes from the manifest's explicit ``meta_tag``
    for the concept's own tag, or from :func:`_swap_tag_identity` for the
    swap field.
    """

    if type(display) is not str or not display.strip():
        raise ConceptSuiteError(f"concept {concept_id} tag is empty")
    if type(canonical) is not str or not canonical.strip():
        raise ConceptSuiteError(f"concept {concept_id} canonical tag is empty")
    route = _CONCEPT_CONDITION_ROUTES.get(concept_type)
    if route is None:
        raise ConceptSuiteError(
            f"concept {concept_id} type {concept_type!r} has no condition route"
        )
    source, role = route
    return CaptionPlan(
        tags=(),
        condition=ConditionRequest(
            source=source,
            role=role,
            tags=(Tag(display.strip(), canonical.strip()),),
        ),
        nl_text=None,
        selected_nl=None,
        all_condition_dropped=False,
        dropout_hits=empty_caption_dropout_hits(),
    )


def _swap_tag_identity(display: str) -> tuple[str, str]:
    """The Concept Manifest v1 canonical contract for the ``swap`` field.

    ``swap.display`` is the manifest value as-is and
    ``swap.canonical`` is ``display.replace(" ", "_")``.  Notes:

    * This is the v1 contract for the manifest ``swap`` field -- **not** a
      generic inverse of ``serialize._display_text``.  Do not apply it to
      natural-language prompts or to arbitrary ``Tag.text`` values.
    * A concept's own canonical identity always comes from its explicit
      ``meta_tag``; only the swap field, which has no independent
      ``swap_meta_tag`` in the v1 schema, uses this rule.
    * The round trip ``display -> canonical -> display`` must be lossless.
      A swap whose display breaks it (for example a literal underscore)
      is rejected so the suite fails closed instead of guessing an ID.

    The shipped concept-120 manifests were audited against this contract
    (every swap passes the round trip) before it was pinned as v1.
    """

    canonical = display.replace(" ", "_")
    if canonical.replace("_", " ") != display:
        raise ConceptSuiteError(
            f"swap tag {display!r} violates the manifest v1 round-trip contract"
        )
    return display, canonical


def _concept_text_plan(
    concept_id: str, display: str, canonical: str, concept_type: str
) -> CaptionPlan:
    """Structured main-text plan for the text pathway.

    The tag rides the main caption body as a structured tag (source
    ``artist``/``character`` by concept type); there is no condition and no
    natural-language text, so the existing serializer alone produces the
    tokenizer-facing text.
    """

    source = _CONCEPT_MAIN_TAG_SOURCES.get(concept_type)
    if source is None:
        raise ConceptSuiteError(
            f"concept {concept_id} type {concept_type!r} has no text-path source"
        )
    if type(display) is not str or not display.strip():
        raise ConceptSuiteError(f"concept {concept_id} tag is empty")
    if type(canonical) is not str or not canonical.strip():
        raise ConceptSuiteError(f"concept {concept_id} canonical tag is empty")
    return CaptionPlan(
        tags=(
            CaptionTag(
                source=source, tag=Tag(display.strip(), canonical.strip())
            ),
        ),
        condition=None,
        nl_text=None,
        selected_nl=None,
        all_condition_dropped=False,
        dropout_hits=empty_caption_dropout_hits(),
    )


def canonical_prompt_cases(
    manifest: ConceptManifest, *, height: int, width: int
) -> tuple[PromptCase, ...]:
    """One prompt case per concept, conditioned on the concept's own tag."""

    cases: list[PromptCase] = []
    for concept in manifest.concepts:
        plan = _concept_caption_plan(
            concept.id, concept.tag, concept.meta_tag, concept.type
        )
        cases.append(
            PromptCase(
                prompt_id=f"{concept.id}.canonical",
                prompt=caption_plan_prompt_text(plan),
                conditions=(),
                seed=_case_seed(manifest.seed, concept.id, "canonical"),
                height=height,
                width=width,
                caption_plan=plan,
            )
        )
    return tuple(cases)


def swap_prompt_cases(
    manifest: ConceptManifest, *, height: int, width: int
) -> tuple[PromptCase, ...]:
    """One prompt case per concept, conditioned on the partner's tag.

    Swap cases reuse the canonical noise stream of the same concept, so a
    swap image differs from its canonical image only in the conditioning
    text.  The swap identity comes from the single v1 contract helper.
    """

    cases: list[PromptCase] = []
    for concept in manifest.concepts:
        display, canonical = _swap_tag_identity(concept.swap.strip())
        plan = _concept_caption_plan(concept.id, display, canonical, concept.type)
        cases.append(
            PromptCase(
                prompt_id=f"{concept.id}.swap",
                prompt=caption_plan_prompt_text(plan),
                conditions=(),
                seed=_case_seed(manifest.seed, concept.id, "canonical"),
                height=height,
                width=width,
                caption_plan=plan,
            )
        )
    return tuple(cases)


def text_canonical_prompt_cases(
    manifest: ConceptManifest, *, height: int, width: int
) -> tuple[PromptCase, ...]:
    """Text pathway: the concept's own tag as a structured main tag.

    Reuses the canonical noise stream, so a text-canonical image differs
    from its condition-canonical image only in the prompt pathway.
    """

    cases: list[PromptCase] = []
    for concept in manifest.concepts:
        plan = _concept_text_plan(
            concept.id, concept.tag, concept.meta_tag, concept.type
        )
        cases.append(
            PromptCase(
                prompt_id=f"{concept.id}.text-canonical",
                prompt=caption_plan_prompt_text(plan),
                conditions=(),
                seed=_case_seed(manifest.seed, concept.id, "canonical"),
                height=height,
                width=width,
                caption_plan=plan,
            )
        )
    return tuple(cases)


def text_swap_prompt_cases(
    manifest: ConceptManifest, *, height: int, width: int
) -> tuple[PromptCase, ...]:
    """Text pathway: the partner tag as a structured main tag.

    Uses the same :func:`_swap_tag_identity` resolution as the condition
    swap, so both pathways carry the identical swap display/canonical
    identity and only the prompt branch differs.
    """

    cases: list[PromptCase] = []
    for concept in manifest.concepts:
        display, canonical = _swap_tag_identity(concept.swap.strip())
        plan = _concept_text_plan(concept.id, display, canonical, concept.type)
        cases.append(
            PromptCase(
                prompt_id=f"{concept.id}.text-swap",
                prompt=caption_plan_prompt_text(plan),
                conditions=(),
                seed=_case_seed(manifest.seed, concept.id, "canonical"),
                height=height,
                width=width,
                caption_plan=plan,
            )
        )
    return tuple(cases)


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


def _concept_document(item: ConceptMetrics) -> dict[str, object]:
    return {
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


@dataclass(frozen=True, slots=True)
class DualPathSuite:
    """Scored dual-path result: one metrics/aggregate pair per pathway."""

    condition_metrics: tuple[ConceptMetrics, ...]
    text_metrics: tuple[ConceptMetrics, ...]
    condition_aggregates: tuple[GroupAggregate, ...]
    text_aggregates: tuple[GroupAggregate, ...]


def score_dual_path(
    *,
    manifest: ConceptManifest,
    condition: dict[str, torch.Tensor],
    text: dict[str, torch.Tensor],
    clip_refs: torch.Tensor,
) -> DualPathSuite:
    """Score both prompt pathways with the existing metric math.

    ``condition`` and ``text`` each hold the L2-normalized CLIP feature
    rows under the keys ``canonical``, ``null`` and ``swap``.  The two
    pathways must share the same null features (the single shared null
    image) and the same reference features; both metric sets are produced
    by the same :func:`compute_concept_metrics` call pattern, so the math
    is identical across pathways and only the canonical/swap inputs differ.
    """

    for name, features in (("condition", condition), ("text", text)):
        for key in ("canonical", "null", "swap"):
            if key not in features:
                raise ConceptSuiteError(
                    f"{name} pathway is missing the {key!r} features"
                )
    if not torch.equal(condition["null"], text["null"]):
        raise ConceptSuiteError(
            "the shared null features must be the same tensor for both pathways"
        )
    condition_metrics = compute_concept_metrics(
        manifest=manifest,
        clip_canonical=condition["canonical"],
        clip_null=condition["null"],
        clip_swap=condition["swap"],
        clip_refs=clip_refs,
    )
    text_metrics = compute_concept_metrics(
        manifest=manifest,
        clip_canonical=text["canonical"],
        clip_null=text["null"],
        clip_swap=text["swap"],
        clip_refs=clip_refs,
    )
    return DualPathSuite(
        condition_metrics=condition_metrics,
        text_metrics=text_metrics,
        condition_aggregates=aggregate_metrics(condition_metrics),
        text_aggregates=aggregate_metrics(text_metrics),
    )


def suite_report_document(
    *,
    manifest: ConceptManifest,
    suite: DualPathSuite,
    provenance: dict[str, object],
) -> dict[str, object]:
    """Machine-readable report document (TOML-safe nested dicts).

    The two prompt pathways are namespaced under ``condition`` and
    ``text`` so no metric alias can silently collide across pathways.
    """

    if len(suite.condition_metrics) != len(manifest.concepts):
        raise ConceptSuiteError(
            "condition report metric count differs from the manifest"
        )
    if len(suite.text_metrics) != len(manifest.concepts):
        raise ConceptSuiteError(
            "text report metric count differs from the manifest"
        )
    meta: dict[str, object] = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "n_concepts": len(suite.condition_metrics),
        "seed": manifest.seed,
        "concept_prompt_protocol": PROTOCOL,
    }
    meta.update(provenance)
    return {
        "suite": meta,
        "condition": {
            "aggregate": {
                agg.group: _group_document(agg)
                for agg in suite.condition_aggregates
            },
            "concepts": [
                _concept_document(item) for item in suite.condition_metrics
            ],
        },
        "text": {
            "aggregate": {
                agg.group: _group_document(agg) for agg in suite.text_aggregates
            },
            "concepts": [
                _concept_document(item) for item in suite.text_metrics
            ],
        },
    }


def _pathway_markdown_section(
    name: str,
    metrics: tuple[ConceptMetrics, ...],
    aggregates: tuple[GroupAggregate, ...],
    weakest: int,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"## {name} protocol")
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
        lines.append(f"### {name}: 最弱 {len(shown)} 个 margin_swap")
        lines.append("")
        lines.append("| id | tag | swap | m_swap | rank |")
        lines.append("|---|---|---|---|---|")
        for item in shown:
            lines.append(
                f"| {item.concept_id} | {item.tag} | {item.swap} "
                f"| {item.margin_swap:.4f} | {item.retrieval_rank} |"
            )
    lines.append("")
    return lines


def render_suite_markdown(
    *,
    suite: DualPathSuite,
    suite_meta: dict[str, object],
    weakest: int = 10,
) -> str:
    """Compact human-facing report with one section per prompt pathway."""

    lines: list[str] = []
    provenance = " ".join(f"{key}={value}" for key, value in suite_meta.items())
    lines.append(f"# concept-suite {provenance}")
    lines.append("")
    lines.extend(
        _pathway_markdown_section(
            "condition",
            suite.condition_metrics,
            suite.condition_aggregates,
            weakest,
        )
    )
    lines.extend(
        _pathway_markdown_section(
            "text", suite.text_metrics, suite.text_aggregates, weakest
        )
    )
    return "\n".join(lines).rstrip("\n") + "\n"


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "PROTOCOL",
    "SUITE_SCHEMA_VERSION",
    "ConceptManifest",
    "ConceptMetrics",
    "ConceptRef",
    "ConceptSpec",
    "ConceptSuiteError",
    "DualPathSuite",
    "GroupAggregate",
    "aggregate_metrics",
    "canonical_prompt_cases",
    "compute_concept_metrics",
    "render_suite_markdown",
    "score_dual_path",
    "suite_report_document",
    "swap_prompt_cases",
    "text_canonical_prompt_cases",
    "text_swap_prompt_cases",
]
