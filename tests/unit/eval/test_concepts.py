import dataclasses
import json
import math
from typing import cast

import pytest
import torch

from sakuramoon.eval.concepts import (
    ConceptManifest,
    ConceptMetrics,
    ConceptSuiteError,
    GroupAggregate,
    aggregate_metrics,
    canonical_prompt_cases,
    compute_concept_metrics,
    render_suite_markdown,
    suite_report_document,
    swap_prompt_cases,
)


def _concepts() -> list[dict[str, object]]:
    return [
        {
            "id": "A001",
            "type": "artist",
            "tier": "high",
            "stratum": 4,
            "tag": "dairi",
            "count": 18655,
            "meta_tag": "dairi",
            "actual_count": 18655,
            "meta_status": "matched",
            "swap": "kantoku",
            "swap_count": 2419,
            "swap_delta": 16236,
            "refs": [
                {"id": 1, "fav": 112, "aesthetics": "excellent"},
                {"id": 2, "fav": 93, "aesthetics": "excellent"},
                {"id": 3, "fav": 80, "aesthetics": None},
            ],
            "replaced_from": None,
        },
        {
            "id": "C061",
            "type": "character",
            "tier": "mid",
            "stratum": 4,
            "tag": "hatsune miku",
            "count": 135838,
            "meta_tag": "hatsune_miku",
            "actual_count": 135838,
            "meta_status": "matched",
            "swap": "hong meiling",
            "swap_count": 27269,
            "swap_delta": 108569,
            "refs": [
                {"id": 4, "fav": 283, "aesthetics": "excellent"},
                {"id": 5, "fav": 282, "aesthetics": "excellent"},
                {"id": 6, "fav": 281, "aesthetics": "excellent"},
            ],
            "replaced_from": None,
        },
    ]


def _manifest_bytes(**top: object) -> bytes:
    document: dict[str, object] = {
        "schema_version": 1,
        "seed": 20260822,
        "concepts": _concepts(),
    }
    document.update(top)
    return json.dumps(document).encode("utf-8")


def _concepts_bytes(concepts: object) -> bytes:
    return _manifest_bytes(concepts=concepts)


def _manifest() -> ConceptManifest:
    return ConceptManifest.from_bytes(_manifest_bytes())


def test_manifest_round_trip() -> None:
    manifest = _manifest()
    assert manifest.seed == 20260822
    assert len(manifest.concepts) == 2
    assert manifest.concepts[0].ref_post_ids == (1, 2, 3)
    assert manifest.concepts[1].swap_delta == 108569
    assert manifest.concepts[1].replaced_from is None


@pytest.mark.parametrize(
    ("top", "match"),
    [
        ({"draw": {}}, "top-level fields"),
        ({"schema_version": 2}, "schema version"),
        ({"seed": -1}, "seed"),
        ({"seed": 1.0}, "seed"),
        ({"concepts": []}, "nonempty"),
    ],
)
def test_manifest_rejects_invalid_top_level(top: dict[str, object], match: str) -> None:
    with pytest.raises(ConceptSuiteError, match=match):
        ConceptManifest.from_bytes(_manifest_bytes(**top))


def test_manifest_rejects_duplicate_ids() -> None:
    concepts = _concepts()
    concepts[1]["id"] = "A001"
    with pytest.raises(ConceptSuiteError, match="unique"):
        ConceptManifest.from_bytes(_concepts_bytes(concepts))


def test_manifest_rejects_missing_and_extra_fields() -> None:
    missing = _concepts()
    del missing[0]["count"]
    with pytest.raises(ConceptSuiteError, match="fields are invalid"):
        ConceptManifest.from_bytes(_concepts_bytes(missing))
    extra = _concepts()
    extra[0]["draw_note"] = "x"
    with pytest.raises(ConceptSuiteError, match="fields are invalid"):
        ConceptManifest.from_bytes(_concepts_bytes(extra))


def test_manifest_rejects_bad_enums_and_swap_contract() -> None:
    bad = _concepts()
    bad[0]["type"] = "painter"
    with pytest.raises(ConceptSuiteError, match="type is invalid"):
        ConceptManifest.from_bytes(_concepts_bytes(bad))
    bad = _concepts()
    bad[0]["tier"] = "rare"
    with pytest.raises(ConceptSuiteError, match="tier is invalid"):
        ConceptManifest.from_bytes(_concepts_bytes(bad))
    bad = _concepts()
    bad[0]["meta_status"] = "fuzzy"
    with pytest.raises(ConceptSuiteError, match="meta_status is invalid"):
        ConceptManifest.from_bytes(_concepts_bytes(bad))
    bad = _concepts()
    bad[0]["swap_delta"] = 1
    with pytest.raises(ConceptSuiteError, match="swap_delta"):
        ConceptManifest.from_bytes(_concepts_bytes(bad))
    bad = _concepts()
    bad[0]["swap"] = "dairi"
    with pytest.raises(ConceptSuiteError, match="cannot swap with itself"):
        ConceptManifest.from_bytes(_concepts_bytes(bad))


def test_manifest_rejects_bad_refs() -> None:
    short = _concepts()
    short_refs = cast(list[dict[str, object]], short[0]["refs"])
    short[0]["refs"] = short_refs[:2]
    with pytest.raises(ConceptSuiteError, match="exactly 3 references"):
        ConceptManifest.from_bytes(_concepts_bytes(short))
    duplicated = _concepts()
    duplicated_refs = cast(list[dict[str, object]], duplicated[0]["refs"])
    duplicated_refs[1] = {"id": 1, "fav": 93, "aesthetics": None}
    with pytest.raises(ConceptSuiteError, match="distinct posts"):
        ConceptManifest.from_bytes(_concepts_bytes(duplicated))
    negative = _concepts()
    negative[0]["count"] = -5
    with pytest.raises(ConceptSuiteError, match="count"):
        ConceptManifest.from_bytes(_concepts_bytes(negative))


def test_prompt_case_builders() -> None:
    manifest = _manifest()
    canonical = canonical_prompt_cases(manifest, height=512, width=512)
    swap = swap_prompt_cases(manifest, height=512, width=512)
    assert [case.prompt_id for case in canonical] == ["A001.canonical", "C061.canonical"]
    assert [case.prompt_id for case in swap] == ["A001.swap", "C061.swap"]
    assert canonical[0].prompt == "dairi"
    assert swap[0].prompt == "kantoku"
    assert all(case.conditions == () for case in canonical + swap)
    for case, swap_case in zip(canonical, swap):
        assert case.seed == swap_case.seed
        assert case.height == swap_case.height == 512
    assert len({case.seed for case in canonical}) == len(canonical)
    again = canonical_prompt_cases(manifest, height=512, width=512)
    assert [case.seed for case in canonical] == [case.seed for case in again]
    other = ConceptManifest.from_bytes(_manifest_bytes(seed=1))
    assert canonical_prompt_cases(other, height=512, width=512)[0].seed != canonical[0].seed


def _normalized_vectors() -> tuple[torch.Tensor, ...]:
    half = math.sqrt(2.0) / 2.0
    return (
        torch.tensor([[half, half, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.6, 0.8, 0.0],
                [0.0, 0.0, 1.0],
                [0.8, 0.6, 0.0],
            ]
        ),
    )


def test_compute_concept_metrics_math() -> None:
    canonical, null, swap, refs = _normalized_vectors()
    manifest = _manifest()
    metrics = compute_concept_metrics(
        manifest=manifest,
        clip_canonical=canonical,
        clip_null=null,
        clip_swap=swap,
        clip_refs=refs,
    )
    first, second = metrics
    assert first.concept_id == "A001"
    assert first.type == "artist" and first.tier == "high"
    # Concept 0: canonical [h, h, 0] vs own refs [e1, e2, e3] -> 2h/3.
    assert first.ref_sim_canonical == pytest.approx(math.sqrt(2.0) / 3.0, abs=1e-6)
    assert first.ref_sim_null == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert first.margin_null == pytest.approx(
        math.sqrt(2.0) / 3.0 - 1.0 / 3.0, abs=1e-6
    )
    assert first.margin_swap == pytest.approx(
        math.sqrt(2.0) / 3.0 - 1.0 / 3.0, abs=1e-6
    )
    # Own max is h; both foreign refs s0/s2 score 1.4h > h, so rank 3.
    assert first.retrieval_rank == 3
    assert not first.hit1
    assert first.hit3
    # Concept 1: canonical [0, 1, 0] vs own refs -> (0.8 + 0 + 0.6) / 3.
    assert second.ref_sim_canonical == pytest.approx(1.4 / 3.0, abs=1e-6)
    assert second.ref_sim_null == pytest.approx(1.4 / 3.0, abs=1e-6)
    assert second.margin_null == pytest.approx(0.0, abs=1e-9)
    # Foreign ref r2 scores 1.0 > best own 0.8, so rank 2.
    assert second.retrieval_rank == 2
    assert not second.hit1
    assert second.hit3


def test_compute_concept_metrics_rejects_bad_features() -> None:
    canonical, null, swap, refs = _normalized_vectors()
    manifest = _manifest()
    unnormalized = canonical.clone()
    unnormalized[0] = torch.tensor([1.0, 1.0, 0.0])
    with pytest.raises(ConceptSuiteError, match="L2-normalized"):
        compute_concept_metrics(
            manifest=manifest,
            clip_canonical=unnormalized,
            clip_null=null,
            clip_swap=swap,
            clip_refs=refs,
        )
    with pytest.raises(ConceptSuiteError, match="shape"):
        compute_concept_metrics(
            manifest=manifest,
            clip_canonical=canonical[:1],
            clip_null=null,
            clip_swap=swap,
            clip_refs=refs,
        )
    nonfinite = canonical.clone()
    nonfinite[0] = torch.tensor([float("nan"), 1.0, 0.0])
    with pytest.raises(ConceptSuiteError, match="nonfinite"):
        compute_concept_metrics(
            manifest=manifest,
            clip_canonical=nonfinite,
            clip_null=null,
            clip_swap=swap,
            clip_refs=refs,
        )


def test_aggregate_metrics_groups() -> None:
    canonical, null, swap, refs = _normalized_vectors()
    manifest = _manifest()
    metrics = compute_concept_metrics(
        manifest=manifest,
        clip_canonical=canonical,
        clip_null=null,
        clip_swap=swap,
        clip_refs=refs,
    )
    aggregates = aggregate_metrics(metrics)
    assert [item.group for item in aggregates] == [
        "overall",
        "artist.high",
        "character.mid",
    ]
    overall = aggregates[0]
    assert overall.n_concepts == 2
    assert overall.mean_margin_null == pytest.approx(
        (math.sqrt(2.0) - 1.0) / 6.0, abs=1e-6
    )
    assert overall.fraction_margin_null_positive == 0.5
    assert overall.fraction_margin_swap_positive == 0.5
    assert overall.mean_retrieval_rank == 2.5
    assert overall.hit1_rate == 0.0
    assert overall.hit3_rate == 1.0
    artist = aggregates[1]
    assert artist.n_concepts == 1
    assert artist.fraction_margin_null_positive == 1.0
    assert artist.mean_retrieval_rank == 3.0
    character = aggregates[2]
    assert character.fraction_margin_null_positive == 0.0
    assert character.mean_retrieval_rank == 2.0


def _default_aggregate() -> GroupAggregate:
    return GroupAggregate(
        group="overall",
        n_concepts=2,
        mean_margin_null=0.1,
        median_margin_null=0.1,
        fraction_margin_null_positive=0.5,
        mean_margin_swap=0.1,
        median_margin_swap=0.1,
        fraction_margin_swap_positive=0.5,
        mean_ref_sim_canonical=0.47,
        mean_retrieval_rank=2.5,
        hit1_rate=0.0,
        hit3_rate=1.0,
    )


def test_suite_report_document() -> None:
    canonical, null, swap, refs = _normalized_vectors()
    manifest = _manifest()
    metrics = compute_concept_metrics(
        manifest=manifest,
        clip_canonical=canonical,
        clip_null=null,
        clip_swap=swap,
        clip_refs=refs,
    )
    document = suite_report_document(
        manifest=manifest,
        metrics=metrics,
        aggregates=aggregate_metrics(metrics),
        provenance={"update": 70000, "growth_alpha": 0.5},
    )
    suite = cast(dict[str, object], document["suite"])
    aggregate = cast(dict[str, object], document["aggregate"])
    concepts = cast(list[dict[str, object]], document["concepts"])
    assert set(suite) == {
        "schema_version",
        "n_concepts",
        "seed",
        "update",
        "growth_alpha",
    }
    assert suite["n_concepts"] == 2
    assert suite["update"] == 70000
    assert set(aggregate) == {"overall", "artist.high", "character.mid"}
    assert len(concepts) == 2
    assert concepts[0]["margin_null"] == round(
        math.sqrt(2.0) / 3.0 - 1.0 / 3.0, 6
    )
    json.dumps(document)


def _valid_metrics() -> tuple[ConceptMetrics, ...]:
    canonical, null, swap, refs = _normalized_vectors()
    return compute_concept_metrics(
        manifest=_manifest(),
        clip_canonical=canonical,
        clip_null=null,
        clip_swap=swap,
        clip_refs=refs,
    )


def test_suite_report_document_rejects_nonfinite() -> None:
    first, second = _valid_metrics()
    metrics = (
        dataclasses.replace(first, ref_sim_canonical=float("nan")),
        second,
    )
    with pytest.raises(ConceptSuiteError, match="not finite"):
        suite_report_document(
            manifest=_manifest(),
            metrics=metrics,
            aggregates=(
                dataclasses.replace(
                    _default_aggregate(),
                    mean_margin_null=float("nan"),
                ),
            ),
            provenance={},
        )


def test_render_suite_markdown() -> None:
    canonical, null, swap, refs = _normalized_vectors()
    manifest = _manifest()
    metrics = compute_concept_metrics(
        manifest=manifest,
        clip_canonical=canonical,
        clip_null=null,
        clip_swap=swap,
        clip_refs=refs,
    )
    aggregates = aggregate_metrics(metrics)
    document = suite_report_document(
        manifest=manifest,
        metrics=metrics,
        aggregates=aggregates,
        provenance={"update": 70000},
    )
    markdown = render_suite_markdown(
        metrics=metrics,
        aggregates=aggregates,
        suite=cast(dict[str, object], document["suite"]),
    )
    assert markdown.startswith("# concept-suite")
    assert "| overall |" in markdown
    assert "| artist.high |" in markdown
    assert "margin_swap" in markdown
    # Both concepts have zero swap margin; the weakest table lists them.
    assert "| A001 | dairi | kantoku |" in markdown
    assert markdown.endswith("\n")
