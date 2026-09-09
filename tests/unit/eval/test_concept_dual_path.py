# pyright: reportPrivateUsage=false
"""Dual-path concept suite: shared runner, shared null, and namespaces.

The in-training suite (``eval.concept_suite``) and the standalone CLI must
share one implementation; these tests pin that behaviour with a fake
evaluator and a fake CLIP model so no GPU or model download is needed.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import cast

import pytest
import torch

from sakuramoon.eval.concept_suite import (
    CONCEPT_IMAGE_STATES,
    _flatten_dual_path,
    run_dual_path_suite,
    save_state_images,
)
from sakuramoon.eval.concepts import (
    ConceptManifest,
    ConceptSuiteError,
    aggregate_metrics,
    canonical_prompt_cases,
    compute_concept_metrics,
    score_dual_path,
    swap_prompt_cases,
    text_canonical_prompt_cases,
    text_swap_prompt_cases,
)
from sakuramoon.eval.features import ClipFeatureModel
from sakuramoon.eval.runtime import TrainingEvaluator
from sakuramoon.eval.spec import PromptCase


def _manifest() -> ConceptManifest:
    document = {
        "schema_version": 1,
        "seed": 20260822,
        "concepts": [
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
                    {"id": 5, "fav": 282, "aesthetics": "excellent"},
                    {"id": 6, "fav": 281, "aesthetics": "excellent"},
                    {"id": 7, "fav": 280, "aesthetics": None},
                ],
                "replaced_from": None,
            },
        ],
    }
    return ConceptManifest.from_bytes(json.dumps(document).encode("utf-8"))


class _FakeConfig:
    class _Train:
        resolution = 64

    train = _Train()


class _FakeEvaluator:
    """Records every generate pass and returns deterministic images."""

    def __init__(self, device: torch.device) -> None:
        self.config = _FakeConfig()
        self.device = device
        self.root = None
        self.growth_alpha = 0.5
        self.calls: list[dict[str, object]] = []

    def generate(
        self, cases: tuple[PromptCase, ...], *, null: bool
    ) -> torch.Tensor:
        generator = torch.Generator().manual_seed(
            1000 + (9 if null else 0)
            + sum(case.seed for case in cases) % 997
        )
        images = torch.randn(len(cases), 3, 8, 8, generator=generator)
        self.calls.append(
            {
                "null": null,
                "n": len(cases),
                "seeds": tuple(case.seed for case in cases),
            }
        )
        return images


class _FakeClip:
    """Deterministic 'CLIP': flattened image rows, L2-normalized."""

    def __init__(self, _root: object, _device: torch.device) -> None:
        pass

    def features(self, images: torch.Tensor) -> torch.Tensor:
        values = images.reshape(images.shape[0], -1).float()
        norm = (values * values).sum(dim=1, keepdim=True).sqrt()
        return values / norm


def test_run_dual_path_suite_generates_five_states_with_one_shared_null() -> None:
    manifest = _manifest()
    evaluator = _FakeEvaluator(torch.device("cpu"))
    refs = torch.stack(
        [
            torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(i))
            for i in range(6)
        ]
    )
    result, images = run_dual_path_suite(
        cast(TrainingEvaluator, evaluator),
        manifest=manifest,
        ref_images=refs,
        batch_size=len(manifest.concepts),
        clip=cast(ClipFeatureModel, _FakeClip(None, torch.device("cpu"))),
    )

    # Exactly five generation passes: four conditional pathways plus one
    # shared null (never a second null).
    null_calls = [call for call in evaluator.calls if call["null"]]
    conditional_calls = [call for call in evaluator.calls if not call["null"]]
    assert len(null_calls) == 1
    assert len(conditional_calls) == 4
    # The shared null rides the canonical noise stream of the same concept.
    canonical_cases = canonical_prompt_cases(manifest, height=64, width=64)
    assert null_calls[0]["seeds"] == tuple(
        case.seed for case in canonical_cases
    )
    # 5 x N concept images, keyed by the single shared state definition.
    assert set(images) == set(CONCEPT_IMAGE_STATES)
    for underscore_key in (
        "condition_canonical",
        "condition_swap",
        "text_canonical",
        "text_swap",
    ):
        assert underscore_key not in images
    for tensor in images.values():
        assert tensor.shape[0] == len(manifest.concepts)
    assert len(result.condition_metrics) == len(manifest.concepts)
    assert len(result.text_metrics) == len(manifest.concepts)
    assert "overall" in [agg.group for agg in result.condition_aggregates]
    assert "overall" in [agg.group for agg in result.text_aggregates]


def test_image_save_uses_the_same_state_contract_as_the_runner() -> None:
    # The CLI's PNG export iterates the same CONCEPT_IMAGE_STATES constant
    # the runner's image dict is keyed by; a fake single-concept draw must
    # produce exactly the five hyphenated PNG names with no KeyError.
    manifest = _manifest()
    images_dir = _tmp_images_dir()
    images = {
        state: torch.zeros(1, 3, 8, 8, dtype=torch.uint8)
        for state in CONCEPT_IMAGE_STATES
    }
    saved = save_state_images(
        images_dir,
        (manifest.concepts[0].id,),
        images,
    )
    assert saved == len(CONCEPT_IMAGE_STATES)
    expected = {f"A001.{state}.png" for state in CONCEPT_IMAGE_STATES}
    assert expected == {
        "A001.condition-canonical.png",
        "A001.condition-swap.png",
        "A001.text-canonical.png",
        "A001.text-swap.png",
        "A001.null.png",
    }
    assert expected == {path.name for path in images_dir.iterdir()}


def test_image_save_fails_fast_on_a_missing_state() -> None:
    images = {
        state: torch.zeros(1, 3, 8, 8, dtype=torch.uint8)
        for state in CONCEPT_IMAGE_STATES
    }
    del images["text-swap"]
    with pytest.raises(KeyError, match="text-swap"):
        save_state_images(
            _tmp_images_dir(),
            ("A001",),
            images,
        )


def _tmp_images_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="concept-states-"))


def test_score_dual_path_requires_the_same_null_features() -> None:
    manifest = _manifest()
    refs = torch.eye(6)
    canonical = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    other = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
    )
    null = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    with pytest.raises(ConceptSuiteError, match="shared null"):
        score_dual_path(
            manifest=manifest,
            condition={"canonical": canonical, "null": null, "swap": other},
            text={"canonical": canonical, "null": other, "swap": other},
            clip_refs=refs,
        )


def test_score_dual_path_math_matches_compute_concept_metrics() -> None:
    manifest = _manifest()
    generator = torch.Generator().manual_seed(7)
    refs = torch.rand(6, 8, generator=generator)
    refs = refs / (refs * refs).sum(dim=1, keepdim=True).sqrt()
    canonical = torch.rand(2, 8, generator=generator)
    null = torch.rand(2, 8, generator=generator)
    swap = torch.rand(2, 8, generator=generator)
    text_canonical = torch.rand(2, 8, generator=generator)
    text_swap = torch.rand(2, 8, generator=generator)
    for features in (
        canonical,
        null,
        swap,
        text_canonical,
        text_swap,
    ):
        features.data = torch.nn.functional.normalize(features, dim=1)

    result = score_dual_path(
        manifest=manifest,
        condition={
            "canonical": canonical,
            "null": null,
            "swap": swap,
        },
        text={
            "canonical": text_canonical,
            "null": null,
            "swap": text_swap,
        },
        clip_refs=refs,
    )
    expected_condition = compute_concept_metrics(
        manifest=manifest,
        clip_canonical=canonical,
        clip_null=null,
        clip_swap=swap,
        clip_refs=refs,
    )
    expected_text = compute_concept_metrics(
        manifest=manifest,
        clip_canonical=text_canonical,
        clip_null=null,
        clip_swap=text_swap,
        clip_refs=refs,
    )
    # Same metric math, only the canonical/swap inputs differ.
    assert result.condition_metrics == expected_condition
    assert result.text_metrics == expected_text
    assert result.condition_aggregates == aggregate_metrics(expected_condition)
    assert result.text_aggregates == aggregate_metrics(expected_text)


def test_flattened_metrics_are_namespaced_per_pathway() -> None:
    manifest = _manifest()
    generator = torch.Generator().manual_seed(11)
    refs = torch.rand(6, 8, generator=generator)
    refs = torch.nn.functional.normalize(refs, dim=1)
    shared_null = torch.nn.functional.normalize(
        torch.rand(2, 8, generator=generator), dim=1
    )
    result = score_dual_path(
        manifest=manifest,
        condition={
            "canonical": torch.nn.functional.normalize(
                torch.rand(2, 8, generator=generator), dim=1
            ),
            "null": shared_null,
            "swap": torch.nn.functional.normalize(
                torch.rand(2, 8, generator=generator), dim=1
            ),
        },
        text={
            "canonical": torch.nn.functional.normalize(
                torch.rand(2, 8, generator=generator), dim=1
            ),
            "null": shared_null,
            "swap": torch.nn.functional.normalize(
                torch.rand(2, 8, generator=generator), dim=1
            ),
        },
        clip_refs=refs,
    )
    flat = _flatten_dual_path(result)
    assert flat
    for key in flat:
        assert key.startswith(("condition/", "text/"))
    assert "condition/overall/mean_margin_null" in flat
    assert "condition/overall/mean_margin_swap" in flat
    assert "text/overall/mean_margin_null" in flat
    assert "text/overall/mean_margin_swap" in flat
    # Both namespaces carry the identical group/field grid.
    condition_suffixes = {
        key[len("condition/") :] for key in flat if key.startswith("condition/")
    }
    text_suffixes = {
        key[len("text/") :] for key in flat if key.startswith("text/")
    }
    assert condition_suffixes == text_suffixes
    # Every key is namespaced as <pathway>/<group>/<field>.
    assert all(key.count("/") >= 2 for key in flat)


def test_prompt_cases_are_the_only_generation_input() -> None:
    # The shared runner must generate from PromptCase objects only; the
    # conditional passes carry distinct prompt ids per pathway/state.
    manifest = _manifest()
    expected_ids: set[str] = set()
    for builder in (
        canonical_prompt_cases,
        swap_prompt_cases,
        text_canonical_prompt_cases,
        text_swap_prompt_cases,
    ):
        for case in builder(manifest, height=64, width=64):
            assert type(case) is PromptCase
            assert case.prompt_id not in expected_ids
            expected_ids.add(case.prompt_id)
    assert len(expected_ids) == 4 * len(manifest.concepts)
