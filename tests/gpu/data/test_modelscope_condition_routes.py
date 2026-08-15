from __future__ import annotations

import json
import os
import tarfile
from collections import Counter
from pathlib import Path
from typing import cast

import pytest

from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    CaptionPlan,
    ConditionSource,
    NlDropoutProbabilities,
    build_caption_plan,
    count_condition_routes,
)
from sakuramoon.data.pipeline import PipelineSampleRejected, rng_identity
from sakuramoon.data.production import parse_modelscope_caption_fields

_REAL_SAMPLE_COUNT = 117
_CYCLE_COUNT = 8


def _real_rows(
    path: Path,
) -> tuple[tuple[tuple[dict[str, object], CaptionFields], ...], int]:
    rows: list[tuple[dict[str, object], CaptionFields]] = []
    corrupted = 0
    with tarfile.open(path, "r") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"JSON member has no payload: {member.name}")
            value = cast(object, json.load(stream))
            if not isinstance(value, dict) or not all(
                isinstance(key, str) for key in value
            ):
                raise TypeError(f"JSON member is not a string-keyed object: {member.name}")
            row = cast(dict[str, object], value)
            try:
                fields = parse_modelscope_caption_fields(row)
            except PipelineSampleRejected as rejected:
                if rejected.reason != "ai_image_corrupted":
                    raise
                corrupted += 1
                continue
            rows.append((row, fields))
            if len(rows) == _REAL_SAMPLE_COUNT:
                break
    if len(rows) != _REAL_SAMPLE_COUNT:
        raise RuntimeError(
            f"real shard contains {len(rows)} JSON rows before EOF, "
            f"expected {_REAL_SAMPLE_COUNT}"
        )
    return tuple(rows), corrupted


def _routed_tags(plan: CaptionPlan) -> Counter[tuple[str, str]]:
    routed: Counter[tuple[str, str]] = Counter(
        (item.source, item.tag.canonical) for item in plan.tags
    )
    if plan.condition is not None:
        tag_source = (
            "artist" if plan.condition.source == "artist_text" else "character"
        )
        routed.update(
            (tag_source, tag.canonical)
            for tag in plan.condition.tags
        )
    return routed


def test_real_modelscope_artist_or_character_routes_are_lossless() -> None:
    shard_value = os.environ.get("SAKURAMOON_REAL_SHARD_PATH")
    if shard_value is None:
        pytest.skip("SAKURAMOON_REAL_SHARD_PATH is not set")
    shard = Path(shard_value)
    if not shard.is_file():
        raise FileNotFoundError(shard)

    probabilities = CaptionDropoutProbabilities(
        tag=0.1,
        candidate_source=0.3,
        nl=NlDropoutProbabilities(0.3, 0.3, 0.3, 0.3, 0.3),
    )
    routes: list[ConditionSource | None] = []
    both_available = 0
    selected_when_both: set[str] = set()

    real_rows, corrupted = _real_rows(shard)
    for row, fields in real_rows:
        sample_id = row.get("id")
        if type(sample_id) is not int or sample_id <= 0:
            raise TypeError("real ModelScope sample ID must be a positive integer")
        for cycle_index in range(_CYCLE_COUNT):
            seed = rng_identity(
                base_seed=20260815,
                stage="S0",
                cycle_index=cycle_index,
                sample_id=sample_id,
            ).caption_seed
            artist_plan = build_caption_plan(
                fields,
                probabilities,
                condition_mode="artist",
                seed=seed,
            )
            mixed_plan = build_caption_plan(
                fields,
                probabilities,
                condition_mode="artist_or_character",
                seed=seed,
            )
            repeated = build_caption_plan(
                fields,
                probabilities,
                condition_mode="artist_or_character",
                seed=seed,
            )

            assert mixed_plan == repeated
            assert mixed_plan.all_condition_dropped == artist_plan.all_condition_dropped
            assert mixed_plan.dropout_hits == artist_plan.dropout_hits
            assert mixed_plan.nl_text == artist_plan.nl_text
            assert mixed_plan.selected_nl == artist_plan.selected_nl
            assert _routed_tags(mixed_plan) == _routed_tags(artist_plan)

            condition = mixed_plan.condition
            source = None if condition is None else condition.source
            routes.append(source)
            if condition is not None:
                tag_source = (
                    "artist"
                    if condition.source == "artist_text"
                    else "character"
                )
                assert not any(item.source == tag_source for item in mixed_plan.tags)

            artist_available = artist_plan.condition is not None
            character_available = any(
                item.source == "character" for item in artist_plan.tags
            )
            if artist_available and character_available:
                both_available += 1
                assert condition is not None
                selected_when_both.add(condition.source)

    counts = count_condition_routes(tuple(routes))
    assert sum(counts.as_mapping().values()) == _REAL_SAMPLE_COUNT * _CYCLE_COUNT
    assert counts.artist_text > 0
    assert counts.character_text > 0
    assert counts.null > 0
    assert corrupted > 0
    assert both_available > 0
    assert selected_when_both == {"artist_text", "character_text"}
