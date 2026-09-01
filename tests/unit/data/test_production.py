from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import cast

import pytest

import sakuramoon.data.production as production_module
from sakuramoon.data.collate import TrainingBatch
from sakuramoon.data.metadata import MetadataFieldMapping, parse_shard_metadata
from sakuramoon.data.production import (
    ConfiguredDataLoader,
    ProductionBatchStreamIdentity,
    ProductionDataError,
    adapt_modelscope_metadata,
    parse_modelscope_caption_fields,
)
from sakuramoon.data.transparent_white import TRANSPARENT_REJECTION_KEYS


class _DepthIterator(Iterator[TrainingBatch]):
    def __init__(self, depth: int) -> None:
        self.depth = depth

    def __next__(self) -> TrainingBatch:
        raise StopIteration

    def ready_batch_depth_snapshot(self) -> int:
        return self.depth


def _stream_identity() -> ProductionBatchStreamIdentity:
    return ProductionBatchStreamIdentity(
        loader=ConfiguredDataLoader(1, 1, 1, False, True),
        dataset_id="leafmoone/webdataset_danbooru_v2@master",
        session_id="test-session",
    )


def _real_row() -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": 71,
        "source": {
            "dataset": "danbooru",
            "dataset_version": "5.9",
            "release": "2026.7",
            "original_path": "danbooru/posts/71.jpg",
        },
        "image": {"format": "webp", "width": 832, "height": 1216},
        "rating": "safe",
        "year": "year 2026, newest",
        "aesthetic": " ",
        "quality": "best",
        "anime_completeness": "polished",
        "anime_classification": "illustration",
        "captions": {"nl2": "A blue-haired character.", "nl3": ""},
        "multicaptions": {
            "long_names": "Alice in a blue dress.",
            "long_no_names": None,
            "short": "A blue-haired character.",
            "vibes": "soft light",
        },
        "tags": {
            "character": ["alice"],
            "copyright": ["original"],
            "general": ["blue_hair", "dress"],
            "artist": ["artist_name"],
        },
        "dropout": {
            "candidate_tags": ["blue_hair"],
            "candidate_source": "popular_tags_intersection",
            "policy_version": "dropout_v1",
        },
        "join": {"multicaptions": "matched", "character_records": "matched"},
        "nsfw": "sfw",
    }


def test_governed_modelscope_adapter_and_caption_parser() -> None:
    raw = _real_row()

    adapted = adapt_modelscope_metadata(raw)
    fields = parse_modelscope_caption_fields(raw)

    assert adapted == {
        "id": 71,
    }
    assert tuple(tag.text for tag in fields.nsfw) == ("sfw",)
    assert tuple(tag.text for tag in fields.character) == ("alice",)
    assert tuple(tag.text for tag in fields.general) == ("blue_hair", "dress")
    assert tuple(tag.text for tag in fields.artists) == ("artist_name",)
    assert fields.candidate_tags == frozenset({"blue_hair"})
    assert fields.nl.long_names == "Alice in a blue dress."
    assert fields.nl.long_no_names is None
    assert fields.nl.short_vibes == "A blue-haired character.\n\nsoft light"
    assert fields.nl.nl2 == "A blue-haired character."
    assert fields.nl.nl3 is None
    assert tuple(tag.text for tag in fields.rating) == ("safe",)
    assert tuple(tag.text for tag in fields.year) == ("year 2026", "newest")
    assert fields.aesthetic == ()
    assert tuple(tag.text for tag in fields.quality) == ("best",)
    assert tuple(tag.text for tag in fields.anime_completeness) == ("polished",)
    assert tuple(tag.text for tag in fields.anime_classification) == (
        "illustration",
    )


@pytest.mark.parametrize("image_format", ["jpg", "jpeg", "png", "webp"])
@pytest.mark.parametrize(
    ("width", "height"),
    [(832, 1216), (None, None)],
)
def test_modelscope_parser_accepts_repository_image_contract(
    image_format: str,
    width: int | None,
    height: int | None,
) -> None:
    raw = _real_row()
    raw["image"] = {"format": image_format, "width": width, "height": height}

    assert adapt_modelscope_metadata(raw) == {"id": 71}
    assert parse_modelscope_caption_fields(raw).quality


def test_modelscope_parser_accepts_null_nsfw_when_character_records_are_missing() -> None:
    raw = _real_row()
    raw["nsfw"] = None
    join = raw["join"]
    assert isinstance(join, dict)
    join["character_records"] = "missing"

    assert parse_modelscope_caption_fields(raw).nsfw == ()


@pytest.mark.parametrize("value", [None, "safe", 2])
def test_governed_modelscope_parser_rejects_invalid_nsfw(value: object) -> None:
    raw = _real_row()
    raw["nsfw"] = value

    with pytest.raises(ProductionDataError, match="nsfw"):
        parse_modelscope_caption_fields(raw)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("year 2021", ("year 2021",)),
        ("year 2024, newest", ("year 2024", "newest")),
        ("year 2017, oldest", ("year 2017", "oldest")),
    ],
)
def test_governed_modelscope_parser_splits_year_tags(
    value: str, expected: tuple[str, ...]
) -> None:
    raw = _real_row()
    raw["year"] = value

    fields = parse_modelscope_caption_fields(raw)

    assert tuple(tag.text for tag in fields.year) == expected


@pytest.mark.parametrize(
    "value",
    [None, "2026", "year 26", "year 2026,newest", "year 2026, middle"],
)
def test_governed_modelscope_parser_rejects_invalid_year(value: object) -> None:
    raw = _real_row()
    raw["year"] = value

    with pytest.raises(ProductionDataError, match="year"):
        parse_modelscope_caption_fields(raw)


def test_governed_modelscope_parser_accepts_blank_nsfw_and_year() -> None:
    raw = _real_row()
    raw["nsfw"] = ""
    raw["year"] = ""

    fields = parse_modelscope_caption_fields(raw)

    assert fields.nsfw == ()
    assert fields.year == ()


def _a2_row() -> dict[str, object]:
    # Mirrors the published artstation-2D (a2) annotation rows: schema v1
    # source with blank release/original_path and a blank string id.
    return {
        "schema_version": 1,
        "id": "",
        "source": {
            "dataset": "artstation-2D",
            "dataset_version": "1",
            "release": "",
            "original_path": "",
        },
        "image": {"format": "jpeg", "width": 1916, "height": 1916},
        "nsfw": "",
        "tags": {
            "general": ["artstation", "1girl"],
            "artist": [],
            "character": [],
            "copyright": [],
        },
        "captions": {"nl2": "", "nl3": ""},
        "multicaptions": {},
        "dropout": {},
        "join": {},
        "rating": "",
        "year": "",
        "aesthetic": "",
        "quality": "low",
        "anime_completeness": "monochrome",
        "anime_classification": "illustration",
    }


def test_governed_modelscope_parser_accepts_2d_annotation_row() -> None:
    raw = _a2_row()

    record = adapt_modelscope_metadata(raw)
    fields = parse_modelscope_caption_fields(raw)

    assert record == {"id": ""}
    assert fields.nsfw == ()
    assert fields.year == ()


def test_shard_metadata_derives_id_for_blank_2d_id() -> None:
    raw = _a2_row()
    fields = MetadataFieldMapping(id_field="id")

    record = parse_shard_metadata(raw, fields=fields, sample_key="a2_filter_4429173")
    assert record.id == 4429173

    fallback = parse_shard_metadata(raw, fields=fields, sample_key="a2_filter_blank")
    assert 0 < fallback.id < 2**53
    again = parse_shard_metadata(raw, fields=fields, sample_key="a2_filter_blank")
    assert again.id == fallback.id


def test_governed_modelscope_parser_rejects_schema_drift() -> None:
    bad_tags = _real_row()
    tags = bad_tags["tags"]
    assert isinstance(tags, dict)
    tags["artist"] = "not-a-list"
    with pytest.raises(ProductionDataError, match=r"tags\.artist must be a list"):
        parse_modelscope_caption_fields(bad_tags)

    bad_candidate = _real_row()
    dropout = bad_candidate["dropout"]
    assert isinstance(dropout, dict)
    dropout["candidate_tags"] = ["valid", 2]
    with pytest.raises(ProductionDataError, match="only strings"):
        parse_modelscope_caption_fields(bad_candidate)

    bad_nl = _real_row()
    multicaptions = bad_nl["multicaptions"]
    assert isinstance(multicaptions, dict)
    multicaptions["short"] = ["not", "text"]
    with pytest.raises(
        ProductionDataError, match=r"multicaptions\.short must be text or null"
    ):
        parse_modelscope_caption_fields(bad_nl)


@pytest.mark.parametrize(
    ("short", "vibes", "expected"),
    [
        (None, None, None),
        ("", "  ", None),
        ("short description", None, "short description"),
        (None, "quiet mood", "quiet mood"),
        (
            " short description ",
            " quiet mood ",
            "short description\n\nquiet mood",
        ),
    ],
)
def test_governed_modelscope_parser_combines_short_and_vibes(
    short: object, vibes: object, expected: str | None
) -> None:
    raw = _real_row()
    multicaptions = raw["multicaptions"]
    assert isinstance(multicaptions, dict)
    multicaptions["short"] = short
    multicaptions["vibes"] = vibes

    fields = parse_modelscope_caption_fields(raw)

    assert fields.nl.short_vibes == expected


def test_governed_modelscope_parser_rejects_corrupted_sample() -> None:
    raw = _real_row()
    raw["ai_image_corrupted"] = "corrupted"

    with pytest.raises(
        production_module.PipelineSampleRejected,
        match="ai_image_corrupted",
    ):
        parse_modelscope_caption_fields(raw)


def test_governed_modelscope_parser_rejects_invalid_corruption_value() -> None:
    raw = _real_row()
    raw["ai_image_corrupted"] = "normal"

    with pytest.raises(ProductionDataError, match="absent or corrupted"):
        parse_modelscope_caption_fields(raw)


def test_accepted_stream_exposes_only_live_iterator_ready_batch_depth() -> None:
    iterator = _DepthIterator(2)
    stream = production_module._issue_batch_stream(  # pyright: ignore[reportPrivateUsage]
        iterator, _stream_identity()
    )

    assert stream.ready_batch_depth_snapshot() == 2
    iterator.depth = 4
    assert stream.ready_batch_depth_snapshot() == 4
    stream.close()
    with pytest.raises(ProductionDataError, match="closed"):
        stream.ready_batch_depth_snapshot()


def test_accepted_stream_fails_when_iterator_has_no_ready_depth_source() -> None:
    stream = production_module._issue_batch_stream(  # pyright: ignore[reportPrivateUsage]
        cast(Iterator[TrainingBatch], iter(())),
        _stream_identity(),
    )

    with pytest.raises(ProductionDataError, match="unavailable"):
        stream.ready_batch_depth_snapshot()
    stream.close()


class _TotalsIterator(Iterator[TrainingBatch]):
    def __init__(self, totals: Mapping[str, int] | None) -> None:
        self._totals = totals

    def __next__(self) -> TrainingBatch:
        raise StopIteration

    def transparent_rejection_totals_snapshot(self) -> Mapping[str, int] | None:
        return self._totals


def _zero_totals() -> dict[str, int]:
    return {key: 0 for key in TRANSPARENT_REJECTION_KEYS}


def test_accepted_stream_publishes_strict_zero_totals_without_ledger() -> None:
    stream = production_module._issue_batch_stream(  # pyright: ignore[reportPrivateUsage]
        cast(Iterator[TrainingBatch], iter(())),
        _stream_identity(),
    )

    assert stream.transparent_rejection_totals() == _zero_totals()
    assert set(stream.transparent_rejection_totals()) == set(
        TRANSPARENT_REJECTION_KEYS
    )
    stream.close()


def test_accepted_stream_passes_explicit_totals_snapshot() -> None:
    stream = production_module._issue_batch_stream(  # pyright: ignore[reportPrivateUsage]
        cast(Iterator[TrainingBatch], iter(())),
        _stream_identity(),
        transparent_rejection_totals_snapshot=lambda: {
            "reject_missing_alpha": 1,
            "reject_special_alpha": 2,
            "reject_conflict_bg": 3,
        },
    )

    assert stream.transparent_rejection_totals() == {
        "reject_missing_alpha": 1,
        "reject_special_alpha": 2,
        "reject_conflict_bg": 3,
    }
    stream.close()


def test_accepted_stream_falls_back_to_duck_typed_totals_snapshot() -> None:
    stream = production_module._issue_batch_stream(  # pyright: ignore[reportPrivateUsage]
        _TotalsIterator(
            {
                "reject_missing_alpha": 5,
                "reject_special_alpha": 0,
                "reject_conflict_bg": 0,
            }
        ),
        _stream_identity(),
    )

    assert stream.transparent_rejection_totals() == {
        "reject_missing_alpha": 5,
        "reject_special_alpha": 0,
        "reject_conflict_bg": 0,
    }
    stream.close()


def test_accepted_stream_rejects_totals_snapshot_with_missing_key() -> None:
    stream = production_module._issue_batch_stream(  # pyright: ignore[reportPrivateUsage]
        _TotalsIterator({"reject_missing_alpha": 1}),
        _stream_identity(),
    )

    with pytest.raises(
        ProductionDataError, match="must carry exactly"
    ):
        stream.transparent_rejection_totals()
    stream.close()


def test_accepted_stream_rejects_negative_total() -> None:
    stream = production_module._issue_batch_stream(  # pyright: ignore[reportPrivateUsage]
        _TotalsIterator(
            {
                "reject_missing_alpha": -1,
                "reject_special_alpha": 0,
                "reject_conflict_bg": 0,
            }
        ),
        _stream_identity(),
    )

    with pytest.raises(
        ProductionDataError, match="nonnegative"
    ):
        stream.transparent_rejection_totals()
    stream.close()


def test_accepted_stream_rejects_non_mapping_totals_snapshot() -> None:
    stream = production_module._issue_batch_stream(  # pyright: ignore[reportPrivateUsage]
        _TotalsIterator(None),
        _stream_identity(),
    )

    with pytest.raises(
        ProductionDataError, match="snapshot is invalid"
    ):
        stream.transparent_rejection_totals()
    stream.close()


def test_accepted_stream_totals_require_live_stream() -> None:
    stream = production_module._issue_batch_stream(  # pyright: ignore[reportPrivateUsage]
        cast(Iterator[TrainingBatch], iter(())),
        _stream_identity(),
    )
    stream.close()

    with pytest.raises(ProductionDataError, match="closed"):
        stream.transparent_rejection_totals()
