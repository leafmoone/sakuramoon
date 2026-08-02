from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import pytest
from PIL import Image

import sakuramoon.data.validation as validation_module
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
)
from sakuramoon.data.modelscope import ModelScopeDatasetTransport
from sakuramoon.data.production import parse_modelscope_caption_fields
from sakuramoon.data.validation import (
    VALIDATION_SELECTION_SEED,
    VALIDATION_SHARD_COUNT,
    VALIDATION_SHARD_PATHS,
    ValidationPromptError,
    ValidationSelection,
    ValidationSelectionError,
    ValidationSelectionExistsError,
    canonical_validation_selection_bytes,
    ensure_validation_selection,
    load_validation_prompt_samples,
    load_validation_selection,
    parse_validation_selection,
    prepare_validation_shards,
    select_validation_shards,
    write_validation_selection,
)


class _Writer(Protocol):
    def write(self, payload: bytes, /) -> int: ...


class _MemoryTransport:
    stream_chunk_bytes = 64

    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies
        self.downloaded: list[str] = []

    def download(
        self, manifest: DatasetManifest, shard: ShardRecord, output: _Writer
    ) -> None:
        assert manifest.shard(shard.path) == shard
        self.downloaded.append(shard.path)
        output.write(self.bodies[shard.path])


def _source() -> DatasetSourceIdentity:
    return DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="master",
    )


def _manifest(bodies: dict[str, bytes]) -> DatasetManifest:
    return DatasetManifest.from_shards(
        _source(),
        tuple(
            ShardRecord(
                path=path,
                bytes=len(body),
                upstream_sha256=hashlib.sha256(body).hexdigest(),
            )
            for path, body in bodies.items()
        ),
    )


def _image_bytes(*, width: int = 47, height: int = 33) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), (20, 40, 60)).save(output, format="PNG")
    return output.getvalue()


def _metadata(
    sample_id: int,
    *,
    captions: dict[str, object] | None = None,
    multicaptions: dict[str, object] | None = None,
    width: int = 47,
    height: int = 33,
) -> bytes:
    return json.dumps(
        {
            "captions": captions if captions is not None else {"nl2": "prompt"},
            "dropout": {"candidate_tags": []},
            "id": sample_id,
            "image": {"height": height, "width": width},
            "multicaptions": multicaptions if multicaptions is not None else {},
            "nsfw": "safe",
            "tags": {
                "artist": ["artist_name"],
                "character": [],
                "copyright": [],
                "general": ["1girl", "blue_hair"],
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _tar_bytes(
    members: list[tuple[str, bytes, str]],
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, payload, kind in members:
            member = tarfile.TarInfo(name)
            member.mtime = 0
            if kind == "file":
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            elif kind == "directory":
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            else:
                raise AssertionError(f"unknown test member kind: {kind}")
    return output.getvalue()


def _valid_tar(
    sample_id: int,
    *,
    captions: dict[str, object] | None = None,
    multicaptions: dict[str, object] | None = None,
    width: int = 47,
    height: int = 33,
) -> bytes:
    return _tar_bytes(
        [
            (
                f"{sample_id}.json",
                _metadata(
                    sample_id,
                    captions=captions,
                    multicaptions=multicaptions,
                    width=width,
                    height=height,
                ),
                "file",
            ),
            (
                f"{sample_id}.png",
                _image_bytes(width=width, height=height),
                "file",
            ),
        ]
    )


def _bodies(
    *,
    body_factory: Callable[[int], bytes] | None = None,
) -> dict[str, bytes]:
    factory = body_factory if body_factory is not None else _valid_tar
    paths = (*VALIDATION_SHARD_PATHS, "release/train-00.tar", "release/train-01.tar")
    return {path: factory(index + 1) for index, path in enumerate(paths)}


def _materialize_selected(
    root: Path,
    manifest: DatasetManifest,
    bodies: dict[str, bytes],
) -> ValidationSelection:
    selection = select_validation_shards(manifest)
    for record in selection.shards:
        path = root / record.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bodies[record.path])
    return selection


def _traversal_tar(sample_id: int) -> bytes:
    return _tar_bytes(
        [
            ("../escape.png", _image_bytes(), "file"),
            ("escape.json", _metadata(sample_id), "file"),
        ]
    )


def _directory_tar(_sample_id: int) -> bytes:
    return _tar_bytes([("folder", b"", "directory")])


def _missing_image_tar(sample_id: int) -> bytes:
    return _tar_bytes([(f"{sample_id}.json", _metadata(sample_id), "file")])


def _missing_json_tar(sample_id: int) -> bytes:
    return _tar_bytes([(f"{sample_id}.png", _image_bytes(), "file")])


def _duplicate_member_tar(sample_id: int) -> bytes:
    return _tar_bytes(
        [
            (f"{sample_id}.png", _image_bytes(), "file"),
            (f"{sample_id}.png", _image_bytes(), "file"),
            (f"{sample_id}.json", _metadata(sample_id), "file"),
        ]
    )


def _duplicate_suffix_tar(sample_id: int) -> bytes:
    return _tar_bytes(
        [
            (f"{sample_id}.JPG", _image_bytes(), "file"),
            (f"{sample_id}.jpg", _image_bytes(), "file"),
            (f"{sample_id}.json", _metadata(sample_id), "file"),
        ]
    )


def _invalid_image_tar(sample_id: int) -> bytes:
    return _tar_bytes(
        [
            (f"{sample_id}.png", b"not-an-image", "file"),
            (f"{sample_id}.json", _metadata(sample_id), "file"),
        ]
    )


def _invalid_metadata_tar(sample_id: int) -> bytes:
    return _tar_bytes(
        [
            (f"{sample_id}.png", _image_bytes(), "file"),
            (f"{sample_id}.json", b"not-json", "file"),
        ]
    )


def _dimension_mismatch_tar(sample_id: int) -> bytes:
    return _tar_bytes(
        [
            (f"{sample_id}.png", _image_bytes(width=48, height=33), "file"),
            (f"{sample_id}.json", _metadata(sample_id), "file"),
        ]
    )


def _sidecar_without_image_tar(sample_id: int) -> bytes:
    return _tar_bytes(
        [
            (f"{sample_id}.txt", b"sidecar", "file"),
            (f"{sample_id}.json", _metadata(sample_id), "file"),
        ]
    )


def _duplicate_id_tar(_sample_id: int) -> bytes:
    return _valid_tar(1)


_INVALID_ARCHIVES: tuple[tuple[Callable[[int], bytes], str], ...] = (
    (_traversal_tar, "path is invalid"),
    (_directory_tar, "regular files"),
    (_missing_image_tar, "missing exactly one image or JSON"),
    (_missing_json_tar, "missing exactly one image or JSON"),
    (_duplicate_member_tar, "duplicate members"),
    (_duplicate_suffix_tar, "duplicate member suffixes"),
    (_invalid_image_tar, "image is invalid"),
    (_invalid_metadata_tar, "metadata JSON is invalid"),
    (_dimension_mismatch_tar, "dimensions differ from decoded image"),
    (_sidecar_without_image_tar, "missing exactly one image or JSON"),
)


def test_selects_exactly_two_distinct_shards_stably_with_seed_44() -> None:
    bodies = _bodies()
    manifest = _manifest(bodies)
    reordered = DatasetManifest.from_shards(_source(), tuple(reversed(manifest.shards)))

    first = select_validation_shards(manifest)
    second = select_validation_shards(reordered)

    assert first == second
    assert first.seed == VALIDATION_SELECTION_SEED == 44
    assert len(first.shards) == VALIDATION_SHARD_COUNT == 2
    assert first.shard_paths == VALIDATION_SHARD_PATHS
    assert set(first.shard_paths) < {item.path for item in manifest.shards}
    with pytest.raises(ValidationSelectionError, match="seed must equal"):
        select_validation_shards(manifest, seed=45)

    missing = {path: body for path, body in bodies.items() if path != first.shard_paths[0]}
    with pytest.raises(ValidationSelectionError, match="fixed validation shard is absent"):
        select_validation_shards(_manifest(missing))


def test_selection_identity_and_encoding_are_canonical() -> None:
    manifest = _manifest(_bodies())
    selection = select_validation_shards(manifest)
    payload = canonical_validation_selection_bytes(selection)

    assert parse_validation_selection(payload) == selection
    assert payload.endswith(b"\n")
    document = json.loads(payload)
    assert set(document) == {
        "manifest_id",
        "schema_version",
        "seed",
        "selection_id",
        "shards",
    }
    assert hashlib.sha256(payload).hexdigest() != selection.selection_id
    with pytest.raises(ValidationSelectionError, match="not canonical"):
        parse_validation_selection(payload.rstrip())


def test_selection_publication_is_atomic_no_clobber_and_reloads(
    tmp_path: Path,
) -> None:
    manifest = _manifest(_bodies())
    selection = select_validation_shards(manifest)
    destination = tmp_path / "state" / "validation-selection.json"

    write_validation_selection(selection, destination)

    assert load_validation_selection(destination, manifest) == selection
    assert ensure_validation_selection(manifest, destination) == selection
    with pytest.raises(ValidationSelectionExistsError, match="already exists"):
        write_validation_selection(selection, destination)
    assert destination.read_bytes() == canonical_validation_selection_bytes(selection)
    assert not tuple(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_selection_publication_rolls_back_when_directory_fsync_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selection = select_validation_shards(_manifest(_bodies()))
    destination = tmp_path / "validation-selection.json"

    def fail_fsync(_path: Path) -> None:
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(validation_module, "_fsync_directory", fail_fsync)
    with pytest.raises(ValidationSelectionError, match="could not be published"):
        write_validation_selection(selection, destination)

    assert not destination.exists()
    assert not tuple(tmp_path.glob(f".{destination.name}.*.tmp"))


def test_existing_selection_rejects_manifest_drift(tmp_path: Path) -> None:
    bodies = _bodies()
    manifest = _manifest(bodies)
    destination = tmp_path / "validation-selection.json"
    write_validation_selection(select_validation_shards(manifest), destination)
    changed = dict(bodies)
    changed[next(iter(changed))] = _valid_tar(99)

    with pytest.raises(ValidationSelectionError, match="manifest_id differs"):
        load_validation_selection(destination, _manifest(changed))


def test_prepare_downloads_and_verifies_both_complete_selected_tars(
    tmp_path: Path,
) -> None:
    bodies = _bodies()
    manifest = _manifest(bodies)
    selection = select_validation_shards(manifest)
    transport = _MemoryTransport(bodies)
    root = (tmp_path / "validation-shards").absolute()

    first = prepare_validation_shards(
        cast(ModelScopeDatasetTransport, transport), manifest, selection, root
    )
    second = prepare_validation_shards(
        cast(ModelScopeDatasetTransport, transport), manifest, selection, root
    )

    assert first.selection == second.selection == selection
    assert first.paths == tuple(root / path for path in selection.shard_paths)
    assert tuple(path.read_bytes() for path in first.paths) == tuple(
        bodies[path] for path in selection.shard_paths
    )
    assert transport.downloaded == list(selection.shard_paths)


@pytest.mark.parametrize(
    "body_factory,error",
    _INVALID_ARCHIVES,
)
def test_strict_loader_rejects_invalid_tar_members_and_pairs(
    tmp_path: Path,
    body_factory: Callable[[int], bytes],
    error: str,
) -> None:
    bodies = _bodies(body_factory=body_factory)
    manifest = _manifest(bodies)
    selection = _materialize_selected(tmp_path.absolute(), manifest, bodies)

    with pytest.raises(ValidationPromptError, match=error):
        load_validation_prompt_samples(
            selection,
            tmp_path.absolute(),
            run_seed=44,
        )


def test_loader_uses_caption_priority_stable_seed_and_metadata_dimensions(
    tmp_path: Path,
) -> None:
    def prompts(sample_id: int) -> bytes:
        return _valid_tar(
            sample_id,
            captions={"z-last": "later", "a-first": "  chosen caption  "},
            multicaptions={"a": "fallback caption"},
            width=47,
            height=33,
        )

    bodies = _bodies(body_factory=prompts)
    manifest = _manifest(bodies)
    root = tmp_path.absolute()
    selection = _materialize_selected(root, manifest, bodies)
    before = {path: (root / path).read_bytes() for path in selection.shard_paths}

    first = load_validation_prompt_samples(selection, root, run_seed=44)
    second = load_validation_prompt_samples(selection, root, run_seed=44)

    assert first == second
    assert len(first) == 2
    assert {item.prompt for item in first} == {"chosen caption"}
    assert {(item.height, item.width) for item in first} == {(33, 47)}
    assert len({item.seed for item in first}) == 2
    assert len({item.prompt_id for item in first}) == 2
    assert {path: (root / path).read_bytes() for path in selection.shard_paths} == before
    with pytest.raises(ValidationPromptError, match="run seed must equal"):
        load_validation_prompt_samples(selection, root, run_seed=45)


def test_loader_falls_back_to_lexicographic_multicaptions(tmp_path: Path) -> None:
    def prompts(sample_id: int) -> bytes:
        return _valid_tar(
            sample_id,
            captions={"a": "", "b": None},
            multicaptions={"z": "later", "a": "fallback"},
        )

    bodies = _bodies(body_factory=prompts)
    manifest = _manifest(bodies)
    root = tmp_path.absolute()
    selection = _materialize_selected(root, manifest, bodies)

    samples = load_validation_prompt_samples(selection, root, run_seed=44)

    assert {item.prompt for item in samples} == {"fallback"}


def test_loader_exposes_typed_tags_when_nl_is_empty(tmp_path: Path) -> None:
    def tags_only(sample_id: int) -> bytes:
        return _valid_tar(
            sample_id,
            captions={"nl2": "", "nl3": None},
            multicaptions={"vibes": ""},
        )

    bodies = _bodies(body_factory=tags_only)
    manifest = _manifest(bodies)
    root = tmp_path.absolute()
    selection = _materialize_selected(root, manifest, bodies)

    samples = load_validation_prompt_samples(
        selection,
        root,
        run_seed=44,
        caption_fields_parser=parse_modelscope_caption_fields,
    )

    assert len(samples) == 2
    assert all(sample.prompt is None for sample in samples)
    assert all(sample.caption_fields is not None for sample in samples)

    def general_tags(sample: validation_module.ValidationPromptSample) -> tuple[str, ...]:
        assert sample.caption_fields is not None
        return tuple(tag.text for tag in sample.caption_fields.general)

    assert {general_tags(sample) for sample in samples} == {("1girl", "blue_hair")}


def test_loader_exposes_typed_existing_nl_caption(tmp_path: Path) -> None:
    bodies = _bodies()
    manifest = _manifest(bodies)
    root = tmp_path.absolute()
    selection = _materialize_selected(root, manifest, bodies)

    samples = load_validation_prompt_samples(
        selection,
        root,
        run_seed=44,
        caption_fields_parser=parse_modelscope_caption_fields,
    )

    assert {sample.prompt for sample in samples} == {"prompt"}
    assert all(sample.caption_fields is not None for sample in samples)
    assert {
        sample.caption_fields.nl.nl2
        for sample in samples
        if sample.caption_fields is not None
    } == {"prompt"}


def test_loader_rejects_global_duplicate_metadata_ids(tmp_path: Path) -> None:
    bodies = _bodies(body_factory=_duplicate_id_tar)
    manifest = _manifest(bodies)
    root = tmp_path.absolute()
    selection = _materialize_selected(root, manifest, bodies)

    with pytest.raises(ValidationPromptError, match="globally unique"):
        load_validation_prompt_samples(selection, root, run_seed=44)
