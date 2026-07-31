from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest
import tomli_w

import sakuramoon.cli.manifest as cli_manifest
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ManifestBuildInventory,
    RemoteShardRecord,
    ShardBuildRecord,
    ShardRecord,
    build_dataset_manifest,
    canonical_build_inventory_bytes,
    parse_dataset_manifest_bytes,
    write_dataset_manifest,
)
from sakuramoon.data.modelscope import (
    DatasetTransportError,
    ModelScopeDatasetTransport,
)

CONTENT = b"synthetic-cli-shard"
REVISION = "c" * 40
EXAMPLE_PATH = Path(__file__).parents[3] / "config/examples/all_options.example.toml"
SYNTHETIC_VALUES: dict[str, object] = {
    "data.source.revision": REVISION,
    "data.manifest.path": "synthetic/train-manifest.jsonl",
    "data.manifest.sha256": "3" * 64,
    "data.cache.low_watermark_gib": 300,
    "data.cache.high_watermark_gib": 400,
    "data.cache.download_concurrency": 2,
    "data.cache.verified_shard_lookahead": 2,
    "data.service.request_timeout_seconds": 30.0,
    "data.service.lease_channel_capacity": 2,
    "data.service.ack_channel_capacity": 2,
    "data.cache.persistent_workers_per_rank": 2,
    "data.cache.ready_batches_per_rank": 2,
    "data.transport.connect_timeout_seconds": 10.0,
    "data.transport.read_timeout_seconds": 30.0,
    "data.transport.max_retries": 2,
    "data.transport.retry_backoff_seconds": 0.0,
    "data.transport.stream_chunk_bytes": 1048576,
    "data.validation.manifest_path": "synthetic/validation-manifest.jsonl",
    "data.validation.manifest_sha256": "4" * 64,
    "checkpoint.slots": 3,
    "stage.local_batch": 1,
    "stage.accumulation": 1,
    "stage.planned_updates": 10,
    "profiling.schedule_updates": 10,
    "benchmark.profile_trace_updates": 5,
    "logging.flush_every_updates": 1,
    "wandb.entity": "synthetic-entity",
    "evaluation.fid.every_successful_updates": 10,
    "evaluation.fid.trend_samples": 100,
    "evaluation.fid.feature_extractor": "synthetic-locked-extractor",
    "evaluation.fid.feature_extractor_version": "synthetic-version",
    "evaluation.fid.preprocess_sha256": "6" * 64,
    "evaluation.fid.real_stats_sha256": "5" * 64,
    "evaluation.is.every_successful_updates": 10,
    "evaluation.is.trend_samples": 100,
    "evaluation.is.splits": 10,
    "evaluation.prompt_manifest_path": "synthetic/prompts.json",
    "evaluation.prompt_manifest_sha256": "7" * 64,
    "evaluation.gpu_index": 0,
    "evaluation.training_paused": True,
}


def _set_path(payload: dict[str, Any], dotted_path: str, value: object) -> None:
    parts = dotted_path.split(".")
    current = payload
    for part in parts[:-1]:
        current = cast(dict[str, Any], current[part])
    current[parts[-1]] = value


def _manifest() -> DatasetManifest:
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision=REVISION,
        license_id="source-license",
        access_terms="source-access-terms",
    )
    shard = ShardRecord(
        path="release-a/000001.tar",
        release="release-a",
        bytes=len(CONTENT),
        sha256=hashlib.sha256(CONTENT).hexdigest(),
        samples=11,
    )
    return DatasetManifest.from_shards(source, (shard,))


def _cli_tree(tmp_path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest_path = workspace / "locks/dataset-manifest.json"
    digest = write_dataset_manifest(_manifest(), manifest_path)

    payload = tomllib.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    for dotted_path, value in SYNTHETIC_VALUES.items():
        _set_path(payload, dotted_path, value)
    _set_path(payload, "data.manifest.path", "locks/dataset-manifest.json")
    _set_path(payload, "data.manifest.sha256", digest)
    config_root = tmp_path / "config"
    config_root.mkdir()
    config_path = config_root / "run.toml"
    config_path.write_text(tomli_w.dumps(payload), encoding="utf-8")
    args = (
        "--config",
        "run.toml",
        "--config-root",
        str(config_root),
        "--root",
        str(workspace),
    )
    return manifest_path, config_path, args


def _build_cli_tree(tmp_path: Path) -> tuple[Path, Path, Path, str, tuple[str, ...]]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inventory = ManifestBuildInventory(
        schema_version=1,
        source=_manifest().source,
        shards=(
            ShardBuildRecord(
                path="release-a/000001.tar",
                release="explicit-release-a",
                samples=11,
            ),
        ),
    )
    inventory_path = workspace / "inputs/build-inventory.json"
    inventory_path.parent.mkdir(parents=True)
    inventory_payload = canonical_build_inventory_bytes(inventory)
    inventory_path.write_bytes(inventory_payload)
    inventory_digest = hashlib.sha256(inventory_payload).hexdigest()
    output_path = workspace / "locks/built-manifest.json"

    payload = tomllib.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    for dotted_path, value in SYNTHETIC_VALUES.items():
        _set_path(payload, dotted_path, value)
    _set_path(payload, "data.manifest.path", "locks/built-manifest.json")
    config_root = tmp_path / "config"
    config_root.mkdir()
    config_path = config_root / "run.toml"
    config_path.write_text(tomli_w.dumps(payload), encoding="utf-8")
    args = (
        "--config",
        "run.toml",
        "--config-root",
        str(config_root),
        "--root",
        str(workspace),
        "--mode",
        "build",
        "--inventory",
        "inputs/build-inventory.json",
        "--inventory-sha256",
        inventory_digest,
        "--output",
        "locks/built-manifest.json",
    )
    return inventory_path, output_path, config_path, inventory_digest, args


def _set_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "synthetic-modelscope-secret")
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-wandb-secret")


def test_local_cli_verifies_config_bound_canonical_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path, _, args = _cli_tree(tmp_path)
    _set_secrets(monkeypatch)

    assert cli_manifest.main((*args, "--mode", "local")) == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert captured.err == ""
    assert report == {
        "bytes": len(CONTENT),
        "manifest_sha256": manifest_digest,
        "mode": "local",
        "ok": True,
        "repo_id": "leafmoone/webdataset_danbooru",
        "revision": REVISION,
        "samples": 11,
        "shards": 1,
    }


def test_remote_cli_uses_exact_transport_and_redacts_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, _, args = _cli_tree(tmp_path)
    _set_secrets(monkeypatch)
    calls: list[tuple[ModelScopeDatasetTransport, object]] = []

    def validate(transport: ModelScopeDatasetTransport, selection: object) -> None:
        calls.append((transport, selection))

    monkeypatch.setattr(cli_manifest, "validate_remote_manifest", validate)
    assert cli_manifest.main((*args, "--mode", "remote")) == 0
    success = capsys.readouterr()
    assert json.loads(success.out)["mode"] == "remote"
    assert len(calls) == 1
    assert type(calls[0][0]) is ModelScopeDatasetTransport

    marker = "raw-remote-url-token-marker"

    def fail(transport: object, selection: object) -> None:
        del transport, selection
        raise DatasetTransportError(marker)

    monkeypatch.setattr(cli_manifest, "validate_remote_manifest", fail)
    assert cli_manifest.main((*args, "--mode", "remote")) == 1
    failure = capsys.readouterr()
    assert json.loads(failure.out) == {
        "error": "remote_inventory_invalid",
        "ok": False,
    }
    assert marker not in failure.out


def test_cli_manifest_hash_drift_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path, _, args = _cli_tree(tmp_path)
    _set_secrets(monkeypatch)
    manifest_path.write_bytes(b"drift")

    assert cli_manifest.main((*args, "--mode", "local")) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "manifest_invalid"


def test_build_cli_publishes_canonical_manifest_without_clobber(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, output_path, _, inventory_digest, args = _build_cli_tree(tmp_path)
    _set_secrets(monkeypatch)
    build_calls = 0

    def build(
        transport: ModelScopeDatasetTransport,
        inventory: ManifestBuildInventory,
    ) -> DatasetManifest:
        nonlocal build_calls
        build_calls += 1
        assert type(transport) is ModelScopeDatasetTransport
        return build_dataset_manifest(
            inventory,
            (
                RemoteShardRecord(
                    path="release-a/000001.tar",
                    bytes=len(CONTENT),
                    sha256=hashlib.sha256(CONTENT).hexdigest(),
                ),
            ),
        )

    monkeypatch.setattr(cli_manifest, "build_remote_dataset_manifest", build)

    assert cli_manifest.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    manifest = parse_dataset_manifest_bytes(output_path.read_bytes())
    assert manifest.shards[0].release == "explicit-release-a"
    assert manifest.shards[0].samples == 11
    assert build_calls == 1
    assert report == {
        "build_inventory_sha256": inventory_digest,
        "bytes": len(CONTENT),
        "manifest_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "mode": "build",
        "ok": True,
        "repo_id": "leafmoone/webdataset_danbooru",
        "revision": REVISION,
        "samples": 11,
        "shards": 1,
    }

    assert cli_manifest.main(args) == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": "manifest_output_exists",
        "ok": False,
    }
    assert build_calls == 1
    assert parse_dataset_manifest_bytes(output_path.read_bytes()) == manifest


def test_build_cli_rejects_inventory_hash_drift_and_output_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, output_path, _, _, args = _build_cli_tree(tmp_path)
    _set_secrets(monkeypatch)

    drifted_hash_args = list(args)
    digest_index = drifted_hash_args.index("--inventory-sha256") + 1
    drifted_hash_args[digest_index] = "f" * 64
    assert cli_manifest.main(drifted_hash_args) == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": "build_inventory_invalid",
        "ok": False,
    }
    assert not output_path.exists()

    drifted_output_args = list(args)
    output_index = drifted_output_args.index("--output") + 1
    drifted_output_args[output_index] = "locks/other-manifest.json"
    assert cli_manifest.main(drifted_output_args) == 2
    assert json.loads(capsys.readouterr().out) == {
        "error": "manifest_path_invalid",
        "ok": False,
    }


def test_build_cli_requires_all_explicit_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, _, _, _, args = _build_cli_tree(tmp_path)
    incomplete = list(args)
    output_index = incomplete.index("--output")
    del incomplete[output_index : output_index + 2]

    assert cli_manifest.main(incomplete) == 2
    assert json.loads(capsys.readouterr().out) == {
        "error": "invalid_arguments",
        "ok": False,
    }


def test_build_cli_redacts_remote_builder_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, output_path, _, _, args = _build_cli_tree(tmp_path)
    _set_secrets(monkeypatch)
    marker = "remote-builder-sensitive-marker"

    def fail(transport: object, inventory: object) -> DatasetManifest:
        del transport, inventory
        raise DatasetTransportError(marker)

    monkeypatch.setattr(cli_manifest, "build_remote_dataset_manifest", fail)
    assert cli_manifest.main(args) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "error": "remote_inventory_invalid",
        "ok": False,
    }
    assert marker not in captured.out
    assert not output_path.exists()
