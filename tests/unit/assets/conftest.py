from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import tomli_w


@dataclass
class SyntheticAssetTree:
    root: Path
    manifest_path: Path
    payload: dict[str, Any]

    def write_manifest(self) -> None:
        self.manifest_path.write_text(tomli_w.dumps(self.payload), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bytes(root: Path, relative: str, value: bytes) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {"bytes": len(value), "sha256": sha256(path)}


def make_reference(
    root: Path,
    relative: str,
    origin: str,
    licenses: list[tuple[str, str, str]],
) -> tuple[str, list[dict[str, object]]]:
    repo = root / relative
    repo.mkdir(parents=True)
    locks: list[dict[str, object]] = []
    for name, scope, license_path in licenses:
        metadata = write_bytes(repo, license_path, f"synthetic {name}\n".encode())
        locks.append(
            {
                "name": name,
                "scope": scope,
                "path": license_path,
                **metadata,
            }
        )
    subprocess.run(("git", "init", "-q", str(repo)), check=True)
    subprocess.run(("git", "-C", str(repo), "remote", "add", "origin", origin), check=True)
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Synthetic Test",
            "-c",
            "user.email=synthetic@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ),
        check=True,
    )
    commit = subprocess.run(
        ("git", "-C", str(repo), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, locks


@pytest.fixture
def synthetic_assets(tmp_path: Path) -> SyntheticAssetTree:
    qwen_config = json.dumps(
        {"text_config": {"num_hidden_layers": 24, "hidden_size": 2048}},
        sort_keys=True,
    ).encode()
    qwen_files = [
        {"path": "config.json", "kind": "config", **write_bytes(tmp_path, "model/qwen/config.json", qwen_config)},
        {"path": "tokenizer.json", "kind": "tokenizer", **write_bytes(tmp_path, "model/qwen/tokenizer.json", b"synthetic tokenizer")},
        {"path": "model.safetensors", "kind": "weights", **write_bytes(tmp_path, "model/qwen/model.safetensors", b"synthetic qwen weights")},
        {"path": "LICENSE", "kind": "license", **write_bytes(tmp_path, "model/qwen/LICENSE", b"synthetic Apache-2.0")},
    ]
    vae_config = json.dumps(
        {"latent_channels": 128, "downsample_factor": 16, "sample_posterior": False},
        sort_keys=True,
    ).encode()
    vae_files = [
        {"path": "config.json", "kind": "config", **write_bytes(tmp_path, "model/vae/config.json", vae_config)},
        {"path": "weights.safetensors", "kind": "weights", **write_bytes(tmp_path, "model/vae/weights.safetensors", b"synthetic vae weights")},
    ]
    database_file = {
        "path": "metadata.db",
        "kind": "database",
        **write_bytes(tmp_path, "db/metadata.db", b"synthetic database fixture"),
    }

    reference_specs = (
        ("reference_hdm", "reference/HDM", "https://example.invalid/HDM.git", [("CC-BY-NC-SA-4.0", "repository", "LICENSE")]),
        ("reference_jlt", "reference/JLT", "https://example.invalid/JLT.git", [("MIT", "repository", "LICENSE")]),
        (
            "reference_krea2",
            "reference/krea-2",
            "https://example.invalid/krea-2.git",
            [
                ("Apache-2.0", "repository root code", "LICENSE.md"),
                ("KREA COMMUNITY LICENSE AGREEMENT", "assets/hf_samples", "assets/hf_samples/LICENSE.pdf"),
            ],
        ),
    )
    references: list[dict[str, object]] = []
    for asset_id, local_path, origin, licenses in reference_specs:
        commit, locks = make_reference(tmp_path, local_path, origin, licenses)
        references.append(
            {
                "asset_id": asset_id,
                "local_path": local_path,
                "origin_url": origin,
                "commit": commit,
                "licenses": locks,
                "tracked_worktree_required_clean": True,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "manifest_revision": 1,
        "models": [
            {
                "asset_id": "qwen_text_encoder",
                "kind": "qwen",
                "local_path": "model/qwen",
                "lock_state": "ready",
                "blockers": [],
                "source": {
                    "repo_id": "spawner/Qwen3_5_2b_claude_heretic_spawner",
                    "revision": "1" * 40,
                    "license_id": "Apache-2.0",
                    "access_terms": "synthetic fixture",
                },
                "summary": {
                    "config_sha256": qwen_files[0]["sha256"],
                    "tokenizer_sha256": qwen_files[1]["sha256"],
                    "layers": 24,
                    "hidden_size": 2048,
                    "dtype": "bfloat16",
                    "frozen": True,
                    "use_cache": False,
                    "visual_path_enabled": False,
                },
                "files": qwen_files,
            },
            {
                "asset_id": "mage_vae",
                "kind": "vae",
                "local_path": "model/vae",
                "lock_state": "ready",
                "blockers": [],
                "source": {
                    "repo_id": "microsoft/Mage-Flow",
                    "revision": "2" * 40,
                    "license_id": "MIT",
                    "access_terms": "synthetic fixture",
                },
                "summary": {
                    "config_sha256": vae_files[0]["sha256"],
                    "latent_channels": 128,
                    "downsample_factor": 16,
                    "sample_posterior": False,
                    "posterior_mean_required": True,
                    "dtype": "bfloat16",
                    "frozen": True,
                },
                "files": vae_files,
            },
        ],
        "databases": [
            {
                "asset_id": "metadata_db",
                "local_path": "db",
                "lock_state": "ready",
                "blockers": [],
                "required_for_runtime": True,
                "source": {
                    "origin_kind": "upstream_repo",
                    "repo_id": "synthetic/metadata",
                    "revision": "3" * 40,
                    "license_id": "MIT",
                    "access_terms": "synthetic fixture",
                },
                "schema_version": "synthetic-v1",
                "files": [database_file],
                "allowed_aggregate_statistics": ["row_count"],
            }
        ],
        "references": references,
    }
    manifest_path = tmp_path / "assets" / "manifest.toml"
    manifest_path.parent.mkdir()
    tree = SyntheticAssetTree(tmp_path, manifest_path, payload)
    tree.write_manifest()
    return tree
