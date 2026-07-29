from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

from sakuramoon.assets import (
    inspect_assets,
    load_manifest,
    require_assets_ready,
)
from sakuramoon.assets.inspect import iter_declared_paths

ROOT = Path(__file__).parents[3]
MANIFEST = ROOT / "assets/manifest.toml"


def test_preflight_has_no_force_or_skip_switch() -> None:
    inspect_parameters = inspect.signature(inspect_assets).parameters
    require_parameters = inspect.signature(require_assets_ready).parameters

    assert set(inspect_parameters) == {"manifest_path", "root"}
    assert set(require_parameters) == {"manifest_path", "root"}


def test_production_manifest_records_ready_models_and_optional_databases_without_io() -> None:
    manifest = load_manifest(MANIFEST)

    assert all(asset.lock_state == "ready" for asset in manifest.models)
    assert all(asset.lock_state == "ready" for asset in manifest.databases)
    assert all(not asset.required_for_runtime for asset in manifest.databases)


def test_all_asset_roots_are_git_ignored_and_payloads_are_not_tracked() -> None:
    manifest = load_manifest(MANIFEST)
    declared = list(iter_declared_paths(manifest))
    result = subprocess.run(
        ("git", "check-ignore", "--stdin"),
        cwd=ROOT,
        input="".join(f"{path}/payload.bin\n" for path in declared),
        capture_output=True,
        check=True,
        text=True,
    )
    ignored = set(result.stdout.splitlines())

    assert ignored == {f"{path}/payload.bin" for path in declared}
    tracked = subprocess.run(
        ("git", "ls-files", "model", "db", "reference"),
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    assert tracked == ""


def test_manifest_contains_no_credential_values_or_signed_urls() -> None:
    payload = MANIFEST.read_text(encoding="utf-8")

    assert "MODELSCOPE_API_TOKEN=" not in payload
    assert "WANDB_API_KEY=" not in payload
    assert "?token=" not in payload.casefold()
    assert "?signature=" not in payload.casefold()
