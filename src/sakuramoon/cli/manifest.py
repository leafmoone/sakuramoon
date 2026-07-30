"""Build or verify an immutable ModelScope dataset manifest."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from sakuramoon.config import ConfigurationError, load_config
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetManifestError,
    DatasetManifestExistsError,
    DatasetManifestPublicationError,
    ManifestBuildInventoryError,
    RemoteManifestBuildError,
    load_dataset_manifest,
    load_manifest_build_inventory,
    write_dataset_manifest,
)
from sakuramoon.data.modelscope import (
    MODELSCOPE_TOKEN_ENVIRONMENT,
    DatasetAuthenticationError,
    DatasetTransportError,
    ModelScopeDatasetTransport,
    build_remote_dataset_manifest,
    validate_remote_manifest,
)


class _CliArgumentError(ValueError):
    """Raised without retaining or rendering attacker-controlled argv text."""


class _ManifestPathError(ValueError):
    """Safe marker for a manifest path outside the selected workspace."""


class _BuildInventoryPathError(ValueError):
    """Safe marker for an invalid build-inventory path."""


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _CliArgumentError("invalid command-line arguments")


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _error(code: str, *, exit_code: int) -> int:
    _emit({"error": code, "ok": False})
    return exit_code


def _sha256_argument(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise argparse.ArgumentTypeError("expected lowercase SHA-256")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Build or verify an immutable ModelScope dataset manifest"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("build", "local", "remote"), required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--inventory-sha256", type=_sha256_argument)
    parser.add_argument("--output", type=Path)
    return parser


def _relative_workspace_path(value: Path) -> Path:
    relative = value
    if relative.is_absolute() or ".." in relative.parts:
        raise _ManifestPathError("dataset manifest path is not workspace-relative")
    if relative == Path("."):
        raise _ManifestPathError("dataset manifest path is empty")
    return relative


def _workspace_root(root: Path) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
    except OSError:
        raise _ManifestPathError("dataset workspace root is inaccessible") from None
    if not resolved_root.is_dir():
        raise _ManifestPathError("dataset workspace root is not a directory")
    return resolved_root


def _workspace_manifest_path(root: Path, configured_path: str) -> Path:
    relative = _relative_workspace_path(Path(configured_path))
    resolved_root = _workspace_root(root)
    candidate = resolved_root / relative
    if not candidate.is_file():
        raise _ManifestPathError("dataset manifest is not a regular file")
    return candidate


def _workspace_build_inventory_path(root: Path, inventory_path: Path) -> Path:
    try:
        relative = _relative_workspace_path(inventory_path)
        candidate = _workspace_root(root) / relative
    except _ManifestPathError:
        raise _BuildInventoryPathError("build inventory path is invalid") from None
    if not candidate.is_file():
        raise _BuildInventoryPathError("build inventory is not a regular file")
    return candidate


def _workspace_output_path(
    root: Path,
    requested_path: Path,
    configured_path: str,
) -> Path:
    requested = _relative_workspace_path(requested_path)
    configured = _relative_workspace_path(Path(configured_path))
    if requested != configured:
        raise _ManifestPathError("build output does not match configured manifest path")
    candidate = _workspace_root(root) / requested
    if candidate.exists() or candidate.is_symlink():
        raise DatasetManifestExistsError(
            "dataset manifest destination already exists"
        )
    return candidate


def _validate_mode_arguments(args: argparse.Namespace) -> None:
    build_values = (args.inventory, args.inventory_sha256, args.output)
    if args.mode == "build":
        if any(value is None for value in build_values):
            raise _CliArgumentError("build mode requires explicit build inputs")
    elif any(value is not None for value in build_values):
        raise _CliArgumentError("build inputs are only valid in build mode")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        _validate_mode_arguments(args)
    except _CliArgumentError:
        return _error("invalid_arguments", exit_code=2)
    except SystemExit as exc:
        if exc.code == 0:
            return 0
        return _error("invalid_arguments", exit_code=2)

    manifest: DatasetManifest
    manifest_digest: str
    inventory_digest: str | None = None
    try:
        loaded = load_config(args.config, config_root=args.config_root)
        if loaded.config.security.modelscope_token_env != MODELSCOPE_TOKEN_ENVIRONMENT:
            raise ConfigurationError(
                "dataset credential environment variable is not approved"
            )
        if args.mode == "build":
            assert isinstance(args.inventory, Path)
            assert isinstance(args.inventory_sha256, str)
            assert isinstance(args.output, Path)
            inventory_path = _workspace_build_inventory_path(args.root, args.inventory)
            inventory_digest = args.inventory_sha256
            inventory = load_manifest_build_inventory(
                inventory_path,
                inventory_digest,
                loaded.config.data.source,
            )
            output_path = _workspace_output_path(
                args.root,
                args.output,
                loaded.config.data.manifest.path,
            )
            transport = ModelScopeDatasetTransport.from_token_environment(
                MODELSCOPE_TOKEN_ENVIRONMENT,
                loaded.config.data.transport,
            )
            manifest = build_remote_dataset_manifest(transport, inventory)
            manifest_digest = write_dataset_manifest(manifest, output_path)
        else:
            manifest_path = _workspace_manifest_path(
                args.root, loaded.config.data.manifest.path
            )
            manifest = load_dataset_manifest(
                manifest_path,
                loaded.config.data.manifest.sha256,
                loaded.config.data.source,
            )
            manifest_digest = loaded.config.data.manifest.sha256
            if args.mode == "remote":
                transport = ModelScopeDatasetTransport.from_token_environment(
                    MODELSCOPE_TOKEN_ENVIRONMENT,
                    loaded.config.data.transport,
                )
                validate_remote_manifest(transport, manifest)
    except ConfigurationError:
        return _error("configuration_invalid", exit_code=2)
    except _ManifestPathError:
        return _error("manifest_path_invalid", exit_code=2)
    except _BuildInventoryPathError:
        return _error("build_inventory_path_invalid", exit_code=2)
    except DatasetAuthenticationError:
        return _error("dataset_authentication_failed", exit_code=2)
    except ManifestBuildInventoryError:
        return _error("build_inventory_invalid", exit_code=1)
    except RemoteManifestBuildError:
        return _error("remote_inventory_invalid", exit_code=1)
    except DatasetManifestExistsError:
        return _error("manifest_output_exists", exit_code=1)
    except DatasetManifestPublicationError:
        return _error("manifest_publication_failed", exit_code=1)
    except DatasetManifestError:
        return _error("manifest_invalid", exit_code=1)
    except DatasetTransportError:
        return _error("remote_inventory_invalid", exit_code=1)
    except OSError:
        return _error("manifest_io_failed", exit_code=2)

    _emit(
        {
            "bytes": manifest.aggregates.bytes,
            "manifest_sha256": manifest_digest,
            "mode": args.mode,
            "ok": True,
            "repo_id": manifest.source.repo_id,
            "revision": manifest.source.revision,
            "samples": manifest.aggregates.samples,
            "shards": manifest.aggregates.shards,
            **(
                {"build_inventory_sha256": inventory_digest}
                if inventory_digest is not None
                else {}
            ),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
