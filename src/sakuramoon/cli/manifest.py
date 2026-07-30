"""Command-line verification for an existing immutable dataset manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from sakuramoon.config import ConfigurationError, load_config
from sakuramoon.data.manifest import DatasetManifestError, load_dataset_manifest
from sakuramoon.data.modelscope import (
    MODELSCOPE_TOKEN_ENVIRONMENT,
    DatasetAuthenticationError,
    DatasetTransportError,
    ModelScopeDatasetTransport,
    validate_remote_manifest,
)


class _CliArgumentError(ValueError):
    """Raised without retaining or rendering attacker-controlled argv text."""


class _ManifestPathError(ValueError):
    """Safe marker for a manifest path outside the selected workspace."""


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _CliArgumentError("invalid command-line arguments")


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _error(code: str, *, exit_code: int) -> int:
    _emit({"error": code, "ok": False})
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Verify an existing immutable ModelScope dataset manifest"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("local", "remote"), required=True)
    return parser


def _workspace_manifest_path(root: Path, configured_path: str) -> Path:
    relative = Path(configured_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise _ManifestPathError("dataset manifest path is not workspace-relative")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError:
        raise _ManifestPathError("dataset workspace root is inaccessible") from None
    if not resolved_root.is_dir():
        raise _ManifestPathError("dataset workspace root is not a directory")
    candidate = resolved_root / relative
    if not candidate.is_file():
        raise _ManifestPathError("dataset manifest is not a regular file")
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except _CliArgumentError:
        return _error("invalid_arguments", exit_code=2)
    except SystemExit as exc:
        if exc.code == 0:
            return 0
        return _error("invalid_arguments", exit_code=2)

    try:
        loaded = load_config(args.config, config_root=args.config_root)
        if loaded.config.security.modelscope_token_env != MODELSCOPE_TOKEN_ENVIRONMENT:
            raise ConfigurationError(
                "dataset credential environment variable is not approved"
            )
        manifest_path = _workspace_manifest_path(
            args.root, loaded.config.data.manifest.path
        )
        manifest = load_dataset_manifest(
            manifest_path,
            loaded.config.data.manifest.sha256,
            loaded.config.data.source,
        )
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
    except DatasetAuthenticationError:
        return _error("dataset_authentication_failed", exit_code=2)
    except DatasetManifestError:
        return _error("manifest_invalid", exit_code=1)
    except DatasetTransportError:
        return _error("remote_inventory_invalid", exit_code=1)
    except OSError:
        return _error("manifest_io_failed", exit_code=2)

    _emit(
        {
            "bytes": manifest.aggregates.bytes,
            "manifest_sha256": loaded.config.data.manifest.sha256,
            "mode": args.mode,
            "ok": True,
            "repo_id": manifest.source.repo_id,
            "revision": manifest.source.revision,
            "samples": manifest.aggregates.samples,
            "shards": manifest.aggregates.shards,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
