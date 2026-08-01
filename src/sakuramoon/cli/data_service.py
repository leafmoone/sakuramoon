"""Launch the independently owned SakuraMoon dataset supply service."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from sakuramoon.config import ConfigurationError, load_config
from sakuramoon.data.cache import CacheQuota, ShardCache, ShardCacheError
from sakuramoon.data.manifest import DatasetManifestError, load_dataset_manifest
from sakuramoon.data.modelscope import (
    MODELSCOPE_TOKEN_ENVIRONMENT,
    DatasetAuthenticationError,
    DatasetTransportError,
    ModelScopeDatasetTransport,
)
from sakuramoon.data.service import (
    DataServiceError,
    DataServiceLimits,
    DataServiceServer,
    DataSupplyService,
)
from sakuramoon.data.service_protocol import DataServiceSessionIdentity
from sakuramoon.storage import StorageValidationError, require_data_service_storage


class _ArgumentError(ValueError):
    pass


class _SafeParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _ArgumentError("invalid arguments")


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeParser(description="Run the SakuraMoon dataset supply service")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def _root_path(root: Path, configured: str) -> Path:
    try:
        base = root.resolve(strict=True)
        relative = Path(configured)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError
        candidate = base / relative
        candidate.resolve(strict=False).relative_to(base)
    except (OSError, ValueError):
        raise DataServiceError("configured service path is invalid") from None
    return candidate.absolute()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        loaded = load_config(args.config, config_root=args.config_root)
        config = loaded.config
        if config.security.modelscope_token_env != MODELSCOPE_TOKEN_ENVIRONMENT:
            raise ConfigurationError("dataset credential variable is not approved")
        root = args.root.resolve(strict=True)
        require_data_service_storage(config, root)
        manifest_path = _root_path(root, config.data.manifest.path)
        manifest = load_dataset_manifest(
            manifest_path,
            config.data.manifest.sha256,
            config.data.source,
        )
        transport = ModelScopeDatasetTransport.from_token_environment(
            MODELSCOPE_TOKEN_ENVIRONMENT, config.data.transport
        )
        cache = ShardCache(
            _root_path(root, config.paths.cache_dir),
            manifest,
            transport,
            CacheQuota(
                config.data.cache.low_watermark_gib * 1024**3,
                config.data.cache.high_watermark_gib * 1024**3,
            ),
        )
        identity = DataServiceSessionIdentity(
            manifest_sha256=config.data.manifest.sha256,
            worker_count=config.data.cache.persistent_workers_per_rank,
        )
        service = DataSupplyService(
            manifest,
            cache,
            _root_path(root, config.data.service.mainset_path),
            Path(config.data.service.ownership_lock_path),
            identity,
            DataServiceLimits(
                download_concurrency=config.data.cache.download_concurrency,
                verified_shard_lookahead=config.data.cache.verified_shard_lookahead,
                lease_channel_capacity=config.data.service.lease_channel_capacity,
                ack_channel_capacity=config.data.service.ack_channel_capacity,
            ),
        )
        server = DataServiceServer(
            service,
            Path(config.data.service.socket_path),
            request_timeout_seconds=config.data.service.request_timeout_seconds,
        )
        stopped = threading.Event()

        def stop(_signum: int, _frame: object) -> None:
            stopped.set()

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        server.serve(
            stopped,
            ready_callback=lambda: _emit(
                {
                    "ok": True,
                    "pid": os.getpid(),
                    "service_sha256": identity.sha256,
                    "status": "listening",
                }
            ),
        )
        return 0
    except (_ArgumentError, SystemExit):
        _emit({"error": "invalid_arguments", "ok": False})
        return 2
    except ConfigurationError:
        _emit({"error": "configuration_invalid", "ok": False})
        return 2
    except DatasetAuthenticationError:
        _emit({"error": "dataset_authentication_failed", "ok": False})
        return 2
    except DatasetManifestError:
        _emit({"error": "dataset_manifest_invalid", "ok": False})
        return 1
    except (
        DataServiceError,
        ShardCacheError,
        DatasetTransportError,
        StorageValidationError,
    ):
        _emit({"error": "data_service_failed", "ok": False})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
