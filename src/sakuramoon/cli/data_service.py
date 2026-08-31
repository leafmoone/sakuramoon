"""Launch the independently owned SakuraMoon dataset supply service."""

from __future__ import annotations

import argparse
import os
import signal
import threading
from collections.abc import Sequence
from pathlib import Path

from sakuramoon.config import load_config
from sakuramoon.data.cache import CacheQuota, ShardCache
from sakuramoon.data.modelscope import (
    MODELSCOPE_TOKEN_ENVIRONMENT,
    ModelScopeDatasetTransport,
    ensure_dataset_manifest,
)
from sakuramoon.data.service import (
    DataServiceError,
    DataServiceLimits,
    DataServiceServer,
    DataSupplyService,
)
from sakuramoon.data.service_protocol import DataServiceSessionIdentity
from sakuramoon.data.validation import (
    VALIDATION_SHARD_COUNT,
    ensure_validation_selection,
    prepare_validation_shards,
    require_published_validation_shards,
)
from sakuramoon.storage import require_data_service_storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SakuraMoon dataset supply service")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def _log(message: str) -> None:
    print(f"[data-server] {message}", flush=True)


def _root_path(
    root: Path, configured: str, *, allow_absolute: bool = False
) -> Path:
    try:
        base = root.resolve(strict=True)
        relative = Path(configured)
        if relative.is_absolute():
            if not allow_absolute:
                raise ValueError
            return relative.absolute()
        if ".." in relative.parts:
            raise ValueError
        candidate = base / relative
        candidate.resolve(strict=False).relative_to(base)
    except (OSError, ValueError):
        raise DataServiceError("configured service path is invalid") from None
    return candidate.absolute()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _log(f"加载配置: {args.config}")
    loaded = load_config(args.config, config_root=args.config_root)
    config = loaded.config
    root = args.root.resolve(strict=True)
    require_data_service_storage(config, root)
    _log("读取数据分片清单")
    manifest_path = _root_path(root, config.data.manifest.path)
    transport = ModelScopeDatasetTransport.from_token_environment(
        MODELSCOPE_TOKEN_ENVIRONMENT, config.data.transport
    )
    _log("下载源: ModelScope（自动使用 https_proxy/HTTPS_PROXY 环境变量，未设置则直连）")
    manifest = ensure_dataset_manifest(
        transport,
        manifest_path,
        config.data.source,
        initialize_if_missing=config.data.manifest.initialize_if_missing,
        refresh_existing=config.data.manifest.refresh_existing,
    )
    _log(f"数据分片: {len(manifest.shards)}")
    if config.data.validation.shard_count == 0:
        # Train-only corpus (e.g. SR_v2): no held-out validation split.
        _log("验证集分片: 0（训练专用语料，跳过验证选择）")
        validation_selection = None
    else:
        validation_selection = ensure_validation_selection(
            manifest,
            _root_path(root, config.data.validation.selection_path),
            expected_shard_count=config.data.validation.shard_count,
        )
        validation_root = _root_path(root, config.data.validation.shard_root)
        if len(validation_selection.shards) >= VALIDATION_SHARD_COUNT:
            require_published_validation_shards(
                manifest, validation_selection, validation_root
            )
        else:
            prepare_validation_shards(
                transport, manifest, validation_selection, validation_root
            )
        _log(f"验证集分片: {len(validation_selection.shards)}")
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
        dataset_id=manifest.dataset_id,
        worker_count=(
            config.data.cache.persistent_workers_per_rank
            * config.distributed.world_size
        ),
    )
    service = DataSupplyService(
        manifest,
        validation_selection,
        cache,
        _root_path(root, config.data.service.mainset_path, allow_absolute=True),
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
        if stopped.is_set():
            _log("再次收到停止信号，立即退出")
            os._exit(130)
        _log("收到停止信号，正在取消下载")
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    server.serve(
        stopped,
        ready_callback=lambda: _log(
            f"监听 {config.data.service.socket_path}，PID={os.getpid()}"
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
