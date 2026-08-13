"""Fail-closed single-GPU production training entry point."""

from __future__ import annotations

import argparse
import os
import signal
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path


def _configure_cuda_allocator() -> str:
    """Enable expandable CUDA segments before importing Torch."""

    current = os.environ.get("PYTORCH_ALLOC_CONF", "")
    options = [
        option.strip()
        for option in current.split(",")
        if option.strip() and not option.strip().startswith("expandable_segments:")
    ]
    options.append("expandable_segments:True")
    configured = ",".join(options)
    os.environ["PYTORCH_ALLOC_CONF"] = configured
    return configured


def _configure_torchinductor_cache(project_root: object) -> Path:
    """Bind TorchInductor to a writable, non-symlinked project cache."""

    if not isinstance(project_root, Path):
        raise TypeError("project_root must be a pathlib.Path")
    resolved_root = project_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"project root is not a directory: {resolved_root}")

    cache_parent = resolved_root / "cache"
    cache_path = cache_parent / "torchinductor"
    if cache_parent.is_symlink() or cache_path.is_symlink():
        raise RuntimeError("TorchInductor cache path may not contain a symlink")
    if cache_parent.exists() and not cache_parent.is_dir():
        raise NotADirectoryError(
            f"project cache parent is not a real directory: {cache_parent}"
        )
    if cache_path.exists() and not cache_path.is_dir():
        raise NotADirectoryError(
            f"TorchInductor cache is not a real directory: {cache_path}"
        )

    current = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if current is not None:
        if not current.strip():
            raise RuntimeError("TORCHINDUCTOR_CACHE_DIR may not be empty")
        configured_path = Path(current)
        if not configured_path.is_absolute():
            raise RuntimeError("TORCHINDUCTOR_CACHE_DIR must be absolute")
        if configured_path.absolute() != cache_path:
            raise RuntimeError(
                "TORCHINDUCTOR_CACHE_DIR conflicts with the project cache: "
                f"configured={configured_path}, required={cache_path}"
            )

    cache_parent.mkdir(exist_ok=True)
    if not cache_parent.is_dir() or cache_parent.is_symlink():
        raise NotADirectoryError(
            f"project cache parent is not a real directory: {cache_parent}"
        )
    cache_path.mkdir(exist_ok=True)
    if not cache_path.is_dir() or cache_path.is_symlink():
        raise NotADirectoryError(
            f"TorchInductor cache is not a real directory: {cache_path}"
        )

    descriptor = -1
    probe_path: Path | None = None
    try:
        descriptor, raw_probe_path = tempfile.mkstemp(
            prefix=".sakuramoon-write-test-",
            dir=cache_path,
        )
        probe_path = Path(raw_probe_path)
        os.close(descriptor)
        descriptor = -1
        probe_path.unlink()
        probe_path = None
    except OSError as exc:
        raise RuntimeError(
            f"TorchInductor cache is not writable: {cache_path}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if probe_path is not None and probe_path.exists():
            probe_path.unlink()

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_path)
    return cache_path


def _install_shutdown_signal_handlers() -> None:
    """Make background/nohup jobs interruptible so owned workers are closed."""

    # Non-interactive shells may start background jobs with SIGINT ignored.
    # Python preserves that inherited disposition unless it is reset here.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SakuraMoon single-GPU training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--evaluate-only",
        type=Path,
        metavar="CHECKPOINT",
        help="compute FID/IS/KID/CMMD from a complete checkpoint without training",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.evaluate_only is not None and (
        args.resume is not None or args.preflight_only
    ):
        parser.error(
            "--evaluate-only cannot be combined with --resume or --preflight-only"
        )
    allocator_config = _configure_cuda_allocator()
    config_root = args.config_root.resolve(strict=True)
    if not config_root.is_dir():
        raise NotADirectoryError(f"config root is not a directory: {config_root}")
    torchinductor_cache = _configure_torchinductor_cache(config_root.parent)
    prefix = "eval" if args.evaluate_only is not None else "train"
    mode = "评估" if args.evaluate_only is not None else "训练"
    print(f"[{prefix}] 进程已启动，配置: {args.config}", flush=True)
    print(f"[{prefix}] CUDA allocator: {allocator_config}", flush=True)
    print(f"[{prefix}] TorchInductor cache: {torchinductor_cache}", flush=True)
    print(f"[{prefix}] 正在加载 Torch、Transformers 和{mode}模块", flush=True)
    import_finished = threading.Event()

    def import_heartbeat() -> None:
        elapsed = 0
        while not import_finished.wait(10.0):
            elapsed += 10
            print(f"[{prefix}] {mode}模块仍在加载: {elapsed}s", flush=True)

    heartbeat = threading.Thread(target=import_heartbeat, daemon=True)
    heartbeat.start()
    try:
        from sakuramoon.train.production import (
            run_production_evaluation,
            run_production_single_gpu,
        )
    finally:
        import_finished.set()
        heartbeat.join()
    print(f"[{prefix}] {mode}模块加载完成", flush=True)
    if args.evaluate_only is not None:
        evaluation = run_production_evaluation(
            args.config,
            config_root=args.config_root,
            repository_root=args.root,
            checkpoint=args.evaluate_only,
        )
        if evaluation is None:
            return 0
        print(
            f"[eval] 完成: update={evaluation.update}, "
            f"FID={evaluation.fid:.4f}, "
            f"IS={evaluation.inception_score_mean:.4f}±"
            f"{evaluation.inception_score_std:.4f}, "
            f"KID={evaluation.kid_mean:.6f}±{evaluation.kid_std:.6f}, "
            f"CMMD={evaluation.cmmd:.6f}, "
            f"generated={evaluation.sample_count}, "
            f"real={evaluation.real_sample_count}",
            flush=True,
        )
        print(f"[eval] 结果: {evaluation.result_path}", flush=True)
        return 0
    result = run_production_single_gpu(
        args.config,
        config_root=args.config_root,
        repository_root=args.root,
        resume=args.resume,
        preflight_only=args.preflight_only,
    )
    print(
        f"[train] 完成: update {result.initial_successful_update} -> "
        f"{result.final_successful_update}",
        flush=True,
    )
    print(f"[train] 模型输出: {result.checkpoint_path}", flush=True)
    print(f"[train] 预检报告: {result.preflight_report}", flush=True)
    return 0


if __name__ == "__main__":
    _install_shutdown_signal_handlers()
    raise SystemExit(main())
