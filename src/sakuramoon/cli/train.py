"""Fail-closed single-GPU production training entry point."""

from __future__ import annotations

import argparse
import os
import threading
from collections.abc import Sequence
from pathlib import Path


def _configure_cuda_allocator() -> str:
    """Enable expandable CUDA segments before importing Torch."""

    current = os.environ.get("PYTORCH_ALLOC_CONF", "")
    options = [
        option.strip()
        for option in current.split(",")
        if option.strip()
        and not option.strip().startswith("expandable_segments:")
    ]
    options.append("expandable_segments:True")
    configured = ",".join(options)
    os.environ["PYTORCH_ALLOC_CONF"] = configured
    return configured


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
        help="compute FID/IS from an existing complete checkpoint without training",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.evaluate_only is not None and (
        args.resume is not None or args.preflight_only
    ):
        parser.error("--evaluate-only cannot be combined with --resume or --preflight-only")
    allocator_config = _configure_cuda_allocator()
    prefix = "eval" if args.evaluate_only is not None else "train"
    mode = "评估" if args.evaluate_only is not None else "训练"
    print(f"[{prefix}] 进程已启动，配置: {args.config}", flush=True)
    print(f"[{prefix}] CUDA allocator: {allocator_config}", flush=True)
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
        print(
            f"[eval] 完成: update={evaluation.update}, "
            f"FID={evaluation.fid:.4f}, "
            f"IS={evaluation.inception_score_mean:.4f}±"
            f"{evaluation.inception_score_std:.4f}",
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
    raise SystemExit(main())
