"""Dedicated bounded engineering-smoke CLI; production training remains gated."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from sakuramoon.engineering_smoke import EngineeringSmokeConfigurationError
from sakuramoon.engineering_smoke.s000 import (
    EngineeringSmokeError,
    run_s000_engineering_smoke,
)


class _ArgumentError(ValueError):
    pass


class _SafeParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _ArgumentError("invalid arguments")


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeParser(description="Run bounded S000 engineering evidence")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        result = run_s000_engineering_smoke(
            args.config,
            config_root=args.config_root,
            repository_root=args.root,
        )
    except (_ArgumentError, SystemExit):
        _emit({"error": "invalid_arguments", "ok": False})
        return 2
    except EngineeringSmokeConfigurationError:
        _emit({"error": "configuration_invalid", "ok": False})
        return 2
    except (EngineeringSmokeError, OSError, RuntimeError, ValueError):
        _emit({"error": "engineering_smoke_failed", "ok": False})
        return 1
    _emit(
        {
            "checkpoint": str(result.checkpoint_path),
            "classification": "synthetic_single_gpu_engineering_only",
            "fresh_process_successful_update": result.fresh_process_successful_update,
            "initial_successful_update": result.initial_successful_update,
            "ok": True,
            "report": str(result.report_path),
            "resolved_config_sha256": result.resolved_config_sha256,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
