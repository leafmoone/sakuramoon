"""Fail-closed single-GPU production training entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from sakuramoon.config import ConfigurationError
from sakuramoon.train.production import (
    ProductionPreflightError,
    ProductionReadinessError,
    ProductionTrainingError,
    run_production_single_gpu,
)


class _ArgumentError(ValueError):
    pass


class _SafeParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _ArgumentError("invalid arguments")


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeParser(description="Run SakuraMoon single-GPU training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        result = run_production_single_gpu(
            args.config,
            config_root=args.config_root,
            repository_root=args.root,
            resume=args.resume,
            preflight_only=args.preflight_only,
        )
    except (_ArgumentError, SystemExit):
        _emit({"error": "invalid_arguments", "ok": False})
        return 2
    except ConfigurationError as error:
        payload: dict[str, object] = {
            "error": "configuration_invalid",
            "ok": False,
        }
        if error.unresolved_bindings:
            payload["unresolved_bindings"] = [
                {
                    "kind": binding.kind,
                    "path": binding.path,
                    "sentinel": binding.sentinel,
                }
                for binding in error.unresolved_bindings
            ]
        _emit(payload)
        return 2
    except ProductionReadinessError as error:
        _emit(
            {
                "blockers": list(error.blockers),
                "error": "production_readiness_blocked",
                "ok": False,
            }
        )
        return 2
    except ProductionPreflightError:
        _emit({"error": "training_preflight_failed", "ok": False})
        return 1
    except ProductionTrainingError:
        _emit({"error": "training_failed", "ok": False})
        return 1
    _emit(
        {
            "checkpoint": str(result.checkpoint_path),
            "final_successful_update": result.final_successful_update,
            "initial_successful_update": result.initial_successful_update,
            "ok": True,
            "preflight_only": result.preflight_only,
            "preflight_report": str(result.preflight_report),
            "resolved_config_sha256": result.resolved_config_sha256,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
