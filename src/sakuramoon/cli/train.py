"""Fail-closed single-GPU training entry point.

The CLI deliberately does not accept workload overrides. D025 owns production data
assembly. C002 has not yet bound the already-confirmed Text/Style decisions into
strict production configuration and assembly, so the command remains fail-closed
without promoting engineering-smoke values.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from sakuramoon.config import ConfigurationError, load_config
from sakuramoon.train.runtime import require_single_gpu_config


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
    return parser


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        loaded = load_config(args.config, config_root=args.config_root)
        require_single_gpu_config(loaded.config)
        if args.resume is not None and (
            not args.resume.is_absolute() or not args.resume.is_dir()
        ):
            raise ConfigurationError("resume must name an absolute raw checkpoint directory")
        raise ConfigurationError(
            "production train assembly requires the C002-governed Text and Style "
            "configuration binding"
        )
    except (_ArgumentError, SystemExit):
        _emit({"error": "invalid_arguments", "ok": False})
        return 2
    except ConfigurationError:
        _emit({"error": "configuration_invalid", "ok": False})
        return 2
    except ValueError:
        _emit({"error": "training_preflight_failed", "ok": False})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
