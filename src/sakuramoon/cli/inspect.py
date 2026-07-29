"""Command-line entry point for explicit asset diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from sakuramoon.assets import (
    AssetPreflightError,
    ManifestError,
    inspect_databases,
    inspect_reference_repositories,
    inspect_runtime_models,
)


class _CliArgumentError(ValueError):
    """Raised without retaining or rendering attacker-controlled argv text."""


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _CliArgumentError("invalid command-line arguments")


def _error(message: str) -> int:
    print(json.dumps({"error": message, "ok": False}, sort_keys=True, separators=(",", ":")))
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description="Inspect immutable SakuraMoon asset locks")
    parser.add_argument("--manifest", type=Path, default=Path("assets/manifest.toml"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--scope",
        choices=("runtime-models", "databases", "references"),
        default="runtime-models",
    )
    parser.add_argument("--asset-id", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except _CliArgumentError:
        return _error("invalid command-line arguments")
    except SystemExit as exc:
        if exc.code == 0:
            return 0
        return _error("invalid command-line arguments")
    try:
        if args.scope == "runtime-models":
            if args.asset_id:
                raise ValueError("--asset-id is only valid for database audits")
            report = inspect_runtime_models(args.manifest, root=args.root)
        elif args.scope == "databases":
            report = inspect_databases(
                args.manifest,
                root=args.root,
                asset_ids=tuple(args.asset_id),
            )
        else:
            if args.asset_id:
                raise ValueError("--asset-id is only valid for database audits")
            report = inspect_reference_repositories(args.manifest, root=args.root)
    except AssetPreflightError:
        return _error("asset preflight failed")
    except ManifestError:
        return _error("asset manifest is invalid")
    except ValueError:
        return _error("invalid asset inspection request")
    except OSError:
        return _error("asset inspection I/O failed")
    print(report.to_json())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
