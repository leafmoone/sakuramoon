"""Command-line entry point for the fail-closed asset preflight."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sakuramoon.assets import ManifestError, inspect_assets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect immutable SakuraMoon asset locks")
    parser.add_argument("--manifest", type=Path, default=Path("assets/manifest.toml"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = inspect_assets(args.manifest, root=args.root)
    except (ManifestError, ValueError) as exc:
        print(json.dumps({"error": str(exc), "ok": False}, sort_keys=True, separators=(",", ":")))
        return 2
    print(report.to_json())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
