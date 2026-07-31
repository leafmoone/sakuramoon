"""Run one bounded SIGKILL fault command and publish process evidence."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from sakuramoon.fault_injection.driver import run_until_ready_and_sigkill
from sakuramoon.fault_injection.schema import FaultScenario

_KILL_SCENARIOS = (
    FaultScenario.DOWNLOAD_INTERRUPTION,
    FaultScenario.MICROBATCH_SIGKILL,
    FaultScenario.OPTIMIZER_SIGKILL,
    FaultScenario.CHECKPOINT_SIGKILL,
)


def _write_evidence(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("fault process evidence already exists")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    published = False
    try:
        with temporary.open("xb") as handle:
            handle.write(
                (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        published = True
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
            temporary.unlink()
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        if published:
            path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=tuple(item.value for item in _KILL_SCENARIOS), required=True
    )
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = tuple(args.command)
    if command and command[0] == "--":
        command = command[1:]
    evidence = run_until_ready_and_sigkill(
        command,
        scenario=FaultScenario(args.scenario),
        timeout_seconds=args.timeout_seconds,
    )
    payload = asdict(evidence)
    payload["scenario"] = evidence.scenario.value
    payload["schema_version"] = 1
    _write_evidence(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
