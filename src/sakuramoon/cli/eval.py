"""Executable, fail-closed checkpoint-driven evaluator entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

from sakuramoon.config import ConfigurationError, load_config
from sakuramoon.eval.extractor import ExtractorContractError
from sakuramoon.eval.generate import GenerationContractError
from sakuramoon.eval.jobs import (
    build_evaluation_jobs,
    write_evaluation_job,
)
from sakuramoon.eval.publisher import EvaluationPublicationError
from sakuramoon.eval.runner import (
    CheckpointSelection,
    EvaluationBlocker,
    EvaluationPreflightError,
    preflight_evaluator,
    run_evaluator,
)
from sakuramoon.eval.spec import ObjectiveProvenance


class _ArgumentError(ValueError):
    pass


class _SafeParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _ArgumentError("invalid arguments")


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeParser(description="Run checkpoint-driven SakuraMoon evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="ROLE=ABSOLUTE_PATH",
    )
    parser.add_argument(
        "--accepted-source-pma",
        type=Path,
        metavar="ABSOLUTE_PATH",
    )
    parser.add_argument("--successful-update", type=int, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trend", action="store_true")
    mode.add_argument("--stage-end", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--engineering-only", action="store_true")
    return parser


def _checkpoint_selection(value: str) -> CheckpointSelection:
    if type(value) is not str or "=" not in value:
        raise _ArgumentError("invalid checkpoint selection")
    raw_role, raw_path = value.split("=", 1)
    if not raw_role or not raw_path:
        raise _ArgumentError("invalid checkpoint selection")
    objective: ObjectiveProvenance = "strict_jlt"
    role_text = raw_role
    if raw_role.startswith("model-only:"):
        role_text, objective_text = raw_role.split(":", 1)
        if objective_text not in ("strict_jlt", "pre_fix"):
            raise _ArgumentError("invalid model-only objective provenance")
        objective = objective_text
    elif ":" in raw_role:
        raise _ArgumentError("invalid checkpoint role")
    if role_text not in ("raw", "model-only", "pma", "accepted"):
        raise _ArgumentError("invalid checkpoint role")
    if role_text == "model-only" and ":" not in raw_role:
        raise _ArgumentError("model-only objective provenance is required")
    path = Path(raw_path)
    if not path.is_absolute() or ".." in path.parts:
        raise _ArgumentError("checkpoint path must be canonical and absolute")
    return CheckpointSelection(
        role=role_text,
        path=path,
        objective_provenance=objective,
    )


def _bind_accepted_source_pma(
    selections: tuple[CheckpointSelection, ...], source_pma: Path | None
) -> tuple[CheckpointSelection, ...]:
    if source_pma is None:
        return selections
    if not source_pma.is_absolute() or ".." in source_pma.parts:
        raise _ArgumentError("accepted source PMA path must be canonical and absolute")
    accepted_indices = tuple(
        index for index, selection in enumerate(selections) if selection.role == "accepted"
    )
    if len(accepted_indices) != 1:
        raise _ArgumentError("accepted source PMA requires one accepted checkpoint")
    index = accepted_indices[0]
    bound = list(selections)
    bound[index] = replace(bound[index], accepted_source_pma=source_pma)
    return tuple(bound)


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        loaded = load_config(args.config, config_root=args.config_root)
        if not loaded.config.evaluation.enabled:
            raise EvaluationPreflightError(
                (EvaluationBlocker("EVALUATION_DISABLED", "evaluation.enabled"),)
            )
        selections = _bind_accepted_source_pma(
            tuple(_checkpoint_selection(value) for value in args.checkpoint),
            args.accepted_source_pma,
        )
        plan = preflight_evaluator(
            loaded,
            repository_root=args.root,
            selections=selections,
            trigger_successful_update=args.successful_update,
            stage_end=args.stage_end,
            engineering_only=args.engineering_only,
        )
        if args.preflight_only:
            _emit(
                {
                    "checkpoint_count": len(plan.checkpoints),
                    "classification": (
                        "synthetic_bounded_engineering_only"
                        if plan.engineering_only
                        else "checkpoint_driven_evaluation"
                    ),
                    "job_count": len(plan.jobs),
                    "ok": True,
                    "plan_id": plan.plan_id,
                    "preflight_only": True,
                    "resolved_config_sha256": loaded.resolved_sha256,
                }
            )
            return 0
        result = run_evaluator(plan)
    except _ArgumentError:
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
    except EvaluationPreflightError as error:
        _emit(
            {
                "blockers": [item.as_mapping() for item in error.blockers],
                "error": "evaluation_preflight_failed",
                "ok": False,
            }
        )
        return 1
    except (
        ExtractorContractError,
        GenerationContractError,
        EvaluationPublicationError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        _emit(
            {
                "error": "evaluation_failed",
                "exception_type": type(error).__name__,
                "ok": False,
            }
        )
        return 1
    _emit(
        {
            "artifact_count": result.artifact_count,
            "checkpoint_count": result.checkpoint_count,
            "classification": result.classification,
            "ok": True,
            "output": str(result.output_path),
            "plan_id": result.plan_id,
            "publication_seconds": result.publication_seconds,
            "started_training": False,
            "total_wall_seconds": result.total_wall_seconds,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_evaluation_jobs",
    "build_parser",
    "main",
    "write_evaluation_job",
]
