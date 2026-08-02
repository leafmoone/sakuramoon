from __future__ import annotations

# pyright: reportPrivateUsage=false
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn, cast

import pytest

from sakuramoon.cli import eval as eval_cli
from sakuramoon.config.load import (
    ConfigurationError,
    LoadedConfig,
    UnresolvedConfigBinding,
)
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.eval.runner import (
    EvaluationBlocker,
    EvaluationClassification,
    EvaluationPlan,
    EvaluationPreflightError,
    EvaluationRunResult,
)


def _loaded() -> LoadedConfig:
    config = cast(
        RuntimeConfig,
        SimpleNamespace(evaluation=SimpleNamespace(enabled=True)),
    )
    return LoadedConfig(config, (), "resolved\n", "a" * 64)


def _disabled_loaded() -> LoadedConfig:
    config = cast(
        RuntimeConfig,
        SimpleNamespace(evaluation=SimpleNamespace(enabled=False)),
    )
    return LoadedConfig(config, (), "resolved\n", "a" * 64)


def test_disabled_cli_stops_before_checkpoint_parsing_or_preflight(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def load_disabled(*_args: object, **_kwargs: object) -> LoadedConfig:
        return _disabled_loaded()

    monkeypatch.setattr(eval_cli, "load_config", load_disabled)

    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("disabled evaluation must stop before checkpoint preflight")

    monkeypatch.setattr(eval_cli, "_checkpoint_selection", forbidden)
    monkeypatch.setattr(eval_cli, "preflight_evaluator", forbidden)
    monkeypatch.setattr(eval_cli, "run_evaluator", forbidden)

    code = eval_cli.main(
        [
            "--config",
            "base.toml",
            "--checkpoint",
            "not-even-a-valid-selection",
            "--successful-update",
            "10",
            "--trend",
        ]
    )

    assert code == 1
    assert json.loads(capsys.readouterr().out) == {
        "blockers": [
            {"code": "EVALUATION_DISABLED", "subject": "evaluation.enabled"}
        ],
        "error": "evaluation_preflight_failed",
        "ok": False,
    }


def test_checkpoint_argument_requires_absolute_role_and_model_provenance() -> None:
    raw = eval_cli._checkpoint_selection("raw=/tmp/raw-checkpoint")
    accepted = eval_cli._checkpoint_selection("accepted=/tmp/release-checkpoint")
    legacy = eval_cli._checkpoint_selection(
        "model-only:pre_fix=/tmp/model-checkpoint"
    )

    assert (raw.role, raw.objective_provenance, raw.path) == (
        "raw",
        "strict_jlt",
        Path("/tmp/raw-checkpoint"),
    )
    assert (legacy.role, legacy.objective_provenance) == ("model-only", "pre_fix")
    bound = eval_cli._bind_accepted_source_pma(
        (raw, accepted), Path("/tmp/accepted-source-pma")
    )
    assert bound[1].accepted_source_pma == Path("/tmp/accepted-source-pma")
    with pytest.raises(ValueError, match="accepted source PMA"):
        eval_cli._bind_accepted_source_pma((raw,), Path("/tmp/source-pma"))
    with pytest.raises(ValueError, match="absolute"):
        eval_cli._bind_accepted_source_pma((accepted,), Path("relative-pma"))
    for value in (
        "raw=relative-checkpoint",
        "model-only=/tmp/model-checkpoint",
        "accepted:pre_fix=/tmp/release",
        "latest=/tmp/checkpoint",
        "raw=/tmp/nested/../checkpoint",
    ):
        with pytest.raises(ValueError, match="invalid|absolute|required"):
            eval_cli._checkpoint_selection(value)


def test_module_help_has_no_preimport_runpy_warning() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "sakuramoon.cli.eval",
            "--help",
        ],
        cwd=repository_root,
        env={"PYTHONPATH": str(repository_root / "src")},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""


def test_cli_preserves_structured_unresolved_config_bindings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bindings = (
        UnresolvedConfigBinding(
            "evaluation.batch_size", "BENCHMARK_EVAL_BATCH_SIZE", "benchmark"
        ),
        UnresolvedConfigBinding(
            "evaluation.fid.real_stats_path",
            "REQUIRED_REAL_STATS_PATH",
            "required",
        ),
    )

    def blocked(*_args: object, **_kwargs: object) -> LoadedConfig:
        raise ConfigurationError("unresolved", unresolved_bindings=bindings)

    monkeypatch.setattr(eval_cli, "load_config", blocked)

    code = eval_cli.main(
        [
            "--config",
            "eval.toml",
            "--checkpoint",
            "raw=/tmp/raw-checkpoint",
            "--successful-update",
            "10",
            "--trend",
        ]
    )

    assert code == 2
    assert json.loads(capsys.readouterr().out) == {
        "error": "configuration_invalid",
        "ok": False,
        "unresolved_bindings": [
            {
                "kind": "benchmark",
                "path": "evaluation.batch_size",
                "sentinel": "BENCHMARK_EVAL_BATCH_SIZE",
            },
            {
                "kind": "required",
                "path": "evaluation.fid.real_stats_path",
                "sentinel": "REQUIRED_REAL_STATS_PATH",
            },
        ],
    }


def test_cli_emits_structured_preflight_blockers_without_running(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_load(*_args: object, **_kwargs: object) -> LoadedConfig:
        return _loaded()

    def fail_run(_plan: EvaluationPlan) -> NoReturn:
        pytest.fail("run_evaluator must not run after failed preflight")

    monkeypatch.setattr(eval_cli, "load_config", fake_load)

    def blocked(*_args: object, **_kwargs: object) -> EvaluationPlan:
        raise EvaluationPreflightError(
            (
                EvaluationBlocker(
                    "FEATURE_EXTRACTOR_IDENTITY_INVALID",
                    "/repo/evaluation/extractor.pt",
                ),
                EvaluationBlocker(
                    "REAL_STATS_IDENTITY_INVALID",
                    "/repo/evaluation/real-stats.safetensors",
                ),
            )
        )

    monkeypatch.setattr(eval_cli, "preflight_evaluator", blocked)
    monkeypatch.setattr(eval_cli, "run_evaluator", fail_run)

    code = eval_cli.main(
        [
            "--config",
            "eval.toml",
            "--checkpoint",
            "raw=/tmp/raw-checkpoint",
            "--successful-update",
            "10",
            "--stage-end",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload == {
        "blockers": [
            {
                "code": "FEATURE_EXTRACTOR_IDENTITY_INVALID",
                "subject": "/repo/evaluation/extractor.pt",
            },
            {
                "code": "REAL_STATS_IDENTITY_INVALID",
                "subject": "/repo/evaluation/real-stats.safetensors",
            },
        ],
        "error": "evaluation_preflight_failed",
        "ok": False,
    }


def test_cli_preflight_only_does_not_start_generation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loaded = _loaded()
    plan = cast(
        EvaluationPlan,
        SimpleNamespace(
            checkpoints=(object(),),
            engineering_only=False,
            jobs=(object(), object()),
            plan_id="evaluation-preflight",
        ),
    )

    def fake_load(*_args: object, **_kwargs: object) -> LoadedConfig:
        return loaded

    def fake_preflight(*_args: object, **_kwargs: object) -> EvaluationPlan:
        return plan

    def fail_run(_plan: EvaluationPlan) -> NoReturn:
        pytest.fail("preflight-only must not generate samples")

    monkeypatch.setattr(eval_cli, "load_config", fake_load)
    monkeypatch.setattr(eval_cli, "preflight_evaluator", fake_preflight)
    monkeypatch.setattr(eval_cli, "run_evaluator", fail_run)

    code = eval_cli.main(
        [
            "--config",
            "eval.toml",
            "--checkpoint",
            "raw=/tmp/raw-checkpoint",
            "--successful-update",
            "10",
            "--trend",
            "--preflight-only",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["preflight_only"] is True
    assert payload["job_count"] == 2


@pytest.mark.parametrize(
    ("extra_args", "expected_engineering_only", "classification"),
    (
        ((), False, "checkpoint_driven_evaluation"),
        (
            ("--engineering-only",),
            True,
            "synthetic_bounded_engineering_only",
        ),
    ),
)
def test_cli_reports_classification_post_commit_and_total_wall_timing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    extra_args: tuple[str, ...],
    expected_engineering_only: bool,
    classification: str,
) -> None:
    loaded = _loaded()
    plan = cast(
        EvaluationPlan,
        SimpleNamespace(engineering_only=expected_engineering_only),
    )
    output = tmp_path / "evaluation-complete"

    def fake_load(*_args: object, **_kwargs: object) -> LoadedConfig:
        return loaded

    def fake_preflight(*_args: object, **_kwargs: object) -> EvaluationPlan:
        assert _kwargs["engineering_only"] is expected_engineering_only
        return plan

    def fake_run(_plan: EvaluationPlan) -> EvaluationRunResult:
        assert _plan.engineering_only is expected_engineering_only
        return EvaluationRunResult(
            plan_id="evaluation-result",
            output_path=output,
            artifact_count=3,
            checkpoint_count=1,
            classification=cast(EvaluationClassification, classification),
            publication_seconds=0.25,
            total_wall_seconds=4.5,
        )

    monkeypatch.setattr(eval_cli, "load_config", fake_load)
    monkeypatch.setattr(eval_cli, "preflight_evaluator", fake_preflight)
    monkeypatch.setattr(eval_cli, "run_evaluator", fake_run)

    code = eval_cli.main(
        [
            "--config",
            "eval.toml",
            "--checkpoint",
            "raw=/tmp/raw-checkpoint",
            "--successful-update",
            "10",
            "--trend",
            *extra_args,
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "artifact_count": 3,
        "checkpoint_count": 1,
        "classification": classification,
        "ok": True,
        "output": str(output),
        "plan_id": "evaluation-result",
        "publication_seconds": 0.25,
        "started_training": False,
        "total_wall_seconds": 4.5,
    }
