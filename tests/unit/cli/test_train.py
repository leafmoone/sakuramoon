from __future__ import annotations

import json
from pathlib import Path

import pytest

import sakuramoon.cli.train as train_cli
from sakuramoon.config import ConfigurationError, UnresolvedConfigBinding
from sakuramoon.train.production import (
    ProductionPreflightError,
    ProductionReadinessError,
    ProductionTrainingError,
    ProductionTrainingResult,
)


def test_train_cli_has_no_workload_override_flags() -> None:
    parser = train_cli.build_parser()
    with pytest.raises(ValueError, match="invalid arguments"):
        parser.parse_args(["--config", "run.toml", "--batch-size", "2"])


def test_train_cli_delegates_exact_lifecycle_arguments_and_emits_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    report = tmp_path / "preflight.json"
    checkpoint.mkdir()
    report.write_text("{}")
    observed: list[dict[str, object]] = []

    def run(config: Path, **kwargs: object) -> ProductionTrainingResult:
        observed.append({"config": config, **kwargs})
        return ProductionTrainingResult("a" * 64, report, checkpoint, 4, 5, False)

    monkeypatch.setattr(train_cli, "run_production_single_gpu", run)

    result = train_cli.main(
        [
            "--config",
            "train_s0.toml",
            "--config-root",
            "config",
            "--root",
            str(tmp_path),
            "--resume",
            str(checkpoint),
        ]
    )

    assert result == 0
    assert observed == [
        {
            "config": Path("train_s0.toml"),
            "config_root": Path("config"),
            "repository_root": tmp_path,
            "resume": checkpoint,
            "preflight_only": False,
        }
    ]
    assert json.loads(capsys.readouterr().out) == {
        "checkpoint": str(checkpoint),
        "final_successful_update": 5,
        "initial_successful_update": 4,
        "ok": True,
        "preflight_only": False,
        "preflight_report": str(report),
        "resolved_config_sha256": "a" * 64,
    }


def test_train_cli_exposes_exact_readiness_blockers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def blocked(*_args: object, **_kwargs: object) -> ProductionTrainingResult:
        raise ProductionReadinessError(("scheduler.warmup_curve", "data.pass"))

    monkeypatch.setattr(train_cli, "run_production_single_gpu", blocked)

    assert train_cli.main(["--config", "train_s0.toml"]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "blockers": ["scheduler.warmup_curve", "data.pass"],
        "error": "production_readiness_blocked",
        "ok": False,
    }


def test_train_cli_preserves_structured_unresolved_config_bindings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bindings = (
        UnresolvedConfigBinding("run.seed", "REQUIRED_S0_SEED", "required"),
        UnresolvedConfigBinding(
            "stage.local_batch", "BENCHMARK_S0_LOCAL_BATCH", "benchmark"
        ),
    )

    def blocked(*_args: object, **_kwargs: object) -> ProductionTrainingResult:
        raise ConfigurationError("unresolved", unresolved_bindings=bindings)

    monkeypatch.setattr(train_cli, "run_production_single_gpu", blocked)

    assert train_cli.main(["--config", "train_s0.toml"]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "error": "configuration_invalid",
        "ok": False,
        "unresolved_bindings": [
            {
                "kind": "required",
                "path": "run.seed",
                "sentinel": "REQUIRED_S0_SEED",
            },
            {
                "kind": "benchmark",
                "path": "stage.local_batch",
                "sentinel": "BENCHMARK_S0_LOCAL_BATCH",
            },
        ],
    }


@pytest.mark.parametrize(
    ("error", "code", "payload"),
    [
        (
            ConfigurationError("invalid"),
            2,
            {"error": "configuration_invalid", "ok": False},
        ),
        (
            ProductionPreflightError("failed"),
            1,
            {"error": "training_preflight_failed", "ok": False},
        ),
        (
            ProductionTrainingError("failed"),
            1,
            {"error": "training_failed", "ok": False},
        ),
    ],
)
def test_train_cli_normalizes_lifecycle_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
    code: int,
    payload: dict[str, object],
) -> None:
    def fail(*_args: object, **_kwargs: object) -> ProductionTrainingResult:
        raise error

    monkeypatch.setattr(train_cli, "run_production_single_gpu", fail)

    assert train_cli.main(["--config", "train_s0.toml"]) == code
    assert json.loads(capsys.readouterr().out) == payload
