from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import sakuramoon.cli.train as train_cli


def test_train_cli_has_no_workload_override_flags() -> None:
    parser = train_cli.build_parser()
    with pytest.raises(ValueError, match="invalid arguments"):
        parser.parse_args(["--config", "run.toml", "--batch-size", "2"])


def test_train_cli_binds_confirmed_model_spec_then_fails_at_downstream_gate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    loaded = SimpleNamespace(config=object())
    observed: list[object] = []

    def fake_load(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return loaded

    def accept_config(_config: object) -> None:
        return None

    def bind_spec(config: object) -> dict[str, object]:
        observed.append(config)
        return {"class": "TrainableComposite"}

    monkeypatch.setattr(train_cli, "load_config", fake_load)
    monkeypatch.setattr(train_cli, "require_single_gpu_config", accept_config)
    monkeypatch.setattr(train_cli, "trainable_composite_spec", bind_spec)

    result = train_cli.main(["--config", "run.toml"])

    assert result == 2
    assert observed == [loaded.config]
    assert json.loads(capsys.readouterr().out) == {
        "error": "configuration_invalid",
        "ok": False,
    }
