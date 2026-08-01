from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import sakuramoon.cli.train as train_cli


def test_train_cli_has_no_workload_override_flags() -> None:
    parser = train_cli.build_parser()
    with pytest.raises(ValueError, match="invalid arguments"):
        parser.parse_args(["--config", "run.toml", "--batch-size", "2"])


def test_train_cli_fails_closed_before_inventing_model_constructor_values(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    loaded = SimpleNamespace(config=object())

    def fake_load(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return loaded

    def accept_config(_config: object) -> None:
        return None

    monkeypatch.setattr(train_cli, "load_config", fake_load)
    monkeypatch.setattr(train_cli, "require_single_gpu_config", accept_config)

    result = train_cli.main(["--config", "run.toml"])

    assert result == 2
    assert json.loads(capsys.readouterr().out) == {
        "error": "configuration_invalid",
        "ok": False,
    }
