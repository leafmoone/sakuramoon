# pyright: reportPrivateUsage=false

from __future__ import annotations

import copy
import inspect
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest
import tomli_w
import torch
from pydantic import ValidationError
from torch import nn

from sakuramoon.cli.engineering_smoke import build_parser
from sakuramoon.conditioning.style_resampler import StyleConditioningOutput
from sakuramoon.conditioning.text_mixer import TextConditioningOutput
from sakuramoon.engineering_smoke.config import (
    EngineeringSmokeConfig,
    EngineeringSmokeConfigurationError,
    load_engineering_smoke_config,
    require_single_gpu_environment,
)
from sakuramoon.engineering_smoke.s000 import (
    _build_composite,
    run_s000_engineering_smoke,
)
from sakuramoon.model.dit import DenseDiT
from sakuramoon.train.step import TrainableComposite, TrainableCompositeInputs

REPOSITORY_ROOT = Path(__file__).parents[3]
CONFIG_ROOT = REPOSITORY_ROOT / "config"
CONFIG_PATH = CONFIG_ROOT / "engineering_smoke_s000.toml"


def _payload() -> dict[str, Any]:
    return tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_config(root: Path, payload: dict[str, Any]) -> Path:
    path = root / "candidate.toml"
    path.write_text(tomli_w.dumps(payload), encoding="utf-8")
    return path


def _tiny_dense() -> DenseDiT:
    return DenseDiT(
        depth=16,
        input_channels=8,
        hidden_size=8,
        intermediate_size=16,
        q_heads=2,
        kv_heads=1,
        head_dim=4,
        rope_nope_dim=0,
        rope_y_dim=2,
        rope_x_dim=2,
        rope_position_scale=1.0,
        rope_theta=10.0,
        norm_eps=1e-6,
        timestep_dim=256,
        size_dim=64,
        aspect_dim=64,
        condition_hidden_size=1024,
        stable_slot_count=24,
        modulation_chunks=6,
        final_modulation_size=16,
        out_channels=8,
        modality_init_std=0.02,
        linear_dtype=torch.float32,
        sensitive_dtype=torch.float32,
        projection_bias=False,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        output_weight_zero_init=True,
        output_bias_zero_init=True,
    )


def test_checked_config_is_strict_engineering_only_dense_s0() -> None:
    loaded = load_engineering_smoke_config(
        Path(CONFIG_PATH.name), config_root=CONFIG_ROOT
    )
    config = loaded.config
    raw = _payload()

    assert config.evidence.classification == "synthetic_single_gpu_engineering_only"
    assert config.run.total_successful_updates == 2
    assert config.stage.depth == 16 and config.stage.resolution == 256
    assert config.kernels.attention_backend == "dense_sdpa_reference"
    assert config.device.world_size == 1
    assert config.evidence.formal_s000 is False
    assert config.evidence.production_cli_unlock is False
    assert "evaluation" not in raw
    assert "benchmark" not in raw
    assert len(loaded.resolved_sha256) == 64


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "missing",
        "zero_updates",
        "string_updates",
        "mismatched_bound",
        "fa4",
        "output_escape",
        "four_gpu",
        "missing_growth_alpha",
        "wrong_noise_boundary",
    ],
)
def test_config_rejects_unknown_missing_wrong_type_and_out_of_scope_values(
    case: str,
) -> None:
    payload = copy.deepcopy(_payload())
    if case == "unknown":
        payload["unknown"] = {"field": True}
    elif case == "missing":
        payload["run"].pop("initial_successful_updates")
    elif case == "zero_updates":
        payload["run"]["initial_successful_updates"] = 0
    elif case == "string_updates":
        payload["run"]["initial_successful_updates"] = "1"
    elif case == "mismatched_bound":
        payload["stage"]["successful_updates"] = 3
    elif case == "fa4":
        payload["kernels"]["attention_backend"] = "fa4_varlen"
    elif case == "output_escape":
        payload["run"]["output_root"] = "runs/s000"
    elif case == "four_gpu":
        payload["device"]["world_size"] = 4
    elif case == "missing_growth_alpha":
        payload["stage"].pop("growth_alpha")
    elif case == "wrong_noise_boundary":
        payload["measurement"]["noise_observation_boundary"] = 0.5
    else:
        raise AssertionError(case)

    with pytest.raises(ValidationError):
        EngineeringSmokeConfig.model_validate(payload)


def test_loader_rejects_unresolved_sentinel_and_root_escape(tmp_path: Path) -> None:
    payload = _payload()
    payload["run"]["run_id"] = "REQUIRED_ENGINEERING_RUN_ID"
    _write_config(tmp_path, payload)

    with pytest.raises(
        EngineeringSmokeConfigurationError, match="unresolved sentinel"
    ):
        load_engineering_smoke_config(Path("candidate.toml"), config_root=tmp_path)
    with pytest.raises(EngineeringSmokeConfigurationError, match="inside config_root"):
        load_engineering_smoke_config(CONFIG_PATH, config_root=tmp_path)


def test_environment_rejects_extra_gpu_and_distributed_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_engineering_smoke_config(
        Path(CONFIG_PATH.name), config_root=CONFIG_ROOT
    ).config
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    require_single_gpu_environment(config)
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(
        EngineeringSmokeConfigurationError, match="distributed launch"
    ):
        require_single_gpu_environment(config)
    monkeypatch.delenv("WORLD_SIZE")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(EngineeringSmokeConfigurationError, match="exactly one"):
        require_single_gpu_environment(config)


def test_cli_and_runner_expose_no_training_semantic_overrides() -> None:
    parser = build_parser()
    parsed = parser.parse_args(["--config", CONFIG_PATH.name])

    assert set(vars(parsed)) == {"config", "config_root", "root"}
    assert set(inspect.signature(run_s000_engineering_smoke).parameters) == {
        "config_path",
        "config_root",
        "repository_root",
    }


def test_checked_config_assembles_exact_dense_16_layer_artifact_on_meta() -> None:
    config = load_engineering_smoke_config(
        Path(CONFIG_PATH.name), config_root=CONFIG_ROOT
    ).config
    composite = _build_composite(config, torch.device("meta"))

    assert type(composite.dit) is DenseDiT
    assert composite.dit.model_metadata()["depth"] == 16
    assert composite.dit.artifact_config()["attention_backend"] == "dense_sdpa"


def test_trainable_composite_routes_homogeneous_batch_to_dense_signature() -> None:
    dense = _tiny_dense()
    composite = TrainableComposite(
        dit=dense,
        text=cast(Any, nn.Identity()),
        style=cast(Any, nn.Identity()),
    )
    inputs = TrainableCompositeInputs(
        qwen_states=torch.empty(0),
        main_token_indices=torch.empty(0, dtype=torch.long),
        main_mask=torch.empty(0, dtype=torch.bool),
        main_token_lengths=(3, 2),
        artist_token_indices=torch.empty(0, dtype=torch.long),
        artist_mask=torch.empty(0, dtype=torch.bool),
        use_null_style=torch.empty(0, dtype=torch.bool),
        active_style_sample_indices=torch.empty(0, dtype=torch.long),
        latents=(torch.randn(8, 2, 2), torch.randn(8, 2, 2)),
        timestep=torch.tensor([0.2, 0.8]),
        size_scale=torch.zeros(2),
        aspect=torch.zeros(2),
        growth_alpha=1.0,
    )
    conditioning = (
        TextConditioningOutput(
            tokens=torch.randn(2, 3, 8),
            mask=torch.tensor([[True, True, True], [True, True, False]]),
            layer_weights=torch.zeros(2, 3, 7),
        ),
        StyleConditioningOutput(
            tokens=torch.randn(2, 4, 8),
            mask=torch.ones(2, 4, dtype=torch.bool),
        ),
    )

    output = composite.forward_dit(inputs, conditioning)

    assert tuple(item.shape for item in output) == ((8, 2, 2), (8, 2, 2))
    assert all(torch.count_nonzero(item) == 0 for item in output)
