"""Governed packed-compile recompile limit (schema + config chains).

Pins the fix-review contract:

- ``KernelsConfig.torch_compile_recompile_limit`` defaults to exactly 8, so
  every existing run keeps the historical torch.compile recompile ceiling;
- the field is a strict positive int in 1..256 (bool/float/str/None and
  out-of-range values are rejected);
- the base / G1 / P25 / v1-canary chains all resolve 8 (default identity);
- the v2 corrected-MBS canary resolves 64 -- its only non-identity
  behavioral difference from the v1 canary besides the reviewed worker
  propagation code;
- the v2 ``[kernels]`` override merges without clobbering any sibling
  kernel value.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from sakuramoon.config.load import load_config
from sakuramoon.config.schema import KernelsConfig

REPOSITORY_ROOT = Path(__file__).parents[3]

_SECRET_ENVIRONMENT = {
    "MODELSCOPE_API_TOKEN": "synthetic-modelscope-secret",
    "WANDB_API_KEY": "synthetic-wandb-secret",
}


def _kernels(**overrides: object) -> KernelsConfig:
    return KernelsConfig(  # type: ignore[arg-type]
        attention_backend="das_fa2_varlen",
        dtype="bfloat16",
        native_gqa=True,
        repeat_kv_heads=False,
        silent_fallback=False,
        **overrides,
    )


def test_default_recompile_limit_is_exactly_eight() -> None:
    kernels = _kernels()

    assert type(kernels.torch_compile_recompile_limit) is int
    assert kernels.torch_compile_recompile_limit == 8


def test_explicit_sixty_four_is_accepted() -> None:
    assert (
        _kernels(torch_compile_recompile_limit=64).torch_compile_recompile_limit
        == 64
    )


def test_bounds_one_and_two_hundred_sixty_are_accepted() -> None:
    assert _kernels(torch_compile_recompile_limit=1).torch_compile_recompile_limit == 1
    assert (
        _kernels(torch_compile_recompile_limit=256).torch_compile_recompile_limit
        == 256
    )


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, -8, 257, 1000, 8.0, 64.0, "8", "64", None],
)
def test_rejects_non_strict_positive_int_values(value: object) -> None:
    with pytest.raises(ValidationError):
        _kernels(torch_compile_recompile_limit=value)


def _resolved_kernels(config_name: str) -> dict:
    loaded = load_config(
        Path(config_name),
        config_root=REPOSITORY_ROOT / "config",
        environment=_SECRET_ENVIRONMENT,
    )
    return tomllib.loads(loaded.resolved_toml)["kernels"]


@pytest.mark.parametrize(
    ("config_name", "expected"),
    [
        ("train_s0.toml", 8),
        ("train_g1.toml", 8),
        ("train_g1_cmuon_production.toml", 8),
        ("train_g1_camera_v2_p25.toml", 8),
        ("train_g1_camera_v2_p25_mirror_canary.toml", 8),
        ("train_g1_camera_v2_p25_mirror_v2_canary.toml", 64),
    ],
)
def test_resolved_recompile_limit_per_config(config_name: str, expected: int) -> None:
    kernels = _resolved_kernels(config_name)

    assert type(kernels["torch_compile_recompile_limit"]) is int
    assert kernels["torch_compile_recompile_limit"] == expected


def test_v2_override_clobbers_no_sibling_kernel_value() -> None:
    v1 = _resolved_kernels("train_g1_camera_v2_p25_mirror_canary.toml")
    v2 = _resolved_kernels("train_g1_camera_v2_p25_mirror_v2_canary.toml")

    for key, value in v1.items():
        if key == "torch_compile_recompile_limit":
            continue
        assert v2[key] == value, key
    assert v2["torch_compile_recompile_limit"] == 64
    assert v1["torch_compile_recompile_limit"] == 8
