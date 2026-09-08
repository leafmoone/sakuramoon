"""Production binding of the governed compile recompile limit.

Pins the fix-review production call site (``_run_accepted_lifecycle`` in
``sakuramoon.train.production``):

- the compile call binds ``recompile_limit=
  config.kernels.torch_compile_recompile_limit`` -- the config-governed
  value, never a hard-coded constant;
- a ``KernelsConfig`` carrying 37 reaches ``compile_packed_dit_blocks``
  as 37 and the dynamo config then reads ``recompile_limit == 37``
  (no GPU kernels are compiled: the same fake-compile harness as
  ``test_distributed_compile``).
"""

from __future__ import annotations

import inspect

import pytest
from test_distributed_compile import _composite
from torch import nn
from torch._dynamo import config as dynamo_config

import sakuramoon.train.production as production_module
from sakuramoon.config.schema import KernelsConfig
from sakuramoon.train.runtime import compile_packed_dit_blocks


def test_production_binds_recompile_limit_from_config_field() -> None:
    source = inspect.getsource(production_module)

    assert (
        "recompile_limit=config.kernels.torch_compile_recompile_limit"
        in source
    )
    for hard_coded in (
        "recompile_limit=8",
        "recompile_limit=64",
        "recompile_limit=37",
    ):
        assert hard_coded not in source, hard_coded


def _compile_on_kernels() -> KernelsConfig:
    return KernelsConfig(
        attention_backend="das_fa2_varlen",
        torch_compile_enabled=True,
        torch_compile_backend="inductor",
        torch_compile_mode="max-autotune-no-cudagraphs",
        torch_compile_dynamic=True,
        torch_compile_recompile_limit=37,
        dtype="bfloat16",
        native_gqa=True,
        repeat_kv_heads=False,
        silent_fallback=False,
    )


def test_config_value_thirty_seven_reaches_compile_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernels = _compile_on_kernels()
    composite = _composite()

    def fake_compile(
        module: nn.Module,
        *args: object,
        **kwargs: object,
    ) -> None:
        assert not args
        module._compiled_call_impl = module.forward

    monkeypatch.setattr(nn.Module, "compile", fake_compile)
    monkeypatch.setattr(dynamo_config, "suppress_errors", False)
    # pin the global dynamo values so teardown restores them (the install
    # path mutates them directly)
    monkeypatch.setattr(
        dynamo_config, "recompile_limit", dynamo_config.recompile_limit
    )
    monkeypatch.setattr(
        dynamo_config,
        "fail_on_recompile_limit_hit",
        dynamo_config.fail_on_recompile_limit_hit,
    )

    # The exact kwarg expressions the production call site uses.
    compile_packed_dit_blocks(
        composite,
        backend=kernels.torch_compile_backend,
        mode=kernels.torch_compile_mode,
        dynamic=kernels.torch_compile_dynamic,
        recompile_limit=kernels.torch_compile_recompile_limit,
    )

    assert dynamo_config.recompile_limit == 37
    assert dynamo_config.fail_on_recompile_limit_hit is True
