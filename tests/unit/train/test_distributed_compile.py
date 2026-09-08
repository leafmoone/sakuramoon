from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn
from torch._dynamo import config as dynamo_config
from torch.nn.parallel import DistributedDataParallel

import sakuramoon.model.attention as attention_module
from sakuramoon.model.attention import (
    FA4VarlenGQAAttention,
    fa4_varlen_attention,
)
from sakuramoon.model.block import PackedDiTBlock
from sakuramoon.model.dit import PackedDiT
from sakuramoon.train.runtime import (
    ActualDitFlopCounter,
    SingleGpuBatchRuntime,
    compile_packed_dit_blocks,
    require_distributed_forward_module,
)
from sakuramoon.train.step import TrainableComposite, TrainableCompositeInputs


class _FakeDistributedDataParallel(DistributedDataParallel):
    def __init__(self, module: nn.Module) -> None:
        nn.Module.__init__(self)
        self.module = module


class _CompiledForward(nn.Module):
    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self._orig_mod = original


class _FakePackedAttention(FA4VarlenGQAAttention):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.q_heads = 2
        self.head_dim = 3
        self.q_proj = nn.Linear(5, 6, bias=False)
        self.k_proj = nn.Linear(5, 3, bias=False)
        self.v_proj = nn.Linear(5, 3, bias=False)
        self.content_gate = nn.Linear(5, 5, bias=False)
        self.out_proj = nn.Linear(5, 5, bias=False)


class _FakePackedBlock(PackedDiTBlock):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.attention = _FakePackedAttention()
        self.mlp = nn.Sequential(
            nn.Linear(5, 7, bias=False),
            nn.Linear(7, 5, bias=False),
        )


class _FakeOutputHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(5, 2)


class _FakePackedDiT(PackedDiT):
    def __init__(
        self,
        block_count: int = 2,
        condition_token_count: int = 8,
    ) -> None:
        nn.Module.__init__(self)
        self.active_slot_ids = tuple(range(block_count))
        self.condition_token_count = condition_token_count
        self.input_projection = nn.Linear(2, 5)
        self.conditioner = nn.Sequential(
            nn.Linear(3, 4),
            nn.Linear(4, 5),
        )
        self.blocks = nn.ModuleDict(
            {
                f"slot_{index:02d}": _FakePackedBlock()
                for index in range(block_count)
            }
        )
        self.output_head = _FakeOutputHead()


class _FakeComposite(TrainableComposite):
    def __init__(self, dit: PackedDiT) -> None:
        nn.Module.__init__(self)
        self.dit = dit


def _composite(
    block_count: int = 2,
    condition_token_count: int = 8,
) -> _FakeComposite:
    return _FakeComposite(_FakePackedDiT(block_count, condition_token_count))


def _inputs() -> TrainableCompositeInputs:
    return TrainableCompositeInputs(
        qwen_states=torch.empty(2, 1, 1, 1),
        main_token_indices=torch.zeros(2, 5, dtype=torch.long),
        main_mask=torch.ones(2, 5, dtype=torch.bool),
        main_token_lengths=(3, 5),
        condition_token_indices=torch.zeros(2, 1, dtype=torch.long),
        condition_mask=torch.ones(2, 1, dtype=torch.bool),
        use_null_condition=torch.zeros(2, dtype=torch.bool),
        active_condition_sample_indices=torch.arange(2),
        latents=(
            torch.empty(2, 2, 3),
            torch.empty(2, 1, 4),
        ),
        image_coordinates=(
            torch.empty(6, 2),
            torch.empty(4, 2),
        ),
        timestep=torch.zeros(2),
        size_scale=torch.zeros(2),
        aspect=torch.zeros(2),
        growth_alpha=1.0,
    )


def test_distributed_forward_returns_eager_ddp_sync_handle() -> None:
    composite = _composite()
    ddp = _FakeDistributedDataParallel(composite)

    assert require_distributed_forward_module(composite, ddp) is ddp


def test_compiled_ddp_wrapper_is_rejected() -> None:
    composite = _composite()
    compiled = _CompiledForward(_FakeDistributedDataParallel(composite))

    with pytest.raises(TypeError, match="uncompiled DistributedDataParallel"):
        require_distributed_forward_module(composite, compiled)


def test_ddp_must_wrap_the_original_composite() -> None:
    composite = _composite()
    ddp = _FakeDistributedDataParallel(_composite())

    with pytest.raises(ValueError, match="original trainable composite"):
        require_distributed_forward_module(composite, ddp)


def test_regional_compile_preserves_parameters_and_state_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composite = _composite()
    before_parameter_ids = tuple(id(item) for item in composite.parameters())
    before_state_keys = tuple(composite.state_dict())
    observed: list[tuple[nn.Module, dict[str, object]]] = []

    def fake_compile(
        module: nn.Module,
        *args: object,
        **kwargs: object,
    ) -> None:
        assert not args
        observed.append((module, dict(kwargs)))
        module._compiled_call_impl = module.forward

    monkeypatch.setattr(nn.Module, "compile", fake_compile)
    monkeypatch.setattr(dynamo_config, "suppress_errors", False)
    monkeypatch.setattr(dynamo_config, "fail_on_recompile_limit_hit", False)
    # pin the global recompile_limit so teardown restores it (the
    # install path mutates the global dynamo config directly)
    monkeypatch.setattr(
        dynamo_config, "recompile_limit", dynamo_config.recompile_limit
    )

    blocks = compile_packed_dit_blocks(
        composite,
        backend="inductor",
        mode="default",
        dynamic=True,
        recompile_limit=8,
    )

    assert blocks == tuple(composite.dit.blocks.values())
    assert before_parameter_ids == tuple(id(item) for item in composite.parameters())
    assert before_state_keys == tuple(composite.state_dict())
    assert dynamo_config.fail_on_recompile_limit_hit is True
    assert dynamo_config.recompile_limit == 8
    assert observed == [
        (
            block,
            {
                "backend": "inductor",
                "mode": "default",
                "fullgraph": False,
                "dynamic": True,
            },
        )
        for block in blocks
    ]


def test_regional_compile_requires_dynamic_shapes() -> None:
    with pytest.raises(ValueError, match="dynamic=true"):
        compile_packed_dit_blocks(
            _composite(),
            backend="inductor",
            mode="default",
            dynamic=False,
            recompile_limit=8,
        )


def test_regional_compile_rejects_python_forward_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composite = _composite()
    first = next(iter(composite.dit.blocks.values()))
    first.register_forward_pre_hook(lambda _module, _inputs: None)
    monkeypatch.setattr(dynamo_config, "suppress_errors", False)

    with pytest.raises(RuntimeError, match="must not carry Python forward hooks"):
        compile_packed_dit_blocks(
            composite,
            backend="inductor",
            mode="default",
            dynamic=True,
            recompile_limit=8,
        )


def test_fa2_is_the_only_explicit_eager_compiler_boundary() -> None:
    disabled = tuple(
        name
        for name, value in vars(attention_module).items()
        if callable(value) and getattr(value, "_torchdynamo_disable", False)
    )

    assert getattr(fa4_varlen_attention, "_torchdynamo_disable", False)
    assert disabled == ("fa4_varlen_attention",)


def test_analytic_packed_flops_match_locked_topology_formula() -> None:
    composite = _composite()
    counter = ActualDitFlopCounter(composite.dit)

    # image projections: 10 * (2*2*5 + 2*5*2) = 400
    # conditioner:       2 * (2*3*4 + 2*4*5) = 128
    # block linears:    34 * 2 blocks * 360 = 24,480
    # attention QK+AV: (17^2 + 17^2) * 2 blocks * (4*2*3) = 27,744
    assert counter.count(_inputs()) == 52_752
    assert all(
        not module._forward_pre_hooks and not module._forward_hooks
        for module in composite.dit.modules()
    )


def test_flop_counter_uses_model_condition_token_count() -> None:
    inputs = _inputs()
    four_token_flops = ActualDitFlopCounter(
        _composite(condition_token_count=4).dit
    ).count(inputs)
    eight_token_flops = ActualDitFlopCounter(
        _composite(condition_token_count=8).dit
    ).count(inputs)

    assert eight_token_flops > four_token_flops


def test_flop_counter_rejects_invalid_condition_token_count() -> None:
    composite = _composite(condition_token_count=0)

    with pytest.raises(ValueError, match="positive integer condition_token_count"):
        ActualDitFlopCounter(composite.dit)


def test_flop_counter_rejects_unaccounted_linear() -> None:
    composite = _composite()
    composite.dit.extra_projection = nn.Linear(5, 5)

    with pytest.raises(RuntimeError, match="unaccounted linear"):
        ActualDitFlopCounter(composite.dit)


def test_runtime_source_contains_no_whole_model_compile_path() -> None:
    source = inspect.getsource(SingleGpuBatchRuntime)

    assert "_compiled_dit_forward" not in source
    assert "torch.compile(" not in source
def _pin_dynamo_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the current global dynamo values so monkeypatch teardown
    restores them even though compile_packed_dit_blocks mutates them
    directly."""

    monkeypatch.setattr(
        dynamo_config, "recompile_limit", dynamo_config.recompile_limit
    )
    monkeypatch.setattr(
        dynamo_config,
        "fail_on_recompile_limit_hit",
        dynamo_config.fail_on_recompile_limit_hit,
    )


@pytest.mark.parametrize("limit", [8, 64])
def test_regional_compile_binds_configured_recompile_limit(
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
) -> None:
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
    _pin_dynamo_config(monkeypatch)

    compile_packed_dit_blocks(
        composite,
        backend="inductor",
        mode="default",
        dynamic=True,
        recompile_limit=limit,
    )

    assert dynamo_config.recompile_limit == limit
    assert dynamo_config.fail_on_recompile_limit_hit is True
    assert dynamo_config.suppress_errors is False


@pytest.mark.parametrize(
    "limit",
    [0, -3, True, False, 8.0, 64.5, "8", "64", None],
)
def test_regional_compile_rejects_invalid_recompile_limit(
    monkeypatch: pytest.MonkeyPatch,
    limit: object,
) -> None:
    monkeypatch.setattr(dynamo_config, "suppress_errors", False)
    _pin_dynamo_config(monkeypatch)
    original = dynamo_config.recompile_limit

    with pytest.raises(
        ValueError, match="recompile_limit must be a positive integer"
    ):
        compile_packed_dit_blocks(
            _composite(),
            backend="inductor",
            mode="default",
            dynamic=True,
            recompile_limit=limit,  # type: ignore[arg-type]
        )

    # fail-closed: a rejected call must not touch the global config
    assert dynamo_config.recompile_limit == original


def test_regional_compile_preserves_accumulated_recompile_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = 256
    monkeypatch.setattr(dynamo_config, "accumulated_recompile_limit", sentinel)
    monkeypatch.setattr(dynamo_config, "suppress_errors", False)
    _pin_dynamo_config(monkeypatch)

    def fake_compile(
        module: nn.Module,
        *args: object,
        **kwargs: object,
    ) -> None:
        module._compiled_call_impl = module.forward

    monkeypatch.setattr(nn.Module, "compile", fake_compile)

    compile_packed_dit_blocks(
        _composite(),
        backend="inductor",
        mode="default",
        dynamic=True,
        recompile_limit=64,
    )

    # the governed field is recompile_limit only; the accumulated counter
    # is a runtime-only record and must never be written by the install
    assert dynamo_config.accumulated_recompile_limit == sentinel
    assert dynamo_config.recompile_limit == 64


def test_regional_compile_requires_suppress_errors_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dynamo_config, "suppress_errors", True)
    _pin_dynamo_config(monkeypatch)

    with pytest.raises(
        RuntimeError, match="error suppression must remain disabled"
    ):
        compile_packed_dit_blocks(
            _composite(),
            backend="inductor",
            mode="default",
            dynamic=True,
            recompile_limit=8,
        )
