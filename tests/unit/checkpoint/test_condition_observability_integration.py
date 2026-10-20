"""Real TorchAO CPU checkpoint boundaries for condition LR plus diagnostics.

The broad four-optimizer/float-and-tensor LR matrix lives in
optim/test_condition_learning_rate.py and optim/test_learning_rate_multipliers.py.
Here only the CUDA construction boundary is replaced, not AdamW or grouping.
Moments and the opaque CUDA SR checkpoint bytes are seeded without executing
CUDA/Triton kernels; these tests do not claim vendor GPU update/SR parity.
"""

from __future__ import annotations

# Checkpoint and scheduler boundaries are intentionally exercised directly.
# pyright: reportPrivateUsage=false
import copy
from io import BytesIO
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch.optim import Optimizer
from torchao.optim import AdamW8bit  # pyright: ignore[reportMissingTypeStubs]
from torchao.optim.subclass_8bit import (  # pyright: ignore[reportMissingTypeStubs]
    OptimState8bit,
)

import sakuramoon.checkpoint.load as loader
import sakuramoon.optim.cmuon as core
from sakuramoon.checkpoint.save import _hybrid_optimizer_schema
from sakuramoon.checkpoint.schema import CheckpointError
from sakuramoon.conditioning.condition_tokens import ConditionTokenEncoder
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.config.load import load_config
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.model.dit import DenseDiT
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit
from sakuramoon.optim.condition import condition_learning_rate
from sakuramoon.optim.groups import ParameterAudit, ParameterSpec
from sakuramoon.optim.stochastic_rounding import StochasticRoundingRNG
from sakuramoon.train import production
from sakuramoon.train.condition_diagnostics_stream import (
    compute_condition_activation_facts,
)
from sakuramoon.train.step import TrainableComposite, TrainableCompositeInputs

ROOT = Path(__file__).parents[3]


def _config(*, split: bool, diagnostics: bool, multiplier: float) -> RuntimeConfig:
    loaded = load_config(
        Path("train_g1_condition_optim_ab.example.toml"),
        config_root=ROOT / "config",
        environment={
            "MODELSCOPE_API_TOKEN": "synthetic-modelscope-secret",
            "WANDB_API_KEY": "synthetic-wandb-secret",
        },
    )
    document = loaded.config.model_dump(
        mode="python", exclude_computed_fields=True, exclude_none=True
    )
    optimizer = document["optimizer"]
    optimizer.update(
        name="hybrid_cmuon", cmuon_lr_multiplier=1.3, adamw_lr_multiplier=0.73
    )
    optimizer.pop("cmuon_guard", None)
    if split:
        optimizer["condition"].update(
            # Interleave membership in both original decay/sensitive groups.
            parameter_prefixes=[
                "condition_tokens",
                "dit.input_projection",
                "dit.conditioner.condition_global_projection",
            ],
            lr_multiplier=multiplier,
        )
    else:
        optimizer.pop("condition")
    document["scheduler"].update(
        name="linear_warmup_constant", warmup_updates=8
    )
    document["condition_diagnostics"] = {"enabled": diagnostics}
    return RuntimeConfig.model_validate(document)


def _composite(*, diagnostics: bool) -> TrainableComposite:
    # 64x64 is the smallest production-quantizable BF16 matrix (4096 values).
    # Global-condition/output-head sensitive parameters remain native FP32.
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        torch.manual_seed(612)  # pyright: ignore[reportUnknownMemberType]
        return TrainableComposite(
            dit=DenseDiT(
                depth=1, active_slot_ids=(0,), input_channels=64, hidden_size=64,
                intermediate_size=64, q_heads=2, kv_heads=1, head_dim=32,
                rope_nope_dim=0, rope_y_dim=16, rope_x_dim=16,
                rope_position_scale=1.0, rope_theta=10.0, norm_eps=1e-6,
                timestep_dim=64, size_dim=64, aspect_dim=64,
                condition_hidden_size=64, stable_slot_count=1,
                modulation_chunks=6, final_modulation_size=128, out_channels=64,
                condition_token_count=2, modality_init_std=0.02,
                linear_dtype=torch.bfloat16, sensitive_dtype=torch.float32,
                projection_bias=False, attention_dropout=0.0, mlp_dropout=0.0,
                output_weight_zero_init=True, output_bias_zero_init=True,
            ),
            text=TextConditioner(
                input_size=64, adapter_size=64, output_size=64, groups=2,
                attention_heads=2, norm_eps=1e-6, mix_gate_init=0.1,
                layer_scale_init=1.0, projection_bias=False,
                linear_dtype=torch.bfloat16, sensitive_dtype=torch.float32,
            ),
            condition_tokens=ConditionTokenEncoder(
                input_size=64, hidden_size=64, intermediate_size=64,
                output_size=64, token_count=2, attention_heads=2,
                norm_eps=1e-6, init_std=0.02, projection_bias=False,
                linear_dtype=torch.bfloat16, sensitive_dtype=torch.float32,
            ),
            condition_diagnostics=diagnostics,
        )


def _cpu_torchao(
    specs: tuple[ParameterSpec, ...], **kwargs: Any
) -> IsolatedAdamW8bit:
    """Use the real TorchAO constructor, bypassing only CUDA device/RNG setup."""
    audit = ParameterAudit(specs)
    assert all(spec.parameter.device.type == "cpu" for spec in specs)
    assert all(
        spec.parameter.numel() >= 4096 and spec.parameter.numel() % 256 == 0
        for spec in audit.decay
    )
    groups = [
        {
            "params": [spec.parameter for spec in subset],
            "param_names": [spec.name for spec in subset],
            "group_name": name,
            "weight_decay": kwargs[decay_key],
        }
        for name, subset, decay_key in (
            ("matrix_decay", audit.decay, "matrix_weight_decay"),
            ("sensitive_no_decay", audit.sensitive, "sensitive_weight_decay"),
        )
    ]
    inner = cast(
        Optimizer,
        AdamW8bit(
            groups, lr=kwargs["lr"], betas=kwargs["betas"], eps=kwargs["eps"],
            weight_decay=0.0, amsgrad=False, block_size=kwargs["block_size"],
            bf16_stochastic_round=kwargs["bf16_stochastic_round"],
        ),
    )
    # CUDA metadata is the real checkpoint contract; no CUDA API is called.
    rng = StochasticRoundingRNG(
        torch.device("cuda:0"), torch.arange(32, dtype=torch.uint8)
    )
    return IsolatedAdamW8bit(inner, audit, rng)


@pytest.fixture(autouse=True)
def _real_cpu_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "_build_adamw8bit_for_specs", _cpu_torchao)


def _build(config: RuntimeConfig) -> tuple[TrainableComposite, core.HybridCMuon]:
    assert config.condition_diagnostics is not None
    module = _composite(diagnostics=config.condition_diagnostics.enabled)
    optimizer = production._build_optimizer(config, module)
    assert isinstance(optimizer, core.HybridCMuon)
    assert isinstance(optimizer.optimizer, AdamW8bit)
    return module, optimizer


def _seed_state(optimizer: core.HybridCMuon) -> None:
    for index, spec in enumerate(optimizer.adamw.audit.specs):
        moments: list[torch.Tensor] = []
        # TorchAO quantizes by size/alignment, including large FP32 sensitive weights.
        quantized = spec.parameter.numel() >= 4096 and spec.parameter.numel() % 256 == 0
        for signed in (True, False):
            if quantized:
                moment = cast(Any, OptimState8bit).zeros(
                    spec.parameter.shape, signed=signed, block_size=256, device="cpu"
                )
                moment.codes.copy_(
                    (torch.arange(moment.codes.numel()).reshape(moment.codes.shape)
                     + 17 * index).remainder(256).to(torch.uint8)
                )
                moment.scale.copy_(
                    (torch.arange(moment.scale.numel()).reshape(moment.scale.shape)
                     + index + 1).float() / 1024
                )
            else:
                values = torch.linspace(
                    -0.05, 0.05, spec.parameter.numel(), dtype=torch.float32
                ).reshape(spec.parameter.shape)
                moment = values + index / 128 if signed else values.square() + 0.01
            moments.append(moment)
        optimizer.optimizer.state[spec.parameter] = {
            "step": torch.tensor(float(index % 3 + 1)),
            "exp_avg": moments[0],
            "exp_avg_sq": moments[1],
        }
    for index, momentum in enumerate(optimizer._momenta.values()):
        momentum.copy_(
            (torch.arange(momentum.numel()).reshape(momentum.shape)
             + index).remainder(13).to(momentum.dtype) / 32
        )
    optimizer.sr_rng.state.copy_(torch.arange(32, dtype=torch.uint8).flip(0))
    state_audit = optimizer.adamw.audit_state()
    assert all(entry.initialized for entry in state_audit)
    assert {entry.state_class for entry in state_audit} == {"OptimState8bit", "Tensor"}


def _snapshot(document: dict[str, Any]) -> dict[str, Any]:
    # OptimState8bit serialization is supported; deepcopy's aten.clone is not.
    with BytesIO() as buffer:
        torch.save(document, buffer)
        buffer.seek(0)
        return cast(
            dict[str, Any], torch.load(buffer, map_location="cpu", weights_only=False)
        )


def _topology(optimizer: core.HybridCMuon) -> list[tuple[Any, ...]]:
    return [
        (
            group["group_name"], group.get("condition_lr_group"),
            tuple(group["param_names"]),
            tuple(id(parameter) for parameter in group["params"]),
            tuple(parameter.dtype for parameter in group["params"]),
        )
        for group in optimizer.optimizer.param_groups
    ]


def _by_name(document: dict[str, Any]) -> dict[str, Any]:
    return {
        name: document["state"][pid]
        for group in document["param_groups"]
        for name, pid in zip(group["param_names"], group["params"], strict=True)
        if pid in document["state"]
    }


def _assert_exact(left: Any, right: Any) -> None:
    assert type(left) is type(right)
    if isinstance(left, OptimState8bit):
        quantized = cast(Any, left)
        assert quantized.dtype == right.dtype
        assert quantized.shape == right.shape
        assert quantized.signed is right.signed
        assert quantized.block_size == right.block_size == 256
        for field in ("codes", "scale", "qmap"):
            _assert_exact(getattr(quantized, field), getattr(right, field))
    elif isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        assert left.device == right.device
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        mapping = cast(dict[str, Any], left)
        assert mapping.keys() == right.keys()
        for key, value in mapping.items():
            _assert_exact(value, right[key])
    elif isinstance(left, (list, tuple)):
        sequence = cast(list[Any] | tuple[Any, ...], left)
        assert len(sequence) == len(right)
        for before, after in zip(sequence, right, strict=True):
            _assert_exact(before, after)
    else:
        assert left == right


def _diagnostic_probe(module: TrainableComposite, optimizer: core.HybridCMuon) -> None:
    before = _snapshot(optimizer.state_dict())
    parameters = copy.deepcopy(module.state_dict())
    identities = tuple(id(parameter) for parameter in module.parameters())
    topology = _topology(optimizer)
    training_rng = torch.get_rng_state().clone()
    inputs = TrainableCompositeInputs(
        qwen_states=torch.linspace(-0.5, 0.5, 2 * 4 * 7 * 64)
        .reshape(2, 4, 7, 64).to(torch.bfloat16),
        main_token_indices=torch.tensor([[0, 1], [0, 1]]),
        main_mask=torch.ones(2, 2, dtype=torch.bool), main_token_lengths=(2, 2),
        condition_token_indices=torch.tensor([[2, 3], [2, 3]]),
        condition_mask=torch.ones(2, 2, dtype=torch.bool),
        use_null_condition=torch.tensor([False, True]),
        active_condition_sample_indices=torch.tensor([0]),
        latents=(torch.zeros(64, 1, 1), torch.zeros(64, 1, 1)),
        image_coordinates=(torch.zeros(1, 3), torch.zeros(1, 3)),
        timestep=torch.tensor([0.25, 0.75]), size_scale=torch.ones(2),
        aspect=torch.zeros(2), growth_alpha=1.0,
    )
    with torch.no_grad():
        module.disable_condition_diagnostics()
        _, disabled = module.forward_conditioning(inputs)
        assert module.last_conditioning is None
        module.enable_condition_diagnostics()
        _, enabled = module.forward_conditioning(inputs)
        assert module.last_conditioning is enabled
        facts = compute_condition_activation_facts(module, inputs, enabled)
        assert facts["active_samples"] == facts["null_samples"] == 1
        _assert_exact(disabled.tokens, enabled.tokens)
        module.disable_condition_diagnostics()
    assert module.last_conditioning is None
    assert tuple(id(parameter) for parameter in module.parameters()) == identities
    assert _topology(optimizer) == topology
    _assert_exact(parameters, module.state_dict())
    _assert_exact(before, optimizer.state_dict())
    assert torch.equal(training_rng, torch.get_rng_state())


def _assert_rates(optimizer: core.HybridCMuon, config: RuntimeConfig, update: int) -> None:
    base = production._config_learning_rate(config, update)
    assert optimizer.learning_rate == base
    assert optimizer.cfg.lr == base * config.optimizer.cmuon_lr_multiplier
    policy = config.optimizer.condition
    assert policy is not None
    rest_rate = base * config.optimizer.adamw_lr_multiplier
    condition_rate = rest_rate * policy.lr_multiplier
    assert condition_learning_rate(optimizer) == pytest.approx(condition_rate)
    assert production._optimizer_learning_rate_matches(optimizer, base, config)
    for group in optimizer.optimizer.param_groups:
        expected = condition_rate if group["condition_lr_group"] else rest_rate
        if isinstance(group["lr"], torch.Tensor):
            assert torch.equal(group["lr"], torch.full_like(group["lr"], expected))
        else:
            assert group["lr"] == expected


@pytest.mark.parametrize("source_split", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_real_torchao_condition_restore_with_diagnostics(
    tmp_path: Path, source_split: bool, diagnostics: bool
) -> None:
    old_config = _config(split=source_split, diagnostics=not diagnostics, multiplier=0.23)
    _, source = _build(old_config)
    _seed_state(source)
    saved = _snapshot(cast(dict[str, Any], source.state_dict()))
    schema = _hybrid_optimizer_schema(source)
    torch.save(saved, tmp_path / "optimizer.pt")

    config = _config(split=True, diagnostics=diagnostics, multiplier=0.41)
    module, current = _build(config)
    # Both decay policy and all effective LRs must come from the NEW config.
    for index, group in enumerate(current.optimizer.param_groups):
        group["weight_decay"] = (index + 1) / 1000
        group["lr"] = torch.tensor(group["lr"], dtype=torch.float32)
    base = production._config_learning_rate(config, 4)
    production._set_optimizer_learning_rate(current, base, config)
    topology = _topology(current)
    runtime_groups = copy.deepcopy(current.optimizer.state_dict()["param_groups"])
    assert len({group["lr"].data_ptr() for group in current.optimizer.param_groups}) == len(topology)
    names = [name for group in topology for name in group[2]]
    assert names == [
        name for group in saved["optimizer"]["param_groups"]
        for name in group["param_names"]
    ]
    assert len(names) == len(set(names)) == len(current.routing.adamw_specs)
    assert len(topology) > 4  # Not condition-at-end buckets or just two groups.
    for previous, following in pairwise(topology):
        if previous[0] == following[0]:
            assert previous[1] is not following[1]

    loader._load_hybrid_state_exact(
        tmp_path, schema, current, source.sr_rng.state_dict(), successful_updates=3
    )
    restored = cast(dict[str, Any], current.state_dict())
    assert _topology(current) == topology
    _assert_exact(runtime_groups, restored["optimizer"]["param_groups"])
    _assert_exact(_by_name(saved["optimizer"]), _by_name(restored["optimizer"]))
    for key in ("routing", "cmuon", "sr_rng", "transition"):
        _assert_exact(saved[key], restored[key])
    assert saved.keys() == restored.keys()  # Diagnostics/multipliers are not persisted.
    assert source.adamw.audit_state() == current.adamw.audit_state()
    _assert_rates(current, config, 4)

    _diagnostic_probe(module, current)
    scheduler = production._SuccessfulUpdateLrScheduler(
        config, current, production._config_learning_rate,
        restored_successful_update=3, fresh=False,
    )
    # Reapplying an attempt rate cannot compound either multiplier.
    production._set_optimizer_learning_rate(current, base, config)
    _assert_rates(current, config, 4)
    scheduler(4)
    _assert_rates(current, config, 5)
    _assert_exact(_by_name(restored["optimizer"]), _by_name(current.optimizer.state_dict()))
    _assert_exact(restored["sr_rng"], current.sr_rng.state_dict())


def test_neutral_b_resume_retains_tags_and_cannot_be_disabled(tmp_path: Path) -> None:
    _, source = _build(_config(split=True, diagnostics=True, multiplier=0.23))
    _seed_state(source)
    schema = _hybrid_optimizer_schema(source)
    torch.save(source.state_dict(), tmp_path / "optimizer.pt")
    neutral_config = _config(split=True, diagnostics=False, multiplier=1.0)
    _, neutral = _build(neutral_config)
    base = production._config_learning_rate(neutral_config, 4)
    production._set_optimizer_learning_rate(neutral, base, neutral_config)
    loader._load_hybrid_state_exact(
        tmp_path, schema, neutral, source.sr_rng.state_dict(), successful_updates=3
    )
    assert [entry[:3] for entry in _topology(neutral)] == [
        entry[:3] for entry in _topology(source)
    ]
    _assert_rates(neutral, neutral_config, 4)
    _assert_exact(
        _by_name(source.optimizer.state_dict()), _by_name(neutral.optimizer.state_dict())
    )
    # docs/condition-optimizer-ab.md: neutral rollback keeps B enabled; turning
    # diagnostics on cannot make a tagged -> legacy topology downgrade legal.
    _, legacy = _build(_config(split=False, diagnostics=True, multiplier=1.0))
    with pytest.raises(CheckpointError, match="group count"):
        loader._load_hybrid_state_exact(
            tmp_path, schema, legacy, source.sr_rng.state_dict(), successful_updates=3
        )
    assert not legacy.optimizer.state
