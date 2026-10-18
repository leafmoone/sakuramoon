"""Focused unit tests for the opt-in per-rank condition diagnostics stream."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from torch import nn

from sakuramoon.conditioning.condition_tokens import ConditionTokenOutput
from sakuramoon.conditioning.global_condition import GlobalConditioner
from sakuramoon.config.schema import (
    ConditionDiagnosticsConfig,
    RuntimeConfig,
)
from sakuramoon.optim.clip import ClipResult
from sakuramoon.train.condition_diagnostics_stream import (
    CONDITION_DIAGNOSTICS_SCHEMA_VERSION,
    ConditionDiagnosticsRecorder,
    # Private helpers are exercised directly: unit tests of the stream's
    # numeric contract (mirrors the repo's targeted-ignore convention).
    _finite_float_or_none,  # pyright: ignore[reportPrivateUsage]
    _grad_share,  # pyright: ignore[reportPrivateUsage]
    _masked_rms,  # pyright: ignore[reportPrivateUsage]
    _per_sample_rms_mean,  # pyright: ignore[reportPrivateUsage]
    _slot_cosine,  # pyright: ignore[reportPrivateUsage]
    build_condition_diagnostic_record,
    compute_condition_activation_facts,
    diagnostics_file_name,
)
from sakuramoon.train.step import (
    SingleGpuStep,
    SingleGpuUpdateResult,
    SingleGpuUpdateState,
    TrainableComposite,
    TrainableCompositeInputs,
)
from sakuramoon.train.tensor_metrics import (
    TrainingSamplingError,
    tensor_cosine,
    tensor_cosine_rows,
)

if TYPE_CHECKING:
    # Type-only: sakuramoon.train.runtime pulls the full training import
    # graph (webdataset, ...) that the CPU unit gate does not install.
    from sakuramoon.train.runtime import SuccessfulTrainingObservation

LINUX_ONLY = pytest.mark.skipif(
    sys.platform != "linux",
    reason="DurableJsonlSink uses POSIX open flags (O_NOFOLLOW)",
)


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


def test_condition_diagnostics_config_defaults_off() -> None:
    # The RuntimeConfig field is optional and absent by default: no section
    # in the TOML means the feature is fully off.
    assert RuntimeConfig.model_fields["condition_diagnostics"].default is None
    default = ConditionDiagnosticsConfig()
    assert default.enabled is False
    assert default.log_every_n == 1
    # One fsync per 100 records by default: this stream fsyncs on the
    # training thread (unlike the async main metrics stream), so the
    # per-record fsync is an explicit opt-in via flush_every_records = 1.
    assert default.flush_every_records == 100
    on = ConditionDiagnosticsConfig(enabled=True, log_every_n=4)
    assert on.enabled is True
    assert on.log_every_n == 4


def test_condition_diagnostics_config_rejects_bad_cadence() -> None:
    with pytest.raises(ValueError, match="log_every_n"):
        ConditionDiagnosticsConfig(enabled=True, log_every_n=0)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_diagnostics_file_name_carries_version() -> None:
    assert diagnostics_file_name() == "condition-diagnostics-v1.jsonl"
    assert diagnostics_file_name() == (
        f"condition-diagnostics-v{CONDITION_DIAGNOSTICS_SCHEMA_VERSION}.jsonl"
    )
    assert diagnostics_file_name(2) == "condition-diagnostics-v2.jsonl"


def test_masked_rms_excludes_masked_elements() -> None:
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    keep = torch.tensor([[True, False], [False, True]])
    result = _masked_rms(values, keep)
    assert result.item() == pytest.approx(math.sqrt((1.0 + 16.0) / 2.0))
    with pytest.raises(ValueError, match="no selected elements"):
        _masked_rms(values, torch.zeros(2, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="shape"):
        _masked_rms(values, torch.ones(4, dtype=torch.bool))


def test_per_sample_rms_mean() -> None:
    # per-row RMS: sqrt(mean(squares)); rows [3,4] -> sqrt(12.5), [6,8] -> sqrt(50)
    values = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
    expected = (math.sqrt(12.5) + math.sqrt(50.0)) / 2.0
    assert _per_sample_rms_mean(values).item() == pytest.approx(expected)


def test_grad_share() -> None:
    assert _grad_share(0.9, 2.0) == pytest.approx(0.45)
    assert _grad_share(0.0, 0.0) is None
    with pytest.raises(ValueError, match="finite"):
        _grad_share(float("inf"), 1.0)


def test_finite_float_or_none_advisory_policy() -> None:
    # Reuses tensor_metrics.metric_float; only the policy differs —
    # nonfinite values become ``null`` instead of raising.
    assert _finite_float_or_none("x", torch.tensor(1.5)) == pytest.approx(1.5)
    assert _finite_float_or_none("x", torch.tensor(float("nan"))) is None
    assert _finite_float_or_none("x", torch.tensor(float("inf"))) is None
    # The RMS sign invariant is kept on top of metric_float (nonfinite
    # values are advisory and never reach the sign check).
    with pytest.raises(ValueError, match="must not be negative"):
        _finite_float_or_none("x_rms", torch.tensor(-1.0))
    assert _finite_float_or_none("x_rms", torch.tensor(float("nan"))) is None


def test_slot_cosine_values() -> None:
    a = torch.tensor([[3.0, 4.0]])
    zero = torch.zeros(1, 2)
    assert _slot_cosine(a, a, zero, torch.tensor([0])) == pytest.approx(1.0)
    b = torch.tensor([[4.0, 3.0]])  # not a multiple -> cosine 24/25
    assert _slot_cosine(a, b, zero, torch.tensor([0])) == pytest.approx(
        24.0 / 25.0
    )
    # Zero-norm vector makes the cosine undefined -> None, not an exception.
    assert _slot_cosine(zero, a, zero, torch.tensor([0])) is None
    # No active samples -> None.
    assert _slot_cosine(a, a, zero, torch.empty(0, dtype=torch.long)) is None
    # The bias shifts both sides identically (shared modulation + per-slot bias).
    bias = torch.tensor([1.0, -2.0])
    expected = _slot_cosine(
        a + bias, b + bias, torch.zeros(2), torch.tensor([0])
    )
    direct = torch.tensor(
        [
            float(
                torch.dot(a[0] + bias, b[0] + bias)
                / (
                    torch.sqrt((a[0] + bias).square().sum())
                    * torch.sqrt((b[0] + bias).square().sum())
                )
            )
        ]
    )
    assert expected == pytest.approx(direct.item())


def _rowwise_slot_cosine_reference(
    shared_with_condition: torch.Tensor,
    shared_base_only: torch.Tensor,
    bias: torch.Tensor,
    active_indices: torch.Tensor,
) -> float | None:
    """The 6a85df8 per-row implementation, kept as the equivalence
    reference: row-by-row ``tensor_cosine`` calls + per-row ``float()``
    conversion + Python-sequential double accumulation."""

    if active_indices.numel() == 0:
        return None
    bias_f = bias.float()
    left = shared_with_condition.index_select(0, active_indices).float()
    left = left + bias_f
    right = shared_base_only.index_select(0, active_indices).float()
    right = right + bias_f
    rows = left.shape[0]
    total = 0.0
    for row in range(rows):
        total += float(tensor_cosine(left[row], right[row]))
    return total / rows


def test_slot_cosine_vectorized_matches_rowwise_reference() -> None:
    """Numerical-equivalence regression (Phase 1 fix 2a, reviewer
    rework): the batched ``_slot_cosine`` (shared primitive
    ``tensor_cosine_rows``) must stay NUMERICALLY equivalent to the old
    per-row scalar path (row-by-row ``tensor_cosine``) on the same
    inputs.  Bitwise identity does NOT hold — the batched reduction
    kernels order float32 additions differently from per-row
    ``torch.dot``; the measured bound is 6.6e-7 absolute over 5671
    random comparisons (2000 seeds/shapes/scales, CPU), so the bound
    asserted here is 1e-6 (10x margin).  That is three orders of
    magnitude below the ~4e-3 weight-update drift these diagnostics
    exist to track."""

    generator = torch.Generator()
    generator.manual_seed(2026)
    for batch, width in [
        (1, 64),
        (5, 32),
        (20, 64),  # production local_batch scale
        (23, 255),
    ]:
        for scale in (1e-4, 1.0, 1e3):
            left = torch.randn(batch, width, generator=generator) * scale
            right = torch.randn(batch, width, generator=generator) * scale
            bias = torch.randn(width, generator=generator)
            for start in range(0, batch, 3):
                active = torch.arange(start, min(start + 4, batch))
                reference = _rowwise_slot_cosine_reference(
                    left, right, bias, active
                )
                actual = _slot_cosine(left, right, bias, active)
                assert reference == pytest.approx(actual, abs=1e-6)


def test_tensor_cosine_rows_matches_scalar_and_error_contract() -> None:
    """Shared batched primitive contract (Phase 1 fix 2a, reviewer
    rework): per-row values within the batched float32-reduction bound
    of the scalar ``tensor_cosine``; same zero-norm error contract; the
    same invalid-shape rejection; returns ``[N]``."""

    left = torch.tensor([[3.0, 4.0, 0.5], [1.0, 2.0, 0.25], [6.0, 8.0, 1.0]])
    right = torch.tensor([[3.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0]])
    rows = tensor_cosine_rows(left, right)
    assert rows.shape == (3,)
    for i in range(3):
        assert rows[i].item() == pytest.approx(
            tensor_cosine(left[i], right[i]).item(), abs=1e-6
        )
    # zero-norm row: same error contract as the scalar primitive
    zero = left.clone()
    zero[1] = 0.0
    with pytest.raises(TrainingSamplingError, match="zero norm"):
        tensor_cosine_rows(zero, right)
    # invalid pairs
    with pytest.raises(TrainingSamplingError, match="non-empty"):
        tensor_cosine_rows(left, left[:, :2])
    empty = left[0:0]
    with pytest.raises(TrainingSamplingError, match="non-empty"):
        tensor_cosine_rows(empty, empty)


def test_slot_cosine_nonfinite_inputs_are_advisory_none() -> None:
    """Fix 1: nonfinite input or result -> None (advisory), never a value
    that could reach the JSONL writer."""

    a = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
    zero = torch.zeros(2)
    nan_left = a.clone()
    nan_left[0, 0] = float("nan")
    assert _slot_cosine(nan_left, a, zero, torch.tensor([0, 1])) is None
    inf_right = a.clone()
    inf_right[1, 1] = float("inf")
    assert _slot_cosine(a, inf_right, zero, torch.tensor([0, 1])) is None
    nan_bias = torch.tensor([1.0, float("nan")])
    assert _slot_cosine(a, a, nan_bias, torch.tensor([0, 1])) is None
    # Finite input with a finite result is still a plain float.
    assert _slot_cosine(a, a, zero, torch.tensor([0, 1])) == pytest.approx(1.0)


def test_record_builder_validates_slot_cosine_finiteness() -> None:
    """Fix 1: the finiteness check covers ALL fact keys (including
    slot_cosine_*), not just the base keys; a valid record must pass
    json.dumps(allow_nan=False)."""

    update = _mini_update_result(pre_condition=0.9, post_condition=0.7)
    for bad in (float("nan"), float("inf"), float("-inf")):
        polluted = dict(_mini_facts(), slot_cosine_03=bad)  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="must be finite or null"):
            build_condition_diagnostic_record(
                successful_update=1,
                rank=0,
                world_size=1,
                update=update,
                activation_facts=polluted,
                slot_ids=(3, 7),
            )
    # A fully-finite record serializes under the sink's allow_nan=False.
    valid = build_condition_diagnostic_record(
        successful_update=1,
        rank=0,
        world_size=1,
        update=update,
        activation_facts=_mini_facts(),
        slot_ids=(3, 7),
    )
    json.dumps(valid, allow_nan=False)
    # ... and null slots (undefined cosine) are fine, as per the contract.
    with_null = build_condition_diagnostic_record(
        successful_update=1,
        rank=0,
        world_size=1,
        update=update,
        activation_facts=dict(_mini_facts(), slot_cosine_07=None),  # type: ignore[dict-arg]
        slot_ids=(3, 7),
    )
    json.dumps(with_null, allow_nan=False)


# ---------------------------------------------------------------------------
# Activation facts (small real conditioner, synthetic inputs)
# ---------------------------------------------------------------------------


def _mini_conditioner() -> GlobalConditioner:
    conditioner = GlobalConditioner(
        timestep_dim=8,
        size_dim=8,
        aspect_dim=8,
        hidden_dim=32,
        model_dim=16,
        slot_count=16,
        active_slot_ids=(3, 7),
        modulation_chunks=6,
        final_modulation_size=32,
        norm_eps=1e-6,
    )
    # Zero-initialized weights would make every modulation vector zero;
    # re-seed for a meaningful cosine signal (values stay deterministic).
    generator = torch.Generator()
    generator.manual_seed(7)  # pyright: ignore[reportUnknownMemberType]
    with torch.no_grad():
        for name, parameter in conditioner.named_parameters():
            if (
                name.startswith(("condition_mlp.", "block_biases."))
                or name == "shared_block_projection.weight"
            ):
                nn.init.normal_(parameter, generator=generator)
    return conditioner


def _mini_composite(
    conditioner: GlobalConditioner, *, active: bool
) -> SimpleNamespace:
    encoder = SimpleNamespace(
        input_size=24,
        null_tokens=nn.Parameter(
            torch.arange(24.0).reshape(3, 8) * 0.25 + 0.5
        ),
    )
    # A real TrainableComposite needs a full DiT; the compute function only
    # touches .condition_tokens / .dit.conditioner / .dit.active_slot_ids.
    composite = SimpleNamespace(
        condition_tokens=encoder,
        dit=SimpleNamespace(
            conditioner=conditioner,
            active_slot_ids=(3, 7),
        ),
    )
    return composite


def _mini_inputs(batch: int) -> TrainableCompositeInputs:
    # C=3 condition slots, input_size=24, 7 qwen layers, L=6 positions.
    generator = torch.Generator()
    generator.manual_seed(11)  # pyright: ignore[reportUnknownMemberType]
    qwen_states = torch.randn(batch, 6, 7, 24, generator=generator)
    # All samples have all 3 condition positions valid (mask all True).
    condition_token_indices = torch.randint(0, 6, (batch, 3))
    condition_mask = torch.ones(batch, 3, dtype=torch.bool)
    return TrainableCompositeInputs(
        qwen_states=qwen_states,
        main_token_indices=torch.zeros(batch, 2, dtype=torch.long),
        main_mask=torch.ones(batch, 2, dtype=torch.bool),
        main_token_lengths=(2,) * batch,
        condition_token_indices=condition_token_indices,
        condition_mask=condition_mask,
        use_null_condition=torch.zeros(batch, dtype=torch.bool),
        active_condition_sample_indices=torch.arange(batch, dtype=torch.long),
        latents=(torch.randn(batch, 4, 8, generator=generator),),
        image_coordinates=(torch.zeros(batch, 4, 2),),
        timestep=torch.rand(batch, generator=generator),
        size_scale=torch.rand(batch, generator=generator),
        aspect=torch.rand(batch, generator=generator),
        growth_alpha=1.0,
    )


def test_compute_condition_activation_facts_matches_manual_math() -> None:
    conditioner = _mini_conditioner()
    composite = _mini_composite(conditioner, active=True)
    batch = 4
    inputs = _mini_inputs(batch)
    # Active samples: 0 and 2; null samples: 1 and 3.
    active_mask = torch.tensor([True, False, True, False])
    tokens = torch.randn(4, 3, 16)
    condition = ConditionTokenOutput(
        tokens=tokens,
        mask=torch.ones(4, 3, dtype=torch.bool),
        active_mask=active_mask,
    )
    facts = compute_condition_activation_facts(
        cast(TrainableComposite, composite), inputs, condition
    )

    assert facts["active_samples"] == 2
    assert facts["null_samples"] == 2

    # Active pre-norm RMS: the encoder's own gather (valid slots only,
    # here the mask is all True so every gathered element counts).
    gathered = inputs.qwen_states[[0, 2]].gather(
        1, inputs.condition_token_indices[[0, 2]][:, :, None, None].expand(-1, -1, 7, 24)
    )
    assert facts["condition_token_rms_active_pre_norm"] == pytest.approx(
        float(gathered.float().square().mean().sqrt())
    )
    # Active post-norm RMS: encoder output tokens of active samples.
    assert facts["condition_token_rms_active_post_norm"] == pytest.approx(
        float(tokens[[0, 2]].float().square().mean().sqrt())
    )
    # Null pre-norm RMS: the learned null_tokens parameter.
    assert facts["condition_token_rms_null_pre_norm"] == pytest.approx(
        float(
            composite.condition_tokens.null_tokens.detach()
            .float()
            .square()
            .mean()
            .sqrt()
        )
    )
    # Null post-norm RMS: tokens of null samples (identity path).
    assert facts["condition_token_rms_null_post_norm"] == pytest.approx(
        float(tokens[[1, 3]].float().square().mean().sqrt())
    )
    # Weight RMS of the global-path RMSNorm.
    assert facts["condition_global_norm_weight_rms"] == pytest.approx(
        float(
            conditioner.condition_global_norm.weight.detach().float()
            .square()
            .mean()
            .sqrt()
        )
    )

    # Global path: recompute through the SAME module and compare.
    with torch.no_grad():
        global_output = conditioner(
            inputs.timestep,
            inputs.size_scale,
            inputs.aspect,
            condition.tokens,
            active_mask,
            (3, 7),
        )
        per_row_base = (
            global_output.base_hidden.float().reshape(4, -1)
            .square()
            .mean(dim=1)
            .sqrt()
        )
        assert facts["base_hidden_rms"] == pytest.approx(
            float(per_row_base.mean())
        )
        per_row_residual = (
            global_output.condition_residual[[0, 2]]
            .float()
            .reshape(2, -1)
            .square()
            .mean(dim=1)
            .sqrt()
        )
        assert facts["condition_residual_rms"] == pytest.approx(
            float(per_row_residual.mean())
        )
        shared_with = conditioner.shared_block_projection(
            global_output.total_hidden
        )
        shared_base = conditioner.shared_block_projection(
            global_output.base_hidden
        )
        for slot_id in (3, 7):
            bias = conditioner.block_biases[f"slot_{slot_id:02d}"]
            left = (shared_with[[0, 2]] + bias).reshape(2, -1)
            right = (shared_base[[0, 2]] + bias).reshape(2, -1)
            expected = float(
                torch.einsum("bd,bd->b", left, right)
                .div(
                    torch.sqrt(left.square().sum(dim=1))
                    * torch.sqrt(right.square().sum(dim=1))
                )
                .mean()
            )
            assert facts[f"slot_cosine_{slot_id:02d}"] == pytest.approx(expected)
        # Null rows are structurally zero residual; the record must agree.
        assert bool(
            (global_output.condition_residual[[1, 3]] == 0).all()
        )


def test_compute_condition_activation_facts_all_null_microbatch() -> None:
    conditioner = _mini_conditioner()
    composite = _mini_composite(conditioner, active=True)
    batch = 2
    inputs = _mini_inputs(batch)
    condition = ConditionTokenOutput(
        tokens=torch.randn(batch, 3, 16),
        mask=torch.ones(batch, 3, dtype=torch.bool),
        active_mask=torch.zeros(batch, dtype=torch.bool),
    )
    facts = compute_condition_activation_facts(
        cast(TrainableComposite, composite), inputs, condition
    )
    assert facts["active_samples"] == 0
    assert facts["null_samples"] == 2
    assert facts["condition_token_rms_active_pre_norm"] is None
    assert facts["condition_token_rms_active_post_norm"] is None
    assert facts["condition_residual_rms"] is None
    assert facts["slot_cosine_03"] is None
    assert facts["slot_cosine_07"] is None
    # Null-side facts stay defined.
    assert isinstance(facts["condition_token_rms_null_post_norm"], float)
    assert isinstance(facts["base_hidden_rms"], float)


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------


def _mini_update_result(
    *,
    pre_condition: float,
    post_condition: float,
    successful_update: int = 1,
) -> SingleGpuUpdateResult:
    return SingleGpuUpdateResult(
        mean_loss=torch.tensor(1.25),
        clip=ClipResult(
            pre_clip_norm=torch.tensor(2.0),
            post_clip_norm=torch.tensor(1.5),
            coefficient=torch.tensor(0.75),
        ),
        condition_encoder_grad_norm=torch.tensor(0.5),
        condition_global_projection_grad_norm=torch.tensor(0.25),
        microbatches=2,
        effective_samples=4,
        state=SingleGpuUpdateState(
            attempted_updates=successful_update,
            successful_updates=successful_update,
            effective_samples=4 * successful_update,
        ),
        condition_grad_norm_pre_clip=torch.tensor(pre_condition),
        condition_grad_norm_post_clip=torch.tensor(post_condition),
    )


def _mini_facts() -> dict[str, float | int | None]:
    return {
        "active_samples": 2,
        "null_samples": 2,
        "condition_token_rms_active_pre_norm": 1.1,
        "condition_token_rms_active_post_norm": 1.2,
        "condition_token_rms_null_pre_norm": 0.7,
        "condition_token_rms_null_post_norm": 0.7,
        "condition_global_norm_weight_rms": 0.5,
        "base_hidden_rms": 2.0,
        "condition_residual_rms": 0.3,
        "slot_cosine_03": 0.95,
        "slot_cosine_07": None,
    }


def test_build_record_versions_and_grad_shares() -> None:
    record = build_condition_diagnostic_record(
        successful_update=17,
        rank=1,
        world_size=2,
        update=_mini_update_result(pre_condition=0.9, post_condition=0.7),
        activation_facts=_mini_facts(),
        slot_ids=(3, 7),
    )
    assert record["schema_version"] == 1
    assert record["successful_update"] == 17
    assert record["rank"] == 1
    assert record["world_size"] == 2
    assert record["microbatches"] == 2
    assert record["effective_samples"] == 4
    # Both clip stages are recorded with their own ratio.
    assert record["condition_grad_share_pre_clip"] == pytest.approx(0.9 / 2.0)
    assert record["condition_grad_share_post_clip"] == pytest.approx(0.7 / 1.5)
    assert record["total_grad_norm_pre_clip"] == pytest.approx(2.0)
    assert record["total_grad_norm_post_clip"] == pytest.approx(1.5)
    # Fixed-key discipline: every fact key is present verbatim.
    for key, value in _mini_facts().items():
        assert record[key] == value
    assert "recorded_at_unix_ns" in record


def test_build_record_rejects_bad_inputs() -> None:
    update = _mini_update_result(pre_condition=0.9, post_condition=0.7)
    facts = _mini_facts()
    # Missing condition grad norms.
    missing = SingleGpuUpdateResult(
        mean_loss=torch.tensor(1.0),
        clip=update.clip,
        condition_encoder_grad_norm=torch.tensor(0.5),
        condition_global_projection_grad_norm=torch.tensor(0.25),
        microbatches=1,
        effective_samples=4,
        state=update.state,
    )
    with pytest.raises(ValueError, match="condition grad norms"):
        build_condition_diagnostic_record(
            successful_update=1,
            rank=0,
            world_size=1,
            update=missing,
            activation_facts=facts,
            slot_ids=(3, 7),
        )
    # Unknown fact key.
    polluted = dict(facts, extra_key=1.0)
    with pytest.raises(ValueError, match="unknown="):
        build_condition_diagnostic_record(
            successful_update=1,
            rank=0,
            world_size=1,
            update=update,
            activation_facts=polluted,
            slot_ids=(3, 7),
        )
    # Slot set mismatch (record contract is fixed per run).
    with pytest.raises(ValueError, match="missing="):
        build_condition_diagnostic_record(
            successful_update=1,
            rank=0,
            world_size=1,
            update=update,
            activation_facts=facts,
            slot_ids=(3, 9),
        )


# ---------------------------------------------------------------------------
# Recorder (POSIX-only durable sink)
# ---------------------------------------------------------------------------


@LINUX_ONLY
def test_recorder_writes_versioned_per_rank_jsonl(tmp_path: Path) -> None:
    for rank, path in {
        0: tmp_path / "rank-0" / diagnostics_file_name(),
        1: tmp_path / "rank-1" / diagnostics_file_name(),
    }.items():
        with ConditionDiagnosticsRecorder(
            path,
            rank=rank,
            world_size=2,
            log_every_n=2,
            flush_every_records=1,
            slot_ids=(3, 7),
        ) as recorder:
            for successful_update in (1, 2, 3, 4):
                recorder.record_update(
                    successful_update=successful_update,
                    update=_mini_update_result(
                        pre_condition=0.9,
                        post_condition=0.7,
                        successful_update=successful_update,
                    ),
                    activation_facts=_mini_facts(),
                )
        lines = path.read_text().splitlines()
        assert len(lines) == 2  # cadence: updates 2 and 4 only
        for line in lines:
            document = json.loads(line)
            assert document["schema_version"] == 1
            assert document["rank"] == rank
            assert document["world_size"] == 2
            assert document["condition_grad_share_pre_clip"] == pytest.approx(
                0.45
            )
        # The two rank files are independent: rank 1's cadence-2 update is 2,
        # never interleaved with rank 0's file.
    assert (tmp_path / "rank-0" / diagnostics_file_name()).exists()
    assert (tmp_path / "rank-1" / diagnostics_file_name()).exists()


@LINUX_ONLY
def test_recorder_rejects_invalid_construction(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="log_every_n"):
        ConditionDiagnosticsRecorder(
            tmp_path / "x.jsonl",
            rank=0,
            world_size=1,
            log_every_n=0,
            flush_every_records=1,
            slot_ids=(3, 7),
        )
    with pytest.raises(ValueError, match="slot_ids"):
        ConditionDiagnosticsRecorder(
            tmp_path / "x.jsonl",
            rank=0,
            world_size=1,
            log_every_n=1,
            flush_every_records=1,
            slot_ids=(),
        )


@LINUX_ONLY
def test_recorder_maybe_record_writes_stored_facts_only_when_cadence_due(
    tmp_path: Path,
) -> None:
    """Cadence gate first: off-cadence updates pay nothing; on-cadence
    updates land the SCALAR facts the runtime's measure() stored (before
    the optimizer step) — no capture, no recompute in the recorder."""

    conditioner = _mini_conditioner()
    composite = cast(
        TrainableComposite, _mini_composite(conditioner, active=True)
    )
    inputs = _mini_inputs(2)
    condition = ConditionTokenOutput(
        tokens=torch.randn(2, 3, 16),
        mask=torch.ones(2, 3, dtype=torch.bool),
        active_mask=torch.ones(2, dtype=torch.bool),
    )
    # Exactly what measure() stores in the last microbatch's measurement.
    stored_facts = compute_condition_activation_facts(
        composite, inputs, condition
    )
    path = tmp_path / diagnostics_file_name()
    with ConditionDiagnosticsRecorder(
        path,
        rank=0,
        world_size=1,
        log_every_n=2,
        flush_every_records=1,
        slot_ids=(3, 7),
    ) as recorder:
        measurement = SimpleNamespace(condition_diagnostics=dict(stored_facts))

        def _observation(successful_update: int) -> object:
            update = _mini_update_result(
                pre_condition=0.9,
                post_condition=0.7,
                successful_update=successful_update,
            )
            return cast(
                "SuccessfulTrainingObservation",
                SimpleNamespace(
                    loop=SimpleNamespace(update=update),
                    microbatches=(measurement,),
                ),
            )

        # Update 1: cadence 2 not due -> nothing is WRITTEN.  The sink
        # creates the (empty) file at construction, so assert zero content.
        recorder.maybe_record(_observation(1))  # type: ignore[arg-type]
        assert path.read_text() == ""
        # Update 2: cadence due -> the stored facts land verbatim.
        recorder.maybe_record(_observation(2))  # type: ignore[arg-type]
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        document = json.loads(lines[0])
        assert document["schema_version"] == CONDITION_DIAGNOSTICS_SCHEMA_VERSION
        assert document["successful_update"] == 2
        assert document["active_samples"] == 2
        for key, value in stored_facts.items():
            assert document[key] == value
        assert document["condition_grad_share_pre_clip"] == pytest.approx(
            0.9 / 2.0
        )


@LINUX_ONLY
def test_recorded_facts_are_computed_before_optimizer_step(
    tmp_path: Path,
) -> None:
    """The landed record carries the measurement's stored facts VERBATIM,
    and those facts are a PRE-step snapshot.

    What this test proves: the recorder lands exactly the scalar facts the
    runtime's ``measure()`` stored on the measurement — and the difference
    between a pre-step snapshot and a post-step recompute is large enough
    to matter (one AdamW lr=1e-4 step moves ``base_hidden_rms`` by ~4e-3
    abs / 3.9e-4 rel; JSON round-trip of the landed floats is exact, so
    ``==`` against the pre-step value is a real check).  It does NOT pin
    the compute site (it feeds the recorder a measurement fake, bypassing
    ``measure()``) — ``test_compute_site_pinned_to_measure_not_observer``
    pins that structurally, because tests/ has no runtime fixture.
    """

    conditioner = _mini_conditioner()
    composite = cast(
        TrainableComposite, _mini_composite(conditioner, active=True)
    )
    inputs = _mini_inputs(2)
    condition = ConditionTokenOutput(
        tokens=torch.randn(2, 3, 16),
        mask=torch.ones(2, 3, dtype=torch.bool),
        active_mask=torch.ones(2, dtype=torch.bool),
    )
    # What measure() stores mid-update, before the step:
    pre_facts = compute_condition_activation_facts(
        composite, inputs, condition
    )
    pre_base = pre_facts["base_hidden_rms"]
    # One optimizer step on the SAME module (the DiT conditioner's
    # parameters) moves the weights — exactly what finish_update does
    # between measure() and the success observer.
    for parameter in conditioner.parameters():
        parameter.grad = torch.full_like(parameter, 0.01)
    torch.optim.AdamW(conditioner.parameters(), lr=1e-4).step()  # pyright: ignore[reportUnknownMemberType]
    post_facts = compute_condition_activation_facts(
        composite, inputs, condition
    )
    post_base = post_facts["base_hidden_rms"]
    # Sanity: the step must actually move the facts, else the test is
    # vacuous (reviewer-measured relative shift ~6.5e-4).
    assert pre_base is not None and post_base is not None
    assert abs(post_base - pre_base) > 1e-5

    # The runtime stored the PRE facts in the last microbatch's
    # measurement; the recorder lands them verbatim.
    measurement = SimpleNamespace(condition_diagnostics=dict(pre_facts))
    update = _mini_update_result(
        pre_condition=0.9, post_condition=0.7, successful_update=1
    )
    observation = cast(
        "SuccessfulTrainingObservation",
        SimpleNamespace(
            loop=SimpleNamespace(update=update),
            microbatches=(measurement,),
        ),
    )
    path = tmp_path / diagnostics_file_name()
    with ConditionDiagnosticsRecorder(
        path,
        rank=0,
        world_size=1,
        log_every_n=1,
        flush_every_records=1,
        slot_ids=(3, 7),
    ) as recorder:
        recorder.maybe_record(observation)  # type: ignore[arg-type]
    document = json.loads(path.read_text().splitlines()[0])
    # The landed values are the pre-step ones, not the post-step ones.
    assert document["base_hidden_rms"] == pre_base
    assert document["base_hidden_rms"] != post_base


def test_compute_site_pinned_to_measure_not_observer() -> None:
    """STRUCTURAL guard: the facts compute call sits inside
    ``SingleGpuBatchRuntime.measure`` and nowhere in the observer hook —
    pinned at METHOD level, not class or module level.

    Why source-level: a behavior-level test cannot pin the compute site
    because tests/ has no runtime fixture — ``SingleGpuBatchRuntime``
    requires CUDA plus the full forward graph, so no test here can drive
    a real update and observe WHEN ``measure()`` ran relative to
    ``optimizer.step()``.  (Root requirement,
    2026-10-17: the guard must go RED if the compute is moved back into
    the observer; verified by temporarily re-adding the call to
    ``maybe_record`` and watching this test fail, then reverting.)

    Why method-level getsource: a class- or module-level source string is
    satisfied by the import line or a comment (``runtime.py`` imports
    ``compute_condition_activation_facts`` at module scope), so the name
    would pass there even if the call lived elsewhere.  Only the METHOD
    body pins "the call happens inside ``measure()``"; the assertion uses
    the call form ``compute_condition_activation_facts(`` (with
    parenthesis) for the same reason.

    Pinning rationale: computing at the observation hook (which runs
    AFTER ``optimizer.step()``) yields a mixed post-step-weight /
    pre-step-token state with no physical meaning.
    """

    import inspect

    from sakuramoon.train.runtime import SingleGpuBatchRuntime

    # (a) The compute call must live in measure()'s body.
    measure_source = inspect.getsource(SingleGpuBatchRuntime.measure)
    assert "compute_condition_activation_facts(" in measure_source
    # (b) The observer hook must NOT carry it.
    recorder_source = inspect.getsource(
        ConditionDiagnosticsRecorder.maybe_record
    )
    assert "compute_condition_activation_facts(" not in recorder_source
    # (c) The recorder constructor must not take the composite back —
    # its presence is what would make a recompute-in-the-observer
    # relapse look plausible (contract regression guard).
    signature = inspect.signature(ConditionDiagnosticsRecorder.__init__)
    assert "composite" not in signature.parameters
    # (d) The cadence number comes from next_growth_update — the same
    # in-flight update number the loop asserts against the landed
    # successful_update — not a self-computed counter.
    runtime_source = Path(inspect.getfile(SingleGpuBatchRuntime)).read_text(
        encoding="utf-8"
    )
    assert "update_number=next_growth_update" in runtime_source
    # (e) The compute runs ONLY on the update's final microbatch (Phase 1
    # fix 2b): one conditioner recompute per due update, off-cadence and
    # non-final microbatches pay nothing.  Pinned in source for the same
    # reason as (a): no runtime fixture can drive measure() in tests/.
    assert "if every is not None and is_final_microbatch:" in measure_source
    assert "is_final_microbatch=microbatch_index" in runtime_source
    assert "config.train.accumulation - 1" in runtime_source
    # (f) The loop hands the microbatch index to the loss_fn (measure
    # wrapper); without it the final-microbatch gate has no signal.
    from sakuramoon.train.loop import SingleGpuTrainingLoop

    loop_source = Path(inspect.getfile(SingleGpuTrainingLoop)).read_text(
        encoding="utf-8"
    )
    assert "self.loss_fn(batch, microbatch_index)" in loop_source


@LINUX_ONLY
def test_recorder_maybe_record_rejects_missing_facts(
    tmp_path: Path,
) -> None:
    """Stream enabled without the runtime facts fails closed on the
    first cadence-due update."""

    with ConditionDiagnosticsRecorder(
        tmp_path / diagnostics_file_name(),
        rank=0,
        world_size=1,
        log_every_n=1,
        flush_every_records=1,
        slot_ids=(3, 7),
    ) as recorder:
        measurement = SimpleNamespace(condition_diagnostics=None)
        update = _mini_update_result(
            pre_condition=0.9, post_condition=0.7, successful_update=1
        )
        observation = cast(
            "SuccessfulTrainingObservation",
            SimpleNamespace(
                loop=SimpleNamespace(update=update),
                microbatches=(measurement,),
            ),
        )
        with pytest.raises(ValueError, match="condition activation facts"):
            recorder.maybe_record(observation)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Step-level opt-in: condition union grad norm pre/post clip
# ---------------------------------------------------------------------------


class _ConditionSideModule(nn.Module):
    """Tiny module whose FQNs match the production condition-side predicate."""

    def __init__(self) -> None:
        super().__init__()
        self.condition_tokens = nn.Linear(4, 4, bias=False)
        self.dit = _ConditionSideDit()
        self.backbone = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # All three parameter groups feed the objective so every one of
        # them accumulates a gradient (the step scans them by FQN).
        return self.backbone(x) + self.condition_tokens(x) + self.dit.conditioner(x)


class _ConditionSideDit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conditioner = nn.Linear(4, 4, bias=False)


class _SgdAdapter:
    def __init__(self, module: nn.Module) -> None:
        self.optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    def step(self) -> None:
        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)


def _run_one_update(
    module: nn.Module, *, condition_grad_share: bool
) -> SingleGpuUpdateResult:
    step = SingleGpuStep(
        module,
        _SgdAdapter(module),
        accumulation_steps=1,
        state=SingleGpuUpdateState.initial(),
        condition_grad_share=condition_grad_share,
    )
    x = torch.randn(4, 4)
    # 4 "samples", each a per-sample scalar objective -> scale = 1/4.
    per_sample = module(x).float().flatten(1).mean(1)
    assert per_sample.requires_grad
    step.backward(per_sample)
    return step.finish_update()


def test_step_condition_grad_share_off_by_default() -> None:
    torch.manual_seed(3)  # pyright: ignore[reportUnknownMemberType]
    module = _ConditionSideModule()
    result = _run_one_update(module, condition_grad_share=False)
    assert result.condition_grad_norm_pre_clip is None
    assert result.condition_grad_norm_post_clip is None


def test_step_condition_grad_share_measures_union_pre_and_post_clip() -> None:
    torch.manual_seed(4)  # pyright: ignore[reportUnknownMemberType]
    module = _ConditionSideModule()
    step = SingleGpuStep(
        module,
        _SgdAdapter(module),
        accumulation_steps=1,
        state=SingleGpuUpdateState.initial(),
        condition_grad_share=True,
    )
    x = torch.randn(4, 4)
    per_sample = module(x).float().flatten(1).mean(1)
    step.backward(per_sample)
    # Raw grads are present and (by construction of the toy module) the
    # global norm exceeds 1, so the clip actually rescales them.
    raw_squares = [
        parameter.grad.square().sum().item()
        for _name, parameter in module.named_parameters()
        if parameter.grad is not None
    ]
    raw_total = math.sqrt(sum(raw_squares))
    # The sample-mean scale (1/4) is applied BEFORE the clip, so the clip
    # coefficient is computed against the SCALED global norm.
    scaled_total = raw_total / 4.0
    coefficient = min(1.0, 1.0 / max(scaled_total, 1e-12))
    expected_pre = math.sqrt(
        sum(
            parameter.grad.square().sum().item()
            for name, parameter in module.named_parameters()
            if (
                "condition_tokens" in name.split(".")
                or "conditioner" in name.split(".")
            )
            and parameter.grad is not None
        )
    )
    result = step.finish_update()
    assert result.condition_grad_norm_pre_clip is not None
    assert result.condition_grad_norm_post_clip is not None
    # scale = 1/samples = 1/4 applied to every gradient before the norm.
    assert result.condition_grad_norm_pre_clip.item() == pytest.approx(
        expected_pre / 4.0, rel=1e-4
    )
    assert float(result.clip.pre_clip_norm.item()) == pytest.approx(
        raw_total / 4.0, rel=1e-4
    )
    assert result.condition_grad_norm_post_clip.item() == pytest.approx(
        expected_pre / 4.0 * coefficient, rel=1e-4
    )
    assert float(result.clip.post_clip_norm.item()) == pytest.approx(
        raw_total / 4.0 * coefficient, rel=1e-4
    )


def test_composite_condition_diagnostics_flag_contract() -> None:
    from sakuramoon.train.step import TrainableComposite

    # The flag is type-validated before any state is written.
    with pytest.raises(ValueError, match="condition_diagnostics"):
        TrainableComposite(
            dit=object(),  # type: ignore[arg-type]
            text=object(),  # type: ignore[arg-type]
            condition_tokens=object(),  # type: ignore[arg-type]
            condition_diagnostics="yes",  # type: ignore[arg-type]
        )
    # Disabled (the default): the capture attribute stays None until a
    # forward runs, so the default path is behavior-identical.
    disabled = TrainableComposite(
        dit=object(),  # type: ignore[arg-type]
        text=object(),  # type: ignore[arg-type]
        condition_tokens=object(),  # type: ignore[arg-type]
    )
    assert disabled.last_conditioning is None
    # Post-construction enablement (the assembly wiring point) flips only
    # the runtime flag: no new parameters, no state_dict entries.
    enabled = TrainableComposite(
        dit=object(),  # type: ignore[arg-type]
        text=object(),  # type: ignore[arg-type]
        condition_tokens=object(),  # type: ignore[arg-type]
    )
    before = set(enabled.state_dict())
    enabled.enable_condition_diagnostics()
    assert set(enabled.state_dict()) == before
    assert set(enabled.parameters()) == set(disabled.parameters())


def test_forward_conditioning_zero_capture_off_then_identity_capture_on(
) -> None:
    """Gate 2 (zero capture): a REAL ``forward_conditioning`` call writes
    nothing while the flag is off, and — once enabled — stores exactly the
    returned condition object (identity, not a copy)."""

    def _condition_stub(*_args: Any) -> ConditionTokenOutput:
        return ConditionTokenOutput(
            tokens=torch.randn(2, 3, 16),
            mask=torch.ones(2, 3, dtype=torch.bool),
            active_mask=torch.ones(2, dtype=torch.bool),
        )

    def _text_stub(*_args: Any) -> object:
        return SimpleNamespace()

    composite = TrainableComposite(
        dit=object(),  # type: ignore[arg-type]
        text=cast(Any, _text_stub),
        condition_tokens=cast(Any, _condition_stub),
    )
    inputs = _mini_inputs(2)
    # Disabled: the forward runs, and NOTHING is captured.
    _text, condition = composite.forward_conditioning(inputs)
    assert isinstance(condition, ConditionTokenOutput)
    assert composite.last_conditioning is None
    # Enabled: the capture is the very object this forward returned.
    composite.enable_condition_diagnostics()
    _text, condition = composite.forward_conditioning(inputs)
    assert composite.last_conditioning is condition
    # A second forward replaces the reference (always the LATEST).
    _text, condition = composite.forward_conditioning(inputs)
    assert composite.last_conditioning is condition
