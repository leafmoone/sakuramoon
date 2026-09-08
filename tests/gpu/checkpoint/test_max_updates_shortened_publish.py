"""Config-authoritative terminal: a SHORTENED invocation publishes its
final checkpoint (DCU).

Locks the production checkpoint publisher against the historical
stage-budget envelope after ``train.max_updates`` became the execution
authority in both directions: a run whose live terminal (N) is BELOW the
historically recorded terminal (H) must still publish every checkpoint it
saves — every saved update is <= N <= H, so the envelope guard in
``ProductionSingleGpuCheckpointPublisher.publish_update`` accepts them —
while updates outside the active run remain fail-closed.

Canonical numbers: checkpoint at C = 130000, historical terminal
H = 168000, live terminal N = 135000 (5000 updates).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sakuramoon.checkpoint.load import read_raw_checkpoint_state
from sakuramoon.checkpoint.policy import CheckpointCadence, CheckpointReason
from sakuramoon.checkpoint.save import save_raw_checkpoint
from sakuramoon.checkpoint.schema import (
    CheckpointIdentity,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.conditioning.condition_tokens import ConditionTokenEncoder
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.model.dit import DenseDiT
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.optim.cmuon import build_hybrid_cmuon
from sakuramoon.train.preflight import (
    ProductionSingleGpuCheckpointPublisher,
    RestoredSingleGpuCheckpoint,
)
from sakuramoon.train.step import SingleGpuUpdateState, TrainableComposite

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

DEVICE = torch.device("cuda", 0)
HIDDEN = 32
DEPTH = 20
# The canonical depth-20 slot set of the current tree (exclusive bound).
STABLE_SLOT_COUNT = max(active_slot_ids(DEPTH)) + 1
RESOLVED_CONFIG = b"[checkpoint]\nfull_every_updates = 10000\n"

# Canonical shortened-invocation example.
CURRENT = 130_000
HISTORICAL = 168_000
LIVE_TERMINAL = 135_000


def _composite() -> TrainableComposite:
    dit = DenseDiT(
        depth=DEPTH,
        input_channels=128,
        hidden_size=HIDDEN,
        intermediate_size=64,
        q_heads=2,
        kv_heads=1,
        head_dim=16,
        rope_nope_dim=0,
        rope_y_dim=8,
        rope_x_dim=8,
        rope_position_scale=1.0,
        rope_theta=10.0,
        norm_eps=1e-6,
        timestep_dim=256,
        size_dim=64,
        aspect_dim=64,
        condition_hidden_size=1024,
        stable_slot_count=STABLE_SLOT_COUNT,
        modulation_chunks=6,
        final_modulation_size=64,
        out_channels=128,
        condition_token_count=8,
        modality_init_std=0.02,
        linear_dtype=torch.bfloat16,
        sensitive_dtype=torch.float32,
        projection_bias=False,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        output_weight_zero_init=True,
        output_bias_zero_init=True,
    ).to(device=DEVICE)  # pyright: ignore[reportArgumentType]
    text = TextConditioner(
        input_size=2048,
        adapter_size=1024,
        output_size=HIDDEN,
        groups=8,
        attention_heads=8,
        norm_eps=1e-6,
        mix_gate_init=0.0,
        layer_scale_init=1.0,
        projection_bias=False,
        linear_dtype=torch.bfloat16,
        sensitive_dtype=torch.float32,
    ).to(device=DEVICE)
    condition = ConditionTokenEncoder(
        input_size=2048,
        hidden_size=1024,
        intermediate_size=2048,
        output_size=HIDDEN,
        token_count=8,
        attention_heads=8,
        norm_eps=1e-6,
        init_std=0.02,
        projection_bias=False,
        linear_dtype=torch.bfloat16,
        sensitive_dtype=torch.float32,
    ).to(device=DEVICE)
    return TrainableComposite(
        dit=dit,
        text=text,
        condition_tokens=condition,
        irepa_alignment=None,
        irepa_tap_slot_id=None,
    )


def _raw_state(update: int) -> RawCheckpointState:
    return RawCheckpointState(
        trainer=SingleGpuUpdateState(update, update, update * 468),
        growth=GrowthCheckpointState(
            active_slot_ids(DEPTH), 1.0, "stage1", 1, 1024, None, None
        ),
        stage_budget=StageBudgetCheckpointState(0, HISTORICAL),
        checkpoint_cadence=CheckpointCadence(update, float(update), 10_000),
    )


def test_shortened_invocation_publishes_final_checkpoint(
    tmp_path: Path,
) -> None:
    module = _composite()
    optimizer = build_hybrid_cmuon(
        module,
        lr=2e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
        sr_seed=1511,
        ns_steps=4,
    )

    # Source checkpoint: 130000 successful updates into a stage whose
    # historically recorded terminal was 168000.
    source = save_raw_checkpoint(
        tmp_path / "ckpt-root",
        CheckpointIdentity(f"raw-{CURRENT}-update-cadence", CURRENT),
        module,
        optimizer,
        _raw_state(CURRENT),
        resolved_config=RESOLVED_CONFIG,
    )

    # The live invocation rebinds to max_updates = 135000 (below the
    # historical 168000): the persisted envelope stays 168000, the live
    # terminal is 135000.
    manifest, state = read_raw_checkpoint_state(source.path)
    restored = RestoredSingleGpuCheckpoint(
        path=source.path.resolve(strict=True),
        manifest=manifest,
        state=state,
        payload_bytes=sum(record.size for record in manifest.files),
        module=module,
        optimizer=optimizer,
    )
    out_root = tmp_path / "out-root"
    out_root.mkdir()
    publisher = ProductionSingleGpuCheckpointPublisher(
        checkpoint_root=out_root,
        resolved_config=RESOLVED_CONFIG,
        module=module,
        optimizer=optimizer,
        restored_checkpoint=restored,
        accepted_checkpoint_ids=frozenset(),
        retention_slots=2,
    )

    # The final checkpoint of the shortened run publishes normally even
    # though the historical terminal (168000) is larger than the live one
    # (135000).
    saved_path = publisher.publish_update(
        SingleGpuUpdateState(LIVE_TERMINAL, LIVE_TERMINAL, LIVE_TERMINAL * 468),
        CheckpointReason.STAGE_FINALIZE,
        CheckpointCadence(LIVE_TERMINAL, float(LIVE_TERMINAL), 10_000),
    )
    saved_manifest, saved_state = read_raw_checkpoint_state(saved_path)
    assert saved_manifest.identity.update == LIVE_TERMINAL
    assert saved_state.trainer.successful_updates == LIVE_TERMINAL
    # The historical envelope is carried forward verbatim (never shrunk).
    assert saved_state.stage_budget.terminal_successful_update == HISTORICAL

    # Fail-closed edges of the envelope guard are unchanged: an update at
    # the restored point and one above the envelope are outside the run.
    with pytest.raises(ValueError, match="outside the active training run"):
        publisher.publish_update(
            SingleGpuUpdateState(CURRENT, CURRENT, CURRENT * 468),
            CheckpointReason.UPDATE_CADENCE,
            CheckpointCadence(CURRENT, float(CURRENT), 10_000),
        )
    with pytest.raises(ValueError, match="outside the active training run"):
        publisher.publish_update(
            SingleGpuUpdateState(
                HISTORICAL + 1, HISTORICAL + 1, (HISTORICAL + 1) * 468
            ),
            CheckpointReason.UPDATE_CADENCE,
            CheckpointCadence(HISTORICAL + 1, float(HISTORICAL + 1), 10_000),
        )
