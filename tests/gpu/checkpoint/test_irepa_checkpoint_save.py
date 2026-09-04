"""Phase 5 iREPA checkpoint save-path contract tests (DCU).

Covers the post-migration RAW save lifecycle unlocked by Phase 5:

* an iREPA-enabled (architecture schema v4) composite publishes RAW
  checkpoints ONLY together with the persisted ``irepa_state.json`` anchor
  document, and the document is republished VERBATIM: it is the ORIGINAL
  migration anchor read once from the resume checkpoint, never a value
  recomputed from the current update;
* the end-to-end chain on the accelerator: real v3 RAW save -> real
  ``migrate_irepa_checkpoint`` -> production publisher bound to the migrated
  checkpoint -> post-migration v4 RAW save whose anchor sidecar is byte
  identical to the migrated one and whose model payload carries the projector;
* fail-closed edges: v4 RAW without the anchor document, v4 RAW with an
  inconsistent document, v3 RAW with an attached document, and v4
  MODEL_ONLY/RELEASE export (projector stripping remains locked).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from sakuramoon.checkpoint.load import read_raw_checkpoint_state
from sakuramoon.checkpoint.migrate_irepa_checkpoint import (
    IREPA_STATE_FILE,
    migrate_irepa_checkpoint,
    read_irepa_state,
)
from sakuramoon.checkpoint.save import save_model_only, save_raw_checkpoint
from sakuramoon.checkpoint.schema import (
    CheckpointCadence,
    CheckpointError,
    CheckpointIdentity,
    CheckpointReason,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.conditioning.condition_tokens import ConditionTokenEncoder
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.config.schema import IRepaConfig
from sakuramoon.model.dit import DenseDiT
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.model.irepa import IRepaAlignment
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
RESOLVED_CONFIG = b"[checkpoint]\nfull_every_updates = 100\n"
SOURCE_UPDATE = 3


def _dit_kwargs() -> dict[str, object]:
    # Locked 128-channel latent I/O and x-prediction head are required by the
    # DiT and checkpoint export contracts; depth 20 carries the canonical G1
    # slot set (includes the iREPA tap slot 8).
    return {
        "depth": DEPTH,
        "input_channels": 128,
        "hidden_size": HIDDEN,
        "intermediate_size": 64,
        "q_heads": 2,
        "kv_heads": 1,
        "head_dim": 16,
        "rope_nope_dim": 0,
        "rope_y_dim": 8,
        "rope_x_dim": 8,
        "rope_position_scale": 1.0,
        "rope_theta": 10.0,
        "norm_eps": 1e-6,
        "timestep_dim": 256,
        "size_dim": 64,
        "aspect_dim": 64,
        "condition_hidden_size": 1024,
        "stable_slot_count": 24,
        "modulation_chunks": 6,
        "final_modulation_size": 64,
        "out_channels": 128,
        "condition_token_count": 8,
        "modality_init_std": 0.02,
        "linear_dtype": torch.bfloat16,
        "sensitive_dtype": torch.float32,
        "projection_bias": False,
        "attention_dropout": 0.0,
        "mlp_dropout": 0.0,
        "output_weight_zero_init": True,
        "output_bias_zero_init": True,
    }


def _composite(v4: bool) -> TrainableComposite:
    dit = DenseDiT(**_dit_kwargs()).to(device=DEVICE)  # pyright: ignore[reportArgumentType]
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
    irepa: IRepaAlignment | None = None
    if v4:
        irepa = IRepaAlignment(HIDDEN).to(device=DEVICE)
    return TrainableComposite(
        dit=dit,
        text=text,
        condition_tokens=condition,
        irepa_alignment=irepa,
        irepa_tap_slot_id=8 if v4 else None,
    )


def _optimizer(module: TrainableComposite):
    # The production G1 optimizer family (hybrid CMuon + AdamW8bit fallback);
    # built pre-step, exactly like the bootstrap checkpoint path.
    return build_hybrid_cmuon(
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


def _raw_state(update: int) -> RawCheckpointState:
    return RawCheckpointState(
        trainer=SingleGpuUpdateState(update, update, update),
        growth=GrowthCheckpointState(
            active_slot_ids(DEPTH), 1.0, "stage1", 1, 1024, None, None
        ),
        stage_budget=StageBudgetCheckpointState(0, 1000),
        checkpoint_cadence=CheckpointCadence(update, float(update), 100),
    )


def _irepa_config() -> IRepaConfig:
    return IRepaConfig(
        enabled=True,
        teacher_id="facebook/PE-Spatial-B16-512",
        tap_slot=8,
        projector_kernel_size=3,
        spatial_norm="zscore",
        loss="cosine",
    )


def _save_v3_checkpoint(tmp_path: Path) -> Path:
    module = _composite(v4=False)
    optimizer = _optimizer(module)
    result = save_raw_checkpoint(
        tmp_path / "ckpt-root",
        CheckpointIdentity(f"raw-{SOURCE_UPDATE}-update-cadence", SOURCE_UPDATE),
        module,
        optimizer,
        _raw_state(SOURCE_UPDATE),
        resolved_config=RESOLVED_CONFIG,
    )
    return result.path


def test_publisher_republishes_original_anchor_after_migration(
    tmp_path: Path,
) -> None:
    source = _save_v3_checkpoint(tmp_path)
    assert not (source / IREPA_STATE_FILE).exists()

    migrated = migrate_irepa_checkpoint(
        source,
        tmp_path / "migrated",
        irepa=_irepa_config(),
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
        migration_seed=20260903,
    )
    assert isinstance(migrated, Path)
    document = read_irepa_state(migrated)
    assert document["start_successful_update"] == SOURCE_UPDATE + 1
    assert document["source_update"] == SOURCE_UPDATE
    # The migration leaves the source checkpoint untouched.
    assert not (source / IREPA_STATE_FILE).exists()

    # The publisher is bound to the MIGRATED checkpoint (production resume):
    # it reads the anchor once, then republishes it on every RAW save.
    v4_module = _composite(v4=True)
    v4_optimizer = _optimizer(v4_module)
    manifest, state = read_raw_checkpoint_state(migrated)
    restored = RestoredSingleGpuCheckpoint(
        path=migrated.resolve(strict=True),
        manifest=manifest,
        state=state,
        payload_bytes=sum(record.size for record in manifest.files),
        module=v4_module,
        optimizer=v4_optimizer,
    )
    out_root = tmp_path / "out-root"
    out_root.mkdir()
    publisher = ProductionSingleGpuCheckpointPublisher(
        checkpoint_root=out_root,
        resolved_config=RESOLVED_CONFIG,
        module=v4_module,
        optimizer=v4_optimizer,
        restored_checkpoint=restored,
        accepted_checkpoint_ids=frozenset(),
        retention_slots=2,
    )
    saved_path = publisher.publish_update(
        SingleGpuUpdateState(SOURCE_UPDATE + 1, SOURCE_UPDATE + 1, SOURCE_UPDATE + 1),
        CheckpointReason.UPDATE_CADENCE,
        CheckpointCadence(SOURCE_UPDATE + 1, float(SOURCE_UPDATE + 1), 100),
    )

    # The ORIGINAL anchor is republished verbatim (never reset to the
    # current update 4): the sidecar bytes are canonical and identical.
    saved_document = read_irepa_state(saved_path)
    assert saved_document == document
    assert (saved_path / IREPA_STATE_FILE).read_bytes() == (
        migrated / IREPA_STATE_FILE
    ).read_bytes()
    saved_manifest, _ = read_raw_checkpoint_state(saved_path)
    assert IREPA_STATE_FILE in {record.path for record in saved_manifest.files}
    # The post-migration payload carries the projector tensors.
    index = json.loads(
        (saved_path / "model" / "model.safetensors.index.json").read_bytes()
    )
    weight_map = index["weight_map"]
    assert "irepa_alignment.projector.weight" in weight_map
    assert "irepa_alignment.projector.bias" in weight_map


def test_v4_raw_save_requires_the_anchor_document(tmp_path: Path) -> None:
    module = _composite(v4=True)
    optimizer = _optimizer(module)
    with pytest.raises(CheckpointError, match="requires the persisted irepa_state"):
        save_raw_checkpoint(
            tmp_path / "root",
            CheckpointIdentity("raw-4-update-cadence", 4),
            module,
            optimizer,
            _raw_state(4),
            resolved_config=RESOLVED_CONFIG,
        )


def test_v4_raw_save_rejects_an_inconsistent_document(tmp_path: Path) -> None:
    module = _composite(v4=True)
    optimizer = _optimizer(module)
    # Self-consistent documents pass; one whose anchor was recomputed from a
    # different update count must be rejected at save time.
    bad = {
        "schema_version": 1,
        "start_successful_update": 5,
        "source_checkpoint_id": "raw-3-update-cadence",
        "source_update": 3,
        "migration_seed": 20260903,
    }
    with pytest.raises(CheckpointError, match="anchor differs"):
        save_raw_checkpoint(
            tmp_path / "root",
            CheckpointIdentity("raw-4-update-cadence", 4),
            module,
            optimizer,
            _raw_state(4),
            resolved_config=RESOLVED_CONFIG,
            irepa_state=bad,
        )
    assert not (tmp_path / "root").exists()


def test_v3_raw_save_rejects_an_attached_anchor(tmp_path: Path) -> None:
    module = _composite(v4=False)
    optimizer = _optimizer(module)
    document = {
        "schema_version": 1,
        "start_successful_update": 4,
        "source_checkpoint_id": "raw-3-update-cadence",
        "source_update": 3,
        "migration_seed": 20260903,
    }
    with pytest.raises(CheckpointError, match="cannot accompany a v3"):
        save_raw_checkpoint(
            tmp_path / "root",
            CheckpointIdentity("raw-4-update-cadence", 4),
            module,
            optimizer,
            _raw_state(4),
            resolved_config=RESOLVED_CONFIG,
            irepa_state=document,
        )


def test_v3_raw_save_without_anchor_is_unchanged(tmp_path: Path) -> None:
    module = _composite(v4=False)
    optimizer = _optimizer(module)
    result = save_raw_checkpoint(
        tmp_path / "root",
        CheckpointIdentity("raw-4-update-cadence", 4),
        module,
        optimizer,
        _raw_state(4),
        resolved_config=RESOLVED_CONFIG,
    )
    assert not (result.path / IREPA_STATE_FILE).exists()
    manifest, _ = read_raw_checkpoint_state(result.path)
    assert IREPA_STATE_FILE not in {record.path for record in manifest.files}


def test_v4_model_only_export_stays_locked(tmp_path: Path) -> None:
    module = _composite(v4=True)
    with pytest.raises(CheckpointError, match="MODEL_ONLY/RELEASE"):
        save_model_only(
            tmp_path / "root",
            CheckpointIdentity("model-4-update-cadence", 4),
            module,
        )
    assert not (tmp_path / "root").exists()
