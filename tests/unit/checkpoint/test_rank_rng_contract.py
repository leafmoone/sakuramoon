"""P2-R per-rank training-RNG contract tests (unit, CPU-safe).

Covers the R1/R2 source contract without touching production training:

  1-8.  capture / validate / restore (structure, bit-exactness, fail-closed)
  9-12. RankRngBundle invariants (frozen gathered input to the save path)
  13-14. RankRngAnchor invariants (validated material only)
  15-24. read_rank_rng_anchor classification matrix:
          legacy rank0 / legacy rank>0 / same-topology / topology-changed
          (rank0 exact vs rank>0 deterministic) + fail-closed cases
  25-26. device-ordinal cross-validation (validate ANOTHER rank's snapshot
          against that rank's saved ordinal; CUDA-gated)
  27-32. full RAW load admission through read_raw_checkpoint_state:
          world2 marker-declared rank files admitted (regression), world1,
          legacy, and fail-closed cases for markerless or mismatched
          rank files
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from sakuramoon.checkpoint.load import (
    read_rank_rng_anchor,
    read_raw_checkpoint_state,
)
from sakuramoon.checkpoint.rng import (
    RankRngAnchor,
    RankRngBundle,
    capture_rank_rng,
    restore_rank_rng,
    validate_rank_rng,
)
from sakuramoon.checkpoint.schema import CheckpointError

REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA/HCU device",
)

BASE_SEED = 20260910


def _seed_everywhere(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _draws(seed: int) -> tuple[list[float], np.ndarray, torch.Tensor]:
    """Deterministic multi-RNG draws used to prove stream identity."""
    import random

    _seed_everywhere(seed)
    py = [random.random() for _ in range(64)]
    npx = np.random.rand(128)
    txc = torch.rand(256)
    return py, npx, txc


# ---------------------------------------------------------------------------
# 1-8. capture / validate / restore
# ---------------------------------------------------------------------------


def test_capture_validate_roundtrip_cpu_bit_exact() -> None:
    import random

    _seed_everywhere(BASE_SEED)
    snapshot = capture_rank_rng()
    before = (
        [random.random() for _ in range(64)],
        np.random.rand(128).copy(),
        torch.rand(256).clone(),
    )
    # advance every stream, restore, and require the NEXT draws to be
    # bit-identical to the continuation taken from the original state.
    random.random()
    np.random.rand(8)
    torch.rand(8)
    restore_rank_rng(snapshot)
    after = (
        [random.random() for _ in range(64)],
        np.random.rand(128),
        torch.rand(256),
    )
    assert after[0] == before[0]
    assert np.array_equal(after[1], before[1])
    assert torch.equal(after[2], before[2])


def test_capture_invariants_host_consistent() -> None:
    snapshot = capture_rank_rng()
    cuda_present = bool(snapshot["cuda_present"].item())
    assert cuda_present == torch.cuda.is_available()
    device_index = int(snapshot["cuda_device_index"].item())
    if cuda_present:
        assert device_index == torch.cuda.current_device()
        assert snapshot["torch_cuda_state"].numel() > 0
    else:
        assert device_index == -1
        assert snapshot["torch_cuda_state"].numel() == 0
    assert snapshot["torch_cpu_state"].numel() > 0
    validate_rank_rng(snapshot)  # a fresh capture always validates


def test_validate_missing_key_fails() -> None:
    snapshot = capture_rank_rng()
    broken = dict(snapshot)
    del broken["python_internal"]
    with pytest.raises(CheckpointError):
        validate_rank_rng(broken)


def test_validate_unknown_key_fails() -> None:
    snapshot = capture_rank_rng()
    broken = dict(snapshot)
    broken["extra_state"] = torch.tensor(0, dtype=torch.int64)
    with pytest.raises(CheckpointError):
        validate_rank_rng(broken)


def test_validate_wrong_scalar_dtype_fails() -> None:
    snapshot = capture_rank_rng()
    broken = dict(snapshot)
    broken["python_version"] = torch.tensor(3, dtype=torch.int32)
    with pytest.raises(CheckpointError):
        validate_rank_rng(broken)


def test_validate_invalid_python_internal_ndim_fails() -> None:
    snapshot = capture_rank_rng()
    broken = dict(snapshot)
    broken["python_internal"] = torch.ones(624, dtype=torch.int64).reshape(24, 26)
    with pytest.raises(CheckpointError):
        validate_rank_rng(broken)


def test_validate_empty_torch_cpu_state_fails() -> None:
    snapshot = capture_rank_rng()
    broken = dict(snapshot)
    broken["torch_cpu_state"] = torch.empty(0, dtype=torch.uint8)
    with pytest.raises(CheckpointError):
        validate_rank_rng(broken)


@REQUIRES_CUDA
def test_validate_cuda_host_mismatch_fails() -> None:
    # A CPU-shaped capture cannot pretend to carry a CUDA state on a CUDA
    # host (and vice versa: the saved availability must match this host).
    snapshot = capture_rank_rng()
    broken = dict(snapshot)
    broken["cuda_present"] = torch.tensor(not torch.cuda.is_available(), dtype=torch.bool)
    with pytest.raises(CheckpointError):
        validate_rank_rng(broken)


# ---------------------------------------------------------------------------
# 9-12. RankRngBundle
# ---------------------------------------------------------------------------


def test_bundle_empty_rejected() -> None:
    with pytest.raises(ValueError):
        RankRngBundle(())


def test_bundle_duplicate_rank_rejected() -> None:
    snap = capture_rank_rng()
    with pytest.raises(ValueError):
        RankRngBundle(((0, snap), (0, snap)))


def test_bundle_negative_or_bool_rank_rejected() -> None:
    snap = capture_rank_rng()
    with pytest.raises(ValueError):
        RankRngBundle(((-1, snap),))
    with pytest.raises(ValueError):
        RankRngBundle(((True, snap),))  # strict int: bool is not rank 1


def test_bundle_non_dict_entry_rejected() -> None:
    with pytest.raises(TypeError):
        RankRngBundle(((0, "not-a-mapping"),))  # type: ignore[arg-type]


def test_bundle_entries_order_and_immutability() -> None:
    snap = capture_rank_rng()
    bundle = RankRngBundle(((1, snap), (0, snap)))
    # entries preserve construction order; the SAVE path sorts before
    # writing rank files.  ranks mirrors the entry order.
    assert bundle.ranks == (1, 0)
    assert tuple(sorted(bundle.ranks)) == (0, 1)
    assert len(bundle.entries) == 2
    with pytest.raises(AttributeError):
        bundle.entries = bundle.entries  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 13-14. RankRngAnchor
# ---------------------------------------------------------------------------


def test_anchor_modes_and_immutability() -> None:
    snap = capture_rank_rng()
    exact = RankRngAnchor(
        "same_topology_exact", source_world_size=2, tensors=snap
    )
    assert exact.mode == "same_topology_exact"
    assert exact.source_world_size == 2
    assert exact.tensors is snap
    for mode in ("topology_changed", "legacy_rank0", "legacy"):
        anchor = RankRngAnchor(mode, source_world_size=1)
        assert anchor.tensors is None
    with pytest.raises(ValueError):
        RankRngAnchor("bogus", source_world_size=1)
    with pytest.raises(AttributeError):
        exact.mode = "legacy"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 15-24. read_rank_rng_anchor classification matrix (hand-built ckpt dirs)
# ---------------------------------------------------------------------------


def _build_ckpt(
    root: Path,
    *,
    ranks: dict[int, dict[str, torch.Tensor]],
    marker: bool,
    marker_world_size: int | None = None,
    marker_ranks: list[int] | None = None,
    marker_schema_version: int = 1,
) -> Path:
    """Minimal RAW checkpoint directory: rng sidecars + marker + manifest."""
    ckpt = root / "ckpt_0_proof"
    rng_dir = ckpt / "train_state" / "rng"
    rng_dir.mkdir(parents=True)
    files: list[dict[str, object]] = []
    for rank in sorted(ranks):
        path = rng_dir / f"rank-{rank}.safetensors"
        save_file(ranks[rank], str(path))
        files.append({"path": f"train_state/rng/rank-{rank}.safetensors", "size": path.stat().st_size})
    if marker:
        world_size = (
            marker_world_size
            if marker_world_size is not None
            else len(ranks)
        )
        marker_path = rng_dir / "rank_set.json"
        marker_path.write_text(
            json.dumps(
                {
                    "ranks": (
                        marker_ranks
                        if marker_ranks is not None
                        else sorted(ranks)
                    ),
                    "schema_version": marker_schema_version,
                    "world_size": world_size,
                },
                sort_keys=True,
            )
            + "\n"
        )
        files.append({"path": "train_state/rng/rank_set.json", "size": marker_path.stat().st_size})
    manifest = {
        "schema_version": 4,
        "kind": "raw",
        "identity": {"checkpoint_id": "proof", "update": 0},
        "files": files,
    }
    (ckpt / "manifest.json").write_text(json.dumps(manifest) + "\n")
    (ckpt / "COMPLETE").write_bytes(b"complete\n")
    return ckpt


def _snap(seed: int) -> dict[str, torch.Tensor]:
    _seed_everywhere(seed)
    np.random.rand(4)
    torch.rand(4)
    return capture_rank_rng()


def test_anchor_legacy_rank0(tmp_path: Path) -> None:
    ckpt = _build_ckpt(tmp_path, ranks={0: _snap(BASE_SEED)}, marker=False)
    anchor = read_rank_rng_anchor(ckpt, rank=0, current_world_size=2)
    assert anchor.mode == "legacy_rank0"
    assert anchor.source_world_size == 1
    assert anchor.tensors is not None


def test_anchor_legacy_rank_above_zero_reseeds(tmp_path: Path) -> None:
    ckpt = _build_ckpt(tmp_path, ranks={0: _snap(BASE_SEED)}, marker=False)
    anchor = read_rank_rng_anchor(ckpt, rank=1, current_world_size=2)
    assert anchor.mode == "legacy"
    assert anchor.tensors is None


def test_anchor_same_topology_exact(tmp_path: Path) -> None:
    snaps = {0: _snap(BASE_SEED), 1: _snap(BASE_SEED + 1)}
    ckpt = _build_ckpt(tmp_path, ranks=snaps, marker=True, marker_world_size=2)
    for rank in (0, 1):
        anchor = read_rank_rng_anchor(ckpt, rank=rank, current_world_size=2)
        assert anchor.mode == "same_topology_exact"
        assert anchor.source_world_size == 2
        assert anchor.tensors is not None
        # each rank reads its OWN snapshot (not rank0's)
        saved = load_file(str(ckpt / "train_state" / "rng" / f"rank-{rank}.safetensors"))
        assert torch.equal(anchor.tensors["python_internal"], saved["python_internal"])


def test_anchor_topology_changed_rank0_keeps_snapshot(tmp_path: Path) -> None:
    snaps = {0: _snap(BASE_SEED), 1: _snap(BASE_SEED + 1)}
    ckpt = _build_ckpt(tmp_path, ranks=snaps, marker=True, marker_world_size=2)
    anchor = read_rank_rng_anchor(ckpt, rank=0, current_world_size=4)
    assert anchor.mode == "topology_changed"
    assert anchor.source_world_size == 2
    assert anchor.tensors is not None


def test_anchor_topology_changed_other_ranks_reseed(tmp_path: Path) -> None:
    snaps = {0: _snap(BASE_SEED), 1: _snap(BASE_SEED + 1)}
    ckpt = _build_ckpt(tmp_path, ranks=snaps, marker=True, marker_world_size=2)
    for rank in (1, 2, 3):
        anchor = read_rank_rng_anchor(ckpt, rank=rank, current_world_size=4)
        assert anchor.mode == "topology_changed"
        assert anchor.source_world_size == 2
        assert anchor.tensors is None


def test_anchor_versioned_rank_not_in_set_fails(tmp_path: Path) -> None:
    snaps = {0: _snap(BASE_SEED), 1: _snap(BASE_SEED + 1)}
    ckpt = _build_ckpt(tmp_path, ranks=snaps, marker=True, marker_world_size=2)
    # rank 2 cannot exist in a 2-rank checkpoint: call with a world that
    # contains it while the marker still declares 0..1 -> topology change
    # would only allow rank0; rank 2 must be a deterministic reseed, and a
    # DIRECT same-world ask for a rank outside the set is fail-closed.
    anchor = read_rank_rng_anchor(ckpt, rank=2, current_world_size=3)
    assert anchor.mode == "topology_changed"
    assert anchor.tensors is None
    with pytest.raises(ValueError):
        read_rank_rng_anchor(ckpt, rank=3, current_world_size=3)


def test_anchor_missing_rank_file_fails(tmp_path: Path) -> None:
    snaps = {0: _snap(BASE_SEED), 1: _snap(BASE_SEED + 1)}
    ckpt = _build_ckpt(tmp_path, ranks=snaps, marker=True, marker_world_size=2)
    (ckpt / "train_state" / "rng" / "rank-1.safetensors").unlink()
    with pytest.raises(CheckpointError):
        read_rank_rng_anchor(ckpt, rank=1, current_world_size=2)


def test_anchor_unreadable_rank_file_fails(tmp_path: Path) -> None:
    snaps = {0: _snap(BASE_SEED), 1: _snap(BASE_SEED + 1)}
    ckpt = _build_ckpt(tmp_path, ranks=snaps, marker=True, marker_world_size=2)
    (ckpt / "train_state" / "rng" / "rank-1.safetensors").write_bytes(b"garbage")
    with pytest.raises(CheckpointError):
        read_rank_rng_anchor(ckpt, rank=1, current_world_size=2)


def test_anchor_malformed_marker_schema_fails(tmp_path: Path) -> None:
    snaps = {0: _snap(BASE_SEED)}
    ckpt = _build_ckpt(
        tmp_path, ranks=snaps, marker=True, marker_world_size=1,
        marker_schema_version=99,
    )
    with pytest.raises(CheckpointError):
        read_rank_rng_anchor(ckpt, rank=0, current_world_size=1)


def test_anchor_marker_file_set_mismatch_fails(tmp_path: Path) -> None:
    snaps = {0: _snap(BASE_SEED), 1: _snap(BASE_SEED + 1)}
    ckpt = _build_ckpt(tmp_path, ranks=snaps, marker=True, marker_world_size=2)
    extra = ckpt / "train_state" / "rng" / "rank-5.safetensors"
    save_file(snaps[0], str(extra))
    with pytest.raises(CheckpointError):
        read_rank_rng_anchor(ckpt, rank=0, current_world_size=2)


def test_anchor_arg_validation(tmp_path: Path) -> None:
    snaps = {0: _snap(BASE_SEED)}
    ckpt = _build_ckpt(tmp_path, ranks=snaps, marker=False)
    with pytest.raises(ValueError):
        read_rank_rng_anchor(ckpt, rank=-1, current_world_size=2)
    with pytest.raises(ValueError):
        read_rank_rng_anchor(ckpt, rank=2, current_world_size=2)
    with pytest.raises(ValueError):
        read_rank_rng_anchor(ckpt, rank=0, current_world_size=0)


# ---------------------------------------------------------------------------
# 25-26. cross-rank device validation (CUDA-gated)
# ---------------------------------------------------------------------------


@REQUIRES_CUDA
def test_cross_rank_validation_against_saved_ordinal(tmp_path: Path) -> None:
    # capture on the current device, then validate with an EXPLICIT
    # expected ordinal equal to the saved one (cross-rank semantics) and
    # with a deliberately wrong ordinal (must fail).
    prev = torch.cuda.current_device()
    snaps = {0: _snap(BASE_SEED), 1: _snap(BASE_SEED + 1)}
    ckpt = _build_ckpt(tmp_path, ranks=snaps, marker=True, marker_world_size=2)
    for rank in (0, 1):
        anchor = read_rank_rng_anchor(ckpt, rank=rank, current_world_size=2)
        assert anchor.mode == "same_topology_exact"
    saved = load_file(
        str(ckpt / "train_state" / "rng" / "rank-0.safetensors")
    )
    saved_index = int(saved["cuda_device_index"].item())
    validate_rank_rng(saved, expected_device_index=saved_index)
    with pytest.raises(CheckpointError):
        validate_rank_rng(saved, expected_device_index=(saved_index + 1) % 2)
    assert saved_index == prev


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# 27-32. Full RAW load admission (regression: the fixed RAW sidecar upper
# bound used to reject marker-declared rank>0 files before rank-set
# validation could establish their legality). These drive the production
# load entry point against complete RAW checkpoints.
# ---------------------------------------------------------------------------

_FULL_UPDATE = 100


def _trainer_growth_documents(
    update: int,
) -> tuple[dict[str, object], dict[str, object]]:
    trainer = {
        "schema_version": 4,
        "attempted_updates": update,
        "successful_updates": update,
        "effective_samples": update * 8,
        "stage_budget": {
            "start_successful_update": 0,
            "terminal_successful_update": 100_000,
        },
        "checkpoint_cadence": {
            "last_successful_update": update,
            "last_wall_clock_unix_seconds": 1_700_000_000.0,
            "every_successful_updates": 100,
        },
    }
    growth = {
        "schema_version": 4,
        "active_slot_ids": [0],
        "alpha": 1.0,
        "stage": "steady",
        "world_size": 1,
        "resolution": 256,
        "ramp_start_successful_update": None,
        "ramp_updates": None,
    }
    return trainer, growth


def _build_full_raw_ckpt(
    root: Path,
    *,
    ranks: dict[int, dict[str, torch.Tensor]],
    marker_world_size: int | None,
    drop_rank_files: tuple[int, ...] = (),
    extra_rank_files: tuple[int, ...] = (),
) -> Path:
    """Complete RAW checkpoint (all required sidecars + the rng face)."""
    ckpt = root / "ckpt_full"
    rng_dir = ckpt / "train_state" / "rng"
    rng_dir.mkdir(parents=True)
    trainer, growth = _trainer_growth_documents(_FULL_UPDATE)
    (ckpt / "resolved_config.toml").write_text("[checkpoint]\n")
    (ckpt / "train_state" / "trainer_state.json").write_text(
        json.dumps(trainer) + "\n"
    )
    (ckpt / "train_state" / "growth_state.json").write_text(
        json.dumps(growth) + "\n"
    )
    (ckpt / "train_state" / "optimizer.pt").write_bytes(b"pt-placeholder")
    (ckpt / "train_state" / "optimizer_schema.json").write_text("{}\n")
    save_file(
        {"state": torch.zeros(2, dtype=torch.float32)},
        str(rng_dir / "optimizer_sr.safetensors"),
    )
    files: list[dict[str, object]] = []
    for rank in sorted(ranks):
        if rank in drop_rank_files:
            continue
        path = rng_dir / f"rank-{rank}.safetensors"
        save_file(ranks[rank], str(path))
        files.append(
            {
                "path": f"train_state/rng/rank-{rank}.safetensors",
                "size": path.stat().st_size,
            }
        )
    for rank in sorted(extra_rank_files):
        path = rng_dir / f"rank-{rank}.safetensors"
        save_file(ranks[0], str(path))
        files.append(
            {
                "path": f"train_state/rng/rank-{rank}.safetensors",
                "size": path.stat().st_size,
            }
        )
    if marker_world_size is not None:
        marker_path = rng_dir / "rank_set.json"
        marker_path.write_text(
            json.dumps(
                {
                    "ranks": list(range(marker_world_size)),
                    "schema_version": 1,
                    "world_size": marker_world_size,
                },
                sort_keys=True,
            )
            + "\n"
        )
        files.append(
            {
                "path": "train_state/rng/rank_set.json",
                "size": marker_path.stat().st_size,
            }
        )
    for relative in (
        "resolved_config.toml",
        "train_state/trainer_state.json",
        "train_state/growth_state.json",
        "train_state/optimizer.pt",
        "train_state/optimizer_schema.json",
        "train_state/rng/optimizer_sr.safetensors",
    ):
        path = ckpt / relative
        files.append({"path": relative, "size": path.stat().st_size})
    model_dir = ckpt / "model"
    model_dir.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    save_file(
        {"some.fqn": torch.zeros(2, dtype=torch.float32)},
        str(model_dir / shard_name),
    )
    (model_dir / "config.json").write_text('{"placeholder": true}\n')
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 16},
                "weight_map": {"some.fqn": shard_name},
            },
            sort_keys=True,
        )
        + "\n"
    )
    nested_paths = (
        "config.json",
        shard_name,
        "model.safetensors.index.json",
    )
    nested_files = [
        {"path": rel, "size": (model_dir / rel).stat().st_size}
        for rel in sorted(nested_paths)
    ]
    (model_dir / "manifest.json").write_text(
        json.dumps({"files": nested_files, "schema_version": 1}) + "\n"
    )
    for relative in (
        "model/manifest.json",
        *(f"model/{rel}" for rel in sorted(nested_paths)),
    ):
        path = ckpt / relative
        files.append({"path": relative, "size": path.stat().st_size})
    files.sort(key=lambda record: record["path"])
    manifest = {
        "schema_version": 4,
        "kind": "raw",
        "identity": {"checkpoint_id": "full", "update": _FULL_UPDATE},
        "files": files,
    }
    (ckpt / "manifest.json").write_text(json.dumps(manifest) + "\n")
    (ckpt / "COMPLETE").write_bytes(b"complete\n")
    return ckpt


def test_raw_world2_full_load_admits_marker_rank_files(tmp_path: Path) -> None:
    """REGRESSION: marker-declared rank>0 sidecars must pass RAW admission."""
    snaps = {rank: _snap(BASE_SEED + rank) for rank in (0, 1)}
    ckpt = _build_full_raw_ckpt(tmp_path, ranks=snaps, marker_world_size=2)
    manifest, state = read_raw_checkpoint_state(ckpt)
    assert manifest.identity.update == _FULL_UPDATE
    assert state.trainer.successful_updates == _FULL_UPDATE


def test_raw_world1_marker_full_load_ok(tmp_path: Path) -> None:
    snaps = {0: _snap(BASE_SEED)}
    ckpt = _build_full_raw_ckpt(tmp_path, ranks=snaps, marker_world_size=1)
    manifest, _state = read_raw_checkpoint_state(ckpt)
    assert manifest.identity.update == _FULL_UPDATE


def test_raw_legacy_full_load_ok(tmp_path: Path) -> None:
    """No marker: the historical rank-0-only publication must load as before."""
    snaps = {0: _snap(BASE_SEED)}
    ckpt = _build_full_raw_ckpt(tmp_path, ranks=snaps, marker_world_size=None)
    read_raw_checkpoint_state(ckpt)


def test_raw_no_marker_rank_above_zero_still_rejected(tmp_path: Path) -> None:
    """Fail-closed guard: rank>0 WITHOUT a marker stays rejected."""
    snaps = {rank: _snap(BASE_SEED + rank) for rank in (0, 1)}
    ckpt = _build_full_raw_ckpt(tmp_path, ranks=snaps, marker_world_size=None)
    with pytest.raises(CheckpointError):
        read_raw_checkpoint_state(ckpt)


def test_raw_marker_missing_declared_rank_file_fails(tmp_path: Path) -> None:
    snaps = {rank: _snap(BASE_SEED + rank) for rank in (0, 1)}
    ckpt = _build_full_raw_ckpt(
        tmp_path, ranks=snaps, marker_world_size=2, drop_rank_files=(1,)
    )
    with pytest.raises(CheckpointError):
        read_raw_checkpoint_state(ckpt)


def test_raw_marker_undeclared_rank_file_fails(tmp_path: Path) -> None:
    snaps = {rank: _snap(BASE_SEED + rank) for rank in (0, 1)}
    ckpt = _build_full_raw_ckpt(
        tmp_path, ranks=snaps, marker_world_size=2, extra_rank_files=(3,)
    )
    with pytest.raises(CheckpointError):
        read_raw_checkpoint_state(ckpt)
