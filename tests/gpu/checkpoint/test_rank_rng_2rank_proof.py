"""P2-R 2-process / 2-rank proof: per-rank RNG separation + bit-exact
stream continuation through a REAL process boundary (R5b + R6).

Two spawned processes (nccl, file init) each:

  1. seed with the historical formula (base + rank*1_000_003 + updates),
     burn streams;
  2. capture the checkpoint-boundary snapshot, then take CONTINUATION
     draws (the reference the restore must reproduce);
  3. gather the boundary snapshot with all_gather_object (the production
     DDP exchange shape);
  4. rank0 publishes a minimal RAW checkpoint (rank files + rank-set
     marker + manifest) — the save path consumes only the gathered value;
  5. ALL ranks destroy their live streams (simulating setup consumption);
  6. every rank reads ITS OWN anchor (read_rank_rng_anchor), binds via
     restore_rank_rng (the exact-anchor branch of the production final
     bind), and re-takes the continuation draws;
  7. continuation must be bit-identical to step 2, PER RANK, with the
     two ranks' streams mutually distinct (separation).

Topology-change probe on the same checkpoint: re-read with
current_world_size=4 — rank0 keeps its snapshot (topology_changed,
tensors present), rank1 reseeds deterministically (tensors None).

Pass = all worker asserts hold and both per-rank JSONs agree.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from sakuramoon.checkpoint.load import read_rank_rng_anchor
from sakuramoon.checkpoint.rng import capture_rank_rng, restore_rank_rng

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2-rank proof requires two visible CUDA/HCU devices",
)

BASE_SEED = 20260910
SIMULATED_UPDATES = 777


def _deterministic_seed(rank: int) -> int:
    # mirrors production _deterministic_rank_rng_seed
    return BASE_SEED + rank * 1_000_003 + SIMULATED_UPDATES


def _draws(rank: int) -> dict[str, object]:
    import random

    py = [random.random() for _ in range(64)]
    npx = np.random.rand(128)
    txc = torch.rand(256)
    txd = torch.randn(256, device=f"cuda:{rank}")
    return {"py": py, "np": npx, "cpu": txc, "cuda": txd}


def _publish_checkpoint(rank: int, out_dir: Path, gathered: list[object]) -> None:
    """rank0-only write delegate over the gathered bundle (frozen value)."""
    if rank != 0:
        return
    ckpt = out_dir / "ckpt_0_proof"
    rng_dir = ckpt / "train_state" / "rng"
    rng_dir.mkdir(parents=True)
    files: list[dict[str, object]] = []
    for r, snap in enumerate(gathered):
        path = rng_dir / f"rank-{r}.safetensors"
        save_file(snap, str(path))
        files.append(
            {"path": f"train_state/rng/rank-{r}.safetensors", "size": path.stat().st_size}
        )
    marker_path = rng_dir / "rank_set.json"
    marker_path.write_text(
        json.dumps(
            {"ranks": [0, 1], "schema_version": 1, "world_size": 2}, sort_keys=True
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


def _worker(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    import random

    import torch.distributed as dist

    dist.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    torch.cuda.set_device(rank)
    seed = _deterministic_seed(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # burn: setup-time consumption before the checkpoint boundary
    random.random()
    np.random.rand(64)
    torch.rand(64)
    torch.randn(64, device=f"cuda:{rank}")

    # 1.5 the checkpoint-boundary snapshot (captured BEFORE the reference
    # draws so the restore can be compared against their continuation)
    boundary_snap = capture_rank_rng()

    # 2. continuation reference (what the restore must reproduce)
    reference = _draws(rank)

    # 3. gather the BOUNDARY snapshot (production DDP exchange shape)
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, boundary_snap)

    # 4. rank0 publishes from the gathered value only
    _publish_checkpoint(rank, Path(out_dir), gathered)
    dist.barrier()

    # 5. destroy the live streams (post-restore setup consumption)
    random.random()
    np.random.rand(64)
    torch.rand(64)
    torch.randn(64, device=f"cuda:{rank}")

    # 6. read THIS rank's anchor + exact-anchor bind (final bind semantics)
    ckpt = Path(out_dir) / "ckpt_0_proof"
    anchor = read_rank_rng_anchor(ckpt, rank=rank, current_world_size=world_size)
    assert anchor.mode == "same_topology_exact", anchor.mode
    assert anchor.tensors is not None
    restore_rank_rng(anchor.tensors)

    # 7. bit-exact continuation, per rank
    restored = _draws(rank)
    assert restored["py"] == reference["py"], f"rank {rank} python stream diverged"
    assert np.array_equal(restored["np"], reference["np"]), f"rank {rank} numpy stream diverged"
    assert torch.equal(restored["cpu"], reference["cpu"]), f"rank {rank} torch cpu stream diverged"
    assert torch.equal(
        restored["cuda"].cpu(), reference["cuda"].cpu()
    ), f"rank {rank} torch cuda stream diverged"

    # separation: the two rank streams must differ (per-rank snapshots)
    gathered_ref = [None] * world_size
    dist.all_gather_object(gathered_ref, reference)
    for other in range(world_size):
        if other == rank:
            continue
        assert gathered_ref[other]["py"] != reference["py"], (
            f"rank {rank} and {other} streams identical — per-rank separation broken"
        )

    # topology-change probe on the SAME checkpoint (2 -> 4 world)
    widened = read_rank_rng_anchor(ckpt, rank=rank, current_world_size=4)
    assert widened.mode == "topology_changed"
    assert widened.source_world_size == 2
    if rank == 0:
        assert widened.tensors is not None
    else:
        assert widened.tensors is None

    result = {"rank": rank, "seed": seed, "mode": anchor.mode, "ok": True}
    (Path(out_dir) / f"per-rank-{rank}.json").write_text(json.dumps(result))
    dist.barrier()
    dist.destroy_process_group()


def test_p2r_2rank_rng_proof(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory(dir=tmp_path) as td:
        init_file = os.path.join(td, "init")
        mp.spawn(_worker, args=(2, init_file, str(tmp_path)), nprocs=2, join=True)
        r0 = json.loads((tmp_path / "per-rank-0.json").read_text())
        r1 = json.loads((tmp_path / "per-rank-1.json").read_text())
    assert r0["ok"] and r1["ok"]
    assert r0["seed"] != r1["seed"]
    assert r0["mode"] == r1["mode"] == "same_topology_exact"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
