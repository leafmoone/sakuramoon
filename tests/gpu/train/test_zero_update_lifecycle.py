# pyright: reportPrivateUsage=false
"""Pure zero-update lifecycle: a COMPLETED resume exits before ANY
training-resource path (DCU).

Locks the production lifecycle contract for a RESUME invocation whose
live config terminal (N) is at or below the restored successful update
(C) with ``preflight_only = false``: after the exact RAW checkpoint
restore and the resume binding, ``_run_accepted_lifecycle`` must return
a zero-update ``ProductionTrainingResult`` WITHOUT touching

  - the frozen encoder loads (Qwen / Mage VAE / VAE torch.compile),
  - the DataServiceClient / batch-stream construction,
  - the training preflight (including the Qwen fast-path probe),
  - torch.compile / accelerator.prepare / calibration / telemetry,
  - any forward / backward / optimizer.step / checkpoint write.

Traps (sentinel exceptions) make the test fail immediately if any of
those boundaries is reached.

Canonical numbers: checkpoint at C = 130000, historical terminal
H = 168000.

  A: N = 120000 (< C)  -> zero-update success, nothing touched
  B: N = 130000 (== C) -> zero-update success, nothing touched
  C: N = 135000 (> C)  -> the shortcut is NOT taken; the first
                          downstream training resource is reached
  D: N = 120000 with preflight_only=true -> the shortcut is NOT taken
  E: the zero-update report content (validated inside A)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, NoReturn

# Single visible device: the production topology check requires
# torch.cuda.device_count() == config.distributed.world_size (1 here).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pytest
import torch

import sakuramoon.encoders.qwen as _qwen_module
from sakuramoon.checkpoint.load import read_raw_checkpoint_state
from sakuramoon.checkpoint.policy import CheckpointCadence
from sakuramoon.checkpoint.save import save_raw_checkpoint
from sakuramoon.checkpoint.schema import (
    CheckpointIdentity,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.config import load_config
from sakuramoon.config.assembly import (
    build_trainable_composite_from_config,
)
from sakuramoon.config.load import LoadedConfig
from sakuramoon.config.schema import EvaluationDisabledConfig
from sakuramoon.model.growth import active_slot_ids
from sakuramoon.train import production
from sakuramoon.train.production import (
    _build_optimizer,
    _run_accepted_lifecycle,
)
from sakuramoon.train.step import (
    SingleGpuUpdateState,
    TrainableComposite,
)

REPOSITORY_ROOT = Path(__file__).parents[3]
DEVICE = torch.device("cuda", 0)
MODEL_ROOT = Path(
    os.environ.get("SAKURAMOON_TEST_MODEL_ROOT", "/sakuramoon-runtime/model")
)

CURRENT = 130_000
HISTORICAL = 168_000
CHECKPOINT_ID = f"raw-{CURRENT}-update-cadence"

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        not MODEL_ROOT.is_dir(),
        reason="local Qwen/VAE assets are required (SAKURAMOON_TEST_MODEL_ROOT)",
    ),
]


class _TrainingResourceReached(RuntimeError):
    """A training-only dependency was reached on a zero-update resume."""


def _trap(name: str) -> Any:
    def _trap_function(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise _TrainingResourceReached(name)

    return _trap_function


def _small_config(loaded: LoadedConfig, max_updates: int) -> Any:
    """A single-process (native), small-model, hybrid-CMuon variant of the
    resolved production config: small DiT/triple sizes, one visible DCU,
    no compile, no evaluation (no CLIP/inception assets required), the
    hybrid CMuon optimizer (valid for the small module)."""

    base = loaded.config
    dit = base.model.dit.model_copy(
        update={
            "hidden_size": 32,
            "intermediate_size": 64,
            "q_heads": 2,
            "kv_heads": 1,
            "head_dim": 16,
        }
    )
    model = base.model.model_copy(
        update={
            "dit": dit,
            # RoPE must partition the (small) head: nope + y + x == head_dim,
            # y == x even.
            "rope": base.model.rope.model_copy(
                update={"head_dim": 16, "nope_dim": 4, "y_dim": 6, "x_dim": 6}
            ),
            # The locked conditioner contract: final_modulation_size ==
            # 2 * hidden_size.
            "head": base.model.head.model_copy(update={"final_modulation_size": 64}),
            "text": base.model.text.model_copy(update={"output_size": 32}),
            "condition_tokens": base.model.condition_tokens.model_copy(
                update={"output_size": 32}
            ),
        }
    )
    # native single-process: effective global batch = local * acc * 1.
    train = base.train.model_copy(
        update={
            "max_updates": max_updates,
            "global_batch": base.train.local_batch * base.train.accumulation,
        }
    )
    return base.model_copy(
        update={
            "model": model,
            "distributed": base.distributed.model_copy(
                update={"backend": "native", "world_size": 1}
            ),
            "train": train,
            "kernels": base.kernels.model_copy(
                update={
                    # DenseDiT (the packed variant is locked to the
                    # production attention geometry).
                    "attention_backend": "dense_sdpa_reference",
                    "torch_compile_enabled": False,
                    "torch_compile_dynamic": False,
                    "vae_torch_compile": False,
                }
            ),
            "optimizer": base.optimizer.model_copy(update={"name": "hybrid_cmuon"}),
            "evaluation": EvaluationDisabledConfig(enabled=False),
        }
    )


def _repository_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    (root / "model").symlink_to(MODEL_ROOT)
    (root / "runs").mkdir(parents=True)
    (root / "artifacts").mkdir()
    (root / "output_model" / "s0").mkdir(parents=True)
    return root


def _checkpoint(config: Any, checkpoint_root: Path) -> tuple[Path, RawCheckpointState]:
    """Save the exact RAW source checkpoint: C successful updates into a
    stage whose historical terminal is H."""

    module = build_trainable_composite_from_config(config, device=DEVICE)
    optimizer = _build_optimizer(config, module, rank=0, world_size=1)
    state = RawCheckpointState(
        trainer=SingleGpuUpdateState(CURRENT, CURRENT, CURRENT * 468),
        growth=GrowthCheckpointState(
            active_slot_ids(config.model.dit.depth),
            1.0,
            "stage1",
            1,
            config.train.resolution,
            None,
            None,
        ),
        stage_budget=StageBudgetCheckpointState(0, HISTORICAL),
        checkpoint_cadence=CheckpointCadence(CURRENT, float(CURRENT), 10_000),
    )
    save = save_raw_checkpoint(
        checkpoint_root,
        CheckpointIdentity(CHECKPOINT_ID, CURRENT),
        module,
        optimizer,
        state,
        resolved_config=b"[test]\n",
    )
    return save.path, state


def _install_traps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail immediately if any training-only dependency is reached."""

    for name in (
        "load_local_qwen",
        "load_local_mage_vae",
        "compile_vae_methods",
        "DataServiceClient",
        "run_single_gpu_preflight",
        "build_single_gpu_preflight_checks",
        "compile_packed_dit_blocks",
        "run_single_gpu_training",
    ):
        monkeypatch.setattr(production, name, _trap(name))
    monkeypatch.setattr(
        production.ProductionPipelineFactory, "from_config", _trap("factory")
    )
    monkeypatch.setattr(_qwen_module, "probe_qwen_fast_path", _trap("qwen_probe"))
    monkeypatch.setattr(
        production._hybrid_cmuon_class(), "step", _trap("optimizer.step")
    )
    monkeypatch.setattr(TrainableComposite, "forward", _trap("forward"))


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loaded: LoadedConfig,
    max_updates: int,
    *,
    preflight_only: bool = False,
    full_traps: bool = True,
) -> Any:
    config = _small_config(loaded, max_updates)
    repo = _repository_root(tmp_path)
    checkpoint, _ = _checkpoint(config, repo / "output_model" / "s0")
    if full_traps:
        _install_traps(monkeypatch)
    else:
        # First downstream training-resource boundary only: proves the
        # normal path was entered (no early exit) without training.
        monkeypatch.setattr(production, "load_local_qwen", _trap("load_local_qwen"))
    result = _run_accepted_lifecycle(
        LoadedConfig(config, loaded.inputs, loaded.resolved_toml),
        repository_root=repo,
        resume=checkpoint,
        preflight_only=preflight_only,
        wall_clock=time.time,
    )
    return result, checkpoint, repo


@pytest.fixture(autouse=True)
def _single_device_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the production topology guard to a single-device view.

    The native single-rank lifecycle uses exactly one DCU (cuda:0).  DTK
    caches the visible device list at the process's first CUDA use, which
    an earlier-imported GPU test module can trigger before this module's
    ``CUDA_VISIBLE_DEVICES`` setdefault takes effect; the production
    ``device_count() == world_size`` guard would then see the host-wide
    count.  The real run never addresses more than cuda:0, so the guard
    is presented the single-device topology the invocation simulates.
    """
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)


@pytest.fixture(scope="module")
def resolved_config() -> LoadedConfig:
    return load_config(
        Path("train_s0.toml"),
        config_root=REPOSITORY_ROOT / "config",
        environment={
            "MODELSCOPE_API_TOKEN": "placeholder-lifecycle",
            "WANDB_API_KEY": "placeholder-lifecycle",
        },
    )


def test_zero_update_below_current_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, resolved_config: LoadedConfig
) -> None:
    result, checkpoint, repo = _run(
        monkeypatch, tmp_path, resolved_config, CURRENT - 10_000
    )
    _, source_state = read_raw_checkpoint_state(checkpoint)

    # Zero-update success: no training, no preflight-only flag.
    assert result.initial_successful_update == CURRENT
    assert result.final_successful_update == CURRENT
    assert result.preflight_only is False
    assert result.checkpoint_path == checkpoint.resolve(strict=True)
    # The resolved config was published (cheap static path, allowed).
    assert result.resolved_config.is_file()

    # Source checkpoint is untouched; nothing new was published.
    _, saved_state = read_raw_checkpoint_state(checkpoint)
    assert saved_state == source_state
    checkpoint_root = repo / "output_model" / "s0"
    assert [p.name for p in checkpoint_root.iterdir()] == [checkpoint.name]

    # Report contract (test E): explicit zero-update identity, and the
    # checks never claim training resources passed.
    report_path = result.preflight_report
    assert report_path.name == f"preflight-{CURRENT}-zero-update.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["passed"] is True
    assert report["dataset_id"] == "not-required-zero-update"
    assert report["checkpoint_update"] == CURRENT
    assert report["checkpoint_id"] == CHECKPOINT_ID
    checks = {item["name"]: item["passed"] for item in report["checks"]}
    assert checks == {
        "resolved_config": True,
        "checkpoint_restore": True,
        "resume_binding": True,
        "terminal_completed": True,
    }
    assert not any("qwen" in name or "data" in name for name in checks)


def test_zero_update_equal_current_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, resolved_config: LoadedConfig
) -> None:
    result, checkpoint, _ = _run(monkeypatch, tmp_path, resolved_config, CURRENT)

    assert result.initial_successful_update == CURRENT
    assert result.final_successful_update == CURRENT
    assert result.preflight_only is False
    assert result.preflight_report.name == f"preflight-{CURRENT}-zero-update.json"
    assert result.checkpoint_path == checkpoint.resolve(strict=True)


def test_above_current_does_not_take_shortcut(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, resolved_config: LoadedConfig
) -> None:
    with pytest.raises(_TrainingResourceReached) as exc:
        _run(
            monkeypatch,
            tmp_path,
            resolved_config,
            CURRENT + 5_000,
            full_traps=False,
        )
    # The first downstream training resource (the Qwen encoder load) was
    # reached: the lifecycle continued into the ordinary training path.
    assert str(exc.value) == "load_local_qwen"


def test_preflight_only_does_not_take_shortcut(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, resolved_config: LoadedConfig
) -> None:
    with pytest.raises(_TrainingResourceReached) as exc:
        _run(
            monkeypatch,
            tmp_path,
            resolved_config,
            CURRENT - 10_000,
            preflight_only=True,
            full_traps=False,
        )
    assert str(exc.value) == "load_local_qwen"
