"""Small real-AdamW RAW continuation test; CPU SR adapter, no vendor claim."""
# pyright: reportPrivateUsage=false
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import sakuramoon.perf.raw_resume as resume_module
from sakuramoon.checkpoint.load import (
    load_inference_artifact,
    read_raw_checkpoint_state,
)
from sakuramoon.checkpoint.save import save_raw_checkpoint
from sakuramoon.checkpoint.schema import (
    CheckpointCadence,
    CheckpointIdentity,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.config import LoadedConfig, load_config
from sakuramoon.config.resolve import resolved_config_bytes
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit
from sakuramoon.optim.groups import ParameterAudit, ParameterSpec
from sakuramoon.optim.stochastic_rounding import StochasticRoundingRNG
from sakuramoon.perf.raw_resume import (
    advance_synthetic,
    validate_paths,
    validate_resume_config,
)
from sakuramoon.train import production
from sakuramoon.train.step import SingleGpuUpdateState, TrainableComposite

ROOT = Path(__file__).parents[3]


def _model(norm: str) -> TrainableComposite:
    spec = importlib.util.spec_from_file_location("_raw_fixture", Path(__file__).parents[1] / "checkpoint/test_final_norm_removal.py")
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return cast(TrainableComposite, helper._composite(norm))


class _CpuSR(StochasticRoundingRNG):
    def run_step(self, step: object) -> object:
        assert callable(step)
        return step()


def _optimizer(model: TrainableComposite, config: RuntimeConfig) -> IsolatedAdamW8bit:
    specs = tuple(ParameterSpec(name, parameter, "sensitive_no_decay", 0.0) for name, parameter in sorted(model.named_parameters()))
    assert all(spec.parameter.numel() < 4096 for spec in specs)
    inner = torch.optim.AdamW([
        {"params": [], "param_names": [], "group_name": "matrix_decay"},
        {"params": [spec.parameter for spec in specs], "param_names": [spec.name for spec in specs], "group_name": "sensitive_no_decay"},
    ], lr=config.scaled_learning_rate(), betas=config.optimizer.betas, eps=config.optimizer.eps, weight_decay=0.0)
    return IsolatedAdamW8bit(inner, ParameterAudit(specs), _CpuSR(torch.device("cuda:0"), torch.arange(16, dtype=torch.uint8)))


def _loaded() -> LoadedConfig:
    loaded = load_config(ROOT / "config/train_g1_hybrid_cmuon_ns4_core.toml", config_root=ROOT / "config", environment={}, validate_secrets=False)
    document = loaded.config.model_dump(mode="python", exclude_none=True, exclude_computed_fields=True)
    document["model"]["dit"]["active_slot_ids"] = (0,)
    document["model"]["dit"]["depth"] = 1
    document["train"].update(local_batch=2, accumulation=1, global_batch=2, max_updates=10)
    document["distributed"].update(backend="native", world_size=1)
    document["optimizer"] = {"name": "torchao_adamw8bit", "base_lr": 0.0001, "reference_batch": 2, "lr_scaling": "none", "betas": (0.9, 0.95), "eps": 1e-8, "block_size": 256, "bf16_stochastic_round": True, "matrix_weight_decay": 0.0, "sensitive_weight_decay": 0.0}
    document["scheduler"] = {"name": "linear_warmup_constant", "warmup_updates": 5}
    document["growth"]["enabled"] = False
    document["checkpoint"]["full_every_updates"] = 1
    config = RuntimeConfig.model_validate(document)
    return LoadedConfig(config, (), resolved_config_bytes(config).decode())


class _Runtime:
    growth_alpha = 1.0
    def __init__(self, module: TrainableComposite) -> None:
        self.module = module
    def set_growth_alpha(self, value: float) -> None:
        self.growth_alpha = value
    def measure(self, _batch: object, **_kwargs: object) -> Any:
        # All real parameters receive finite gradients and real optimizer updates.
        loss = torch.stack([parameter.float().square().mean() for parameter in self.module.parameters()]).sum()
        return SimpleNamespace(per_sample_loss=loss.expand(2))


def _phase(source: Path, destination: Path, updates: int, identity: str) -> Path:
    loaded = _loaded()
    validate_paths(source, destination)
    validate_resume_config(loaded, source, updates)
    module = _model("none")
    optimizer = _optimizer(module, loaded.config)
    restored = production._restore_checkpoint(loaded, checkpoint=source, module=module, optimizer=optimizer, learning_rate_for_update=production._config_learning_rate)
    return advance_synthetic(loaded=loaded, restored=restored, runtime=_Runtime(module), batches=[None] * updates, updates=updates, output_root=destination, checkpoint_id=identity)


def test_two_updates_separate_save_then_fresh_process_one_update(tmp_path: Path) -> None:
    loaded = _loaded()
    source_model = _model("rmsnorm")
    optimizer = _optimizer(source_model, loaded.config)
    for spec in optimizer.audit.specs:
        optimizer.optimizer.state[spec.parameter] = {"step": torch.tensor(1.0), "exp_avg": torch.full_like(spec.parameter, 0.2), "exp_avg_sq": torch.full_like(spec.parameter, 0.3)}
    state = RawCheckpointState(SingleGpuUpdateState(1, 1, 2), GrowthCheckpointState((0,), 1.0, "synthetic", 1, 256, None, None), StageBudgetCheckpointState(0, 10), CheckpointCadence(1, 1.0, 1))
    source = save_raw_checkpoint(tmp_path / "output_model" / "source", CheckpointIdentity("old", 1), source_model, optimizer, state, resolved_config=loaded.resolved_toml.encode()).path
    untouched = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    saved = _phase(source, tmp_path / "output_model" / "phase1", 2, "new")
    _, after = read_raw_checkpoint_state(saved)
    assert after.trainer == SingleGpuUpdateState(3, 3, 6)
    saved_config = tomllib.loads((saved / "resolved_config.toml").read_text())
    assert saved_config["run"]["run_id"] == "synthetic-new"
    assert saved_config["paths"]["checkpoint_dir"] == str(saved.parent)
    assert all(Path(value).is_relative_to(saved.parent) for value in saved_config["paths"].values())
    assert saved_config["optimizer"] == tomllib.loads(loaded.resolved_toml)["optimizer"]
    assert after.checkpoint_cadence.last_successful_update == 3
    actual_model = load_inference_artifact(saved, CheckpointIdentity("new", 3), device="cpu")
    assert not torch.equal(actual_model.state_dict()["dit.output_head.projection.weight"], source_model.state_dict()["dit.output_head.projection.weight"])
    assert "dit.output_head.norm.weight" not in actual_model.state_dict()
    saved_optimizer = cast(dict[str, Any], torch.load(saved / "train_state/optimizer.pt", weights_only=True))
    assert all(entry["step"].item() == 3 for entry in saved_optimizer["state"].values())
    assert saved_optimizer["param_groups"][1]["lr"] == pytest.approx(production._config_learning_rate(loaded.config, 4))
    child = subprocess.run([sys.executable, "-c", "import runpy,sys;from pathlib import Path;scope=runpy.run_path(sys.argv[1]);scope['_phase'](Path(sys.argv[2]),Path(sys.argv[3]),1,'fresh')", str(Path(__file__).resolve()), str(saved), str(tmp_path / "output_model" / "phase2")], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1"}, capture_output=True, text=True, check=True)
    assert "synthetic update=4" in child.stdout
    final = tmp_path / "output_model" / "phase2" / "ckpt_4_fresh"
    _, fresh = read_raw_checkpoint_state(final)
    assert fresh.trainer == SingleGpuUpdateState(4, 4, 8)
    assert untouched == {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}


def test_output_paths_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "output_model" / "source"
    source.mkdir(parents=True)
    with pytest.raises(ValueError, match="overlap"):
        validate_paths(source, source / "nested")
    with pytest.raises(FileExistsError):
        validate_paths(source, source)
    link = tmp_path / "output_model" / "link"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        validate_paths(source, link / "new")


def test_numeric_and_growth_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    loaded = _loaded()
    config = loaded.config.model_copy(update={"growth": loaded.config.growth.model_copy(update={"enabled": True})})
    loaded = LoadedConfig(config, (), resolved_config_bytes(config).decode())
    (tmp_path / "model").mkdir()
    (tmp_path / "model/config.json").write_text(json.dumps({"architecture": {"dit": {}}}))
    (tmp_path / "resolved_config.toml").write_text(loaded.resolved_toml)
    state = RawCheckpointState(SingleGpuUpdateState(1, 1, 2), GrowthCheckpointState((0,), 1.0, "synthetic", 1, 256, 0, 1, (0,)), StageBudgetCheckpointState(0, 10), CheckpointCadence(1, 1.0, 1))
    def read_state(_path: Path) -> tuple[None, RawCheckpointState]:
        return None, state
    monkeypatch.setattr(resume_module, "read_raw_checkpoint_state", read_state)
    assert validate_resume_config(loaded, tmp_path, 2).growth == state.growth
    for section, change in (("optimizer", {"base_lr": 0.3}), ("kernels", {"attention_backend": "dense_sdpa_reference"})):
        config = loaded.config.model_copy(update={section: getattr(loaded.config, section).model_copy(update=change)})
        altered = LoadedConfig(config, (), resolved_config_bytes(config).decode())
        with pytest.raises(ValueError, match="unchanged"):
            validate_resume_config(altered, tmp_path, 2)
    active_growth = replace(state.growth, ramp_updates=10)
    state = replace(state, growth=replace(active_growth, alpha=resume_module.canonical_growth_alpha(active_growth, 1)))
    with pytest.raises(ValueError, match="active growth"):
        validate_resume_config(loaded, tmp_path, 2)


def test_optimizer_diagnostic_destinations_are_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAKURAMOON_FORENSIC_DIR", "previous")
    optimizer = SimpleNamespace(hard_fail_artifact_root=Path("old"), legacy_forensic_dir=Path("old"), emergency_capsule_root=Path("old"), lr=0.125)
    resume_module.isolate_optimizer_diagnostics(optimizer, tmp_path)
    assert optimizer.lr == 0.125
    assert all(getattr(optimizer, name).is_relative_to(tmp_path) for name in ("hard_fail_artifact_root", "legacy_forensic_dir", "emergency_capsule_root"))
    assert Path(os.environ["SAKURAMOON_FORENSIC_DIR"]).is_relative_to(tmp_path)
