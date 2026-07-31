from __future__ import annotations

import hashlib
import json
import multiprocessing
import random
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from safetensors.torch import (
    load_file,  # pyright: ignore[reportUnknownVariableType]
    save_file,  # pyright: ignore[reportUnknownVariableType]
)
from torch import nn

from sakuramoon.checkpoint import (
    CheckpointIdentity,
    CheckpointKind,
    GrowthCheckpointState,
    RawCheckpointState,
    load_inference_artifact,
    load_model_directory,
    load_model_only,
    load_raw_checkpoint,
    save_pma10,
    save_raw_checkpoint,
)
from sakuramoon.checkpoint.rng import capture_rank_rng
from sakuramoon.checkpoint.schema import CheckpointError
from sakuramoon.conditioning.style_resampler import StyleResampler
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.fault_injection import select_complete_raw_parent
from sakuramoon.model.dit import DenseDiT
from sakuramoon.model.growth import BASE_SLOT_IDS
from sakuramoon.objective.flow import flow_matching_loss
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit, build_adamw8bit
from sakuramoon.optim.clip import clip_grad_norm_fp32
from sakuramoon.train.step import SingleGpuUpdateState, TrainableComposite

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_RESOLVED_CONFIG = b'[run]\nname = "t044-gpu"\n'
_CONFIG_SHA256 = hashlib.sha256(_RESOLVED_CONFIG).hexdigest()


def _identity(checkpoint_id: str, update: int, schema_sha256: str) -> CheckpointIdentity:
    return CheckpointIdentity(
        checkpoint_id=checkpoint_id,
        update=update,
        config_sha256=_CONFIG_SHA256,
        dependency_sha256="b" * 64,
        parameter_schema_sha256=schema_sha256,
    )


def _raw_state(update: int) -> RawCheckpointState:
    return RawCheckpointState(
        trainer=SingleGpuUpdateState(update, update, update * 4),
        growth=GrowthCheckpointState(
            BASE_SLOT_IDS, 1.0, "S0", 1, 256, None, None
        ),
    )


def _compact_composite() -> TrainableComposite:
    return TrainableComposite(
        dit=DenseDiT(  # pyright: ignore[reportArgumentType]
            depth=16,
            input_channels=128,
            hidden_size=256,
            intermediate_size=256,
            q_heads=8,
            kv_heads=2,
            head_dim=32,
            rope_nope_dim=8,
            rope_y_dim=12,
            rope_x_dim=12,
            rope_position_scale=16.0,
            rope_theta=1000.0,
            norm_eps=1e-6,
            timestep_dim=256,
            size_dim=64,
            aspect_dim=64,
            condition_hidden_size=1024,
            stable_slot_count=24,
            modulation_chunks=6,
            final_modulation_size=512,
            out_channels=128,
            modality_init_std=0.02,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
            projection_bias=False,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            output_weight_zero_init=True,
            output_bias_zero_init=True,
        ),
        text=TextConditioner(
            input_size=256,
            adapter_size=256,
            output_size=256,
            groups=8,
            attention_heads=8,
            norm_eps=1e-6,
            mix_gate_init=0.0,
            layer_scale_init=1.0,
            projection_bias=False,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
        ),
        style=StyleResampler(
            input_size=256,
            hidden_size=256,
            intermediate_size=256,
            output_size=256,
            query_count=4,
            attention_heads=8,
            norm_eps=1e-6,
            init_std=0.02,
            projection_bias=False,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
        ),
    ).cuda()


def _optimizer(module: nn.Module, seed: int) -> IsolatedAdamW8bit:
    return build_adamw8bit(
        module,
        lr=2e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
        sr_seed=seed,
    )


def _update(module: TrainableComposite, optimizer: IsolatedAdamW8bit) -> None:
    for spec in optimizer.audit.specs:
        spec.parameter.grad = torch.randn_like(spec.parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _enable_fixed_batch_gradient_paths(module: TrainableComposite) -> None:
    hidden_size = module.dit.hidden_size
    with torch.no_grad():
        for bias in module.dit.conditioner.block_biases.values():
            bias[2 * hidden_size : 3 * hidden_size].fill_(0.05)
            bias[5 * hidden_size : 6 * hidden_size].fill_(0.05)
        projection = module.dit.output_head.projection
        projection.weight.copy_(
            torch.linspace(
                -0.001,
                0.001,
                projection.weight.numel(),
                device=projection.weight.device,
                dtype=projection.weight.dtype,
            ).reshape_as(projection.weight)
        )
        projection.bias.copy_(
            torch.linspace(
                -0.0001,
                0.0001,
                projection.bias.numel(),
                device=projection.bias.device,
                dtype=projection.bias.dtype,
            )
        )


def _fixed_batch_step(
    module: TrainableComposite,
    optimizer: IsolatedAdamW8bit,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    device = next(module.parameters()).device
    qwen_states = (
        torch.arange(2 * 5 * 7 * 256, device=device, dtype=torch.float32)
        .remainder(97)
        .sub_(48.0)
        .div_(97.0)
        .reshape(2, 5, 7, 256)
        .to(torch.bfloat16)
    )
    main_indices = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], device=device, dtype=torch.long
    )
    main_mask = torch.ones((2, 3), device=device, dtype=torch.bool)
    artist_indices = torch.tensor(
        [[3, 4], [0, 0]], device=device, dtype=torch.long
    )
    artist_mask = torch.tensor(
        [[True, True], [False, False]], device=device, dtype=torch.bool
    )
    use_null = torch.tensor([False, True], device=device, dtype=torch.bool)
    active_style_samples = torch.tensor([0], device=device, dtype=torch.long)
    latent = (
        torch.arange(2 * 128 * 2 * 2, device=device, dtype=torch.float32)
        .remainder(113)
        .sub_(56.0)
        .div_(113.0)
        .reshape(2, 128, 2, 2)
        .to(torch.bfloat16)
    )
    timestep = torch.tensor([0.25, 0.90], device=device, dtype=torch.float32)
    size_scale = torch.tensor([0.0, 0.0], device=device, dtype=torch.float32)
    aspect = torch.tensor([0.0, 0.0], device=device, dtype=torch.float32)

    text = module.text(qwen_states, main_indices, main_mask)
    style = module.style(
        qwen_states,
        artist_indices,
        artist_mask,
        use_null,
        active_style_samples,
    )
    output = module.dit(
        latent,
        text.tokens,
        text.mask,
        style.tokens,
        timestep,
        size_scale,
        aspect,
        growth_alpha=1.0,
    )
    clean = latent.float().mul(0.75).add(0.125)
    loss = flow_matching_loss(
        output,
        latent,
        clean,
        timestep,
        t_eps=0.05,
        noise_observation_boundary=0.95,
    )
    loss.loss.backward()  # pyright: ignore[reportUnknownMemberType]
    gradients = {
        name: parameter.grad.detach().cpu().contiguous().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }
    parameter_names = tuple(name for name, _parameter in module.named_parameters())
    assert tuple(sorted(gradients)) == tuple(sorted(parameter_names))
    clip = clip_grad_norm_fp32(module.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    tensors = {
        "clip_coefficient": clip.coefficient.detach().cpu(),
        "clip_post_norm": clip.post_clip_norm.detach().cpu(),
        "clip_pre_norm": clip.pre_clip_norm.detach().cpu(),
        "mean_loss": loss.loss.detach().cpu(),
        "output": output.detach().cpu().contiguous(),
        "per_sample_loss": loss.per_sample.detach().cpu(),
        "rng_torch_cpu": torch.rand(3),
        "rng_torch_cuda": torch.rand(3, device=device).cpu(),
    }
    tensors.update({f"gradient::{name}": value for name, value in gradients.items()})
    metadata: dict[str, object] = {
        "gradient_names": sorted(gradients),
        "numpy": float(np.random.random()),
        "python": random.random(),
    }
    return tensors, metadata


def _assert_nested_state_equal(left: object, right: object) -> None:
    if type(left).__name__ == "OptimState8bit":
        assert type(right).__name__ == "OptimState8bit"
        for attribute in ("block_size", "signed"):
            assert getattr(left, attribute) == getattr(right, attribute)
        for attribute in ("codes", "qmap", "scale"):
            torch.testing.assert_close(
                getattr(left, attribute),
                getattr(right, attribute),
                atol=0,
                rtol=0,
            )
        return
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        left_mapping = cast(dict[object, object], left)
        right_mapping = cast(dict[object, object], right)
        assert set(left_mapping) == set(right_mapping)
        for key in left_mapping:
            _assert_nested_state_equal(left_mapping[key], right_mapping[key])
        return
    if isinstance(left, list):
        assert isinstance(right, list)
        left_list = cast(list[object], left)
        right_list = cast(list[object], right)
        assert len(left_list) == len(right_list)
        for left_item, right_item in zip(left_list, right_list, strict=True):
            _assert_nested_state_equal(left_item, right_item)
        return
    if isinstance(left, tuple):
        assert isinstance(right, tuple)
        left_tuple = cast(tuple[object, ...], left)
        right_tuple = cast(tuple[object, ...], right)
        assert len(left_tuple) == len(right_tuple)
        for left_item, right_item in zip(left_tuple, right_tuple, strict=True):
            _assert_nested_state_equal(left_item, right_item)
        return
    assert type(right) is type(left) and right == left


def _fresh_process_worker(checkpoint: str, output_root: str) -> None:
    module = _compact_composite()
    optimizer = _optimizer(module, 9999)
    identity = _identity("fresh-input", 1, optimizer.audit.schema_sha256)
    load_raw_checkpoint(Path(checkpoint), module, optimizer, identity)
    draws = {
        "torch_cpu": torch.rand(3),
        "torch_cuda": torch.rand(3, device="cuda").cpu(),
    }
    metadata = {"numpy": float(np.random.random()), "python": random.random()}
    _update(module, optimizer)
    output_identity = _identity("fresh-output", 2, optimizer.audit.schema_sha256)
    save_raw_checkpoint(
        Path(output_root), output_identity, module, optimizer, _raw_state(2),
        resolved_config=_RESOLVED_CONFIG,
    )
    save_file(draws, str(Path(output_root) / "fresh-draws.safetensors"))
    (Path(output_root) / "fresh-draws.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )


def _fixed_batch_resume_worker(checkpoint: str, output_root: str) -> None:
    module = _compact_composite()
    optimizer = _optimizer(module, 9999)
    input_identity = _identity("fixed-input", 1, optimizer.audit.schema_sha256)
    load_raw_checkpoint(Path(checkpoint), module, optimizer, input_identity)
    tensors, metadata = _fixed_batch_step(module, optimizer)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output / "fixed-step.safetensors"))
    (output / "fixed-step.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    output_identity = _identity("fixed-output", 2, optimizer.audit.schema_sha256)
    save_raw_checkpoint(
        output,
        output_identity,
        module,
        optimizer,
        _raw_state(2),
        resolved_config=_RESOLVED_CONFIG,
    )


def test_real_torchao_raw_checkpoint_safe_load_matches_next_update(tmp_path: Path) -> None:
    random.seed(1210)
    np.random.seed(1211)
    torch.manual_seed(1212)  # pyright: ignore[reportUnknownMemberType]
    source = _compact_composite()
    source_optimizer = _optimizer(source, 1213)
    _update(source, source_optimizer)
    identity = _identity("gpu", 1, source_optimizer.audit.schema_sha256)
    state = _raw_state(1)
    checkpoint_parameter = next(source.parameters()).detach().cpu().clone()
    result = save_raw_checkpoint(
        tmp_path, identity, source, source_optimizer, state,
        resolved_config=_RESOLVED_CONFIG,
    )
    raw_manifest = json.loads((result.path / "manifest.json").read_bytes())
    assert raw_manifest["schema_version"] == 2
    assert (result.path / "resolved_config.toml").read_bytes() == _RESOLVED_CONFIG
    assert not (result.path / "train_state/data_state.json").exists()

    expected_draws = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
        torch.rand(3, device="cuda"),
    )
    _update(source, source_optimizer)
    expected_model = {
        name: tensor.detach().cpu().clone() for name, tensor in source.state_dict().items()
    }
    expected_sr = source_optimizer.sr_rng.state.clone()

    restored = _compact_composite()
    restored_optimizer = _optimizer(restored, 9999)
    loaded_state = load_raw_checkpoint(
        result.path, restored, restored_optimizer, identity
    )
    actual_draws = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
        torch.rand(3, device="cuda"),
    )
    _update(restored, restored_optimizer)

    assert loaded_state == state
    assert actual_draws[:2] == expected_draws[:2]
    torch.testing.assert_close(actual_draws[2], expected_draws[2], atol=0, rtol=0)
    torch.testing.assert_close(actual_draws[3], expected_draws[3], atol=0, rtol=0)
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), expected_model[name], atol=0, rtol=0)
    assert torch.equal(restored_optimizer.sr_rng.state, expected_sr)
    assert all(spec.step == 2 for spec in restored_optimizer.audit_state())

    standalone = tmp_path / "standalone-model"
    shutil.copytree(result.path / "model", standalone)
    inference_module, inference_identity, inference_kind = load_model_directory(
        standalone, device="cuda"
    )
    assert inference_identity == identity
    assert inference_kind is CheckpointKind.RAW
    torch.testing.assert_close(
        next(inference_module.parameters()).cpu(),
        checkpoint_parameter,
        atol=0,
        rtol=0,
    )

    with pytest.raises(CheckpointError, match="kind"):
        load_model_only(result.path, restored, identity)


@pytest.mark.parametrize(
    ("resolved_config", "message"),
    [
        (b"", "nonempty"),
        (b"[run\n", "valid UTF-8 TOML"),
        (b'[run]\nname = "wrong"\n', "hash"),
    ],
)
def test_raw_save_rejects_invalid_or_unbound_resolved_config(
    tmp_path: Path, resolved_config: bytes, message: str
) -> None:
    source = _compact_composite()
    optimizer = _optimizer(source, 1214)
    identity = _identity("bad-config", 0, optimizer.audit.schema_sha256)

    with pytest.raises(ValueError, match=message):
        save_raw_checkpoint(
            tmp_path,
            identity,
            source,
            optimizer,
            _raw_state(0),
            resolved_config=resolved_config,
        )

    assert list(tmp_path.iterdir()) == []


def test_lazy_and_lagging_optimizer_state_round_trips(tmp_path: Path) -> None:
    source = _compact_composite()
    source_optimizer = _optimizer(source, 1221)
    primary = next(
        spec for spec in source_optimizer.audit.specs if spec.name == "dit.input_projection.weight"
    )
    conditional = next(
        spec for spec in source_optimizer.audit.specs if spec.name == "style.null_tokens"
    )
    for spec in (primary, conditional):
        spec.parameter.grad = torch.ones_like(spec.parameter)
    source_optimizer.step()
    source_optimizer.zero_grad(set_to_none=True)
    primary.parameter.grad = torch.full_like(primary.parameter, 0.5)
    source_optimizer.step()
    source_optimizer.zero_grad(set_to_none=True)
    identity = _identity("lazy", 2, source_optimizer.audit.schema_sha256)
    result = save_raw_checkpoint(
        tmp_path, identity, source, source_optimizer, _raw_state(2),
        resolved_config=_RESOLVED_CONFIG,
    )

    restored = _compact_composite()
    restored_optimizer = _optimizer(restored, 9999)
    load_raw_checkpoint(result.path, restored, restored_optimizer, identity)
    audit = {spec.name: spec for spec in restored_optimizer.audit_state()}

    assert audit[primary.name].step == 2
    assert audit[conditional.name].step == 1
    assert sum(spec.initialized for spec in audit.values()) == 2
    for optimizer in (source_optimizer, restored_optimizer):
        by_name = {spec.name: spec for spec in optimizer.audit.specs}
        by_name[primary.name].parameter.grad = torch.full_like(
            by_name[primary.name].parameter, -0.25
        )
        by_name[conditional.name].parameter.grad = torch.full_like(
            by_name[conditional.name].parameter, 0.75
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    for source_parameter, restored_parameter in zip(
        source.parameters(), restored.parameters(), strict=True
    ):
        torch.testing.assert_close(
            source_parameter, restored_parameter, atol=0, rtol=0
        )
    assert restored_optimizer.audit_state() == source_optimizer.audit_state()


@pytest.mark.parametrize(
    ("relative", "mutation"),
    [
        ("train_state/optimizer.pt", "bitflip"),
        ("train_state/trainer_state.json", "missing"),
        ("resolved_config.toml", "bitflip"),
        ("train_state/data_state.json", "legacy"),
        ("opaque.bin", "legacy"),
        ("model/data_state.json", "legacy"),
        ("model/opaque.bin", "legacy"),
        ("train_state/growth_state.json", "missing"),
        ("train_state/rng/rank-0.safetensors", "bitflip"),
        ("train_state/rng/optimizer_sr.safetensors", "missing"),
    ],
)
def test_raw_sidecar_failure_precedes_all_state_changes(
    tmp_path: Path, relative: str, mutation: str
) -> None:
    source = _compact_composite()
    source_optimizer = _optimizer(source, 1222)
    _update(source, source_optimizer)
    identity = _identity("fault", 1, source_optimizer.audit.schema_sha256)
    complete = save_raw_checkpoint(
        tmp_path / "source", identity, source, source_optimizer, _raw_state(1),
        resolved_config=_RESOLVED_CONFIG,
    ).path
    damaged = tmp_path / "damaged" / complete.name
    damaged.parent.mkdir()
    shutil.copytree(complete, damaged)
    target_path = damaged / relative
    if mutation == "missing":
        target_path.unlink()
    elif mutation == "legacy":
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"{}\n")
        manifest_path = damaged / "manifest.json"
        manifest = cast(dict[str, object], json.loads(manifest_path.read_bytes()))
        files = cast(list[dict[str, object]], manifest["files"])
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
                "size": target_path.stat().st_size,
            }
        )
        files.sort(key=lambda record: cast(str, record["path"]))
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
    else:
        body = bytearray(target_path.read_bytes())
        body[-1] ^= 1
        target_path.write_bytes(body)

    target = _compact_composite()
    target_optimizer = _optimizer(target, 1223)
    parameters_before = tuple(parameter.detach().clone() for parameter in target.parameters())
    rng_before = capture_rank_rng()
    with pytest.raises(
        CheckpointError,
        match="file set|checksum|sidecar|model manifest|model payload",
    ):
        load_raw_checkpoint(damaged, target, target_optimizer, identity)

    assert not target_optimizer.optimizer.state
    assert all(
        torch.equal(parameter, before)
        for parameter, before in zip(target.parameters(), parameters_before, strict=True)
    )
    rng_after = capture_rank_rng()
    assert all(torch.equal(rng_before[key], rng_after[key]) for key in rng_before)


def test_fresh_process_resume_matches_next_update(tmp_path: Path) -> None:
    random.seed(1230)
    np.random.seed(1231)
    torch.manual_seed(1232)  # pyright: ignore[reportUnknownMemberType]
    source = _compact_composite()
    source_optimizer = _optimizer(source, 1233)
    _update(source, source_optimizer)
    identity = _identity("fresh-input", 1, source_optimizer.audit.schema_sha256)
    checkpoint = save_raw_checkpoint(
        tmp_path / "input", identity, source, source_optimizer, _raw_state(1),
        resolved_config=_RESOLVED_CONFIG,
    ).path
    expected_metadata = {"numpy": float(np.random.random()), "python": random.random()}
    expected_draws = {"torch_cpu": torch.rand(3), "torch_cuda": torch.rand(3, device="cuda")}
    _update(source, source_optimizer)
    expected_model = tuple(parameter.detach().cpu().clone() for parameter in source.parameters())
    expected_audit = source_optimizer.audit_state()
    expected_sr = source_optimizer.sr_rng.state.clone()
    del source_optimizer, source
    torch.cuda.empty_cache()

    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_fresh_process_worker,
        args=(str(checkpoint), str(tmp_path / "output")),
    )
    process.start()
    process.join(timeout=180)
    assert process.exitcode == 0
    draws = load_file(tmp_path / "output/fresh-draws.safetensors", device="cpu")
    metadata = json.loads((tmp_path / "output/fresh-draws.json").read_bytes())
    assert metadata == expected_metadata
    torch.testing.assert_close(draws["torch_cpu"], expected_draws["torch_cpu"], atol=0, rtol=0)
    torch.testing.assert_close(draws["torch_cuda"], expected_draws["torch_cuda"].cpu(), atol=0, rtol=0)

    restored = _compact_composite()
    restored_optimizer = _optimizer(restored, 1234)
    output_identity = _identity("fresh-output", 2, restored_optimizer.audit.schema_sha256)
    load_raw_checkpoint(
        tmp_path / "output/ckpt_2_fresh-output",
        restored,
        restored_optimizer,
        output_identity,
    )
    for parameter, expected in zip(restored.parameters(), expected_model, strict=True):
        torch.testing.assert_close(parameter.cpu(), expected, atol=0, rtol=0)
    assert restored_optimizer.audit_state() == expected_audit
    assert torch.equal(restored_optimizer.sr_rng.state, expected_sr)


def test_service_decoupled_resume_matches_fixed_external_batch(
    tmp_path: Path,
) -> None:
    random.seed(1240)
    np.random.seed(1241)
    torch.manual_seed(1242)  # pyright: ignore[reportUnknownMemberType]
    source = _compact_composite()
    _enable_fixed_batch_gradient_paths(source)
    source_optimizer = _optimizer(source, 1243)
    _fixed_batch_step(source, source_optimizer)
    input_identity = _identity(
        "fixed-input", 1, source_optimizer.audit.schema_sha256
    )
    checkpoint = save_raw_checkpoint(
        tmp_path / "fixed-input",
        input_identity,
        source,
        source_optimizer,
        _raw_state(1),
        resolved_config=_RESOLVED_CONFIG,
    ).path
    expected_tensors, expected_metadata = _fixed_batch_step(
        source, source_optimizer
    )

    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_fixed_batch_resume_worker,
        args=(str(checkpoint), str(tmp_path / "fixed-output")),
    )
    process.start()
    process.join(timeout=180)
    assert process.exitcode == 0

    actual_tensors = load_file(
        tmp_path / "fixed-output/fixed-step.safetensors", device="cpu"
    )
    actual_metadata = json.loads(
        (tmp_path / "fixed-output/fixed-step.json").read_bytes()
    )
    assert actual_metadata == expected_metadata
    assert set(actual_tensors) == set(expected_tensors)
    assert len(actual_metadata["gradient_names"]) == len(
        tuple(source.named_parameters())
    )
    for name, expected in expected_tensors.items():
        torch.testing.assert_close(
            actual_tensors[name], expected, atol=0, rtol=0
        )

    restored = _compact_composite()
    restored_optimizer = _optimizer(restored, 1244)
    output_identity = _identity(
        "fixed-output", 2, restored_optimizer.audit.schema_sha256
    )
    loaded_state = load_raw_checkpoint(
        tmp_path / "fixed-output/ckpt_2_fixed-output",
        restored,
        restored_optimizer,
        output_identity,
    )
    assert loaded_state == _raw_state(2)
    for name, parameter in restored.named_parameters():
        torch.testing.assert_close(
            parameter,
            dict(source.named_parameters())[name],
            atol=0,
            rtol=0,
        )
    _assert_nested_state_equal(
        source_optimizer.optimizer.state_dict(),
        restored_optimizer.optimizer.state_dict(),
    )
    torch.testing.assert_close(
        source_optimizer.sr_rng.state,
        restored_optimizer.sr_rng.state,
        atol=0,
        rtol=0,
    )


def test_pma10_real_composite_loads_as_fresh_inference_artifact(
    tmp_path: Path,
) -> None:
    source = _compact_composite()
    optimizer = _optimizer(source, 1236)
    raw_sources: list[Path] = []
    for update in range(1, 11):
        with torch.no_grad():
            for tensor in source.state_dict(keep_vars=True).values():
                tensor.fill_(float(update))
        raw_sources.append(
            save_raw_checkpoint(
                tmp_path,
                _identity(f"pma-source-{update:02d}", update, optimizer.audit.schema_sha256),
                source,
                optimizer,
                _raw_state(update),
                resolved_config=_RESOLVED_CONFIG,
            ).path
        )

    identity = _identity("pma-roundtrip", 10, optimizer.audit.schema_sha256)
    result = save_pma10(tmp_path, identity, tuple(raw_sources))
    restored = load_inference_artifact(result.path, identity, device="cuda")

    assert type(restored) is TrainableComposite
    for tensor in restored.state_dict().values():
        torch.testing.assert_close(
            tensor,
            torch.full_like(tensor, 5.5),
            atol=0,
            rtol=0,
        )


def test_fault_recovery_selector_requires_exact_complete_raw_parent(
    tmp_path: Path,
) -> None:
    module = _compact_composite()
    optimizer = _optimizer(module, 1235)
    _update(module, optimizer)
    identity = _identity("fault-parent", 1, optimizer.audit.schema_sha256)
    checkpoint = save_raw_checkpoint(
        tmp_path, identity, module, optimizer, _raw_state(1),
        resolved_config=_RESOLVED_CONFIG,
    ).path

    selected = select_complete_raw_parent(
        tmp_path, checkpoint_id="fault-parent", successful_update=1
    )

    assert selected.path == checkpoint
    assert selected.identity == identity
    with pytest.raises(CheckpointError, match="exact COMPLETE"):
        select_complete_raw_parent(
            tmp_path, checkpoint_id="fault-parent", successful_update=2
        )
