from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import tomli_w

from sakuramoon.config import (
    S0_GOVERNED_SEMANTIC_BLOCKERS,
    S0_RUNTIME_INTEGRATION_BLOCKERS,
    S0CapacitySweepRow,
    validate_s0_capacity_sweep_matrix,
)
from sakuramoon.config.load import (
    ConfigurationError,
    load_config,
    unresolved_config_bindings,
)
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.eval.spec import PromptCase, PromptManifest
from sakuramoon.train.preflight import (
    PreflightError,
    require_evaluation_identities,
    require_logging_checkpoint_contracts,
)


def test_unresolved_bindings_preserve_exact_path_token_and_kind(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    secret_environment: dict[str, str],
) -> None:
    payload = copy.deepcopy(valid_payload)
    payload["run"]["seed"] = "REQUIRED_S0_SEED"
    payload["stage"]["local_batch"] = "BENCHMARK_S0_LOCAL_BATCH"
    path = tmp_path / "candidate.toml"
    path.write_text(tomli_w.dumps(payload), encoding="utf-8")

    bindings = unresolved_config_bindings(Path(path.name), config_root=tmp_path)

    assert tuple(
        (binding.path, binding.sentinel, binding.kind) for binding in bindings
    ) == (
        ("run.seed", "REQUIRED_S0_SEED", "required"),
        ("stage.local_batch", "BENCHMARK_S0_LOCAL_BATCH", "benchmark"),
    )
    with pytest.raises(ConfigurationError) as captured:
        load_config(Path(path.name), config_root=tmp_path, environment=secret_environment)
    assert captured.value.unresolved_bindings == bindings
    assert "run.seed=REQUIRED_S0_SEED" in str(captured.value)
    assert "stage.local_batch=BENCHMARK_S0_LOCAL_BATCH" in str(captured.value)


def _row(
    candidate_id: str,
    workers: int,
    *,
    ready_batches: int | None = None,
) -> S0CapacitySweepRow:
    return S0CapacitySweepRow(
        candidate_id=candidate_id,
        local_batch=1,
        accumulation=1,
        global_batch=1,
        activation_checkpoint_mode="none",
        cache_low_watermark_gib=1,
        cache_high_watermark_gib=2,
        download_concurrency=workers,
        verified_shard_lookahead=workers,
        persistent_workers_per_rank=workers,
        ready_batches_per_rank=workers if ready_batches is None else ready_batches,
        lease_channel_capacity=workers,
        ack_channel_capacity=workers,
    )


def test_capacity_matrix_is_explicit_and_covers_governed_worker_axis() -> None:
    rows = (_row("workers-1", 1), _row("workers-2", 2), _row("workers-3", 3))

    assert validate_s0_capacity_sweep_matrix(rows) is rows

    with pytest.raises(ValueError, match="1/2/3"):
        validate_s0_capacity_sweep_matrix(rows[:2])
    with pytest.raises(ValueError, match="unique explicit rows"):
        validate_s0_capacity_sweep_matrix((rows[0], rows[0], rows[2]))
    with pytest.raises(ValueError, match="row is invalid"):
        _row("partial-ready-depth", 2, ready_batches=3)


def test_non_toml_readiness_blockers_are_stable_and_do_not_invent_values() -> None:
    assert tuple(blocker.code for blocker in S0_GOVERNED_SEMANTIC_BLOCKERS) == (
        "S0_WARMUP_FUNCTION_UNRESOLVED",
        "S0_PASS_INDEX_OWNERSHIP_UNRESOLVED",
        "S0_FORMAL_PROMPT_CONDITION_CONTRACT_UNRESOLVED",
    )
    assert tuple(blocker.code for blocker in S0_RUNTIME_INTEGRATION_BLOCKERS) == (
        "S0_LIVE_READY_QUEUE_DEPTH_UNBOUND",
        "S0_DIT_FLOPS_OBSERVATION_UNBOUND",
    )
    rendered = repr(
        (S0_GOVERNED_SEMANTIC_BLOCKERS, S0_RUNTIME_INTEGRATION_BLOCKERS)
    )
    assert "pass_index" in rendered
    assert "flat condition strings cannot be guessed" in rendered
    assert "ready-batch capacity is not an observation" in rendered
    assert "planned_dit_flops and planned_updates cannot be used to derive it" in rendered


def _config_with_evaluation_files(
    root: Path, payload: dict[str, Any], *, prompt_count: int = 100
) -> RuntimeConfig:
    synthetic = root / "synthetic"
    synthetic.mkdir()
    prompts = PromptManifest(
        tuple(
            PromptCase(
                f"p{index:05d}",
                f"prompt {index}",
                (),
                index,
                256,
                256,
            )
            for index in range(prompt_count)
        )
    )
    files = {
        "prompt_manifest": (synthetic / "prompts.json", prompts.canonical_bytes()),
        "feature_extractor": (synthetic / "extractor.safetensors", b"extractor\n"),
        "preprocess": (synthetic / "preprocess.json", b"preprocess\n"),
        "real_stats": (synthetic / "real-stats.npz", b"real-stats\n"),
    }
    for path, content in files.values():
        path.write_bytes(content)
    evaluation = payload["evaluation"]
    fid = evaluation["fid"]
    fid["acceptance_samples"] = 100
    evaluation["is"]["acceptance_samples"] = 100
    evaluation["prompt_manifest_sha256"] = prompts.sha256
    fid["feature_extractor_sha256"] = hashlib.sha256(
        files["feature_extractor"][1]
    ).hexdigest()
    fid["preprocess_sha256"] = hashlib.sha256(files["preprocess"][1]).hexdigest()
    fid["real_stats_sha256"] = hashlib.sha256(files["real_stats"][1]).hexdigest()
    return RuntimeConfig.model_validate(payload)


def test_evaluation_preflight_verifies_all_local_identities_and_prompt_capacity(
    tmp_path: Path, valid_payload: dict[str, Any]
) -> None:
    config = _config_with_evaluation_files(tmp_path, valid_payload)

    identities = require_evaluation_identities(config, tmp_path)

    assert tuple(item.role for item in identities) == (
        "prompt manifest",
        "feature extractor",
        "preprocess",
        "real stats",
    )
    assert all(item.size > 0 and item.path.is_file() for item in identities)


def test_evaluation_preflight_hard_fails_hash_symlink_and_undersized_prompt(
    tmp_path: Path, valid_payload: dict[str, Any]
) -> None:
    config = _config_with_evaluation_files(tmp_path, valid_payload)
    extractor = tmp_path / config.evaluation.fid.feature_extractor_path
    extractor.write_bytes(b"changed\n")
    with pytest.raises(PreflightError, match="feature extractor.*SHA-256"):
        require_evaluation_identities(config, tmp_path)

    extractor.unlink()
    target = tmp_path / "extractor-target"
    target.write_bytes(b"extractor\n")
    extractor.symlink_to(target)
    with pytest.raises(PreflightError, match="feature extractor.*symlink"):
        require_evaluation_identities(config, tmp_path)

    isolated = tmp_path / "undersized"
    isolated.mkdir()
    undersized_payload = copy.deepcopy(valid_payload)
    undersized = _config_with_evaluation_files(
        isolated, undersized_payload, prompt_count=99
    )
    with pytest.raises(PreflightError, match="cannot cover"):
        require_evaluation_identities(undersized, isolated)


def test_logging_checkpoint_preflight_rejects_nondurable_existing_file(
    tmp_path: Path, valid_payload: dict[str, Any]
) -> None:
    config = RuntimeConfig.model_validate(valid_payload)
    checkpoint_root, local_path, retry_path = require_logging_checkpoint_contracts(
        config, tmp_path
    )
    assert checkpoint_root.is_dir()
    assert local_path != retry_path

    local_path.write_text("existing\n", encoding="utf-8")
    os.chmod(local_path, 0o644)
    with pytest.raises(PreflightError, match="mode 0600"):
        require_logging_checkpoint_contracts(config, tmp_path)
