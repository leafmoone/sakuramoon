from __future__ import annotations

import copy
import os
from dataclasses import replace
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
from sakuramoon.train.preflight import (
    PreflightError,
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
    accumulation: int = 4,
    ready_batches: int | None = None,
) -> S0CapacitySweepRow:
    return S0CapacitySweepRow(
        candidate_id=candidate_id,
        local_batch=1,
        accumulation=accumulation,
        global_batch=accumulation,
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


def test_capacity_row_uses_explicit_accumulation_for_global_batch() -> None:
    base = _row("workers-1", 1)

    alternate = _row("alternate-accumulation", 1, accumulation=2)
    assert alternate.accumulation == 2
    assert alternate.global_batch == 2

    with pytest.raises(ValueError, match="row is invalid"):
        replace(
            base,
            candidate_id="wrong-global-batch",
            global_batch=8,
        )


def test_non_toml_readiness_blockers_are_stable_and_do_not_invent_values() -> None:
    assert S0_GOVERNED_SEMANTIC_BLOCKERS == ()
    assert S0_RUNTIME_INTEGRATION_BLOCKERS == ()


def test_logging_checkpoint_preflight_rejects_nondurable_existing_file(
    tmp_path: Path, valid_payload: dict[str, Any]
) -> None:
    valid_payload["checkpoint"]["full_every_updates"] = 37
    valid_payload["checkpoint"]["slots"] = 5
    valid_payload["storage"]["checkpoint_copies"] = 6
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
