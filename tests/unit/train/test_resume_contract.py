"""B1 tests for the fail-closed learning-rate / batch-protocol resume contract.

The transparent canary cut-over may only resume a checkpoint whose saved
optimizer learning rate and global batch protocol equal the configured
values exactly; anything else is a protocol break and must fail closed.
"""

from __future__ import annotations

import pytest

from sakuramoon.train.preflight import (
    RESUME_CONTRACT_BATCH_LEAVES,
    PreflightError,
    require_resume_contract,
)

BASE = (
    "[stage]\n"
    "global_batch = 760\n"
    "local_batch = 19\n"
    "accumulation = 20\n"
    "\n"
    "[optimizer]\n"
    "base_lr = 0.0000475\n"
)

# 0.0000475 * 760 / 256 (the 2-dcu canary rate) and
# 0.0000475 * 400 / 256 (the live 1-dcu p50 canary rate).
TWO_GROUP_LRS = (0.000141015625, 0.000141015625)
LIVE_CANARY_LR = (0.00007421875,)


def test_pinned_batch_leaves() -> None:
    assert RESUME_CONTRACT_BATCH_LEAVES == ("stage.global_batch",)


def test_matching_contract_passes() -> None:
    require_resume_contract(
        BASE, BASE, TWO_GROUP_LRS, TWO_GROUP_LRS
    )


def test_source_global_batch_drift_fails() -> None:
    drifted = BASE.replace("global_batch = 760", "global_batch = 800")
    with pytest.raises(PreflightError, match="batch protocol drift"):
        require_resume_contract(BASE, drifted, TWO_GROUP_LRS, TWO_GROUP_LRS)


def test_current_global_batch_drift_fails() -> None:
    drifted = BASE.replace("global_batch = 760", "global_batch = 400")
    with pytest.raises(PreflightError, match="batch protocol drift"):
        require_resume_contract(BASE, drifted, LIVE_CANARY_LR, LIVE_CANARY_LR)


def test_missing_leaf_in_source_fails() -> None:
    without_stage = BASE.replace("[stage]\nglobal_batch = 760\n", "")
    with pytest.raises(PreflightError, match="cannot locate stage.global_batch"):
        require_resume_contract(
            without_stage, BASE, TWO_GROUP_LRS, TWO_GROUP_LRS
        )


def test_missing_leaf_in_current_fails() -> None:
    without_stage = BASE.replace("[stage]\nglobal_batch = 760\n", "")
    with pytest.raises(PreflightError, match="cannot locate stage.global_batch"):
        require_resume_contract(
            BASE, without_stage, TWO_GROUP_LRS, TWO_GROUP_LRS
        )


def test_saved_learning_rate_drift_fails() -> None:
    # The source checkpoint ran the live 1-dcu canary rate; the configured
    # rate is the 2-dcu 760-batch value -> exact-equality drift, group 0.
    with pytest.raises(PreflightError, match="learning-rate drift at optimizer group 0"):
        require_resume_contract(
            BASE, BASE, LIVE_CANARY_LR, (0.000141015625,)
        )


def test_second_group_drift_is_reported() -> None:
    drifted = (TWO_GROUP_LRS[0], 0.00001)
    with pytest.raises(PreflightError, match="group 1"):
        require_resume_contract(BASE, BASE, drifted, TWO_GROUP_LRS)


def test_group_count_mismatch_fails() -> None:
    with pytest.raises(PreflightError, match="group count differs"):
        require_resume_contract(BASE, BASE, (0.000141015625,), TWO_GROUP_LRS)


def test_missing_saved_rates_fail_closed() -> None:
    with pytest.raises(PreflightError, match="saved optimizer"):
        require_resume_contract(BASE, BASE, None, TWO_GROUP_LRS)


def test_unparseable_source_fails() -> None:
    with pytest.raises(PreflightError, match="cannot parse"):
        require_resume_contract("{not toml", BASE, TWO_GROUP_LRS, TWO_GROUP_LRS)


def test_unparseable_current_fails() -> None:
    with pytest.raises(PreflightError, match="cannot parse"):
        require_resume_contract(BASE, "global_batch", TWO_GROUP_LRS, TWO_GROUP_LRS)
