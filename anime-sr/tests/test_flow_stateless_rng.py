"""M4-final P0: stateless rank-specific flow-target RNG (flow_step_seed).

The train flow draw must be a pure function of (global_seed, step, rank):
bit-identical across fresh processes/generators, independent per rank,
step-varying, crash-resume exact without any process-global RNG snapshot,
and non-polluting to the global RNG.
"""

from __future__ import annotations

import torch
from anime_sr.config.schema import Config, FlowSpec
from anime_sr.train.latent_flow import build_flow_targets, flow_step_seed

GLOBAL_SEED = 42  # SRDataset default (fixed dataset-level constant)

# zero_fraction=0.0 makes every sample draw a non-zero sigma -> the source
# noise (and hence rt/v_star) is ALWAYS rank/step dependent: no flaky edge
# where an all-zero sigma batch makes two ranks coincidentally equal.
_CFG = Config(flow=FlowSpec(train_sigma_zero_fraction=0.0, train_sigma_noise_range=[0.02, 0.15]))


def _targets(step: int, rank: int, seed: int = GLOBAL_SEED, b: int = 8):
    """One train step's flow target under the stateless per-rank seed
    (the same call the trainer makes each step)."""
    z_hr = torch.zeros(b, 1, 4, 4)  # deterministic inputs
    z_lr = torch.full((b, 1, 4, 4), 0.25)
    g = torch.Generator(device="cpu")
    g.manual_seed(flow_step_seed(seed, step, rank))
    return build_flow_targets(z_hr, z_lr, _CFG, generator=g, device="cpu")


def test_same_step_rank_exact() -> None:
    a = _targets(step=123, rank=0)
    b = _targets(step=123, rank=0)
    for x, y in zip(a, b):
        assert torch.equal(x, y), "same (step, rank) must be bit-identical"


def test_different_rank_differs() -> None:
    r0 = _targets(step=123, rank=0)
    r1 = _targets(step=123, rank=1)
    assert not torch.equal(r0[0], r1[0]), "rt must differ across ranks"
    assert not torch.equal(r0[1], r1[1]), "v_star must differ across ranks"


def test_different_step_differs() -> None:
    s1 = _targets(step=100, rank=0)
    s2 = _targets(step=101, rank=0)
    assert not torch.equal(s1[0], s2[0]), "rt must differ across steps"


def test_resume_sequence_exact() -> None:
    """N+M continuous vs N -> crash -> resume -> M: the flow target
    SEQUENCE is exact — the resumed tail (fresh generator, same
    (step, rank) seeds) reproduces the continuous run bit-for-bit."""
    N, M = 7, 5
    continuous = [_targets(step=i, rank=1) for i in range(N, N + M)]
    crash_at = 3  # crash 3 steps into the window; resume rebuilds from ckpt
    resumed = [_targets(step=i, rank=1) for i in range(N + crash_at, N + M)]
    for cont, res in zip(continuous[crash_at:], resumed):
        for x, y in zip(cont, res):
            assert torch.equal(x, y), "resumed flow targets must be bit-exact"


def test_ddp_ranks_distinct() -> None:
    for step in (0, 1, 12345, 374999):
        assert flow_step_seed(GLOBAL_SEED, step, 0) != flow_step_seed(
            GLOBAL_SEED, step, 1
        ), f"rank seeds must be distinct at step {step}"


def test_generator_does_not_pollute_global_rng() -> None:
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    _targets(step=5, rank=0)
    _targets(step=6, rank=1)
    after = torch.get_rng_state()
    assert torch.equal(before, after), "global CPU RNG must be untouched"
