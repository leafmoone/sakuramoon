"""M3 latent flow loop unit tests (CPU-only; no HCU / VAE weights needed).

Covers the deterministic-schedule contract (plan §11.5), the flow-target
draws (plan §5), the validation metrics (plan §13 M3 checklist) and the
FlowSampler adapter.
"""

import torch
from anime_sr.config.schema import Config, LatentFlowSpec
from anime_sr.train.latent_flow import (
    _LatentVelocity,
    build_flow_targets,
    latent_sample_index,
    latent_val_metrics,
    velocity_cosine,
)


def test_latent_flow_spec_defaults() -> None:
    lf = LatentFlowSpec()
    assert lf.batch_size == 8
    assert lf.save_every_steps == 1_000
    assert lf.val_every_steps == 5_000
    assert lf.val_samples == 8
    assert lf.prefetch_depth == 2  # double-buffered M3 default
    assert lf.out_dir == "output_model/latent-flow"
    cfg = Config()
    assert isinstance(cfg.latent_flow, LatentFlowSpec)
    cfg2 = Config(latent_flow=LatentFlowSpec(batch_size=4, prefetch_depth=0))
    cfg2.validate_all()
    assert cfg2.latent_flow.batch_size == 4
    assert cfg2.latent_flow.prefetch_depth == 0  # sync canary mode


def test_prefetch_depth_quad_config() -> None:
    """M1 #8 Phase I override: quad buffering via [latent_flow] prefetch_depth."""
    cfg = Config(latent_flow=LatentFlowSpec(prefetch_depth=4))
    cfg.validate_all()
    assert cfg.latent_flow.prefetch_depth == 4


def test_schedule_matches_m2_formula() -> None:
    n, bs, world = 10_000, 8, 2
    for s in (0, 1, 24, 25, 99):
        for r in range(world):
            for i in range(bs):
                expect = (s * (bs * world) + r * bs + i) % n
                assert latent_sample_index(s, r, i, bs, world, n) == expect
    # wrap-around on a small set
    assert latent_sample_index(3, 0, 2, 8, 1, 5) == (3 * 8 + 0 + 2) % 5


def test_flow_targets_reproducible_with_seeded_generator() -> None:
    cfg = Config()
    z_hr = torch.randn(4, 128, 8, 8) * 2.0
    z_lr = torch.randn(4, 128, 8, 8) * 0.5
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = build_flow_targets(z_hr, z_lr, cfg, generator=g1)
    b = build_flow_targets(z_hr, z_lr, cfg, generator=g2)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    rt, v_star, sigma, t = a
    # identity: rt = (1-t) r0 + t delta and v* = delta - r0
    #  =>  r0 = rt - t * v*   (delta - v* = r0)
    delta = z_hr - z_lr
    r0 = rt - t.view(-1, 1, 1, 1) * v_star
    assert torch.allclose(r0 + v_star, delta, atol=1e-6)
    # zero-sigma samples must have r0 == 0 exactly (deterministic path)
    zero = sigma == 0
    assert zero.any()
    assert torch.equal(r0[zero], torch.zeros_like(r0[zero]))
    assert torch.allclose(sigma[~zero], sigma[~zero] * 1.0)  # nonzero branch
    nz = sigma[~zero]
    assert (nz >= 0.02 - 1e-6).all() and (nz <= 0.15 + 1e-6).all()
    # t is a uniform time sample in [0, 1)
    assert (t >= 0).all() and (t < 1.0).all()


def test_flow_targets_sigma_mix_stats() -> None:
    cfg = Config()
    z_hr = torch.randn(64, 128, 8, 8)
    z_lr = torch.randn(64, 128, 8, 8)
    _, _, sigma, _ = build_flow_targets(z_hr, z_lr, cfg)
    zero_frac = (sigma == 0).float().mean().item()
    assert 0.5 < zero_frac < 0.95  # 0.75 +/- sampling noise (plan §5.6)


def test_val_metrics_perfect_and_anchor() -> None:
    z_hr = torch.randn(4, 128, 8, 8)
    z_lr = z_hr + 0.5
    m1 = latent_val_metrics(z_hr, z_lr, z_hr)  # perfect prediction
    assert m1["l1"] == 0.0
    assert m1["toward_frac"] == 1.0
    m0 = latent_val_metrics(z_hr, z_lr, z_lr)  # pure anchor: never closer
    assert m0["toward_frac"] == 0.0
    # half the batch perfect, the other half overshooting (farther than
    # the anchor) -> exactly 0.5 toward fraction
    z_hat = torch.empty_like(z_hr)
    z_hat[:2] = z_hr[:2]
    z_hat[2:] = z_lr[2:] + 2.0 * (z_hr[2:] - z_lr[2:])
    m_half = latent_val_metrics(z_hr, z_lr, z_hat)
    assert m_half["toward_frac"] == 0.5


def test_velocity_cosine() -> None:
    v = torch.randn(4, 128, 8, 8)
    assert abs(velocity_cosine(v, v) - 1.0) < 1e-5
    assert velocity_cosine(v, -v) < -0.99
    v2 = torch.zeros_like(v)
    assert velocity_cosine(v, v2) == 0.0


def test_val_lq_batching_for_interpolate_and_vae() -> None:
    """Regression (M3 smoke crash at val step 5000): ``degrade_hr`` returns
    the UNBATCHED LQ ``[3, h, w]``; both ``F.interpolate`` and the frozen
    VAE's ``encode`` require a batch dim, so the val path must add one.
    Feeding the 3D tensor straight into ``F.interpolate`` raises
    "Input and output must have the same number of spatial dimensions"."""
    import torch.nn.functional as F
    from anime_sr.data.degradation import degrade_hr

    cfg = Config()
    # HR crops must be multiples of 64 (data-contract §1); smallest is 64
    hr_crop = torch.rand(3, 64, 64, dtype=torch.float32) * 2.0 - 1.0
    lq, _ = degrade_hr(
        hr_crop,
        cfg,
        global_seed=1,
        sample_id="s0",
        data_cycle=0,
        exposure_index=0,
    )
    # degrade_hr contract: unbatched [3, H/4, W/4]
    assert lq.dim() == 3 and lq.shape == (3, 16, 16)

    bucket_hr = 64
    # the fix: batch dim for interpolate, drop it afterwards
    lq_up = F.interpolate(
        lq.unsqueeze(0), size=(bucket_hr, bucket_hr), mode="bicubic"
    ).squeeze(0)
    assert lq_up.shape == (3, bucket_hr, bucket_hr)
    # the frozen VAE encode takes [B, 3, H, W]
    enc_in = lq_up.unsqueeze(0)
    assert enc_in.dim() == 4 and enc_in.shape == (1, 3, bucket_hr, bucket_hr)


def test_latent_velocity_adapter() -> None:
    from torch import nn

    class _FakeTrunk(nn.Module):
        """Stand-in for the trunk signature (rt, z_lr, t, sigma, ...)."""

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, rt, z_lr, t, sigma):
            self.calls += 1
            return rt + z_lr

    fake = _FakeTrunk()
    adapter = _LatentVelocity(fake)  # type: ignore[arg-type]
    rt = torch.zeros(2, 4, 4, 4)
    z_lr = torch.ones(2, 4, 4, 4)
    t = torch.full((2,), 0.5)
    sigma = torch.zeros(2)
    out = adapter(rt, t, sigma, z_lr)  # FlowSampler protocol: (rt, t, sigma, cond)
    assert torch.allclose(out, rt + z_lr)
    assert fake.calls == 1


def _no_autocast():
    from contextlib import nullcontext

    return nullcontext()


def test_endpoint_consistency_exact_field() -> None:
    """Revised M3 #3: with the exact (constant) field, delta_hat_t == delta at
    every probe time, so every endpoint L1 is 0."""
    from anime_sr.train.latent_flow import endpoint_consistency
    from torch import nn

    z_hr = torch.randn(4, 8, 8, 8)
    z_lr = torch.randn(4, 8, 8, 8)
    delta = z_hr - z_lr

    class _ExactField(nn.Module):
        def forward(self, rt, z_lr_, t, sigma):
            return delta

    ep = endpoint_consistency(
        _ExactField(), z_hr, z_lr, _no_autocast, torch.device("cpu")
    )
    assert set(ep) == {"ep_l1_t0", "ep_l1_t25", "ep_l1_t50", "ep_l1_t75"}
    for k, v in ep.items():
        assert v < 1e-5, f"{k}={v}"


def test_trajectory_deviation_exact_field() -> None:
    """Revised M3 #3: with the exact field every solver trajectory lands on
    r_true(t) = t*delta (r0 = 0), so D_t == 0 at every sub-step end."""
    from anime_sr.train.latent_flow import trajectory_deviation
    from torch import nn

    z_hr = torch.randn(4, 8, 8, 8)
    z_lr = torch.randn(4, 8, 8, 8)
    delta = z_hr - z_lr

    class _ExactField(nn.Module):
        def forward(self, rt, z_lr_, t, sigma):
            return delta

    dev = torch.device("cpu")
    for solver, n in (("euler", 4), ("heun", 4)):
        d = trajectory_deviation(
            _ExactField(), z_hr, z_lr, solver=solver, n_steps=n,
            autocast=_no_autocast, device=dev,
        )
        assert list(d) == [f"D_t{round((k + 1) / n * 100):03d}" for k in range(n)]
        for k, v in d.items():
            assert v < 1e-5, f"{solver} {k}={v}"


def test_endpoint_deviation_zero_field_internal_consistency() -> None:
    """Zero model (v = 0): 1-step L1, the endpoint L1 at t=0 and the
    trajectory D at t=1 (euler, N=1) must all equal |z_lr - z_hr| — the
    revised-#3 metrics are internally consistent (revised M3 #3)."""
    from anime_sr.train.latent_flow import endpoint_consistency, trajectory_deviation
    from torch import nn

    z_hr = torch.randn(2, 8, 8, 8)
    z_lr = torch.randn(2, 8, 8, 8)
    l1_1 = (z_lr - z_hr).abs().mean().item()

    class _ZeroField(nn.Module):
        def forward(self, rt, z_lr_, t, sigma):
            return torch.zeros_like(rt)

    dev = torch.device("cpu")
    ep = endpoint_consistency(_ZeroField(), z_hr, z_lr, _no_autocast, dev)
    d1 = trajectory_deviation(
        _ZeroField(), z_hr, z_lr, solver="euler", n_steps=1,
        autocast=_no_autocast, device=dev,
    )
    assert abs(ep["ep_l1_t0"] - l1_1) < 1e-6  # endpoint at t=0 == 1-step L1
    assert abs(d1["D_t100"] - l1_1) < 1e-6  # D at t=1 (euler N=1) == 1-step L1
    # and the zero field degrades gracefully: later probes are no worse than t=0
    assert ep["ep_l1_t75"] <= ep["ep_l1_t0"] + 1e-6
