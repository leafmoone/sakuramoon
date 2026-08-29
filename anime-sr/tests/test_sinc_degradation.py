"""P1-fix sinc degradation: windowed ideal low-pass kernel contract.

M4-prep work order (2026-08-29): the legacy truncated-sinc used
``sin(pi*x/f)/(pi*x)`` with center ``1/(2f)`` — half the correct center
tap (the limit of the sinc at n=0 is ``1/f``) — and derived the cutoff
from the Gaussian sigma so fc could exceed Nyquist (fc > 0.5 is not a
low-pass at all). The kernel is now the standard ideal low-pass

    h[n] = sin(2*pi*fc*n)/(pi*n),  h[0] = 2*fc,   fc in (0, 0.5]

truncated at ~4 zero crossings, Hann-windowed, DC-gain 1.

Guarantees under test:
  * kernel sum ~= 1 (DC gain, flat crops stay flat);
  * lower fc attenuates the Nyquist checkerboard more (stronger low-pass);
  * a low-fc sinc kills a high-frequency checkerboard (energy collapse);
  * the impulse response is NOT the identity (it is a real blur);
  * deterministic: same fc -> bit-identical kernel; same exposure seed ->
    bit-identical sampled sinc_fc; non-sinc exposures keep sinc_fc = 0.
"""

from __future__ import annotations

import pytest
import torch
from anime_sr.config.schema import Config
from anime_sr.data.degradation import (
    DegradationParams,
    _blur_plan,
    _conv2d_separable,
    _sinc_kernel_1d,
    apply_degradation,
    exposure_seed,
    sample_params,
)

CFG = Config()


def _sinc_params(fc: float) -> DegradationParams:
    """A params record forced to blur_kind='sinc' with a given cutoff."""
    base = sample_params(exposure_seed(1, "sinc-base", 0, 0), CFG).to_dict()
    base.update(profile="P4_severe", blur_kind="sinc", blur_sigma=1.0, sinc_fc=fc)
    return DegradationParams(**base)


def test_sinc_kernel_dc_gain_one() -> None:
    """kernel sum ~= 1 across the legal fc range (flat field stays flat)."""
    for fc in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        k = _sinc_kernel_1d(fc)
        assert abs(k.sum().item() - 1.0) < 1e-6, f"DC gain != 1 at fc={fc}"
        assert k.size(2) % 2 == 1 and k.size(2) >= 5  # odd, non-trivial


def test_sinc_kernel_not_identity() -> None:
    """The impulse response is a spread blur, not a delta."""
    k = _sinc_kernel_1d(0.15).squeeze()
    assert k.numel() > 1
    assert k.abs().max().item() < 1.0  # no single tap holds all the mass


def test_sinc_kernel_deterministic() -> None:
    for fc in (0.05, 0.2, 0.5):
        assert torch.equal(_sinc_kernel_1d(fc), _sinc_kernel_1d(fc))
    assert not torch.equal(_sinc_kernel_1d(0.1), _sinc_kernel_1d(0.4))


def test_sinc_clamps_to_legal_range() -> None:
    """fc outside (0, 0.5] is clamped, never an error."""
    k_hi = _sinc_kernel_1d(7.0)  # clamps to 0.5 (Nyquist)
    assert abs(k_hi.sum().item() - 1.0) < 1e-6
    # a brickwall AT Nyquist is the identity: single center tap = 1
    # (sin(pi*n) == 0 for every integer n != 0)
    assert k_hi.size(2) == 9 and k_hi.squeeze().abs().max().item() == pytest.approx(1.0, abs=1e-6)
    k_lo = _sinc_kernel_1d(0.0)  # clamps to 1e-3 -> very wide kernel
    assert k_lo.size(2) > 100
    # a sub-Nyquist cutoff is a genuine spread blur (max tap < 1)
    k_mid = _sinc_kernel_1d(0.4)
    assert k_mid.size(2) >= 5 and k_mid.abs().max().item() < 1.0


def _checkerboard(size: int = 128) -> torch.Tensor:
    """Pure Nyquist content: alternating ±1 on a pixel grid (no RNG)."""
    i = torch.arange(size)
    cb = ((i.view(-1, 1) + i.view(1, -1)) % 2).float() * 2.0 - 1.0
    return cb.view(1, 1, size, size).expand(1, 3, size, size).contiguous()


def test_lower_fc_attenuates_checkerboard_more() -> None:
    """fc=0.05 must kill the Nyquist checkerboard far more than fc=0.4."""
    x = _checkerboard()
    e_in = x.square().sum().item()
    eats: list[float] = []
    for fc in (0.05, 0.15, 0.40):
        out = _conv2d_separable(x, _sinc_kernel_1d(fc))
        eats.append(out.square().sum().item())
    # monotone: lower fc -> stronger low-pass -> less residual energy
    assert eats[0] < eats[1] < eats[2], f"non-monotone attenuation {eats}"
    # low fc collapses the high-frequency energy by >100x
    assert eats[0] < e_in / 100.0, (
        f"fc=0.05 left {eats[0] / e_in:.3%} of the Nyquist energy"
    )
    # near-Nyquist fc still passes an order of magnitude more energy than
    # the mid cutoff (a 2D corner frequency is attenuated on BOTH axes of
    # the separable pass, so absolute energy is low even at fc=0.4 — the
    # relative ordering is the contract)
    assert eats[2] > 10.0 * eats[1], (
        f"fc=0.4 should pass >10x the residual of fc=0.15: {eats}"
    )


def test_sinc_flat_field_stays_flat() -> None:
    c = -0.31
    x = torch.full((1, 3, 64, 64), c, dtype=torch.float32)
    for fc in (0.05, 0.2):
        out = _conv2d_separable(x, _sinc_kernel_1d(fc))
        assert (out - c).abs().max().item() < 1e-6, "flat field must stay flat"


def test_sampled_sinc_fc_in_legal_range() -> None:
    """Every sampled sinc exposure carries fc in (0, 0.5]; others 0.0."""
    n_sinc = 0
    for i in range(600):
        seed = exposure_seed(77, f"r{i}", 0, 0)
        p = sample_params(seed, CFG)
        if p.blur_kind == "sinc":
            n_sinc += 1
            assert 0.0 < p.sinc_fc <= 0.5, f"illegal fc={p.sinc_fc} for {p}"
        else:
            assert p.sinc_fc == 0.0, f"non-sinc exposure carries fc={p.sinc_fc}"
    assert n_sinc >= 10, f"expected some sinc exposures in 600 draws, saw {n_sinc}"


def test_sample_params_deterministic_including_sinc_fc() -> None:
    s = exposure_seed(3, "det", 2, 5)
    p1 = sample_params(s, CFG)
    p2 = sample_params(s, CFG)
    assert p1 == p2  # includes sinc_fc


def test_sinc_end_to_end_finite() -> None:
    """apply_degradation with a forced sinc exposure: finite, clamped,
    exact HR/4 (the separable path on the new kernel)."""
    g = torch.Generator().manual_seed(11)
    hr = (torch.rand(3, 64, 64, generator=g) * 2.0 - 1.0).contiguous()
    for fc in (0.08, 0.25):
        p = _sinc_params(fc)
        kernel, k1 = _blur_plan(p)
        assert kernel is None and k1 is not None  # separable 1D form
        out, _ = apply_degradation(hr, p, noise_seed=0, dither_seed=0)
        assert out.shape == (3, 16, 16)
        assert torch.isfinite(out).all()
        assert (out.abs() <= 1.0 + 1e-6).all()
    # different cutoffs must give different LQs (the fc actually matters)
    o1, _ = apply_degradation(hr, _sinc_params(0.05), noise_seed=0, dither_seed=0)
    o2, _ = apply_degradation(hr, _sinc_params(0.45), noise_seed=0, dither_seed=0)
    assert not torch.equal(o1, o2)
