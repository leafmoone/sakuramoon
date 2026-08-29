"""P2-prep unit tests: SampleEMA (sample-rate-based weight EMA).

Covers the decay^k recurrence, variable-batch equivalence, fp32 shadow /
bf16 live dtypes, state-dict round-trip, and apply/restore bridging.
Pure CPU; no VAE / weights involved.
"""

from __future__ import annotations

from typing import cast

import torch
from anime_sr.train.ema_sample import SampleEMA
from torch import nn


def _tiny() -> nn.Sequential:
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 4))
    return m


def test_exact_recurrence_single_steps() -> None:
    """Static-param step is a no-op; a moved-param step is exact in fp32
    when n == ref_samples (beta = decay**1, op order replicated)."""
    m = _tiny()
    decay, ref = 0.9, 4
    e = SampleEMA(m, decay, ref)

    b0 = e.update(m, ref)
    assert b0 == decay  # 0.9**1 is exactly representable
    live = dict(m.named_parameters())
    for fqn, shadow in e._shadow.items():
        # static-param step ages the EMA by exactly the fp32 recurrence
        # 0.9*e0 + 0.1*p with e0 == p: not a bit-exact no-op in floating
        # point, so check tight allclose, not equality.
        assert torch.allclose(shadow, live[fqn].to(torch.float32), rtol=1e-7, atol=1e-7)

    # move the live weights, one more update: e = decay*e0 + (1-decay)*p1
    e0 = {f: s.clone() for f, s in e._shadow.items()}
    for p in m.parameters():
        p.data.add_(0.5)
    p1 = {f: p.detach().to(torch.float32) for f, p in m.named_parameters()}
    e.update(m, ref)
    for fqn, shadow in e._shadow.items():
        # same op order as update(): mul_ then add_ with the same scalars ->
        # bit-exact against the reference recurrence
        expected = e0[fqn].mul(decay).add_(p1[fqn], alpha=1.0 - decay)
        assert torch.equal(shadow, expected)


def test_sample_proportional_decay_equivalence() -> None:
    """One update with n=2*ref ages the EMA like two consecutive n=ref
    updates with a static param in between (exact identity; fp32 1ulp)."""
    decay, ref = 0.9, 4
    m_single = _tiny()
    e_single = SampleEMA(m_single, decay, ref)
    for p in m_single.parameters():
        p.data.add_(0.25)
    beta = e_single.update(m_single, 2 * ref)
    assert abs(beta - decay * decay) < 1e-12  # 0.9**2 == 0.8

    m_step = _tiny()  # same seed -> same start weights
    e_step = SampleEMA(m_step, decay, ref)
    for p in m_step.parameters():
        p.data.add_(0.25)
    e_step.update(m_step, ref)
    e_step.update(m_step, ref)
    for fqn, e_t in e_single._shadow.items():
        assert torch.allclose(e_t, e_step._shadow[fqn], rtol=1e-6, atol=1e-7), fqn


def test_n_samples_total_accumulates() -> None:
    m = _tiny()
    e = SampleEMA(m, 0.9, 4)
    e.update(m, 4)
    e.update(m, 8)
    assert e.n_samples_total == 12


def test_bf16_live_fp32_shadow() -> None:
    m = _tiny().to(dtype=torch.bfloat16)
    e = SampleEMA(m, 0.9, 4)
    for t in e._shadow.values():
        assert t.dtype == torch.float32
    p = cast(nn.Linear, m[0]).weight
    p.data.fill_(1.0)
    e.update(m, 4)
    assert e._shadow["0.weight"].dtype == torch.float32
    prev = e.apply(m)
    # apply() = cast-copy of the fp32 shadow into the live (bf16) param
    assert torch.equal(cast(nn.Linear, m[0]).weight, e._shadow["0.weight"].to(torch.bfloat16))
    assert cast(nn.Linear, m[0]).weight.dtype == torch.bfloat16
    assert e._shadow["0.weight"].dtype == torch.float32  # shadow stays fp32
    assert prev["0.weight"].dtype == torch.bfloat16
    e.restore(m, prev)


def test_state_dict_roundtrip_and_config_guard() -> None:
    m = _tiny()
    e = SampleEMA(m, 0.9, 4)
    e.update(m, 4)
    sd = e.state_dict()
    assert sd["decay"] == 0.9 and sd["ref_samples"] == 4 and sd["n_samples_total"] == 4
    for t in sd["params"].values():
        assert t.dtype == torch.float32
    e2 = SampleEMA(_tiny(), 0.9, 4)  # different live weights
    e2.load_state_dict(sd)
    for fqn in e._shadow:
        assert torch.equal(e._shadow[fqn], e2._shadow[fqn])
    assert e2.n_samples_total == 4
    try:
        e2.load_state_dict({**sd, "decay": 0.5})
        raise AssertionError("config mismatch must raise")
    except ValueError:
        pass


def test_param_set_mismatch_raises() -> None:
    m = _tiny()
    e = SampleEMA(m, 0.9, 4)
    m2 = nn.Sequential(nn.Linear(8, 8))  # different param set
    try:
        e.update(m2, 4)
        raise AssertionError("param mismatch must raise")
    except ValueError:
        pass


def test_ddp_unwrap_consistency() -> None:
    """DDP-style wrapper: shadow keys stay bare-fqn; update through the
    wrapper (SimpleNamespace mimics ``.module`` unwrapping)."""
    import types

    m = _tiny()
    w = types.SimpleNamespace(module=m)
    e = SampleEMA(w, 0.9, 4)  # type: ignore[arg-type]
    assert set(e._shadow) == {"0.weight", "0.bias", "2.weight", "2.bias"}
    e.update(w, 4)  # type: ignore[arg-type]
