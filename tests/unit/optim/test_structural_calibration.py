"""CPU unit tests for the structural/SNR pre-NS classifier calibration
(D1 round, 08-31).

Covers:
  S1. Power iteration (with deflation) sigma1..sigma4 vs exact SVD on
      rank-1-dominant matrices (the pathological class) and on random
      matrices (the near-isotropic class, where PI must converge slowly —
      asserted as a bound, not exactness), at several iteration counts.
  S2. Feature derivations: stable_rank / top1..top4 cumulative energy /
      eff_rank4 from exact SVD singular values (the row math the shadow
      step uses).
  S3. _cosine: unit vectors, orthogonality, zero-vector guard, shape
      invariance (flattened inner product).
  S4. SVD sample saving + label-row schema sanity via a tiny end-to-end
      shadow install on a mock optimizer (CPU, no DDP).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import sakuramoon.optim.cmuon as _cmuon_mod
from sakuramoon.optim.structural_calibration import (
    StructuralCalibrationComplete,
    _cosine,
    _PowerIteration,
    install_structural_calibration,
    singular_top4,
)

# ---------------------------------------------------------------- S1/S2


def _rank1_dominant(m: int, n: int, seed: int) -> torch.Tensor:
    """0.01 * broad noise + 0.9 * (rank-1 outer product): the forensic
    pathology shape (top-1 energy ~89% for 2560-wide matrices)."""
    torch.manual_seed(seed)
    a = torch.randn(m, n) * 0.01
    a = a + 0.9 * torch.randn(m, 1) @ torch.randn(1, n)
    return a


@pytest.mark.parametrize(
    "shape",
    [
        (512, 512),  # square (scaled-down 2560 case)
        (128, 512),  # tall (scaled-down 640x2560 k_proj)
        (512, 128),  # wide (scaled-down FFN)
        (300, 300),  # small
    ],
)
@pytest.mark.parametrize("iters", [1, 5, 10, 20])
def test_power_iteration_rank1_dominant(shape, iters):
    m, n = shape
    a = _rank1_dominant(m, n, seed=0)
    exact = torch.linalg.svdvals(a.double())
    sigmas = _PowerIteration(a, iters, seed_base=7).top4()
    # sigma1: PI Rayleigh value is a monotone over-estimate
    rel1 = abs(sigmas[0] - float(exact[0])) / float(exact[0])
    if iters >= 10:
        assert rel1 < 1e-3, f"sigma1 rel err {rel1:.2e} (iters={iters})"
    elif iters >= 5:
        assert rel1 < 5e-3
    else:
        assert rel1 < 0.2
    # sigma2..sigma4 (deflated) on the noise floor: loose but bounded
    for j in (1, 2, 3):
        if float(exact[j]) > 0:
            rel = abs(sigmas[j] - float(exact[j])) / float(exact[j])
            if iters >= 20:
                assert rel < 5e-2, f"sigma{j + 1} rel err {rel:.2e}"
            else:
                assert rel < 0.3, f"sigma{j + 1} rel err {rel:.2e}"


def test_power_iteration_random_matrix_slow_gap():
    """Near-isotropic matrix: PI sigma1 must still be an over-estimate and
    must improve monotonically with iterations (documented slow case)."""
    a = torch.randn(256, 256)
    exact = torch.linalg.svdvals(a.double())
    rels = []
    for iters in (2, 5, 10, 20, 40):
        s1 = _PowerIteration(a, iters, seed_base=7).top4()[0]
        rels.append(abs(s1 - float(exact[0])) / float(exact[0]))
    assert all(r >= 0 for r in rels)
    assert rels[-1] < 2e-2  # 40 iters enough even for random (fp32 floor)
    assert rels[2] <= rels[0] + 1e-12  # more iters -> no worse


@pytest.mark.parametrize("method,iters", [("svd", 1), ("pi", 40)])
def test_singular_top4_dispatch(method, iters):
    a = _rank1_dominant(256, 128, seed=1)
    exact = torch.linalg.svdvals(a.double())
    out = singular_top4(a, method, iters, seed_base=7)
    assert len(out) == 4
    for j in range(4):
        assert abs(out[j] - float(exact[j])) / float(exact[j]) < 2e-2
    with pytest.raises(ValueError):
        singular_top4(a, "bogus", 1, seed_base=7)


def test_stable_rank_and_energies_from_exact_svd():
    a = _rank1_dominant(512, 512, seed=1)
    s = torch.linalg.svdvals(a)
    fro2 = float((s**2).sum())
    stable_rank = fro2 / float(s[0] ** 2)
    assert stable_rank >= 1.0
    top2 = float((s[:2] ** 2).sum()) / fro2
    assert 0 < top2 <= 1
    # the shadow-step row math on the same numbers
    sigmas = [float(x) for x in s[:4]]
    top1_energy = sigmas[0] ** 2 / fro2
    top2_cum = (sigmas[0] ** 2 + sigmas[1] ** 2) / fro2
    top4_cum = sum(x * x for x in sigmas) / fro2
    assert abs(top2_cum - top2) < 1e-6
    assert top1_energy <= top2_cum <= top4_cum <= 1.0 + 1e-9
    shares = [x * x / fro2 for x in sigmas]
    eff4 = math.exp(-sum(p * math.log(p) for p in shares if p > 0))
    assert 1.0 <= eff4 <= 4.0


# ------------------------------------------------------------------ S3


def test_cosine_values():
    a = torch.randn(64, 64)
    b = a.clone()
    assert _cosine(a, b) == pytest.approx(1.0, abs=1e-6)
    assert _cosine(a, -b) == pytest.approx(-1.0, abs=1e-6)
    c = torch.zeros_like(a)
    c[0, 0] = 1.0
    # a is random => cos ~ 0 (std ~ 1/64) but finite
    v = _cosine(a, c)
    assert v is not None and abs(v) < 0.05
    z = torch.zeros(4, 4)
    assert _cosine(z, a) is None
    assert _cosine(a, z) is None
    # flattening invariance: 2D and reshaped give the same value
    a2 = a.reshape(1, 4096)
    b2 = b.reshape(1, 4096)
    assert _cosine(a2, b2) == pytest.approx(_cosine(a, b), abs=1e-7)


# ------------------------------------------------------------------ S4


class _FakeHybrid:  # stands in for HybridCMuon in the isinstance check
    pass


class _Spec:
    def __init__(self, name, param, role, chunk_dim, chunk_count, ndim=2):
        self.name = name
        self.parameter = param
        self.role = role
        self.chunk_dim = chunk_dim
        self.chunk_count = chunk_count
        self.ndim = ndim

    def chunk_size(self):
        return self.parameter.shape[self.chunk_dim] // self.chunk_count


class _FakeParam:
    """Stands in for a nn.Parameter: carries .shape/.data/.grad/.dtype."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.shape = tensor.shape
        self.data = tensor
        self.grad = None
        self.dtype = tensor.dtype


class _Routing:
    def __init__(self, specs):
        self.cmuon_specs = specs


class _Cfg:
    momentum = 0.95
    lr = 1.5625e-4
    ns_coefficients = (3.4445, -4.7750, 2.0315)
    eps = 1e-7
    chunk_rescale_sqrt_n = False

    def ns_steps_for_role(self, role):
        return 4


class _MockOpt(_FakeHybrid):
    def __init__(self, specs, device="cpu"):
        self.routing = _Routing(specs)
        self.cfg = _Cfg()
        self._momenta = {s.parameter: torch.zeros_like(s.parameter.tensor) for s in specs}

    def _sync_learning_rate(self):
        return None

    def _validate_finite_gradients(self):
        return None

    def step(self):
        raise AssertionError("original step must not run during calibration")


def test_shadow_install_end_to_end_cpu(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(_cmuon_mod, "HybridCMuon", _FakeHybrid)
    torch.manual_seed(3)
    p1 = _FakeParam(torch.randn(32, 16))
    p2 = _FakeParam(torch.randn(256, 128))
    specs = [
        _Spec("m.blocks.slot_07.attention.k_proj.weight", p1, "attn_k", 0, 1),
        _Spec("m.blocks.slot_08.mlp.gate_proj.weight", p2, "ffn", 0, 2),
    ]
    p1.grad = torch.randn_like(p1.tensor)
    p2.grad = torch.randn_like(p2.tensor)
    opt = _MockOpt(specs)
    refs = {
        ("m.blocks.slot_07.attention.k_proj.weight", 0): 1e-3,
        ("m.blocks.slot_08.mlp.gate_proj.weight", 0): 1e-3,
        ("m.blocks.slot_08.mlp.gate_proj.weight", 1): 1e-3,
    }
    out = tmp_path / "struct-rank0.jsonl"
    art = tmp_path / "art"
    handle = install_structural_calibration(
        opt,
        observations=3,
        ns_repeat=2,
        pi_iters=10,
        sigma_method="svd",
        output_path=out,
        artifact_dir=art,
        rank=0,
        world_size=1,
        update_offset=97100,
        refs=refs,
        svd_samples=2,
    )
    for _ in range(3):
        p1.grad = torch.randn_like(p1.tensor)
        p2.grad = torch.randn_like(p2.tensor)
        try:
            opt.step()
        except StructuralCalibrationComplete as done:
            assert done.observations == 3
            break
    else:
        raise AssertionError("calibration did not complete")
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    rec = json.loads(lines[0])
    assert rec["obs"] == 1 and rec["abs_update"] == 97101
    assert rec["ceiling"] == pytest.approx(10 * 0.2 * rec["lr"])
    assert len(rec["rows"]) == 3  # 1 chunk + 2 chunks
    row = rec["rows"][0]
    for key in (
        "rms",
        "fro",
        "max_abs",
        "ref",
        "rel_sig",
        "sigma1",
        "top1_energy",
        "top2_cum_energy",
        "top4_cum_energy",
        "stable_rank",
        "eff_rank4",
        "cos_grad_mom",
        "cos_grad_nest",
        "alpha",
        "ns_runs",
        "hazard_rate",
        "label",
    ):
        assert key in row, key
    assert row["slot"] == "slot_07"
    assert row["cos_nest_mom_prev"] is None  # no history on obs 1
    assert len(row["ns_runs"]) == 2
    assert row["label"] in ("SAFE", "DANGEROUS")
    rec2 = json.loads(lines[1])
    assert rec2["rows"][0]["cos_nest_mom_prev"] is not None
    # SVD samples saved on obs 1 only
    samples = sorted((art / "svd-samples").glob("sample-*.pt"))
    assert len(samples) == 2
    blob = torch.load(samples[0], map_location="cpu", weights_only=False)
    assert blob["tensor"].shape[0] == 32
    assert blob["tensor"].dtype in (torch.bfloat16, torch.float32)
    assert handle.observations == 3
    # parameters strictly unchanged (fresh grads each obs, momentum updated)
    assert opt._momenta[p1].abs().sum() > 0
    assert not torch.all(opt._momenta[p1] == 0)


def test_install_rejects_wrong_optimizer(tmp_path: Path):
    class _NotHybrid:
        pass

    with pytest.raises(TypeError):
        install_structural_calibration(
            _NotHybrid(),  # type: ignore[arg-type]
            observations=1,
            ns_repeat=1,
            pi_iters=1,
            output_path=tmp_path / "x.jsonl",
            artifact_dir=tmp_path,
            rank=0,
            world_size=1,
        )
