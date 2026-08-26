"""FrozenMageVAE wrapper tests (plan §4): shape, determinism, freeze, grad path.

Runs on CUDA when available, else CPU (small 512x512 inputs keep CPU cheap).
Skipped when the Mage-VAE weights are absent (set ANIME_SR_VAE_PATH).
"""

from __future__ import annotations

import pytest
import torch
from anime_sr.vae import FrozenMageVAE
from conftest import VAE_WEIGHTS, requires_vae

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def vae() -> FrozenMageVAE:
    return FrozenMageVAE(str(VAE_WEIGHTS), device=DEVICE, dtype=torch.bfloat16)


def _rand_input(size: int = 512) -> torch.Tensor:
    return torch.randn(1, 3, size, size, device=DEVICE, dtype=torch.bfloat16).clamp(-1.0, 1.0)


@requires_vae
def test_geometry_and_frozen(vae: FrozenMageVAE) -> None:
    assert vae.latent_channels == 128
    assert vae.downsample_factor == 16
    params = list(vae.parameters())
    assert len(params) > 0
    assert all(not p.requires_grad for p in params)
    assert vae.training is False


@requires_vae
def test_encode_shape_and_determinism(vae: FrozenMageVAE) -> None:
    x = _rand_input(512)
    z1 = vae.encode(x)
    z2 = vae.encode(x)
    assert z1.shape == (1, 128, 32, 32)
    assert torch.isfinite(z1).all()
    # deterministic posterior mean: bit-identical across calls (plan §2.1)
    assert torch.equal(z1, z2)


@requires_vae
def test_encode_rejects_bad_multiples(vae: FrozenMageVAE) -> None:
    with pytest.raises(ValueError):
        vae.encode(_rand_input(500))


@requires_vae
def test_decode_shape_and_range(vae: FrozenMageVAE) -> None:
    z = vae.encode(_rand_input(512))
    img = vae.decode(z)
    assert img.shape == (1, 3, 512, 512)
    assert torch.isfinite(img).all()
    assert img.min() >= -1.5 and img.max() <= 1.5  # t=0 decode stays near [-1, 1]


@requires_vae
def test_decode_with_grad_backprops_to_z(vae: FrozenMageVAE) -> None:
    """Stage-II contract (plan §12.6): grads flow through the *frozen* decoder."""
    z = vae.encode(_rand_input(512))
    z = z.detach().requires_grad_(True)
    out = vae.decode_with_grad(z)
    out.float().pow(2).mean().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0
    # decoder parameters remain frozen (no grad buffers)
    assert all(p.grad is None for p in vae.parameters())


@requires_vae
def test_fingerprint(vae: FrozenMageVAE) -> None:
    fp = vae.fingerprint()
    assert fp["component"] == "mage-vae"
    assert fp["latent_channels"] == 128
    assert fp["frozen"] is True
    assert len(fp["ckpt_sha256"]) == 64
    assert fp["n_params_m"] > 50  # DConv encoder + CoD decoder, well beyond a 12M pixel encoder
