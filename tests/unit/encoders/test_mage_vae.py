from __future__ import annotations

import torch
from torch import nn

from sakuramoon.encoders.mage_vae import FrozenMageVAE


class _FakeMageVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            image.shape[0],
            128,
            image.shape[2] // 16,
            image.shape[3] // 16,
            dtype=image.dtype,
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            latent.shape[0],
            3,
            latent.shape[2] * 16,
            latent.shape[3] * 16,
            dtype=latent.dtype,
        )


def test_wrapper_freezes_and_uses_mean_shape_contract() -> None:
    backend = _FakeMageVAE()
    vae = FrozenMageVAE(backend)
    image = torch.zeros(2, 3, 32, 48, dtype=torch.bfloat16, requires_grad=True)

    latent = vae.encode(image)
    reconstruction = vae.decode(latent)

    assert latent.shape == (2, 128, 2, 3)
    assert reconstruction.shape == image.shape
    assert latent.dtype == torch.bfloat16
    assert not latent.requires_grad
    assert not backend.weight.requires_grad
    assert not vae.training


def test_train_keeps_backend_in_eval_mode() -> None:
    backend = _FakeMageVAE()
    vae = FrozenMageVAE(backend)

    vae.train()

    assert not vae.training
    assert not backend.training
