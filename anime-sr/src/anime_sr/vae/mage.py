"""Frozen Mage-VAE wrapper for anime-sr (plan §4, §2.1).

The Mage-VAE is the *only* pretrained component this project reuses (plan
§8): it maps HR pixel space to the 128-channel 1/16 latent space and, for
Stage II and inference, decodes latents back to pixels. All of it is frozen.

Guarantees (plan §2.1, §12.6):
  * ``encode`` returns the deterministic posterior mean
    (``sample_posterior=False``) under ``no_grad`` — the encoder is only
    ever run HR→latent, weights frozen.
  * ``decode_with_grad`` runs the same t=0 single-step decode as the vendored
    ``MageVAE.decode`` but WITHOUT ``no_grad``, so Stage-II pixel/edge/
    perceptual losses backprop through the frozen decoder into the U-Flow
    output ``z_hat``. Decoder parameters remain frozen.
  * Every parameter has ``requires_grad=False`` after construction.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypedDict

import torch
from torch import nn

from anime_sr.vae.mage_vae_impl import MageVAE


class VAEFingerprint(TypedDict):
    """Component record for the checkpoint/release manifest (plan §17.3)."""

    component: str
    ckpt_path: str
    ckpt_sha256: str
    latent_channels: int
    downsample_factor: int
    n_params_m: float
    frozen: bool


class FrozenMageVAE(nn.Module):
    """Frozen Mage-VAE: deterministic encode + t=0 single-step decode."""

    def __init__(
        self,
        ckpt_path: str | Path,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Mage-VAE weights not found: {ckpt_path}")
        vae = MageVAE(str(ckpt_path), sample_posterior=False).to(device=device, dtype=dtype)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        self.vae = vae
        self.ckpt_path = ckpt_path
        self.eval()  # wrapper flag too (training is never used, but keep it honest)

    # ------------------------------------------------------------------
    # geometry / dtype
    # ------------------------------------------------------------------
    @property
    def latent_channels(self) -> int:
        """128 (plan §4)."""
        return self.vae.latent_channels

    @property
    def downsample_factor(self) -> int:
        """16 (plan §4)."""
        return self.vae.downsample_factor

    @property
    def device(self) -> torch.device:
        return self.vae.device

    @property
    def dtype(self) -> torch.dtype:
        return self.vae.dtype

    # ------------------------------------------------------------------
    # encode: deterministic posterior mean, no_grad (plan §2.1)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: [B, 3, H, W] in [-1, 1], H/W multiples of 16.

        Returns [B, 128, H/16, W/16].
        """
        return self.vae.encode(x)

    # ------------------------------------------------------------------
    # decode: inference path, no_grad
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``z``: [B, 128, h, w] -> [B, 3, 16h, 16w] in [-1, 1]."""
        return self.vae.decode(z)

    # ------------------------------------------------------------------
    # decode with gradients: Stage II (plan §12.6)
    # ------------------------------------------------------------------
    def decode_with_grad(self, z: torch.Tensor) -> torch.Tensor:
        """Same t=0 single-step decode as :meth:`decode`, without ``no_grad``.

        Stage-II losses backprop through the frozen decoder into ``z`` (and
        hence into the U-Flow); decoder parameters stay frozen.
        """
        impl = self.vae
        cond = impl.decoder_model.y_embedder.decoder(z)
        b = z.shape[0]
        h = z.shape[2] * impl.downsample_factor
        w = z.shape[3] * impl.downsample_factor
        noise = torch.zeros(b, 3, h, w, device=z.device, dtype=z.dtype)
        t = torch.zeros(b, device=z.device, dtype=z.dtype)
        return impl.decoder_model.forward(noise, t, cond)

    # ------------------------------------------------------------------
    # optional optimization
    # ------------------------------------------------------------------
    def fold_t0_cache(self) -> None:
        """Constant-fold the adaLN modulation MLPs at t=0 (both pathways
        always run at t=0). Numerically identical, cheaper."""
        self.vae._freeze_adaln_cache()

    # ------------------------------------------------------------------
    # release-manifest fingerprint (plan §17.3)
    # ------------------------------------------------------------------
    def weights_sha256(self, chunk_bytes: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with self.ckpt_path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_bytes), b""):
                h.update(chunk)
        return h.hexdigest()

    def fingerprint(self) -> VAEFingerprint:
        """Component record for the checkpoint/release manifest (plan §17.3)."""
        n_params = sum(p.numel() for p in self.parameters())
        return {
            "component": "mage-vae",
            "ckpt_path": str(self.ckpt_path),
            "ckpt_sha256": self.weights_sha256(),
            "latent_channels": self.latent_channels,
            "downsample_factor": self.downsample_factor,
            "n_params_m": round(n_params / 1e6, 3),
            "frozen": True,
        }


def load_frozen_vae(
    path: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> FrozenMageVAE:
    """Factory that hard-fails on a missing/empty weights path (repo rule:
    缺失时直接报错)."""
    if not path:
        raise ValueError(
            "VAE weights path is empty: set [vae].path in the config overlay"
        )
    return FrozenMageVAE(path, device=device, dtype=dtype)
