"""anime_sr: pure-image 4x anime super-resolution on the Mage-VAE latent space.

Independent codebase (plan §8): the only reused pretrained component is the
frozen Mage-VAE (``anime_sr/vae/mage_vae_impl.py``, vendored MIT code).
v1 has no text/CFG/T2I/cross-attn/ControlNet/GAN/DWT/OCR-loss.

Authoritative spec: ``anime-sr/docs/plan-v2.0.md`` (frozen 2026-08-26) and
``anime-sr/docs/design.md``.
"""

__version__ = "0.1.0"
