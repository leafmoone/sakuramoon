"""Pre-encoded frozen Mage-VAE HR latent store (plan §4, §20 step 2).

Each sample keeps exactly one latent:

    z_hr = E_Mage(hr_crop)   [128, B/16, B/16]  (plan §4.3, deterministic
    posterior mean, encoder frozen)

stored as little-endian fp16 bytes (CHW row-major, torch layout). The LQ
anchor ``z_lr = E_Mage(Bicubic4x(lq))`` is NOT pre-encoded here: it depends
on the per-exposure degradation draw (plan §4.3) and is computed at data
time by the flow trainer (§M4 async prefetch absorbs the encoder cost).

Storage layout (resume-safe; byte-size checks only — repo data-service
discipline, same as the codec bank):

    latents_dir/
      index-v1.json        {"version": 1, "bucket_hr": int, "dtype": "fp16",
                           "channels": 128, "latent_bytes": int,
                           "samples": {sample_id: {"file": "z/<sid>.bin",
                                                    "bytes": int}}}
      z/<sample_id>.bin

``sample_id`` is a danbooru id (safe as a filename); the index is rewritten
atomically at the end of a build.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

INDEX_NAME = "index-v1.json"
LATENT_DIR = "z"


def _expected_latent_bytes(bucket_hr: int, channels: int) -> int:
    g = bucket_hr // 16
    return channels * g * g * 2  # fp16


class LatentStore:
    """One-sample-per-file z_hr store (build + resume + read)."""

    def __init__(
        self, out_dir: str | Path, bucket_hr: int, channels: int = 128
    ) -> None:
        self.root = Path(out_dir)
        self.bucket_hr = int(bucket_hr)
        if bucket_hr % 16 != 0:
            raise ValueError(f"bucket_hr {bucket_hr} must be a multiple of the 16x VAE")
        self.channels = int(channels)
        self.expected_bytes = _expected_latent_bytes(bucket_hr, channels)
        self.z_dir = self.root / LATENT_DIR

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def write(self, sample_id: str, z_hr: torch.Tensor) -> bool:
        """Atomically store one latent. Returns True if written, False if
        skipped (existing file with the expected byte size)."""
        if tuple(z_hr.shape) != (
            self.channels,
            self.bucket_hr // 16,
            self.bucket_hr // 16,
        ):
            raise ValueError(
                f"latent shape {tuple(z_hr.shape)} != expected "
                f"({self.channels}, {self.bucket_hr // 16}, {self.bucket_hr // 16})"
            )
        self.z_dir.mkdir(parents=True, exist_ok=True)
        p = self.z_dir / f"{sample_id}.bin"
        if p.is_file() and p.stat().st_size == self.expected_bytes:
            return False
        raw = z_hr.detach().contiguous().to(torch.float16).cpu().numpy().tobytes()
        if len(raw) != self.expected_bytes:
            raise RuntimeError(f"byte-size mismatch for {sample_id}: {len(raw)}")
        tmp = p.with_name(f"{p.name}.{os.getpid()}.part")
        try:
            tmp.write_bytes(raw)
            os.replace(tmp, p)
        finally:
            tmp.unlink(missing_ok=True)
        return True

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------
    def read(self, sample_id: str) -> torch.Tensor:
        """Load one latent as fp16 (host). Hard-fails on a missing/mis-sized
        file — the trainer must never see a silent skip (M1 gate)."""
        p = self.z_dir / f"{sample_id}.bin"
        if not p.is_file():
            raise FileNotFoundError(f"latent missing for {sample_id} in {self.z_dir}")
        raw = p.read_bytes()
        if len(raw) != self.expected_bytes:
            raise RuntimeError(
                f"latent for {sample_id} is {len(raw)} bytes, expected {self.expected_bytes}"
            )
        g = self.bucket_hr // 16
        z = (
            torch.frombuffer(bytearray(raw), dtype=torch.float16)
            .view(1, self.channels, g, g)[0]
            .clone()
        )
        return z

    def has(self, sample_id: str) -> bool:
        p = self.z_dir / f"{sample_id}.bin"
        return p.is_file() and p.stat().st_size == self.expected_bytes

    # ------------------------------------------------------------------
    # index
    # ------------------------------------------------------------------
    def finalize_index(self, sample_ids: list[str]) -> Path:
        """Write index-v1.json atomically for the samples in this store."""
        self.root.mkdir(parents=True, exist_ok=True)
        samples = {}
        for sid in sorted(sample_ids):
            if not self.has(sid):
                raise RuntimeError(
                    f"cannot finalize index: latent for {sid} is incomplete"
                )
            samples[sid] = {
                "file": f"{LATENT_DIR}/{sid}.bin",
                "bytes": self.expected_bytes,
            }
        doc = {
            "version": 1,
            "bucket_hr": self.bucket_hr,
            "dtype": "fp16",
            "channels": self.channels,
            "latent_bytes": self.expected_bytes,
            "n_samples": len(samples),
            "samples": samples,
        }
        p = self.root / INDEX_NAME
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc), encoding="utf-8")
        os.replace(tmp, p)
        return p


def read_index(root: str | Path) -> dict:
    """Load and sanity-check index-v1.json (version, bucket, byte sizes)."""
    p = Path(root) / INDEX_NAME
    if not p.is_file():
        raise FileNotFoundError(f"latent index missing: {p} (run encode_latents first)")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("version") != 1:
        raise ValueError(f"unsupported latent index version: {doc.get('version')}")
    if doc.get("dtype") != "fp16":
        raise ValueError(f"unsupported latent dtype: {doc.get('dtype')}")
    return doc


__all__ = [
    "INDEX_NAME",
    "LATENT_DIR",
    "LatentStore",
    "read_index",
]
