"""Real-codec degradation bank (plan §11.4, step 6).

The online degradation chain (``degradation.py``) approximates the codec
stage with block quantization + chroma subsampling. The codec bank holds
*real* codec encode/decode passes of HR crops, built offline so training
workers never launch ffmpeg (§11.4: 1-2 versions per crop, 10-20% batch
share, 50k-100k crops at full scale).

Variant axes (§11.4):

- codec families: webp, avif, h264, h265, av1, mpeg4
- chroma: 4:2:0 / 4:2:2 (yuv422p only where the encoder supports it)
- limited/full range mismatch (yuv codecs only: full-range data declared
  as limited range → the encoder's wrong conversion leaves a
  brightness/contrast shift after decode)
- double transcode (second encode pass over the decoded first pass)

Storage layout (resume-safe; byte-size checks only — repo data-service
discipline):

    bank_dir/
      index-v1.json          {"version": 1, "samples": {sample_id:
                              [{variant_id, bytes, lq_w, lq_h, codec,
                               pix_fmt, range_mismatch, passes, quality}]}}
      variants/<variant_id>.bin   raw RGB24 LQ bytes (lq_h * lq_w * 3)

``variant_id`` is the sha1 of the recipe tuple, so a rebuild with the same
recipe produces identical ids (skip-on-size-match resume) and the runtime
lookup is deterministic in the exposure seed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INDEX_NAME = "index-v1.json"
VARIANT_DIR = "variants"

# codec -> (ffmpeg encoder, extension/muxer, quality flag, quality range,
#           supports yuv422p, supports range-mismatch)
ENCODE_PROFILES: dict[str, tuple[str, str, str, tuple[int, int], bool, bool]] = {
    "webp": ("libwebp", "webp", "quality", (50, 95), False, False),
    "avif": ("libaom-av1", "ivf", "crf", (20, 45), False, False),
    "h264": ("libx264", "h264", "crf", (18, 40), True, True),
    "h265": ("libx265", "h265", "crf", (20, 42), True, True),
    "av1": ("libaom-av1", "ivf", "crf", (20, 45), False, False),
    "mpeg4": ("mpeg4", "mp4", "q", (3, 12), False, True),  # -q:v 3..12 (lower = better)
}

_CODEC_EXT: dict[str, str] = {
    "webp": ".webp",
    "avif": ".ivf",
    "h264": ".h264",
    "h265": ".h265",
    "av1": ".ivf",
    "mpeg4": ".mp4",
}


@dataclass(frozen=True)
class CodecVariant:
    """One real-codec LQ recipe for one crop (deterministic)."""

    codec: str
    pix_fmt: str  # yuv420p | yuv422p
    range_mismatch: bool
    passes: int  # 1 or 2 (double transcode)
    quality: int

    @property
    def variant_id(self) -> str:
        key = f"{self.codec}|{self.pix_fmt}|{self.range_mismatch}|{self.passes}|{self.quality}"
        return hashlib.sha1(key.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["variant_id"] = self.variant_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CodecVariant:
        return cls(
            codec=d["codec"],
            pix_fmt=d["pix_fmt"],
            range_mismatch=bool(d["range_mismatch"]),
            passes=int(d["passes"]),
            quality=int(d["quality"]),
        )


def _int_from_seed(seed: int, mod: int, salt: str) -> int:
    """Deterministic 0..mod-1 draw from the exposure seed (salted blake2b)."""
    h = hashlib.blake2b(f"{salt}|{seed}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") % mod


def sample_variants(seed: int, k: int) -> list[CodecVariant]:
    """k deterministic codec variants for one crop, all from the exposure seed.

    Each draw picks codec family, quality within the family band, chroma,
    range-mismatch and pass count; axes the codec does not support are
    forced off, so the recipe is always buildable.
    """
    out: list[CodecVariant] = []
    families = list(ENCODE_PROFILES)
    for j in range(k):
        codec = families[_int_from_seed(seed, len(families), f"cb|{j}|codec")]
        _enc, _ext, _qflag, (q_lo, q_hi), chroma_ok, range_ok = ENCODE_PROFILES[codec]
        quality = q_lo + _int_from_seed(seed, q_hi - q_lo + 1, f"cb|{j}|q")
        pix_fmt = "yuv422p" if (chroma_ok and _int_from_seed(seed, 2, f"cb|{j}|c") == 0) else "yuv420p"
        mismatch = range_ok and _int_from_seed(seed, 10, f"cb|{j}|r") == 0  # ~10% where supported
        passes = 2 if _int_from_seed(seed, 10, f"cb|{j}|p") == 0 else 1  # ~10%
        out.append(CodecVariant(codec, pix_fmt, mismatch, passes, quality))
    return out


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("ffmpeg not found on PATH (build the codec bank on a host with ffmpeg)")
    return path


def _encode_pass(ffmpeg: str, src: str, dst: str, v: CodecVariant) -> None:
    """One encode pass of the still frame ``src`` -> container ``dst``."""
    enc, _ext, qflag, _, _, range_ok = ENCODE_PROFILES[v.codec]
    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    if v.codec in ("h264", "h265", "mpeg4") and range_ok:
        # input is full-range RGB from a png:
        #   normal      → declare full range (-color_range 1)
        #   mismatch    → declare limited range (-color_range 2); the encoder
        #                  applies the wrong full/limited conversion
        cmd += ["-color_range", "2" if v.range_mismatch else "1"]
    cmd += ["-f", "image2", "-framerate", "1", "-i", src, "-frames:v", "1", "-c:v", enc]
    if qflag == "quality":
        cmd += ["-quality", str(v.quality), "-sharpness", "0"]
    elif qflag == "q":
        cmd += ["-q:v", str(v.quality)]
    else:
        cmd += ["-crf", str(v.quality)]
    cmd += ["-pix_fmt", v.pix_fmt, dst]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed ({v.codec} {v.variant_id}): {r.stderr.strip()[:400]}")


def _decode_to_raw(ffmpeg: str, src: str, out_path: Path, w: int, h: int) -> None:
    """Decode ``src`` (first frame), resize to w x h, dump raw RGB24."""
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", src, "-frames:v", "1",
        "-vf", f"scale={w}:{h}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {r.stderr.strip()[:400]}")


def encode_variant(
    crop_rgb: np.ndarray,
    v: CodecVariant,
    lq_w: int,
    lq_h: int,
    workdir: Path,
) -> tuple[Path, int]:
    """Encode/decode one variant of an HR crop -> raw RGB24 LQ on disk.

    ``crop_rgb`` is uint8 [H, W, 3] RGB. Returns (raw_path, n_bytes) where
    raw is ``lq_h * lq_w * 3`` bytes (the LQ pixels, codec artifacts intact).
    """
    from PIL import Image  # builder-side; PIL is a pipeline dependency

    ffmpeg = _ffmpeg()
    workdir.mkdir(parents=True, exist_ok=True)
    src_png = workdir / f"{v.variant_id}-in.png"
    Image.fromarray(crop_rgb).save(src_png)
    ext = _CODEC_EXT[v.codec]
    stage = workdir / f"{v.variant_id}{ext}"
    _encode_pass(ffmpeg, str(src_png), str(stage), v)
    if v.passes == 2:
        # second pass: decode pass 1, re-encode at the same recipe
        mid_raw = workdir / f"{v.variant_id}-p1raw"
        _decode_to_raw(ffmpeg, str(stage), mid_raw, crop_rgb.shape[1], crop_rgb.shape[0])
        arr = np.frombuffer(mid_raw.read_bytes(), dtype=np.uint8).reshape(crop_rgb.shape[0], crop_rgb.shape[1], 3)
        mid_png = workdir / f"{v.variant_id}-p1.png"
        Image.fromarray(arr).save(mid_png)
        _encode_pass(ffmpeg, str(mid_png), str(stage), v)
    raw_out = workdir / f"{v.variant_id}-out.raw"
    _decode_to_raw(ffmpeg, str(stage), raw_out, lq_w, lq_h)
    return raw_out, lq_w * lq_h * 3


class CodecBank:
    """Runtime consumer: deterministic per-sample variant lookup."""

    def __init__(self, bank_dir: str | Path) -> None:
        self.dir = Path(bank_dir)
        p = self.dir / INDEX_NAME
        if not p.exists():
            raise FileNotFoundError(f"codec bank index missing: {p} (run cli.build_codec_bank)")
        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc.get("version") != 1:
            raise ValueError(f"unknown codec bank index version: {doc.get('version')}")
        self.by_sample: dict[str, list[dict]] = doc["samples"]
        self.variants_dir = self.dir / VARIANT_DIR

    def __len__(self) -> int:
        return len(self.by_sample)

    def variants_for(self, sample_id: str, seed: int, k: int = 1) -> list[np.ndarray]:
        """k LQ uint8 [h, w, 3] arrays for one sample (deterministic pick)."""
        entries = self.by_sample.get(sample_id)
        if not entries:
            raise KeyError(f"sample {sample_id} not in codec bank")
        picks = [entries[_int_from_seed(seed, len(entries), f"cbpick|{j}")] for j in range(k)]
        out: list[np.ndarray] = []
        for e in picks:
            p = self.variants_dir / f"{e['variant_id']}.bin"
            n = e["lq_h"] * e["lq_w"] * 3
            if p.stat().st_size != n:
                raise RuntimeError(f"codec bank variant size mismatch: {p} ({p.stat().st_size} != {n})")
            arr = np.fromfile(str(p), dtype=np.uint8, count=n).reshape(e["lq_h"], e["lq_w"], 3)
            out.append(arr)
        return out

    def bank_fraction_hit(self, sample_id: str, seed: int, fraction: float) -> bool:
        """Deterministic 10-20% batch selection (per sample + exposure seed)."""
        threshold = fraction * 10_000 + 0.5  # round-half-up for 0 <= fraction <= 1
        return _int_from_seed(seed, 10_000, f"cbfrac|{sample_id}") < int(threshold)
