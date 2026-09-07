"""Vertical Bottom Supervision Audit - stage 2: full-canvas rebuild + VAE latents.

Reconstructs, for every pair, the production full canvas from the frozen
source image and freezes the TOP/BOTTOM crop latents.  The pipeline is the
literal production camera branch (src/sakuramoon/data/pipeline.py,
train_g1_camera_v2_p25.toml - camera_viewport enabled, no spatial policy, no
transparent-white policy):

    bytes -> PIL Image.open(BytesIO).load()
          -> normalize_image(image)          # image_ops: exif_transpose + RGB
          -> canvas = normalized.resize((256, F), resample=LANCZOS)
          -> crop (0, k, 256, k+256)         # k = k_start (TOP) / k_end (BOTTOM)
          -> uint8 CHW tensor
          -> bf16 / 127.5 - 1  ->  [1,3,256,256]
          -> FrozenMageVAE.encode           # [1,128,16,16] bf16

Reads ONLY: pair-manifest.json (frozen), frozen source images under
/tmp/camera-vertical-bottom/sources/, frozen anchor unit bundles, the frozen
local VAE asset.  Writes ONLY under /tmp/camera-vertical-bottom/:
    latents/pair-NNNN-{top,bottom}.pt
    images/pair-NNNN-{top,bottom}.png, images/canvas-NNNN.png
    anchor-parity.json, encode-pairs.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contracts as C


def log(msg: str, log_path: Path) -> None:
    line = f"[encode-pairs {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="worktree (PYTHONPATH src)")
    ap.add_argument("--out-root", type=Path, default=C.AUDIT_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="0 = all pairs")
    args = ap.parse_args()

    log_path = args.out_root / "encode-pairs.log"
    lat_dir = args.out_root / "latents"
    img_dir = args.out_root / "images"
    lat_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    with open(args.out_root / "pair-manifest.json", "r", encoding="utf-8") as fh:
        pm = json.load(fh)
    if pm.get("status") != "OK":
        log(f"FATAL: pair manifest status {pm.get('status')}", log_path)
        return 2
    pairs = pm["pairs"]
    if args.limit:
        pairs = pairs[: args.limit]
    log(f"pairs to encode: {len(pairs)} (device={args.device})", log_path)

    import torch
    from PIL import Image

    from sakuramoon.data.image_ops import normalize_image
    from sakuramoon.encoders.mage_vae import load_local_mage_vae

    device = torch.device(args.device)
    vae = load_local_mage_vae(C.RUNTIME_ROOT, device=device)
    # spec s19: pair latent cache must record VAE hash / RGB hash / latent hash
    vae_hash = C.sha256_file(C.VAE_DIR / "diffusion_pytorch_model.safetensors")
    log(f"VAE loaded (frozen, bf16) vae_sha256={vae_hash[:16]}...", log_path)

    parity: list[dict] = []
    n_done = 0
    n_skip = 0
    for row in pairs:
        i = row["pair_index"]
        geo = row["geometry"]
        F = geo["full_height"]
        src = args.out_root / row["source_image_rel"]
        outs = {
            side: lat_dir / f"pair-{i:04d}-{side}.pt"
            for side in ("top", "bottom")
        }
        receipts = {
            side: lat_dir / f"pair-{i:04d}-{side}.json"
            for side in ("top", "bottom")
        }
        pngs = {
            side: img_dir / f"pair-{i:04d}-{side}.png"
            for side in ("top", "bottom")
        }
        canvas_png = img_dir / f"canvas-{i:04d}.png"
        if (
            all(o.is_file() for o in outs.values())
            and all(r.is_file() for r in receipts.values())
            and all(p.is_file() for p in pngs.values())
        ):
            n_skip += 1
            continue
        img_bytes = src.read_bytes()
        image = Image.open(BytesIO(img_bytes))
        image.load()
        normalized = normalize_image(image)
        canvas = normalized.resize((C.VIEWPORT, F), resample=Image.Resampling.LANCZOS)
        crops = {}
        for side, box in (("top", tuple(geo["crop_box_top"])), ("bottom", tuple(geo["crop_box_bottom"]))):
            crop = canvas.crop(box)
            if crop.mode != "RGB":
                raise RuntimeError(f"pair {i} {side} crop not RGB")
            crops[side] = crop
        if not canvas_png.is_file():
            canvas.save(canvas_png)
        for side, crop in crops.items():
            if not pngs[side].is_file():
                crop.save(pngs[side])
            tensor = torch.frombuffer(
                bytearray(crop.tobytes()), dtype=torch.uint8
            ).reshape(C.VIEWPORT, C.VIEWPORT, 3).permute(2, 0, 1).contiguous()
            x = tensor.to(dtype=torch.bfloat16, device=device)
            x = x / 127.5 - 1.0
            with torch.no_grad():
                x0 = vae.encode(x.unsqueeze(0))
            if tuple(x0.shape) != (1, 128, 16, 16):
                raise RuntimeError(f"pair {i} {side} encode shape {tuple(x0.shape)}")
            if not outs[side].is_file():
                torch.save(x0.cpu(), outs[side])
            if not receipts[side].is_file():
                C.write_frozen(receipts[side], {
                    "pair_index": i,
                    "side": side,
                    "vae_sha256": vae_hash,
                    "rgb_sha256": C.sha256_file(pngs[side]),
                    "latent_sha256": C.sha256_file(outs[side]),
                    "shape": [1, 128, 16, 16],
                    "dtype": "bfloat16",
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
            del x, x0
        # anchor parity vs the frozen unit bundle (x0 of the anchor side)
        anchor = row["anchor_unit"]
        b = torch.load(anchor["unit_file"], map_location="cpu", weights_only=False)
        if b["unit"] != anchor["unit"]:
            raise RuntimeError(f"pair {i} bundle unit mismatch {b['unit']} != {anchor['unit']}")
        bundle_sha = C.sha256_file(Path(anchor["unit_file"]))
        if bundle_sha != anchor["unit_file_sha256"]:
            raise RuntimeError(f"pair {i} bundle sha mismatch")
        a_side = "top" if row["geometry"]["anchor_is_top"] else "bottom"
        mine = torch.load(outs[a_side], map_location="cpu", weights_only=False)
        ref = b["x0"]
        a = mine.float()
        r = ref.float()
        diff = (a - r).abs()
        rms = float(diff.pow(2).mean().sqrt())
        ref_rms = float(r.pow(2).mean().sqrt())
        rel = rms / ref_rms if ref_rms > 0 else 0.0
        parity.append({
            "pair_index": i,
            "unit": anchor["unit"],
            "anchor_side": a_side,
            "bitexact": bool(torch.equal(mine, ref)),
            "maxabs": float(diff.max()),
            "rms": rms,
            "rel_rms": rel,
        })
        n_done += 1
        if n_done % 64 == 0:
            log(f"encoded {n_done}/{len(pairs)}", log_path)
        del b, mine, ref, a, r, diff

    bitexact = sum(1 for p in parity if p["bitexact"])
    worst = max((p["maxabs"] for p in parity), default=0.0)
    worst_rel = max((p["rel_rms"] for p in parity), default=0.0)
    doc = {
        "label": "VERTICAL BOTTOM SUPERVISION ANCHOR PARITY",
        "n_pairs_checked": len(parity),
        "n_bitexact": bitexact,
        "n_skipped_existing": n_skip,
        "maxabs_worst": worst,
        "rel_rms_worst": worst_rel,
        "pipeline": (
            "source bytes -> PIL load -> normalize_image(exif_transpose+RGB) -> "
            "resize((256,F),LANCZOS) -> crop(0,k,256,k+256) -> uint8 CHW -> "
            "bf16/127.5-1 -> FrozenMageVAE.encode"
        ),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parity": parity,
    }
    C.write_frozen(args.out_root / "anchor-parity.json", doc)
    log(
        f"ANCHOR PARITY: {bitexact}/{len(parity)} bitexact, worst maxabs={worst:.3e}, "
        f"worst rel_rms={worst_rel:.3e} (skipped {n_skip})",
        log_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
