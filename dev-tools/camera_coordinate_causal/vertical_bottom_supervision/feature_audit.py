"""Vertical Bottom Supervision Audit - stage 4: content feature audit.

Frozen content features for every pair, computed from the SAME full canvas
and the SAME TOP/BOTTOM crops as the scoring pipeline (reconstructed in
memory from the frozen source bytes, identical PIL operations):

  PE-Spatial-B16-512 (frozen, bf16, strict load, MATH SDPA - iprea-branch
  vendored tower, see pe_ref/): dense patch features
      features = visual(image)              # [1, 1+256, 768]
      dense    = features[:, 1:, :]          # strip CLS, row-major 16x16
  for the full canvas (256xF resized to 256x256 LANCZOS, pre-registered) and
  for each crop.  Cross-view correspondence uses the p25c method (spec s33):
  mutual nearest-neighbor patch matching between crop and full-canvas grids
  (fp32, L2-normalized cosine), matched cosine mean, inlier ratio n/256,
  retained = inlier_ratio * matched_cos.  No invented method.

  CLIP ViT-L/14 @ 336 (frozen, fp32, local asset only): image features for
  full/top/bottom (processor standard 336 resize) and pairwise cosine
  similarities sim(full, top), sim(full, bottom).

  CLIP TEXT (pre-registered, fail-soft): the EXACT production caption is
  reconstructed from the shard metadata with the production functions
  (parse_modelscope_caption_fields -> build_caption_plan(seed=_domain_seed(
  base_seed, stage, 0, sample_id, "caption")) -> serialize_caption with the
  production tokenizer + framing contract), verified BIT-EXACT against the
  frozen anchor unit bundles for --text-validate-n pairs (input ids, dense
  length, main/condition indices, use_null_condition).  Any mismatch or
  missing dependency marks CLIP_TEXT = NOT_AVAILABLE (image+PE continue).
  The CLIP text input is the qwen-tokenizer decode of the reconstructed
  caption ids, truncated to 77 tokens by the CLIP tokenizer.

The content-balanced subset (spec s37) is the pre-registered lowest-50%
|content asymmetry| of pairs, computed from content fields ONLY (the
selector receives pair indices + one asymmetry scalar per pair; no causal
field can enter), and is written BEFORE any analysis runs.

Reads ONLY: frozen source images, frozen anchor bundles (validation), the
frozen local CLIP / PE assets, the worktree production source (imports).
Writes ONLY under /tmp/camera-vertical-bottom/features/ + feature-audit.log.
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
    line = f"[feature-audit {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------- canvas / crop reconstruction (identical to encode stage) ----------------

def _pil_canvas(row: dict, src_bytes: bytes):
    from PIL import Image

    from sakuramoon.data.image_ops import normalize_image

    geo = row["geometry"]
    F = geo["full_height"]
    image = Image.open(BytesIO(src_bytes))
    image.load()
    normalized = normalize_image(image)
    canvas = normalized.resize((C.VIEWPORT, F), resample=Image.Resampling.LANCZOS)
    top = canvas.crop(tuple(geo["crop_box_top"]))
    bot = canvas.crop(tuple(geo["crop_box_bottom"]))
    return canvas, top, bot


def _to_bf16_tensor(pil, device):
    import torch

    tensor = torch.frombuffer(
        bytearray(pil.tobytes()), dtype=torch.uint8
    ).reshape(pil.height, pil.width, 3).permute(2, 0, 1).contiguous()
    x = tensor.to(dtype=torch.bfloat16, device=device)
    return (x / 127.5 - 1.0).unsqueeze(0)


# ---------------- PE MNN (p25c method, spec s33) ----------------

def mnn_match(crop_feats: object, full_feats: object) -> dict:
    """Mutual nearest-neighbor patch matching (fp32 cosine), p25c method.

    crop_feats/full_feats: [256, 768] float tensors (row-major 16x16 grid).
    Returns {n, matched_cos, inlier_ratio, retained} (matched_cos=0.0 if no
    mutual matches).  Deterministic argmax (first max on ties).
    """
    cf = crop_feats.float()
    ff = full_feats.float()
    cf = cf / (cf.norm(dim=1, keepdim=True) + 1e-12)
    ff = ff / (ff.norm(dim=1, keepdim=True) + 1e-12)
    sim = cf @ ff.t()
    nn_cf = sim.argmax(dim=1)          # crop j -> full i
    nn_fc = sim.argmax(dim=0)          # full i  -> crop j
    mutual = []
    for j in range(cf.shape[0]):
        i = int(nn_cf[j])
        if int(nn_fc[i]) == j:
            mutual.append((j, i))
    n = len(mutual)
    if n == 0:
        return {"n": 0, "matched_cos": 0.0, "inlier_ratio": 0.0, "retained": 0.0}
    matched_cos = float(sim[ [j for j, _ in mutual], [i for _, i in mutual] ].mean())
    inlier_ratio = n / float(cf.shape[0])
    return {
        "n": n,
        "matched_cos": matched_cos,
        "inlier_ratio": inlier_ratio,
        "retained": inlier_ratio * matched_cos,
    }


# ---------------- PE encoder (iprea vendored, frozen) ----------------

def build_pe_encoder(device, log_path: Path):
    from dataclasses import asdict as _asdict

    import torch
    from pe_ref import PE_VISION_CONFIG, VisionTransformer

    if C.sha256_file(C.PE_WEIGHTS) != C.PE_APPROVED_SHA:
        raise RuntimeError("PE weights sha mismatch (asset changed)")
    if C.PE_WEIGHTS.stat().st_size != C.PE_APPROVED_SIZE:
        raise RuntimeError("PE weights size mismatch (asset changed)")
    visual = VisionTransformer(**_asdict(PE_VISION_CONFIG["PE-Spatial-B16-512"]))
    raw = torch.load(C.PE_WEIGHTS, weights_only=True)
    if isinstance(raw, dict) and ("state_dict" in raw or "weights" in raw):
        raw = raw.get("state_dict", raw.get("weights"))
    norm = {k.replace("module.", ""): v for k, v in raw.items()}
    if any(k.startswith("visual.") for k in norm):
        norm = {k.replace("visual.", ""): v for k, v in norm.items() if "visual" in k}
    missing, unexpected = visual.load_state_dict(norm, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"PE strict load failed: missing={missing} unexpected={unexpected}")
    visual = visual.to(device=device, dtype=torch.bfloat16).eval()
    for p in visual.parameters():
        p.requires_grad_(False)
    log("PE-Spatial-B16-512 loaded (frozen bf16, strict)", log_path)
    return visual


def pe_features(visual, image_bf16: object) -> object:
    """[1,3,256,256] bf16 -> dense [1,256,768] (CLS stripped, prod semantics)."""
    import torch

    with torch.no_grad():
        feats = visual(image_bf16)
    if tuple(feats.shape) != (1, 1 + 16 * 16, 768):
        raise RuntimeError(f"PE token layout {tuple(feats.shape)}")
    return feats[:, 1:, :]


# ---------------- CLIP (frozen, fp32) ----------------

def build_clip(device, log_path: Path):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(str(C.CLIP_DIR), local_files_only=True).to(device=device, dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(str(C.CLIP_DIR), local_files_only=True)
    log("CLIP ViT-L/14@336 loaded (frozen fp32)", log_path)
    return model, processor


def _cos(a, b) -> float:
    a = a.float()
    b = b.float()
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return float((a * b).sum(dim=-1).mean())


# ---------------- caption reconstruction (production-verbatim) ----------------

class CaptionReconstructor:
    def __init__(self, repo: Path, device_name: str):
        from transformers import AutoTokenizer

        from sakuramoon.config import load_config
        from sakuramoon.data.caption import (
            CaptionDropoutProbabilities,
            NlDropoutProbabilities,
            build_caption_plan,
        )
        from sakuramoon.data.pipeline import _domain_seed
        from sakuramoon.data.production import parse_modelscope_caption_fields
        from sakuramoon.data.serialize import (
            EXPECTED_PREFIX_TOKENS,
            EXPECTED_SUFFIX_TOKENS,
            FramingContract,
            serialize_caption,
        )

        config = load_config(
            repo / "config" / "train_g1_camera_v2_p25.toml",
            config_root=repo / "config",
            validate_secrets=False,
        )
        runtime = config.config  # LoadedConfig.config == RuntimeConfig
        dropout = runtime.caption.dropout
        self.probabilities = CaptionDropoutProbabilities(
            condition_route=dropout.condition_route,
            condition_only=dropout.condition_only,
            tag=dropout.tag,
            candidate_source=dropout.candidate_source,
            nl=NlDropoutProbabilities(
                long_names=dropout.nl.long_names,
                long_no_names=dropout.nl.long_no_names,
                short_vibes=dropout.nl.short_vibes,
                nl2=dropout.nl.nl2,
                nl3=dropout.nl.nl3,
            ),
        )
        self.condition_mode = runtime.caption.condition_mode
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(C.QWEN_DIR), local_files_only=True
        )
        if self.tokenizer.pad_token_id != 248044:
            raise RuntimeError("qwen tokenizer pad id changed")
        self.framing = FramingContract(
            EXPECTED_PREFIX_TOKENS, EXPECTED_SUFFIX_TOKENS, self.tokenizer.pad_token_id
        )
        self.base_seed = runtime.run.seed
        self.stage = runtime.stage.name
        self._build_caption_plan = build_caption_plan
        self._serialize = serialize_caption
        self._parse = parse_modelscope_caption_fields
        self._domain_seed = _domain_seed
        self._device_name = device_name

    def reconstruct(self, json_bytes: bytes, sample_id: int) -> dict:
        raw = json.loads(json_bytes.decode("utf-8"))
        fields = self._parse(raw)
        seed = self._domain_seed(self.base_seed, self.stage, 0, int(sample_id), "caption")
        plan = self._build_caption_plan(
            fields, self.probabilities, condition_mode=self.condition_mode, seed=seed
        )
        cap = self._serialize(plan, self.tokenizer, self.framing)
        return {
            "input_ids": [int(v) for v in cap.input_ids],
            "dense_length": int(cap.dense_length),
            "main_token_indices": [int(v) for v in cap.main_token_indices],
            "condition_token_indices": [int(v) for v in cap.condition_token_indices],
            "use_null_condition": bool(cap.use_null_condition),
            "text": self.tokenizer.decode(list(cap.input_ids), skip_special_tokens=False),
        }

    def verify_against_bundle(self, rec: dict, bundle: dict) -> list[str]:
        """Bit-exact integer comparison against the frozen anchor bundle."""
        import torch

        problems = []
        ids = bundle["input_ids"][0]
        n = len(rec["input_ids"])
        if int(rec["dense_length"]) != int(bundle["dense_length"]):
            problems.append(f"dense_length {rec['dense_length']} != {int(bundle['dense_length'])}")
        seg = ids[:n]
        mine = torch.tensor(rec["input_ids"], dtype=torch.long)
        if bool((seg != mine).any()):
            k = int((seg != mine).nonzero().flatten()[0])
            problems.append(f"input_ids differ at {k}: {int(seg[k])} vs {rec['input_ids'][k]}")
        if [int(v) for v in bundle["main_token_indices"][0]] != rec["main_token_indices"]:
            problems.append("main_token_indices differ")
        if [int(v) for v in bundle["condition_token_indices"][0]] != rec["condition_token_indices"]:
            problems.append("condition_token_indices differ")
        if bool(bundle["use_null_condition"][0]) != rec["use_null_condition"]:
            problems.append("use_null_condition differs")
        return problems


# ---------------- main ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=C.AUDIT_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--text-validate-n", type=int, default=3)
    ap.add_argument("--skip-text", action="store_true")
    args = ap.parse_args()

    log_path = args.out_root / "feature-audit.log"
    feat_dir = args.out_root / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    with open(args.out_root / "pair-manifest.json", "r", encoding="utf-8") as fh:
        pm = json.load(fh)
    pairs = pm["pairs"]
    log(f"pairs: {len(pairs)} device={args.device}", log_path)

    import torch

    device = torch.device(args.device)
    if not torch.cuda.is_available():
        log("FATAL: no CUDA/HCU device", log_path)
        return 2

    # ---- PE ----
    visual = build_pe_encoder(device, log_path)

    # ---- CLIP ----
    clip_model, clip_processor = build_clip(device, log_path)

    # ---- caption reconstructor ----
    cap_rec = None
    text_status = "NOT_AVAILABLE"
    if not args.skip_text:
        try:
            cap_rec = CaptionReconstructor(args.repo, str(device))
            text_status = "AVAILABLE"
            log("caption reconstructor ready (production-verbatim)", log_path)
        except Exception as e:  # noqa: BLE001  fail-soft (pre-registered)
            text_status = f"NOT_AVAILABLE:{type(e).__name__}:{e}"
            cap_rec = None
            log(f"CLIP_TEXT NOT_AVAILABLE: {text_status}", log_path)

    # ---- validation of caption reconstruction vs frozen bundles ----
    if cap_rec is not None:
        n_ok = 0
        n_checked = 0
        for row in pairs:
            if n_checked >= args.text_validate_n:
                break
            b = torch.load(row["anchor_unit"]["unit_file"], map_location="cpu", weights_only=False)
            rec = cap_rec.reconstruct(
                (args.out_root / row["source_image_rel"]).with_suffix(".json").read_bytes(),
                row["sample_id"],
            )
            problems = cap_rec.verify_against_bundle(rec, b)
            n_checked += 1
            if problems:
                text_status = f"NOT_AVAILABLE:bundle_mismatch:{problems[:3]}"
                cap_rec = None
                log(f"CLIP_TEXT NOT_AVAILABLE (validation): {problems[:3]}", log_path)
                break
            n_ok += 1
        if cap_rec is not None:
            log(f"caption validation: {n_ok}/{n_checked} bit-exact vs frozen bundles", log_path)

    pe_rows: list[dict] = []
    clip_rows: list[dict] = []
    text_rows: list[dict] = []

    for row in pairs:
        i = row["pair_index"]
        from PIL import Image as _PILImage

        src_bytes = (args.out_root / row["source_image_rel"]).read_bytes()
        canvas, top, bot = _pil_canvas(row, src_bytes)
        full256 = canvas.resize(
            (C.VIEWPORT, C.VIEWPORT), resample=_PILImage.Resampling.LANCZOS
        )

        # PE features (256x256 grids, 16x16 patches)
        with torch.inference_mode():
            f_full = pe_features(visual, _to_bf16_tensor(full256, device)).cpu()
            f_top = pe_features(visual, _to_bf16_tensor(top, device)).cpu()
            f_bot = pe_features(visual, _to_bf16_tensor(bot, device)).cpu()
        pe_top = mnn_match(f_top.reshape(256, 768), f_full.reshape(256, 768))
        pe_bot = mnn_match(f_bot.reshape(256, 768), f_full.reshape(256, 768))
        pe_rows.append({
            "pair_index": i,
            "top": pe_top,
            "bottom": pe_bot,
            "delta": {
                "matched_cos": pe_top["matched_cos"] - pe_bot["matched_cos"],
                "inlier_ratio": pe_top["inlier_ratio"] - pe_bot["inlier_ratio"],
                "retained": pe_top["retained"] - pe_bot["retained"],
            },
        })

        # CLIP image
        pil_images = [full256, top, bot]
        inputs = clip_processor(images=pil_images, return_tensors="pt").to(device)
        with torch.inference_mode():
            img_feats = clip_model.get_image_features(inputs.pixel_values)
        # transformers 5.x: get_image_features returns a ModelOutput;
        # pooler_output is the 768-d shared-embedding space (normalized for cosine)
        f0 = img_feats.pooler_output[0].cpu()
        f1 = img_feats.pooler_output[1].cpu()
        f2 = img_feats.pooler_output[2].cpu()
        sim_full_top = _cos(f0, f1)
        sim_full_bot = _cos(f0, f2)
        clip_rows.append({
            "pair_index": i,
            "sim_full_top": sim_full_top,
            "sim_full_bot": sim_full_bot,
            "delta": sim_full_top - sim_full_bot,
        })

        # CLIP text
        if cap_rec is not None:
            rec = cap_rec.reconstruct(
                (args.out_root / row["source_image_rel"]).with_suffix(".json").read_bytes(),
                row["sample_id"],
            )
            text_inputs = clip_processor.tokenizer(
                rec["text"], return_tensors="pt", padding=True,
                truncation=True, max_length=77,
            ).to(device)
            with torch.inference_mode():
                text_feat = clip_model.get_text_features(text_inputs.input_ids)
            tf = text_feat.pooler_output[0].cpu()
            text_rows.append({
                "pair_index": i,
                "sim_text_top": _cos(tf, f1),
                "sim_text_bot": _cos(tf, f2),
                "delta": _cos(tf, f1) - _cos(tf, f2),
                "text_len": len(rec["text"]),
            })
            del text_inputs, text_feat, tf
        del f0, f1, f2, img_feats, inputs, f_full, f_top, f_bot
        if (i + 1) % 64 == 0:
            log(f"features {i + 1}/{len(pairs)}", log_path)

    # ---- content asymmetry + balanced subset (content fields ONLY) ----
    asym = []
    for row in pairs:
        i = row["pair_index"]
        c = next(x for x in clip_rows if x["pair_index"] == i)
        p = next(x for x in pe_rows if x["pair_index"] == i)
        t = next((x for x in text_rows if x["pair_index"] == i), None)
        if t is not None:
            asym.append(abs(t["delta"]))
        else:
            asym.append(abs(c["delta"]) + abs(p["delta"]["retained"]))
    subset_idx = C.content_balanced_subset(pairs, asym, fraction=0.5)

    C.write_frozen(feat_dir / "pe-audit.json", {
        "label": "VERTICAL BOTTOM SUPERVISION PE AUDIT",
        "model": "PE-Spatial-B16-512 (frozen bf16, strict, MATH SDPA)",
        "weights_sha256": C.PE_APPROVED_SHA,
        "grid": "256x256 -> 16x16 patches, full canvas resized 256x256 LANCZOS",
        "method": "p25c mutual-nearest-neighbor: matched cosine mean + inlier ratio n/256 + retained",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": pe_rows,
    })
    C.write_frozen(feat_dir / "clip-audit.json", {
        "label": "VERTICAL BOTTOM SUPERVISION CLIP AUDIT",
        "model": "CLIP ViT-L/14@336 (frozen fp32, local asset)",
        "text_status": text_status,
        "text_validation": f"{args.text_validate_n} pairs bit-exact vs frozen bundles" if text_status == "AVAILABLE" else text_status,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image_rows": clip_rows,
        "text_rows": text_rows,
    })
    C.write_frozen(feat_dir / "content-balanced-subset.json", {
        "label": "VERTICAL BOTTOM SUPERVISION CONTENT BALANCED SUBSET",
        "rule": "pre-registered lowest-50% |content asymmetry|; asymmetry = "
                "|CLIP text delta| when CLIP_TEXT available else "
                "|CLIP image delta| + |PE retained delta| (content fields only)",
        "n_total": len(pairs),
        "n_selected": len(subset_idx),
        "pair_indices": subset_idx,
        "asymmetry": [round(float(v), 9) for v in asym],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    log("FEATURE AUDIT DONE", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
