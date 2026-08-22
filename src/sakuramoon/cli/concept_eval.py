"""Run the standalone concept-conditioning suite on a saved checkpoint.

For every concept in the suite manifest the CLI generates three images from
one shared noise stream (canonical tag text, fully dropped condition, and the
swap partner's tag text), extracts CLIP features for the generated images and
the Danbooru reference posts, and reports the unified-sign margins,
reference similarity, and self-retrieval ranking.

Reference posts are downloaded from Danbooru on first use and cached under
the manifest's ``refs/`` directory (stdlib urllib only, no new dependencies).

Usage:
    python -m sakuramoon.cli.concept_eval \
        --config train_g1.toml --config-root config --root /sakuramoon-runtime \
        --checkpoint /sakuramoon-runtime/output_model/g1/ckpt_70000_...
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import tomli_w

from sakuramoon.cli.generation_eval import _resolve_alpha

if TYPE_CHECKING:
    import torch

    from sakuramoon.eval.concepts import ConceptManifest
    from sakuramoon.eval.features import ClipFeatureModel
    from sakuramoon.eval.runtime import TrainingEvaluator
    from sakuramoon.eval.spec import PromptCase
    from sakuramoon.train.step import TrainableComposite

_SUITE_DIR = "data/concept-benchmarks/concept-120-v1"
_DANBOORU_API = "https://danbooru.donewtf.k/api/v1/posts.json"
_USER_AGENT = "sakuramoon-concept-suite/1.0"
_HTTP_TIMEOUT = 120
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_REF_PAGE_SIZE = 100
_REF_API_DELAY_S = 0.2
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone concept-conditioning suite")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="complete checkpoint directory (manifest.json + model/ + train_state/)",
    )
    parser.add_argument(
        "--update",
        type=int,
        default=None,
        help="suite run label (defaults to the checkpoint manifest update)",
    )
    parser.add_argument(
        "--growth-alpha",
        type=float,
        default=None,
        help="growth alpha (defaults to train_state/growth_state.json when present)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=f"suite manifest (default: {_SUITE_DIR}/manifest.json under --root)",
    )
    parser.add_argument(
        "--refs-root",
        type=Path,
        default=None,
        help="reference image cache (default: <manifest directory>/refs)",
    )
    parser.add_argument(
        "--no-fetch-refs",
        action="store_true",
        help="fail instead of downloading missing reference posts from Danbooru",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="do not save the generated PNG images",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=40,
        help="chunk size for generation and CLIP passes (default: 40)",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="suite output directory (default: <evaluation output_dir>/concept-suite)",
    )
    return parser


def _http_get(url: str, *, max_bytes: int | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
            payload = response.read(max_bytes + 1) if max_bytes is not None else response.read()
    except OSError as error:
        raise RuntimeError(f"reference download failed for {url}: {error}") from error
    if max_bytes is not None and len(payload) > max_bytes:
        raise RuntimeError(
            f"reference image exceeds {_MAX_IMAGE_BYTES} bytes: {url}"
        )
    return payload


def _fetch_reference_urls(post_ids: tuple[int, ...]) -> dict[int, str]:
    """Resolve Danbooru post ids to download URLs (prefers file_url)."""

    urls: dict[int, str] = {}
    for start in range(0, len(post_ids), _REF_PAGE_SIZE):
        batch = post_ids[start : start + _REF_PAGE_SIZE]
        query = urllib.parse.urlencode(
            {
                "ids": ",".join(str(post_id) for post_id in batch),
                "per_page": str(len(batch)),
                "fields[0]": "file_url",
                "fields[1]": "sample_url",
            }
        )
        posts: object = json.loads(_http_get(f"{_DANBOORU_API}?{query}"))
        if type(posts) is not list:
            raise RuntimeError("Danbooru API returned an unexpected document")
        for post in cast(list[object], posts):
            if type(post) is not dict:
                continue
            record = cast(dict[str, object], post)
            post_id = record.get("id")
            if type(post_id) is not int:
                continue
            file_url = record.get("file_url")
            sample_url = record.get("sample_url")
            chosen = file_url if type(file_url) is str and file_url else sample_url
            if type(chosen) is str and chosen:
                urls[post_id] = chosen
        if start + _REF_PAGE_SIZE < len(post_ids):
            time.sleep(_REF_API_DELAY_S)
    return urls


def ensure_reference_images(
    manifest: ConceptManifest,
    refs_root: Path,
    *,
    fetch: bool,
) -> tuple[tuple[Path, ...], ...]:
    """Resolve every concept reference to a cached file, downloading as needed."""

    refs_root.mkdir(parents=True, exist_ok=True)
    index_path = refs_root / "refs-index.json"
    index: dict[str, str] = {}
    if index_path.is_file():
        raw: object = json.loads(index_path.read_bytes())
        if type(raw) is not dict:
            raise RuntimeError(f"reference index is invalid: {index_path}")
        index = cast(dict[str, str], raw)
    required = tuple(
        post_id
        for concept in manifest.concepts
        for post_id in concept.ref_post_ids
    )
    missing = tuple(
        post_id
        for post_id in dict.fromkeys(required)
        if (
            str(post_id) not in index
            or not (refs_root / index[str(post_id)]).is_file()
        )
    )
    if missing:
        if not fetch:
            raise RuntimeError(
                f"{len(missing)} reference images are missing and --no-fetch-refs is set: "
                f"{missing[:8]}{'...' if len(missing) > 8 else ''}"
            )
        print(
            f"[concept-eval] 下载缺失参考图: {len(missing)} 张 (Danbooru → {refs_root})",
            flush=True,
        )
        urls = _fetch_reference_urls(missing)
        unavailable = tuple(post_id for post_id in missing if post_id not in urls)
        if unavailable:
            raise RuntimeError(
                f"Danbooru returned no usable file for posts: {unavailable[:8]}"
            )
        for position, post_id in enumerate(missing, start=1):
            url = urls[post_id]
            payload = _http_get(url, max_bytes=_MAX_IMAGE_BYTES)
            suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
            if suffix not in _IMAGE_SUFFIXES:
                suffix = ".jpg"
            target = refs_root / f"{post_id}{suffix}"
            target.write_bytes(payload)
            index[str(post_id)] = target.name
            if position % 50 == 0 or position == len(missing):
                print(
                    f"[concept-eval] 参考图进度: {position}/{len(missing)}",
                    flush=True,
                )
        temporary = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, index_path)
    resolved: list[tuple[Path, ...]] = []
    for concept in manifest.concepts:
        paths: list[Path] = []
        for post_id in concept.ref_post_ids:
            entry = index.get(str(post_id))
            if entry is None:
                raise RuntimeError(f"reference post {post_id} is unavailable")
            path = refs_root / entry
            if not path.is_file():
                raise RuntimeError(f"reference image is missing: {path}")
            paths.append(path)
        resolved.append(tuple(paths))
    return tuple(resolved)


def load_reference_images(
    ref_paths: tuple[tuple[Path, ...], ...], *, resolution: int
) -> torch.Tensor:
    """Decode reference posts through the online real-image preprocessing."""

    import torch
    from PIL import Image
    from torch.nn import functional

    images: list[torch.Tensor] = []
    for paths in ref_paths:
        for path in paths:
            try:
                with Image.open(path) as image:
                    tensor = functional.pil_to_tensor(image.convert("RGB"))
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    f"reference image cannot be decoded: {path}"
                ) from error
            images.append(
                functional.interpolate(
                    tensor.unsqueeze(0).float(),
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .round()
                .clamp(0.0, 255.0)
                .to(torch.uint8)
            )
    return torch.stack(images)


def _generate_chunked(
    evaluator: TrainingEvaluator,
    cases: tuple[PromptCase, ...],
    *,
    batch_size: int,
    null: bool,
) -> torch.Tensor:
    """Run the evaluator's generation pass in bounded chunks (per-pass progress)."""

    import torch

    label = "null" if null else "canonical/swap"
    chunks: list[torch.Tensor] = []
    for start in range(0, len(cases), batch_size):
        chunk = cases[start : start + batch_size]
        print(
            f"[concept-eval] 生成 {label} 批次 {start + 1}-{start + len(chunk)}/"
            f"{len(cases)}",
            flush=True,
        )
        chunks.append(evaluator.generate(chunk, null=null).cpu())
    return torch.cat(chunks)


def _extract_features(
    clip: ClipFeatureModel,
    images: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    import torch

    chunks: list[torch.Tensor] = []
    for start in range(0, images.shape[0], batch_size):
        chunks.append(
            clip.features(images[start : start + batch_size]).cpu()
        )
    return torch.cat(chunks)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    current = os.environ.get("PYTORCH_ALLOC_CONF", "")
    options = [
        value
        for value in current.split(",")
        if value and not value.startswith("expandable_segments:")
    ]
    options.append("expandable_segments:True")
    os.environ["PYTORCH_ALLOC_CONF"] = ",".join(options)

    import torch

    from sakuramoon.checkpoint.load import (
        load_inference_artifact,
        read_checkpoint_manifest,
    )
    from sakuramoon.config import load_config
    from sakuramoon.encoders.mage_vae import load_local_mage_vae
    from sakuramoon.encoders.qwen import load_local_qwen
    from sakuramoon.eval.concepts import (
        ConceptManifest,
        aggregate_metrics,
        canonical_prompt_cases,
        compute_concept_metrics,
        render_suite_markdown,
        suite_report_document,
        swap_prompt_cases,
    )
    from sakuramoon.eval.features import CLIP_MODEL_ID, ClipFeatureModel
    from sakuramoon.eval.runtime import TrainingEvaluator

    root = args.root.resolve(strict=True)
    config_root = (
        args.config_root if args.config_root.is_absolute() else root / args.config_root
    )
    loaded = load_config(
        args.config,
        config_root=config_root,
        validate_secrets=False,
    )
    config = loaded.config
    if config.evaluation.enabled is not True:
        raise ValueError("the concept suite requires evaluation.enabled=true")
    evaluation = config.evaluation
    checkpoint = args.checkpoint.resolve(strict=True)
    manifest_meta = read_checkpoint_manifest(checkpoint)
    update = args.update if args.update is not None else manifest_meta.identity.update
    if update < 0:
        raise ValueError("update must be nonnegative")
    alpha = _resolve_alpha(checkpoint, args.growth_alpha)

    resolution = config.stage.resolution
    if resolution <= 0 or resolution % 16:
        raise ValueError("stage resolution must be a positive multiple of 16")
    if args.generation_batch_size <= 0:
        raise ValueError("generation batch size must be positive")

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError(
            "the concept suite requires exactly one visible CUDA device"
        )
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    manifest_path = (
        args.manifest if args.manifest is not None else Path(_SUITE_DIR) / "manifest.json"
    )
    manifest_path = (
        manifest_path if manifest_path.is_absolute() else root / manifest_path
    )
    manifest = ConceptManifest.from_json(manifest_path)
    refs_root = (
        args.refs_root
        if args.refs_root is not None
        else manifest_path.parent / "refs"
    )
    refs_root = refs_root if refs_root.is_absolute() else root / refs_root

    if args.output_subdir is not None:
        output_subdir = Path(args.output_subdir)
        base_dir = (
            output_subdir
            if output_subdir.is_absolute()
            else root / output_subdir
        )
    else:
        base_dir = root / evaluation.output_dir / "concept-suite"
    run_dir = base_dir / f"update-{update}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[concept-eval] 加载检查点 {checkpoint.name} (update={update}, "
        f"growth_alpha={alpha}, n_concepts={len(manifest.concepts)})",
        flush=True,
    )
    composite = cast(
        "TrainableComposite",
        load_inference_artifact(checkpoint, manifest_meta.identity, device=device),
    )
    qwen = load_local_qwen(root, device)
    vae = load_local_mage_vae(root, device)
    evaluator = TrainingEvaluator(
        config,
        repository_root=root,
        composite=composite,
        qwen=qwen,
        vae=vae,
        device=device,
        growth_alpha=alpha,
    )

    print(f"[concept-eval] 参考图缓存: {refs_root}", flush=True)
    ref_paths = ensure_reference_images(
        manifest, refs_root, fetch=not args.no_fetch_refs
    )
    ref_images = load_reference_images(ref_paths, resolution=resolution)

    canonical_cases = canonical_prompt_cases(
        manifest, height=resolution, width=resolution
    )
    swap_cases = swap_prompt_cases(manifest, height=resolution, width=resolution)
    batch_size = args.generation_batch_size

    canonical_images = _generate_chunked(
        evaluator, canonical_cases, batch_size=batch_size, null=False
    )
    null_images = _generate_chunked(
        evaluator, canonical_cases, batch_size=batch_size, null=True
    )
    swap_images = _generate_chunked(
        evaluator, swap_cases, batch_size=batch_size, null=False
    )

    print("[concept-eval] 提取 CLIP 特征", flush=True)
    clip = ClipFeatureModel(root, device)
    clip_canonical = _extract_features(clip, canonical_images, batch_size=batch_size)
    clip_null = _extract_features(clip, null_images, batch_size=batch_size)
    clip_swap = _extract_features(clip, swap_images, batch_size=batch_size)
    clip_refs = _extract_features(clip, ref_images, batch_size=batch_size)

    print("[concept-eval] 计算指标", flush=True)
    metrics = compute_concept_metrics(
        manifest=manifest,
        clip_canonical=clip_canonical,
        clip_null=clip_null,
        clip_swap=clip_swap,
        clip_refs=clip_refs,
    )
    aggregates = aggregate_metrics(metrics)
    provenance: dict[str, object] = {
        "update": update,
        "growth_alpha": alpha,
        "checkpoint": checkpoint.name,
        "resolution": resolution,
        "sampling_profile": evaluation.sampling_profile,
        "clip_model_id": CLIP_MODEL_ID,
    }
    document = suite_report_document(
        manifest=manifest,
        metrics=metrics,
        aggregates=aggregates,
        provenance=provenance,
    )

    report_path = run_dir / "report.toml"
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(tomli_w.dumps(document).encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, report_path)
    markdown_path = run_dir / "report.md"
    markdown_path.write_bytes(
        render_suite_markdown(
            metrics=metrics,
            aggregates=aggregates,
            suite=cast(dict[str, object], document["suite"]),
        ).encode("utf-8")
    )

    if not args.no_images:
        import numpy as np
        from PIL import Image

        images_dir = run_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for concept, canon, nulled, swapped in zip(
            manifest.concepts,
            canonical_images,
            null_images,
            swap_images,
        ):
            for variant, image in (
                ("canonical", canon),
                ("null", nulled),
                ("swap", swapped),
            ):
                array = image.permute(1, 2, 0).numpy()
                Image.fromarray(np.ascontiguousarray(array)).save(
                    images_dir / f"{concept.id}.{variant}.png"
                )
        print(f"[concept-eval] 生成图像: {images_dir} (3 x {len(manifest.concepts)})", flush=True)

    print(f"[concept-eval] 报告: {report_path}", flush=True)
    print(render_suite_markdown(metrics=metrics, aggregates=aggregates, suite=cast(dict[str, object], document["suite"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
