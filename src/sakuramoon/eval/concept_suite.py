"""In-training concept-conditioning suite (optional, after the FID pass).

Runs the same five-metric suite as the standalone ``concept_eval`` CLI, but
against the *live* model held by a :class:`TrainingEvaluator`, so the
current training state is scored on the fixed concept draw inside the
regular evaluation cadence.  The suite is best-effort by design: any
failure is reported on the log and swallowed so it can never abort a
training run.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import cast

import tomli_w
import torch

from sakuramoon.eval.concepts import (
    ConceptManifest,
    GroupAggregate,
    aggregate_metrics,
    canonical_prompt_cases,
    compute_concept_metrics,
    render_suite_markdown,
    suite_report_document,
    swap_prompt_cases,
)
from sakuramoon.eval.features import CLIP_MODEL_ID, ClipFeatureModel
from sakuramoon.eval.runtime import TrainingEvaluator

__all__ = ["run_concept_suite"]


def load_reference_images(
    ref_paths: tuple[tuple[Path, ...], ...], *, resolution: int
) -> torch.Tensor:
    """Decode reference posts through the online real-image preprocessing."""

    from PIL import Image
    from torch.nn import functional
    from torchvision.transforms import functional as tvis_f

    images: list[torch.Tensor] = []
    for paths in ref_paths:
        for path in paths:
            try:
                with Image.open(path) as image:
                    tensor = tvis_f.pil_to_tensor(image.convert("RGB"))
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


def resolve_reference_images(
    manifest: ConceptManifest, refs_root: Path
) -> tuple[tuple[Path, ...], ...]:
    """Resolve every concept reference to a cached file; never downloads."""

    import json

    index_path = refs_root / "refs-index.json"
    index: dict[str, str] = {}
    if index_path.is_file():
        raw: object = json.loads(index_path.read_bytes())
        if type(raw) is not dict:
            raise RuntimeError(f"reference index is invalid: {index_path}")
        index = cast(dict[str, str], raw)
    missing: list[str] = []
    for concept in manifest.concepts:
        for post_id in concept.ref_post_ids:
            entry = index.get(str(post_id))
            if entry is None or not (refs_root / entry).is_file():
                missing.append(f"{concept.id}:{post_id}")
    if missing:
        raise RuntimeError(
            f"{len(missing)} concept references are missing under "
            f"{refs_root}: {', '.join(missing[:8])}"
        )
    return tuple(
        tuple(
            refs_root / index[str(post_id)]
            for post_id in concept.ref_post_ids
        )
        for concept in manifest.concepts
    )


def _generate_chunked(
    evaluator: TrainingEvaluator,
    cases: tuple[object, ...],
    *,
    batch_size: int,
    null: bool,
) -> torch.Tensor:
    """Run the evaluator's generation pass in bounded chunks."""

    label = "null" if null else "canonical/swap"
    chunks: list[torch.Tensor] = []
    for start in range(0, len(cases), batch_size):
        chunk = cases[start : start + batch_size]
        print(
            f"[concept-suite] 生成 {label} 批次 {start + 1}-{start + len(chunk)}/"
            f"{len(cases)}",
            flush=True,
        )
        chunks.append(evaluator.generate(chunk, null=null).cpu())
    return torch.cat(chunks)


def _extract_features(
    clip: ClipFeatureModel, images: torch.Tensor, *, batch_size: int
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, images.shape[0], batch_size):
        chunks.append(
            clip.features(images[start : start + batch_size]).cpu()
        )
    return torch.cat(chunks)


def _flatten_aggregates(
    aggregates: tuple[GroupAggregate, ...],
) -> dict[str, float]:
    """Flatten all group aggregates into ``{group/field: value}`` metrics."""

    flat: dict[str, float] = {}
    for aggregate in aggregates:
        fields = asdict(aggregate)
        for name, value in fields.items():
            if name != "group":
                flat[f"{aggregate.group}/{name}"] = float(value)
    return flat


def run_concept_suite(
    evaluator: TrainingEvaluator,
    *,
    update: int,
    manifest: ConceptManifest,
    refs_root: Path,
    run_dir: Path,
    batch_size: int = 40,
) -> dict[str, float]:
    """Score the live model on the concept draw; write reports; flat metrics.

    The evaluator's current ``growth_alpha`` and sampling profile are used,
    so the suite stays aligned with the FID pass of the same update.
    """

    evaluation = evaluator.evaluation
    config = evaluator.config
    resolution = config.stage.resolution

    print(f"[concept-suite] 参考图缓存: {refs_root}", flush=True)
    ref_paths = resolve_reference_images(manifest, refs_root)
    ref_images = load_reference_images(ref_paths, resolution=resolution)

    canonical_cases = canonical_prompt_cases(
        manifest, height=resolution, width=resolution
    )
    swap_cases = swap_prompt_cases(manifest, height=resolution, width=resolution)

    canonical_images = _generate_chunked(
        evaluator, canonical_cases, batch_size=batch_size, null=False
    )
    null_images = _generate_chunked(
        evaluator, canonical_cases, batch_size=batch_size, null=True
    )
    swap_images = _generate_chunked(
        evaluator, swap_cases, batch_size=batch_size, null=False
    )

    print("[concept-suite] 提取 CLIP 特征", flush=True)
    clip = ClipFeatureModel(evaluator.root, evaluator.device)
    clip_canonical = _extract_features(
        clip, canonical_images, batch_size=batch_size
    )
    clip_null = _extract_features(clip, null_images, batch_size=batch_size)
    clip_swap = _extract_features(clip, swap_images, batch_size=batch_size)
    clip_refs = _extract_features(clip, ref_images, batch_size=batch_size)

    print("[concept-suite] 计算指标", flush=True)
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
        "growth_alpha": evaluator.growth_alpha,
        "checkpoint": "in-training",
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

    run_dir.mkdir(parents=True, exist_ok=True)
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
    print(f"[concept-suite] 报告: {report_path}", flush=True)
    return _flatten_aggregates(aggregates)
