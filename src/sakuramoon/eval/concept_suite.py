"""In-training concept-conditioning suite (optional, after the FID pass).

Runs the dual-path concept suite (condition pathway + text pathway with a
shared null) against the *live* model held by a :class:`TrainingEvaluator`,
so the current training state is scored on the fixed concept draw inside
the regular evaluation cadence.  The suite is best-effort by design: any
failure is reported on the log and swallowed so it can never abort a
training run.

The standalone ``concept_eval`` CLI calls the same
:func:`run_dual_path_suite` implementation, so the two entry points can
never drift apart.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import cast

import tomli_w
import torch

from sakuramoon.eval.concepts import (
    ConceptManifest,
    DualPathSuite,
    canonical_prompt_cases,
    render_suite_markdown,
    score_dual_path,
    suite_report_document,
    swap_prompt_cases,
    text_canonical_prompt_cases,
    text_swap_prompt_cases,
)
from sakuramoon.eval.features import CLIP_MODEL_ID, ClipFeatureModel
from sakuramoon.eval.runtime import TrainingEvaluator
from sakuramoon.eval.spec import PromptCase

__all__ = [
    "CONCEPT_IMAGE_STATES",
    "run_concept_suite",
    "run_dual_path_suite",
    "save_state_images",
]

# The single definition of the concept image states.  Both the runner's
# image dict and the standalone CLI's PNG export use these exact names, so
# there is no second naming layer (no underscore/hyphen aliasing).
CONCEPT_IMAGE_STATES: tuple[str, ...] = (
    "condition-canonical",
    "condition-swap",
    "text-canonical",
    "text-swap",
    "null",
)


def save_state_images(
    images_dir: Path,
    concept_ids: tuple[str, ...],
    images: Mapping[str, torch.Tensor],
) -> int:
    """Export one PNG per concept and state; returns the file count.

    Iterates the shared :data:`CONCEPT_IMAGE_STATES` against the runner's
    image dict, so a missing state is a hard ``KeyError`` instead of a
    silently missing PNG.  Images must be uint8 ``[N, 3, H, W]``.
    """

    import numpy as np
    from PIL import Image

    images_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for index, concept_id in enumerate(concept_ids):
        for state in CONCEPT_IMAGE_STATES:
            image = images[state][index]
            array = image.permute(1, 2, 0).numpy()
            Image.fromarray(np.ascontiguousarray(array)).save(
                images_dir / f"{concept_id}.{state}.png"
            )
            count += 1
    return count


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
    cases: tuple[PromptCase, ...],
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


def run_dual_path_suite(
    evaluator: TrainingEvaluator,
    *,
    manifest: ConceptManifest,
    ref_images: torch.Tensor,
    batch_size: int,
    clip: ClipFeatureModel | None = None,
) -> tuple[DualPathSuite, dict[str, torch.Tensor]]:
    """Generate the five per-concept states and score both pathways.

    Five images per concept: condition-canonical, condition-swap,
    text-canonical, text-swap, and one shared null (generated exactly
    once from the canonical noise stream and reused by both metric sets).
    Generation stays bounded by ``batch_size``; the ``clip`` model may be
    injected (unit tests) and is built on demand otherwise.
    """

    resolution = evaluator.config.train.resolution
    condition_canonical_cases = canonical_prompt_cases(
        manifest, height=resolution, width=resolution
    )
    condition_swap_cases = swap_prompt_cases(
        manifest, height=resolution, width=resolution
    )
    text_canonical_cases = text_canonical_prompt_cases(
        manifest, height=resolution, width=resolution
    )
    text_swap_cases = text_swap_prompt_cases(
        manifest, height=resolution, width=resolution
    )

    condition_canonical_images = _generate_chunked(
        evaluator, condition_canonical_cases, batch_size=batch_size, null=False
    )
    condition_swap_images = _generate_chunked(
        evaluator, condition_swap_cases, batch_size=batch_size, null=False
    )
    text_canonical_images = _generate_chunked(
        evaluator, text_canonical_cases, batch_size=batch_size, null=False
    )
    text_swap_images = _generate_chunked(
        evaluator, text_swap_cases, batch_size=batch_size, null=False
    )
    # The shared null: one generation pass from the canonical noise
    # stream, consumed by both the condition and the text metrics.
    null_images = _generate_chunked(
        evaluator, condition_canonical_cases, batch_size=batch_size, null=True
    )

    print("[concept-suite] 提取 CLIP 特征", flush=True)
    clip_model = (
        clip if clip is not None else ClipFeatureModel(evaluator.root, evaluator.device)
    )
    ref_features = _extract_features(
        clip_model, ref_images, batch_size=batch_size
    )
    null_features = _extract_features(
        clip_model, null_images, batch_size=batch_size
    )
    print("[concept-suite] 计算指标", flush=True)
    result = score_dual_path(
        manifest=manifest,
        condition={
            "canonical": _extract_features(
                clip_model, condition_canonical_images, batch_size=batch_size
            ),
            "null": null_features,
            "swap": _extract_features(
                clip_model, condition_swap_images, batch_size=batch_size
            ),
        },
        text={
            "canonical": _extract_features(
                clip_model, text_canonical_images, batch_size=batch_size
            ),
            "null": null_features,
            "swap": _extract_features(
                clip_model, text_swap_images, batch_size=batch_size
            ),
        },
        clip_refs=ref_features,
    )
    images = {
        "condition-canonical": condition_canonical_images,
        "condition-swap": condition_swap_images,
        "text-canonical": text_canonical_images,
        "text-swap": text_swap_images,
        "null": null_images,
    }
    return result, images


def _flatten_dual_path(result: DualPathSuite) -> dict[str, float]:
    """Flatten both group aggregates into namespaced metrics.

    Every key carries its pathway prefix (``condition/...`` or
    ``text/...``) so the two protocols can never alias each other in the
    telemetry stream.
    """

    flat: dict[str, float] = {}
    for namespace, aggregates in (
        ("condition", result.condition_aggregates),
        ("text", result.text_aggregates),
    ):
        for aggregate in aggregates:
            fields = asdict(aggregate)
            for name, value in fields.items():
                if name != "group":
                    flat[f"{namespace}/{aggregate.group}/{name}"] = float(value)
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

    config = evaluator.config
    resolution = config.train.resolution

    print(f"[concept-suite] 参考图缓存: {refs_root}", flush=True)
    ref_paths = resolve_reference_images(manifest, refs_root)
    ref_images = load_reference_images(ref_paths, resolution=resolution)

    result, _images = run_dual_path_suite(
        evaluator,
        manifest=manifest,
        ref_images=ref_images,
        batch_size=batch_size,
    )
    provenance: dict[str, object] = {
        "update": update,
        "growth_alpha": evaluator.growth_alpha,
        "checkpoint": "in-training",
        "resolution": resolution,
        "sampling_profile": evaluator.evaluation.sampling_profile,
        "clip_model_id": CLIP_MODEL_ID,
    }
    document = suite_report_document(
        manifest=manifest, suite=result, provenance=provenance
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
            suite=result,
            suite_meta=cast(dict[str, object], document["suite"]),
        ).encode("utf-8")
    )
    print(f"[concept-suite] 报告: {report_path}", flush=True)
    return _flatten_dual_path(result)
