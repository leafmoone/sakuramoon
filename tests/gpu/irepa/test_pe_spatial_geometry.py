"""17-bucket rectangular geometry hard gate for the frozen PE teacher.

The bucket vocabulary is taken dynamically from the real training config
and the bucket implementation (never hand-copied): both stage edges of
the 17-shape family (256 = the G1 training stage, 512 = the
512-equivalent base including 512x512) must pass through the frozen
teacher with an exact ``[B, (H/16)*(W/16), 768]`` output — no resize, no
feature interpolation, no pad-to-square.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sakuramoon.config import load_config
from sakuramoon.data.buckets import BucketShape, generate_base_buckets, scale_buckets
from sakuramoon.encoders.pe_spatial import (
    FrozenPESpatialEncoder,
    prepare_teacher_targets,
)

REPOSITORY_ROOT = Path(__file__).parents[3]

SECRET_ENVIRONMENT = {
    "MODELSCOPE_API_TOKEN": "synthetic-modelscope-secret",
    "WANDB_API_KEY": "synthetic-wandb-secret",
}

DEVICE = torch.device("cuda", 0)

_LOADED = load_config(
    Path("train_g1.toml"),
    config_root=REPOSITORY_ROOT / "config",
    environment=SECRET_ENVIRONMENT,
)
BASE_BUCKETS = generate_base_buckets(_LOADED.config.data.buckets)
EDGE_SHAPES: dict[int, tuple[BucketShape, ...]] = {
    edge: scale_buckets(BASE_BUCKETS, edge) for edge in (256, 512)
}
ALL_SHAPES = [
    (shape.height, shape.width)
    for edge in (256, 512)
    for shape in EDGE_SHAPES[edge]
]


@pytest.fixture(scope="module")
def teacher() -> FrozenPESpatialEncoder:
    return FrozenPESpatialEncoder.load_asset(
        REPOSITORY_ROOT,
        "model/pe_spatial_b16_512",
        device=DEVICE,
    )


def _synthetic_image(height: int, width: int, *, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    images = (
        2.0
        * torch.rand(1, 3, height, width, generator=generator, device=DEVICE)
        - 1.0
    )
    return images.to(torch.bfloat16)


@pytest.mark.parametrize(("height", "width"), ALL_SHAPES)
def test_every_bucket_produces_exact_grid(
    teacher: FrozenPESpatialEncoder, height: int, width: int
) -> None:
    output = prepare_teacher_targets(teacher, _synthetic_image(height, width))

    grid_h, grid_w = height // 16, width // 16
    assert output.image_shape == (grid_h, grid_w)
    assert tuple(output.patch_features.shape) == (1, grid_h * grid_w, 768)
    assert output.patch_features.dtype is torch.bfloat16
    assert bool(torch.isfinite(output.patch_features.float()).all())


def test_bucket_family_is_dynamic_and_divisible() -> None:
    assert len(BASE_BUCKETS) == 17
    for edge, shapes in EDGE_SHAPES.items():
        assert len(shapes) == 17
        assert len(set(shapes)) == 17
        for shape in shapes:
            assert shape.height % 16 == 0, (edge, shape)
            assert shape.width % 16 == 0, (edge, shape)


def test_required_extremes_are_covered() -> None:
    edge512 = EDGE_SHAPES[512]
    assert BucketShape(512, 512) in edge512
    landscape = max(edge512, key=lambda shape: shape.width / shape.height)
    portrait = BucketShape(landscape.width, landscape.height)
    assert landscape in edge512
    assert portrait in edge512
    assert landscape.width / landscape.height > 2.0


def test_repeated_forward_is_bitwise_deterministic(
    teacher: FrozenPESpatialEncoder
) -> None:
    images = _synthetic_image(512, 512, seed=7)
    first = prepare_teacher_targets(teacher, images)
    second = prepare_teacher_targets(teacher, images)

    assert torch.equal(first.patch_features, second.patch_features)


def test_teacher_device_mismatch_rejected(
    teacher: FrozenPESpatialEncoder
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two accelerators")
    images = _synthetic_image(128, 128).to(torch.device("cuda", 1))
    with pytest.raises(RuntimeError, match="teacher device"):
        prepare_teacher_targets(teacher, images)
