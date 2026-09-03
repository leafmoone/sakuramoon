"""Reference parity and token-order contract for the frozen PE teacher.

The SakuraMoon teacher runs the official PE vision modules directly
(vendored, Apache-2.0; see src/sakuramoon/pe_spatial/NOTICE), so
"reference parity" is exercised as: strict state-dict load with zero
missing/unexpected keys, loader equivalence with the official
normalization path, bitwise-equal forward output against the official
module path, and an explicit 2D-RoPE raster-order contract
(token index ``t = y * grid_w + x``) matching the SakuraMoon latent
raster order used by the PackedDiT image spans.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from sakuramoon.assets.pe_spatial import (
    PE_SPATIAL_TEACHER_FILENAME,
    require_local_pe_spatial_teacher,
)
from sakuramoon.encoders.pe_spatial import (
    FrozenPESpatialEncoder,
    _normalize_checkpoint_state_dict,
    prepare_teacher_targets,
)
from sakuramoon.pe_spatial.config import PE_VISION_CONFIG
from sakuramoon.pe_spatial.pe_vision import VisionTransformer
from sakuramoon.pe_spatial.rope import RotaryEmbedding

REPOSITORY_ROOT = Path(__file__).parents[3]
DEVICE = torch.device("cuda", 0)


@pytest.fixture(scope="module")
def weight_path() -> Path:
    return require_local_pe_spatial_teacher(
        REPOSITORY_ROOT, "model/pe_spatial_b16_512"
    )


@pytest.fixture(scope="module")
def teacher(weight_path: Path) -> FrozenPESpatialEncoder:
    return FrozenPESpatialEncoder.load_asset(
        REPOSITORY_ROOT,
        "model/pe_spatial_b16_512",
        device=DEVICE,
    )


def _build_official_like_visual() -> VisionTransformer:
    return VisionTransformer(**asdict(PE_VISION_CONFIG["PE-Spatial-B16-512"]))


def test_strict_load_covers_exactly_the_module_keys(weight_path: Path) -> None:
    raw = torch.load(weight_path, weights_only=True)
    inner: dict[str, torch.Tensor] = raw
    if "state_dict" in raw:
        inner = raw["state_dict"]  # type: ignore[typeddict-item]
    elif "weights" in raw:
        inner = raw["weights"]  # type: ignore[typeddict-item]
    normalized = _normalize_checkpoint_state_dict(inner)

    visual = _build_official_like_visual()
    module_keys = set(visual.state_dict().keys())
    assert set(normalized.keys()) == module_keys


def test_official_loader_path_matches_strict_loader(
    teacher: FrozenPESpatialEncoder, weight_path: Path
) -> None:
    official = _build_official_like_visual()
    official.load_ckpt(str(weight_path), verbose=False)  # pyright: ignore[reportUnknownMemberType]
    official = official.to(device=DEVICE, dtype=torch.bfloat16)  # pyright: ignore[reportUnknownMemberType]
    official.eval()

    frozen = teacher.visual
    assert frozen.use_cls_token == official.use_cls_token  # pyright: ignore[reportUnknownMemberType]
    for (name_a, tensor_a), (name_b, tensor_b) in zip(
        sorted(frozen.state_dict().items()),
        sorted(official.state_dict().items()),
        strict=True,
    ):
        assert name_a == name_b
        assert torch.equal(tensor_a.to(DEVICE), tensor_b.to(DEVICE)), name_a


@pytest.mark.parametrize(
    ("height", "width"),
    [(256, 256), (512, 512), (384, 512)],
)
def test_forward_parity_with_official_path(
    teacher: FrozenPESpatialEncoder, height: int, width: int
) -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(11)
    images = (
        2.0 * torch.rand(2, 3, height, width, generator=generator, device=DEVICE) - 1.0
    ).to(torch.bfloat16)

    wrapped = prepare_teacher_targets(teacher, images)
    official_tokens = teacher.visual(images)  # pyright: ignore[reportUnknownMemberType]
    grid_h, grid_w = height // 16, width // 16

    assert tuple(official_tokens.shape) == (2, 1 + grid_h * grid_w, 768)
    assert torch.equal(wrapped.patch_features, official_tokens[:, 1:, :])
    assert wrapped.patch_features.dtype is torch.bfloat16
    max_abs = (
        (wrapped.patch_features.float() - official_tokens[:, 1:, :].float()).abs()
    )
    assert float(max_abs.max()) == 0.0


def test_token_order_is_row_major_raster(teacher: FrozenPESpatialEncoder) -> None:
    grid_h, grid_w = 6, 9  # 96x144
    images = (
        2.0 * torch.rand(1, 3, 16 * grid_h, 16 * grid_w, device=DEVICE) - 1.0
    ).to(torch.bfloat16)
    with torch.no_grad():
        teacher.visual(images)  # pyright: ignore[reportUnknownMemberType]

    freq = teacher.visual.rope.freq  # pyright: ignore[reportUnknownMemberType]
    assert tuple(freq.shape) == (1, 1 + grid_h * grid_w, 64)
    # CLS token sits at coordinate (0, 0): zero rotation.
    assert bool((freq[0, 0] == 0).all())

    rope = RotaryEmbedding(32)
    base = rope.freqs.detach()  # pyright: ignore[reportUnknownMemberType]
    assert tuple(base.shape) == (16,)
    for y in range(grid_h):
        for x in range(grid_w):
            token_index = 1 + y * grid_w + x
            # use_cls_token=True offsets coordinates by +1 (CLS at 0,0);
            # the x half (first 32 dims) rotates by the x position, the y
            # half by the y position, interleaved (repeat r=2).
            expected_x = (base * (x + 1)).repeat_interleave(2)
            expected_y = (base * (y + 1)).repeat_interleave(2)
            want = torch.cat([expected_x, expected_y]).to(DEVICE)
            assert torch.equal(freq[0, token_index], want), (y, x)


def test_rope_grid_is_rectangular_not_square_padded(
    teacher: FrozenPESpatialEncoder
) -> None:
    images = (
        2.0 * torch.rand(1, 3, 256, 512, device=DEVICE) - 1.0
    ).to(torch.bfloat16)
    with torch.no_grad():
        teacher.visual(images)  # pyright: ignore[reportUnknownMemberType]

    freq = teacher.visual.rope.freq  # pyright: ignore[reportUnknownMemberType]
    grid_h, grid_w = 256 // 16, 512 // 16
    assert tuple(freq.shape) == (1, 1 + grid_h * grid_w, 64)
    assert teacher.visual.rope.grid_size == (grid_h, grid_w)  # pyright: ignore[reportUnknownMemberType]


def test_teacher_not_part_of_trainable_state(
    teacher: FrozenPESpatialEncoder
) -> None:
    # The teacher must never leak grad state into training.
    assert all(p.requires_grad is False for p in teacher.parameters())
    output = prepare_teacher_targets(
        teacher,
        (2.0 * torch.rand(1, 3, 128, 128, device=DEVICE) - 1.0).to(torch.bfloat16),
    )
    assert output.patch_features.requires_grad is False
    assert output.patch_features.grad_fn is None
    assert PE_SPATIAL_TEACHER_FILENAME == "PE-Spatial-B16-512.pt"
