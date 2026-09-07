"""Guarantees for the vertical mirror pair training-path contracts.

These tests exercise the pure (CPU, deterministic) pieces of the
single-GPU runtime's mirror branch and the loss-reduction contract. They
never construct ``SingleGpuBatchRuntime`` (which requires CUDA); instead
they pin the exact index layout and the reduction math that the runtime
uses, so the hard design gates are proven at the unit level:

* one mirror pair is ONE logical sample:
  ``0.5 * L_original + 0.5 * L_mirror`` (total source weight 1.0);
* the layout keeps ascending logical order and each mirror view
  immediately follows its original view;
* the same logical row is reused for both physical views (so the runtime
  draws one timestep and one noise tensor per logical sample and the two
  views share them);
* conditioning is row-duplicated, not re-encoded (Qwen forward stays one
  per logical sample);
* the disabled path (no mirror flags) is the identity (bit-identical).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sakuramoon.data.camera_viewport import CameraMirrorPayload
from sakuramoon.train.runtime import (
    MirrorPhysicalLayout,
    _mirror_view_images,
    _physical_active_condition_indices,
    _physical_view_images,
    _reduce_mirror_observation,
    _RuntimeLoss,
    _size_conditions,
    reduce_mirror_pair_loss,
)

# ---------------------------------------------------------------------------
# MirrorPhysicalLayout
# ---------------------------------------------------------------------------


def test_layout_identity_when_no_pairs() -> None:
    layout = MirrorPhysicalLayout.build((False, False, False))
    assert layout.logical_count == 3
    assert layout.physical_count == 3
    assert layout.phys_to_log == (0, 1, 2)
    assert layout.logical_orig_phys == (0, 1, 2)
    assert layout.pair_orig_phys == ()
    assert layout.mirror_phys == ()
    assert layout.pair_logical_rows == ()
    assert layout.physical_source_order == (0, 1, 2)


def test_layout_all_pairs() -> None:
    layout = MirrorPhysicalLayout.build((True, True, True))
    assert layout.logical_count == 3
    assert layout.physical_count == 6
    # Each logical row i occupies physical rows (2i, 2i+1).
    assert layout.phys_to_log == (0, 0, 1, 1, 2, 2)
    assert layout.logical_orig_phys == (0, 2, 4)
    assert layout.pair_orig_phys == (0, 2, 4)
    assert layout.mirror_phys == (1, 3, 5)
    assert layout.pair_logical_rows == (0, 1, 2)
    # Source concatenation is [logical views 0..2; mirror views 3..5]; the
    # physical order pulls each mirror from the tail segment.
    assert layout.physical_source_order == (0, 3, 1, 4, 2, 5)


def test_layout_mixed_pairs() -> None:
    layout = MirrorPhysicalLayout.build((False, True, False, True))
    assert layout.physical_count == 6
    assert layout.phys_to_log == (0, 1, 1, 2, 3, 3)
    assert layout.logical_orig_phys == (0, 1, 3, 4)
    assert layout.pair_orig_phys == (1, 4)
    assert layout.mirror_phys == (2, 5)
    assert layout.pair_logical_rows == (1, 3)
    # Source concat [logical 0..3; mirror 4,5]; physical pulls mirrors from
    # the tail: logical1's mirror is source 4, logical3's mirror is source 5.
    assert layout.physical_source_order == (0, 1, 4, 2, 3, 5)


def test_layout_source_order_covers_every_view_once() -> None:
    for flags in (
        (True,),
        (False, True),
        (True, False, True, False, True),
    ):
        layout = MirrorPhysicalLayout.build(flags)
        bound = layout.logical_count + len(layout.pair_orig_phys)
        assert sorted(layout.physical_source_order) == list(range(bound))
        assert layout.physical_count == bound


def test_layout_rejects_non_boolean_flags() -> None:
    with pytest.raises(ValueError):
        MirrorPhysicalLayout.build((0, 1))  # type: ignore[arg-type]


def test_layout_rejects_corrupted_construction() -> None:
    good = MirrorPhysicalLayout.build((True, False))
    # Mirror view must immediately follow its original view.
    with pytest.raises(ValueError):
        MirrorPhysicalLayout(
            logical_count=2,
            physical_count=3,
            phys_to_log=(0, 1, 0),
            logical_orig_phys=(0, 1),
            pair_orig_phys=(0,),
            mirror_phys=(2,),
            physical_source_order=good.physical_source_order,
        )
    # Original view must point at the right logical row.
    with pytest.raises(ValueError):
        MirrorPhysicalLayout(
            logical_count=2,
            physical_count=3,
            phys_to_log=(0, 0, 1),
            logical_orig_phys=(1, 2),
            pair_orig_phys=(0,),
            mirror_phys=(1,),
            physical_source_order=good.physical_source_order,
        )
    # Source order must cover every view exactly once.
    with pytest.raises(ValueError):
        MirrorPhysicalLayout(
            logical_count=2,
            physical_count=3,
            phys_to_log=(0, 0, 1),
            logical_orig_phys=(0, 2),
            pair_orig_phys=(0,),
            mirror_phys=(1,),
            physical_source_order=(0, 0, 2),
        )
    # Physical count must equal logical + pairs.
    with pytest.raises(ValueError):
        MirrorPhysicalLayout(
            logical_count=2,
            physical_count=4,
            phys_to_log=(0, 0, 0, 1),
            logical_orig_phys=(0, 2),
            pair_orig_phys=(0,),
            mirror_phys=(1,),
            physical_source_order=(0, 1, 2, 3),
        )


# ---------------------------------------------------------------------------
# reduce_mirror_pair_loss  (spec section 23 fixtures)
# ---------------------------------------------------------------------------


def test_ordinary_sample_keeps_its_loss() -> None:
    # ordinary: L = 4 -> logical loss 4
    layout = MirrorPhysicalLayout.build((False,))
    assert reduce_mirror_pair_loss(
        torch.tensor([4.0]), layout
    ).tolist() == [4.0]


def test_mirror_pair_is_weighted_average() -> None:
    # mirror: L_a = 2, L_b = 6 -> logical loss (2 + 6) / 2 = 4
    layout = MirrorPhysicalLayout.build((True,))
    assert reduce_mirror_pair_loss(
        torch.tensor([2.0, 6.0]), layout
    ).tolist() == [4.0]


def test_weighting_proves_logical_not_physical_mean() -> None:
    # The discriminating fixture: physical-view mean of (10, 0, 2) is 4.0,
    # but the logical mean is (10 + 1) / 2 = 5.5.
    layout = MirrorPhysicalLayout.build((False, True))
    # physical rows: [ordinary 10, pair-original 0, pair-mirror 2]
    per_physical = torch.tensor([10.0, 0.0, 2.0])
    logical = reduce_mirror_pair_loss(per_physical, layout)
    assert logical.tolist() == [10.0, 1.0]
    assert logical.mean().item() == pytest.approx(5.5)
    # The (incorrect) all-physical-view mean would be 4.0.
    assert per_physical.mean().item() == pytest.approx(4.0)
    assert logical.mean().item() != per_physical.mean().item()


def test_batch_weight_contract_two_logical_samples() -> None:
    # Spec 11: Batch A (2 ordinary) and Batch B (1 ordinary + 1 pair) both
    # carry exactly two logical sample weights.
    batch_a = MirrorPhysicalLayout.build((False, False))
    batch_b = MirrorPhysicalLayout.build((False, True))
    # Batch A: logical loss = [4, 4] -> 2 weights, mean 4.
    loss_a = reduce_mirror_pair_loss(torch.tensor([4.0, 4.0]), batch_a)
    # Batch B: [ordinary 4, pair (2, 6) -> 4] -> 2 weights, mean 4.
    loss_b = reduce_mirror_pair_loss(
        torch.tensor([4.0, 2.0, 6.0]), batch_b
    )
    assert loss_a.numel() == 2
    assert loss_b.numel() == 2
    assert loss_a.numel() == loss_b.numel()
    assert loss_a.mean().item() == pytest.approx(4.0)
    assert loss_b.mean().item() == pytest.approx(4.0)


def test_reduce_identity_is_bit_identical() -> None:
    layout = MirrorPhysicalLayout.build((False, False))
    values = torch.tensor([1.5, 2.5])
    assert torch.equal(reduce_mirror_pair_loss(values, layout), values)


def test_reduce_requires_1d_matching_length() -> None:
    layout = MirrorPhysicalLayout.build((True,))
    with pytest.raises(ValueError):
        reduce_mirror_pair_loss(torch.tensor([[2.0, 6.0]]), layout)
    with pytest.raises(ValueError):
        reduce_mirror_pair_loss(torch.tensor([2.0]), layout)


def test_reduce_carries_gradient_to_both_views() -> None:
    layout = MirrorPhysicalLayout.build((True,))
    x = torch.tensor([2.0, 6.0], requires_grad=True)
    loss = reduce_mirror_pair_loss(x, layout)
    assert loss.numel() == 1
    loss.sum().backward()
    # d(0.5*x0 + 0.5*x1)/dx = (0.5, 0.5): both physical views receive an
    # equal 0.5 weight within the one logical sample.
    assert x.grad is not None
    assert x.grad.tolist() == [0.5, 0.5]


# ---------------------------------------------------------------------------
# _reduce_mirror_observation  (logical-domain high/low bucketing)
# ---------------------------------------------------------------------------


def _physical_loss(per_physical: torch.Tensor) -> _RuntimeLoss:
    # The physical bucketing the stacked/per-row loss path would produce if
    # every view were high-noise (t < boundary): high = total, low = 0.
    return _RuntimeLoss(
        per_physical,
        per_physical.sum(),
        torch.tensor(float(per_physical.numel())),
        torch.tensor(0.0),
        torch.tensor(0.0),
    )


def test_observation_rebuckets_on_logical_timesteps() -> None:
    layout = MirrorPhysicalLayout.build((False, True))
    per_physical = torch.tensor([10.0, 0.0, 2.0])
    logical = _reduce_mirror_observation(
        _physical_loss(per_physical), layout,
        torch.tensor([0.1, 0.99]), 0.95,
    )
    assert logical.per_sample.tolist() == [10.0, 1.0]
    # logical row 0 (t=0.1) is high-noise, logical row 1 (t=0.99) is low.
    assert logical.high_noise_loss_sum.item() == pytest.approx(10.0)
    assert logical.high_noise_sample_count.item() == 1
    assert logical.low_noise_loss_sum.item() == pytest.approx(1.0)
    assert logical.low_noise_sample_count.item() == 1
    # Conservation: high + low == total logical loss.
    total = logical.high_noise_loss_sum + logical.low_noise_loss_sum
    assert total.item() == pytest.approx(logical.per_sample.sum().item())


def test_observation_pair_shares_one_bucket() -> None:
    # Both views of a pair share the logical timestep, so both physical
    # views of the pair fall into the same bucket.
    layout = MirrorPhysicalLayout.build((True, False))
    # physical rows: [pair-original 2, pair-mirror 6, ordinary 10]
    per_physical = torch.tensor([2.0, 6.0, 10.0])
    logical = _reduce_mirror_observation(
        _physical_loss(per_physical), layout,
        torch.tensor([0.99, 0.1]), 0.95,
    )
    assert logical.per_sample.tolist() == [4.0, 10.0]
    # logical row 0 (the pair, t=0.99) is entirely low-noise.
    assert logical.low_noise_loss_sum.item() == pytest.approx(4.0)
    assert logical.low_noise_sample_count.item() == 1
    # logical row 1 (ordinary, t=0.1) is high-noise.
    assert logical.high_noise_loss_sum.item() == pytest.approx(10.0)
    assert logical.high_noise_sample_count.item() == 1


def test_observation_requires_matching_logical_timesteps() -> None:
    layout = MirrorPhysicalLayout.build((True, False))
    with pytest.raises(ValueError):
        _reduce_mirror_observation(
            _physical_loss(torch.tensor([2.0, 6.0, 10.0])),
            layout,
            torch.tensor([0.1]),  # one t for two logical rows
            0.95,
        )


# ---------------------------------------------------------------------------
# Shared timestep / noise / conditioning expansion contract (spec 24 / 25)
# ---------------------------------------------------------------------------


def test_expansion_shares_timestep_and_noise_per_pair() -> None:
    # The runtime draws one timestep and one noise tensor per LOGICAL sample
    # and expands them to physical rows via index_select(phys_to_log). Both
    # views of a pair must receive the identical value (torch.equal);
    # different logical samples stay independent.
    layout = MirrorPhysicalLayout.build((True, False, True))
    assert layout.phys_to_log == (0, 0, 1, 2, 2)
    expand = torch.as_tensor(layout.phys_to_log, dtype=torch.long)

    t_logical = torch.tensor([0.20, 0.50, 0.80])
    t_physical = t_logical.index_select(0, expand)
    # Pair 0 (logical 0): original row 0 and mirror row 1 share t.
    assert torch.equal(t_physical[0], t_physical[1])
    # Pair 2 (logical 2): original row 3 and mirror row 4 share t.
    assert torch.equal(t_physical[3], t_physical[4])
    # Ordinary logical 1 is untouched.
    assert torch.equal(t_physical[2], t_logical[1])
    # Distinct logical samples are distinct.
    assert t_physical[0].item() != t_physical[2].item()

    noise_logical = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    noise_physical = noise_logical.index_select(0, expand)
    assert torch.equal(noise_physical[0], noise_physical[1])
    assert torch.equal(noise_physical[3], noise_physical[4])
    assert not torch.equal(noise_physical[0], noise_physical[2])


def test_conditioning_is_row_duplicated_not_reencoded() -> None:
    # Conditioning rows are duplicated by index_select(phys_to_log), so the
    # number of UNIQUE logical rows the encoders see stays equal to the
    # logical batch size (Qwen forward count == 1 per logical sample, never
    # per physical view).
    layout = MirrorPhysicalLayout.build((True, False, True))
    logical_rows = torch.arange(
        layout.logical_count * 4, dtype=torch.float32
    ).reshape(layout.logical_count, 4)
    expand = torch.as_tensor(layout.phys_to_log, dtype=torch.long)
    physical_rows = logical_rows.index_select(0, expand)
    assert physical_rows.shape[0] == layout.physical_count
    # Every physical row equals its logical source row exactly.
    for phys, logical in enumerate(layout.phys_to_log):
        assert torch.equal(physical_rows[phys], logical_rows[logical])
    # Unique logical rows == logical count, not physical count.
    unique = {
        tuple(row.tolist()) for row in physical_rows
    }
    assert len(unique) == layout.logical_count
    assert layout.physical_count > layout.logical_count


# ---------------------------------------------------------------------------
# _physical_active_condition_indices
# ---------------------------------------------------------------------------


def test_active_condition_indices_remap() -> None:
    layout = MirrorPhysicalLayout.build((True, False, True))
    # Logical active rows {0, 2} -> physical rows {0,1,3,4}.
    active = torch.tensor([0, 2], dtype=torch.long)
    remapped = _physical_active_condition_indices(active, layout)
    assert remapped.tolist() == [0, 1, 3, 4]
    # Ordinary-only active row {1} -> physical {2}.
    assert (
        _physical_active_condition_indices(
            torch.tensor([1], dtype=torch.long), layout
        ).tolist()
        == [2]
    )
    # Empty stays empty.
    assert (
        _physical_active_condition_indices(
            torch.empty(0, dtype=torch.long), layout
        ).numel()
        == 0
    )


def test_active_condition_indices_rejects_bad_dtype() -> None:
    layout = MirrorPhysicalLayout.build((False,))
    with pytest.raises(ValueError):
        _physical_active_condition_indices(
            torch.tensor([0.0]), layout
        )
    with pytest.raises(ValueError):
        _physical_active_condition_indices(
            torch.tensor([[0]]), layout
        )


# ---------------------------------------------------------------------------
# _mirror_view_images  (uint8 -> bf16 training normalization)
# ---------------------------------------------------------------------------


def _payload(value: int) -> CameraMirrorPayload:
    return CameraMirrorPayload(
        mirror_image=torch.full((3, 8, 8), value, dtype=torch.uint8),
        crop_box=(0, 0, 8, 8),
        signed_pixel_center_shift=-16.0,
        camera_shift_y=-0.25,
        normalized_offset=0.9,
    )


def test_mirror_view_images_none_when_empty() -> None:
    batch = SimpleNamespace(mirror=(None, None))
    assert _mirror_view_images(batch, device=torch.device("cpu")) is None


def test_mirror_view_images_normalizes_like_originals() -> None:
    # The exact ordinary training normalization: uint8 -> bf16 -> /127.5
    # -> -1, so a mirror view enters the VAE identically to an original.
    batch = SimpleNamespace(mirror=(_payload(0), _payload(255)))
    views = _mirror_view_images(batch, device=torch.device("cpu"))
    assert views is not None
    assert views.shape == (2, 3, 8, 8)
    assert views.dtype == torch.bfloat16
    expected_0 = (
        torch.zeros(3, 8, 8, dtype=torch.bfloat16)
        .div(127.5)
        .sub(1.0)
    )
    expected_255 = (
        torch.full((3, 8, 8), 255, dtype=torch.bfloat16)
        .div(127.5)
        .sub(1.0)
    )
    assert torch.equal(views[0], expected_0)
    assert torch.equal(views[1], expected_255)
    # Range check: 0 -> -1.0, 255 -> +1.0 (uniform tensors).
    assert views[0].abs().max().item() == pytest.approx(1.0)
    assert views[1].abs().max().item() == pytest.approx(1.0)


def test_mirror_view_images_skips_none_rows() -> None:
    batch = SimpleNamespace(mirror=(None, _payload(128), None))
    views = _mirror_view_images(batch, device=torch.device("cpu"))
    assert views is not None
    assert views.shape == (1, 3, 8, 8)


def test_mirror_view_images_rejects_non_channel_first() -> None:
    bad = CameraMirrorPayload.__new__(CameraMirrorPayload)
    # Bypass __post_init__ to craft a non-CHW stacked shape.
    object.__setattr__(
        bad, "mirror_image", torch.zeros(8, 8, 3, dtype=torch.uint8)
    )
    object.__setattr__(bad, "crop_box", (0, 0, 8, 8))
    object.__setattr__(bad, "signed_pixel_center_shift", 0.0)
    object.__setattr__(bad, "camera_shift_y", 0.0)
    object.__setattr__(bad, "normalized_offset", 0.5)
    batch = SimpleNamespace(mirror=(bad,))
    with pytest.raises(ValueError):
        _mirror_view_images(batch, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# _physical_view_images  (interleave original + mirror)
# ---------------------------------------------------------------------------


def test_physical_view_images_interleave() -> None:
    layout = MirrorPhysicalLayout.build((True, False, True))
    # logical views: rows 0,1,2 ; mirror views: rows 3,4 (for logical 0,2)
    logical = torch.arange(12, dtype=torch.float32).reshape(3, 1, 1, 4)
    mirrors = torch.arange(8, dtype=torch.float32).reshape(2, 1, 1, 4) + 100
    physical = _physical_view_images(logical, mirrors, layout)
    assert physical.shape == (5, 1, 1, 4)
    # Physical order per physical_source_order = (0, 3, 1, 2, 4).
    expected = torch.cat((logical, mirrors), dim=0).index_select(
        0, torch.tensor(layout.physical_source_order, dtype=torch.long)
    )
    assert torch.equal(physical, expected)


def test_physical_view_images_rejects_shape_mismatch() -> None:
    layout = MirrorPhysicalLayout.build((True,))
    logical = torch.zeros(1, 1, 1, 4, dtype=torch.float32)
    wrong_count = torch.zeros(2, 1, 1, 4, dtype=torch.float32)
    with pytest.raises(ValueError):
        _physical_view_images(logical, wrong_count, layout)
    wrong_shape = torch.zeros(1, 1, 1, 9, dtype=torch.float32)
    with pytest.raises(ValueError):
        _physical_view_images(logical, wrong_shape, layout)


# ---------------------------------------------------------------------------
# _size_conditions  (count override for the physical view count)
# ---------------------------------------------------------------------------


def _dummy_batch(height: int, width: int, rows: int) -> SimpleNamespace:
    return SimpleNamespace(
        target_height=height,
        target_width=width,
        images=torch.zeros(rows, 3, height, width),
    )


def test_size_conditions_default_uses_logical_rows() -> None:
    batch = _dummy_batch(512, 512, rows=4)
    size_scale, aspect = _size_conditions(
        batch, device=torch.device("cpu")  # type: ignore[arg-type]
    )
    assert size_scale.shape == (4,)
    assert aspect.shape == (4,)
    # 512x512 -> size_scale = 0.5*log2(512*512/512/512) = 0, aspect 0.
    assert size_scale.abs().max().item() == pytest.approx(0.0)
    assert aspect.abs().max().item() == pytest.approx(0.0)


def test_size_conditions_count_override_for_physical_views() -> None:
    batch = _dummy_batch(512, 512, rows=4)
    size_scale, aspect = _size_conditions(
        batch, device=torch.device("cpu"), count=6  # type: ignore[arg-type]
    )
    assert size_scale.shape == (6,)
    assert aspect.shape == (6,)
    # One value per physical view, identical broadcast for a uniform batch.
    assert (size_scale == size_scale[0]).all().item()
    assert (aspect == aspect[0]).all().item()


# ---------------------------------------------------------------------------
# Active-condition device routing (canary readiness fix A)
#
# The mirror branch of prepare() must remap the ALREADY device-local
# active-condition indices (the batch attribute is CPU-side). The helper
# must preserve the input device, stay 1-D long, and work for the empty
# active set.
# ---------------------------------------------------------------------------


def test_active_condition_remap_preserves_cpu_device() -> None:
    layout = MirrorPhysicalLayout.build((True, False, True))
    active = torch.tensor([0, 2], dtype=torch.long)  # CPU input
    physical = _physical_active_condition_indices(active, layout)
    assert physical.device.type == "cpu"
    assert physical.dtype == torch.long
    assert physical.ndim == 1
    # Logical active rows {0,2} with mirror flags {T,F,T}: logical 0
    # expands to physical 0 (original) + 1 (mirror), logical 2 expands to
    # physical 3 (original) + 4 (mirror).
    assert tuple(physical.tolist()) == (0, 1, 3, 4)


def test_active_condition_remap_empty_set_stays_on_device() -> None:
    layout = MirrorPhysicalLayout.build((True, True))
    active = torch.zeros(0, dtype=torch.long)
    physical = _physical_active_condition_indices(active, layout)
    assert physical.device.type == "cpu"
    assert physical.dtype == torch.long
    assert physical.ndim == 1
    assert physical.numel() == 0


def test_active_condition_remap_rejects_bad_input() -> None:
    layout = MirrorPhysicalLayout.build((True,))
    with pytest.raises(ValueError):
        _physical_active_condition_indices(
            torch.zeros(2, 1, dtype=torch.long), layout
        )
    with pytest.raises(ValueError):
        _physical_active_condition_indices(torch.tensor([0.0]), layout)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HCU device required"
)
def test_active_condition_remap_preserves_cuda_device() -> None:
    layout = MirrorPhysicalLayout.build((True, False, True))
    active = torch.tensor([0, 2], dtype=torch.long, device="cuda")
    physical = _physical_active_condition_indices(active, layout)
    # Device parity with the input (and therefore with qwen_states in the
    # runtime, where both live on self.device).
    assert physical.device == active.device
    assert physical.device.type == "cuda"
    assert tuple(physical.cpu().tolist()) == (0, 1, 3, 4)
