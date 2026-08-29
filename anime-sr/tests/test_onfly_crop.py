"""P1-4 on-fly dynamic crop: train-path crop-box contract.

The train fetch used to pin every crop to (0,0). That is REQUIRED in store
mode — the pre-encoded z_hr store was built from ``box_seed(sample_id, 0, 0)``
(encode_latents.py), so a moving crop would silently desync LQ from z_hr.
In on-fly mode there is no store (z_hr is encoded by the consumer from the
exact crop), so the crop follows the §11.5 exposure identity:

    data_cycle     = step // exposure_per_cycle
    exposure_index = step % exposure_per_cycle

which makes the box a pure function of (sample_id, cycle, exposure):
resume to the same step reproduces the same crop. Val keeps its own pinned
contract separately (``_VAL_CYCLE``/``_VAL_EXPOSURE``), untouched here.

Guarantees under test:
  * on-fly: same step -> identical crop, even after a simulated restart;
  * on-fly: many distinct exposures -> many distinct crops (the box moves);
  * store: the crop is ALWAYS the pinned (0,0)-identity box, any step;
  * train/val contracts are separate (train identity comes from the step).
"""

from __future__ import annotations

from anime_sr.data.buckets import crop_box
from anime_sr.data.latent_store import LatentStore
from anime_sr.data.pipeline import SampleMeta, SRDataset, box_seed
from anime_sr.train import latent_flow as lf

EPC = 25  # _EXPOSURE_PER_CYCLE


def _meta(sid: str = "s-0001") -> SampleMeta:
    # 1600x1200 web image: room for the 1024 box to move (577 x 177 windows)
    return SampleMeta(
        sample_id=sid, shard="0", rel_path=f"{sid}.webp",
        width=1600, height=1200, is_validation=False,
    )


class _StubDS(SRDataset):
    """Minimal SRDataset surface: the REAL crop math (SRDataset.crop is a
    one-liner over crop_box/box_seed with self.bucket.hr), no shards —
    SRDataset.__init__ is skipped on purpose (needs index/webp dirs)."""

    def __init__(self, bucket_hr: int = 1024) -> None:
        self.bucket_hr = bucket_hr
        self.calls: list[tuple[str, int, int]] = []

    def crop(
        self, meta: SampleMeta, data_cycle: int, exposure_index: int
    ) -> tuple[int, int]:
        self.calls.append((meta.sample_id, data_cycle, exposure_index))
        return crop_box(
            meta.width, meta.height, self.bucket_hr,
            box_seed(meta.sample_id, data_cycle, exposure_index),
        )


class _StubStore(LatentStore):
    """Truthy stand-in for the store: _train_crop_box only checks `is not
    None` (it never reads from the store on the crop decision)."""

    def __init__(self) -> None:
        pass


def test_onfly_same_step_same_crop() -> None:
    """The box is a pure function of (sample, step): repeat fetches of the
    same step — including after a simulated "restart" at that step — agree
    exactly, and the box always sits inside the valid window."""
    ds = _StubDS()
    meta = _meta()
    for step in (0, 3 * EPC + 7, EPC + 11, 4242):
        a = lf._train_crop_box(ds, meta, None, step, EPC)
        b = lf._train_crop_box(ds, meta, None, step, EPC)
        assert a == b, f"non-deterministic crop at step {step}"
    boxes = {lf._train_crop_box(ds, meta, None, s, EPC) for s in range(30)}
    for x, y in boxes:
        assert 0 <= x <= meta.width - 1024 and 0 <= y <= meta.height - 1024


def test_onfly_exposures_move_the_crop() -> None:
    """100 distinct (cycle, exposure) identities on a 1600x1200 image must
    not all collapse to one box — the crop really moves with the exposure."""
    ds = _StubDS()
    meta = _meta()
    boxes = {
        lf._train_crop_box(ds, meta, None, EPC * c + i, EPC)
        for c in range(4)
        for i in range(EPC)
    }
    assert len(boxes) > 10, f"only {len(boxes)} distinct boxes in 100 exposures"
    for x, y in boxes:
        assert 0 <= x <= meta.width - 1024 and 0 <= y <= meta.height - 1024


def test_onfly_resume_stream_reproducible() -> None:
    """The box stream at step s depends only on s: simulate a run, 'crash',
    restart at step 300 — the boxes for steps 300..349 must be identical
    to the first run's (the resume contract)."""
    ds1, ds2 = _StubDS(), _StubDS()
    metas = [_meta(f"s-{i:04d}") for i in range(8)]
    stream1 = [
        lf._train_crop_box(ds1, metas[step % 8], None, step, EPC)
        for step in range(300, 350)
    ]
    stream2 = [
        lf._train_crop_box(ds2, metas[step % 8], None, step, EPC)
        for step in range(300, 350)  # resume: same step -> same crop
    ]
    assert stream1 == stream2


def test_store_mode_always_pinned() -> None:
    """Store mode must keep the pre-encoded z_hr alignment: for EVERY step
    the dataset is asked for the pinned (sample, 0, 0) identity — and the
    returned box is exactly the (0,0)-identity box."""
    ds = _StubDS()
    meta = _meta()
    store = _StubStore()
    for step in (0, 1, EPC - 1, EPC, 4321):
        box = lf._train_crop_box(ds, meta, store, step, EPC)
        assert box == crop_box(
            meta.width, meta.height, 1024, box_seed(meta.sample_id, 0, 0)
        )
    assert all(c == (meta.sample_id, 0, 0) for c in ds.calls), (
        f"store mode must pin (sample, 0, 0), saw {set(ds.calls)}"
    )


def test_train_val_contracts_separate() -> None:
    """The train crop identity is derived from the STEP (cycle, exposure),
    not a constant: same exposure in two different cycles is a different
    crop draw, so val's fixed (cycle, exposure) contract stays separate."""
    ds = _StubDS()
    meta = _meta()
    lf._train_crop_box(ds, meta, None, 5, EPC)  # (cycle 0, exp 5)
    lf._train_crop_box(ds, meta, None, EPC + 5, EPC)  # (cycle 1, exp 5)
    assert (meta.sample_id, 0, 5) in ds.calls
    assert (meta.sample_id, 1, 5) in ds.calls
