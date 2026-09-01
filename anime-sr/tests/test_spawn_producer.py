"""Spawn producer (2026-09-01): torch-free worker + bit-exactness contract.

The 09-01 salt9 death loop (host vm.overcommit_memory=2, shared ~453 GiB
CommitLimit) killed the forked process producer (each fork counted the
parent's torch commit); the thread producer survived the VA budget but
GIL-shares with the training loop (0.6-0.7 it/s vs 1.5-1.6 canary).
``producer="spawn"`` is the middle path: fresh torch-free worker
processes (numpy/PIL only).

Contract under test:
  * the spawn worker's module graph must NEVER contain torch (the VA
    budget depends on it — a stray import regresses the whole fix);
  * ``load_hr_numpy`` + the consumer's exact decode_hr conversion is
    BIT-EXACT with ``SRDataset.decode_hr`` (full and cropped);
  * the crop box the worker computes is the trainer's exact rule
    (``crop_box`` + ``box_seed`` over the dynamic exposure identity);
  * a real spawn Pool round-trips (init + apply_async + result).
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
from anime_sr import torchfree_fetch as tf
from anime_sr.config.schema import Config
from anime_sr.data.buckets import crop_box
from anime_sr.data.index import build_index, scan_shard_full
from anime_sr.data.pipeline import (
    SRDataset,
    box_seed,
)
from anime_sr.train.latent_flow import degrade_hr
from PIL import Image

CFG = Config()

_CASES = [
    (3001, 1200, 900, "sfw"),
    (3002, 1400, 1200, "sfw"),
]


def _make_shard(tmp_path: Path, flat: str = "shard-3_2024-000007.tar") -> Path:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tfh:
        for cid, meta_w, meta_h, nsfw in _CASES:
            stem = f"danbooru/5.9/3_2024/{cid}"
            img_bytes = io.BytesIO()
            rng = np.random.default_rng(cid)
            Image.fromarray(
                rng.integers(0, 255, (meta_h // 2, meta_w // 2, 3), dtype=np.uint8)
            ).save(img_bytes, format="WEBP")
            img_bytes = img_bytes.getvalue()
            meta = json.dumps(
                {
                    "schema_version": 1,
                    "id": cid,
                    "source": {
                        "dataset": "danbooru",
                        "release": "3_2024",
                        "original_path": f"3_2024/{cid}.webp",
                    },
                    "image": {"format": "webp", "width": meta_w, "height": meta_h},
                    "nsfw": nsfw,
                    "tags": {"general": ["anime"]},
                }
            )
            t1 = tarfile.TarInfo(f"{stem}.webp")
            t1.size = len(img_bytes)
            tfh.addfile(t1, io.BytesIO(img_bytes))
            t2 = tarfile.TarInfo(f"{stem}.json")
            t2.size = len(meta)
            tfh.addfile(t2, io.BytesIO(meta.encode()))
    p = tmp_path / flat
    p.write_bytes(buf.getvalue())
    return p


def _dataset(tmp_path: Path) -> tuple[SRDataset, Path]:
    """SRDataset in tar-direct mode with the shard pinned (like the driver)."""
    shard = _make_shard(tmp_path)
    index_dir = tmp_path / "index"
    build_index(
        [str(shard)],
        CFG,
        index_dir,
        scan=lambda sp: scan_shard_full(sp, "shard-3_2024-000007.tar"),
    )
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    (pinned / "shard-3_2024-000007.tar").write_bytes(shard.read_bytes())
    ds = SRDataset(
        index_dir, tmp_path / "webp", CFG, bucket_hr=512,
        split="train", tar_dir=pinned,
    )
    return ds, pinned


def test_worker_module_source_is_torch_free() -> None:
    """The VA budget of the spawn workers: the worker module may not
    import torch (numpy/PIL/stdlib only) — the fresh-interpreter probe
    below is the authoritative import-graph check."""
    src = Path(tf.__file__).read_text()
    assert "import torch" not in src
    assert "from torch" not in src


def test_worker_probe_in_fresh_interpreter(tmp_path: Path) -> None:
    """Authoritative VA-budget probe: a FRESH python (spawn worker's real
    environment) runs the worker's own module probe and must not have
    torch loaded (the probe lives in the torchfree module, so the worker
    never imports this test module and its torch imports)."""
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        processes=1, initializer=tf.init_worker, initargs=(None, None, 512)
    ) as pool:
        mods = pool.apply(tf._probe_modules, ())
    assert "torch" not in mods, f"torch loaded in torch-free worker: {mods}"
    assert "numpy" in mods and "PIL" in mods


def _convert(crop: np.ndarray) -> torch.Tensor:
    """The consumer's EXACT decode_hr conversion (latent_flow spawn branch):
    the per-element uint8->fp32 linear map, CHW."""
    return (
        torch.from_numpy(crop).permute(2, 0, 1).float().mul(2.0 / 255.0).sub(1.0)
    ).contiguous()


def test_load_hr_numpy_bitexact_with_decode_hr(tmp_path: Path) -> None:
    ds, pinned = _dataset(tmp_path)
    for meta in ds.samples:
        t_full, _st = ds.decode_hr_timed(meta)
        arr, st = tf.load_hr_numpy(meta, pinned, None)
        assert set(st) == {"shard", "decode"}
        t_np = (
            torch.from_numpy(arr).permute(2, 0, 1).float().mul(2.0 / 255.0).sub(1.0)
        )
        assert torch.equal(t_full, t_np), f"not bit-exact for {meta.sample_id}"
        assert t_full.shape == (3, meta.height, meta.width)


def test_crop_path_bitexact_with_trainer_crop(tmp_path: Path) -> None:
    """worker numpy crop + consumer convert == trainer torch slice (the
    per-element linear map commutes with the slice)."""
    ds, pinned = _dataset(tmp_path)
    epc = 25
    for meta in ds.samples:
        for step in (0, 3, 27, 140):
            t_full, _ = ds.decode_hr_timed(meta)
            x, y = crop_box(
                meta.width, meta.height, 512,
                box_seed(meta.sample_id, step // epc, step % epc),
            )
            ref = t_full[..., y : y + 512, x : x + 512].contiguous()
            crop_np, st = tf.fetch_crop_numpy_worker(
                meta, step, epc, tar_dir=pinned, webp_dir=None, bucket_hr=512
            )
            assert crop_np.shape == (512, 512, 3) and crop_np.dtype == np.uint8
            assert set(st) == {"shard", "decode", "crop"}
            assert torch.equal(_convert(crop_np), ref), (
                f"crop not bit-exact for {meta.sample_id} step {step}"
            )


def test_degradation_bitexact_consumer_side(tmp_path: Path) -> None:
    """The spawn consumer runs the SAME seeded degrade on its hr_crop as
    the thread producer: identical lq for the same (sample, step)."""
    ds, pinned = _dataset(tmp_path)
    epc = 25
    for meta in ds.samples:
        for step in (0, 27):
            t_full, _ = ds.decode_hr_timed(meta)
            x, y = crop_box(
                meta.width, meta.height, 512,
                box_seed(meta.sample_id, step // epc, step % epc),
            )
            hr_ref = t_full[..., y : y + 512, x : x + 512].contiguous()
            crop_np, _ = tf.fetch_crop_numpy_worker(
                meta, step, epc, tar_dir=pinned, webp_dir=None, bucket_hr=512
            )
            lq_ref, _ = degrade_hr(
                hr_ref, CFG, global_seed=ds.global_seed, sample_id=meta.sample_id,
                data_cycle=step // epc, exposure_index=step % epc,
            )
            lq_new, _ = degrade_hr(
                _convert(crop_np), CFG, global_seed=ds.global_seed,
                sample_id=meta.sample_id, data_cycle=step // epc,
                exposure_index=step % epc,
            )
            assert torch.equal(lq_ref, lq_new), (
                f"lq not bit-exact for {meta.sample_id} step {step}"
            )


def test_spawn_pool_roundtrip(tmp_path: Path) -> None:
    import multiprocessing

    ds, pinned = _dataset(tmp_path)
    epc = 25
    with multiprocessing.get_context("spawn").Pool(
        processes=2,
        initializer=tf.init_worker,
        initargs=(str(pinned), None, 512),
    ) as pool:
        futs = [
            pool.apply_async(
                tf.fetch_crop_numpy, ((meta, step, epc),)
            )
            for meta, step in zip(ds.samples, (0, 3))
        ]
        results = [f.get() for f in futs]
    for (crop, st), meta in zip(results, ds.samples):
        assert crop.shape == (512, 512, 3)
        assert set(st) == {"shard", "decode", "crop"}
        t_full, _ = ds.decode_hr_timed(meta)
        x, y = crop_box(meta.width, meta.height, 512, box_seed(meta.sample_id, 0, 0))
        ref = t_full[..., y : y + 512, x : x + 512].contiguous()
        assert torch.equal(_convert(crop), ref)
