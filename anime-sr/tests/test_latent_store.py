"""z_hr latent store (plan §4.3): round-trip, resume, failure modes, index.

Pure CPU test with a small synthetic latent (4ch, 64px bucket -> 4x4 grid);
no VAE weights or GPU required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from anime_sr.cli.encode_latents import select_crops
from anime_sr.data.latent_store import INDEX_NAME, LatentStore, read_index
from anime_sr.data.pipeline import SampleMeta


def _store(tmp_path: Path, **kw: int) -> LatentStore:
    kwargs: dict[str, int] = {"bucket_hr": 64, "channels": 4}
    kwargs.update(kw)
    return LatentStore(tmp_path, **kwargs)


def _z(i: int) -> torch.Tensor:
    g = 4
    z = torch.arange(i, i + 4 * g * g, dtype=torch.float32).view(4, g, g)
    return (z % 17 - 8) / 8.0  # values inside the fp16-safe range


def test_round_trip(tmp_path: Path) -> None:
    st = _store(tmp_path)
    z = _z(1)
    assert st.write("s1", z)
    out = st.read("s1")
    assert out.dtype == torch.float16
    assert out.shape == (4, 4, 4)
    assert torch.equal(out, z.to(torch.float16))
    assert st.has("s1")


def test_resume_skip(tmp_path: Path) -> None:
    st = _store(tmp_path)
    assert st.write("s1", _z(1))
    assert not st.write("s1", _z(999))  # existing size-valid file -> skip
    assert torch.equal(st.read("s1"), _z(1).to(torch.float16))


def test_read_missing_hard_fails(tmp_path: Path) -> None:
    st = _store(tmp_path)
    with pytest.raises(FileNotFoundError):
        st.read("nope")
    assert not st.has("nope")


def test_mis_sized_file_hard_fails(tmp_path: Path) -> None:
    st = _store(tmp_path)
    st.write("s1", _z(1))
    (st.z_dir / "s1.bin").write_bytes(b"\x00" * 8)
    with pytest.raises(RuntimeError):
        st.read("s1")


def test_write_shape_checked(tmp_path: Path) -> None:
    st = _store(tmp_path)
    with pytest.raises(ValueError, match="latent shape"):
        st.write("s1", torch.zeros(4, 2, 4, dtype=torch.float16))


def test_finalize_index_and_read(tmp_path: Path) -> None:
    st = _store(tmp_path)
    st.write("a", _z(1))
    st.write("b", _z(2))
    p = st.finalize_index(["a", "b"])
    assert p.name == INDEX_NAME
    doc = read_index(st.root)
    assert doc["version"] == 1
    assert doc["dtype"] == "fp16"
    assert doc["n_samples"] == 2
    assert doc["samples"]["a"]["file"] == "z/a.bin"
    assert st.expected_bytes == 4 * 4 * 4 * 2


def test_finalize_index_refuses_incomplete(tmp_path: Path) -> None:
    st = _store(tmp_path)
    st.write("a", _z(1))
    with pytest.raises(RuntimeError, match="incomplete"):
        st.finalize_index(["a", "b"])


def test_read_index_missing(tmp_path: Path) -> None:
    st = _store(tmp_path)
    with pytest.raises(FileNotFoundError):
        read_index(st.root)


def test_bucket_must_be_multiple_of_16(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="multiple of the 16x VAE"):
        LatentStore(tmp_path, bucket_hr=100)


def test_select_crops_order_and_cap() -> None:
    sids = [f"danbooru/{i}" for i in range(10)]
    metas = [
        SampleMeta(
            sample_id=s,
            shard="shard-000",
            rel_path=f"danbooru/{s.rsplit('/', 1)[-1]}.webp",
            width=1024,
            height=1024,
            is_validation=False,
        )
        for s in sids
    ]
    a = select_crops(metas, 4, salt="cbank")
    assert a == select_crops(metas, 4, salt="cbank")  # deterministic
    assert len(a) == 4
    # ordering must follow the CLI's key: blake2b(salt|sid) as little-endian u64
    full = [
        s
        for s in sorted(
            sids,
            key=lambda s: int.from_bytes(
                hashlib.blake2b(f"cbank|{s}".encode(), digest_size=8).digest(), "little"
            ),
        )
    ]
    assert [metas[i].sample_id for i in select_crops(metas, 10, salt="cbank")] == full
    with pytest.raises(SystemExit, match="--n-crops"):
        select_crops(metas, 11)
