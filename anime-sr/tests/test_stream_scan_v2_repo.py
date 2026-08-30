"""Unit tests for tools/stream_scan_v2_repo.py (offline, synthetic tar).

The walk logic (tar header parsing, offset arithmetic, windowed buffer,
webp/json handler dispatch) is exercised against a synthetic tar byte stream
served by an in-memory fake range feed — no network, no production data.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

from anime_sr.data.index import _parse_meta, webp_header_size

TOOL = Path(__file__).resolve().parents[1] / "tools" / "stream_scan_v2_repo.py"
_spec = importlib.util.spec_from_file_location("stream_scan_v2_repo", TOOL)
assert _spec is not None and _spec.loader is not None
st = importlib.util.module_from_spec(_spec)
sys.modules["stream_scan_v2_repo"] = st
_spec.loader.exec_module(st)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _vp8x_header(w: int, h: int) -> bytes:
    hdr = bytearray(64)
    hdr[0:4] = b"RIFF"
    hdr[8:12] = b"WEBP"
    hdr[12:16] = b"VP8X"
    hdr[16:20] = (10).to_bytes(4, "little")
    hdr[24:27] = (w - 1).to_bytes(3, "little")
    hdr[27:30] = (h - 1).to_bytes(3, "little")
    return bytes(hdr)


class FakeFeed:
    """In-memory range feed: get(s, e) -> data[s:e+1]."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.n_req = 0

    def get(self, s: int, e: int) -> bytes:
        self.n_req += 1
        return self.data[s : e + 1]


def _make_tar_bytes(members: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def _doc(sid: str, **kw) -> bytes:
    d = {
        "id": sid,
        "image": {"width": kw.get("w", 1200), "height": kw.get("h", 800)},
        "nsfw": "sfw",
        "source": {"release": "1_2024"},
        "tags": {"general": []},
    }
    if "tier" in kw:
        d["quality"] = kw["tier"]
    return json.dumps(d).encode()


# ---------------------------------------------------------------------------
# parse_tar_header / pad512
# ---------------------------------------------------------------------------
def test_pad512() -> None:
    assert st.pad512(0) == 0
    assert st.pad512(1) == 512
    assert st.pad512(512) == 512
    assert st.pad512(513) == 1024
    assert st.pad512(2000) == 2048


def test_parse_tar_header_roundtrip() -> None:
    data = _make_tar_bytes([("a/b/1.json", b"x" * 700)])
    name, size = st.parse_tar_header(data[0:512])
    assert name == "a/b/1.json"
    assert size == 700
    # the end-of-archive zero block
    assert st.parse_tar_header(b"\x00" * 512) is None
    assert st.parse_tar_header(b"\x00" * 10) is None  # short header at EOF


# ---------------------------------------------------------------------------
# walk_shard (synthetic shard, windowed buffer)
# ---------------------------------------------------------------------------
def test_walk_shard_dispatch_and_pairing() -> None:
    members: list[tuple[str, bytes]] = [
        ("danbooru/5.9/1_2024/101.webp", _vp8x_header(1200, 800)),
        ("danbooru/5.9/1_2024/101.json", _doc("101", tier="good")),
        # reversed order (json before webp): pairing must not depend on order
        ("danbooru/5.9/1_2024/102.json", _doc("102", w=1410, h=2048)),
        ("danbooru/5.9/1_2024/102.webp", _vp8x_header(1410, 2048)),
        # a big json (larger than the 8192B window) forces a top-up fetch
        ("danbooru/5.9/1_2024/103.webp", _vp8x_header(1600, 1200)),
        (
            "danbooru/5.9/1_2024/103.json",
            json.dumps(
                {
                    "id": "103",
                    "image": {"width": 1600, "height": 1200},
                    "nsfw": "sfw",
                    "source": {"release": "1_2024"},
                    "tags": {"general": [f"tag_{i:04d}" for i in range(2000)]},
                }
            ).encode(),
        ),
        # an unrelated member is skipped
        ("README.txt", b"hello"),
    ]
    data = _make_tar_bytes(members)
    feed = FakeFeed(data)
    webp_dims: dict[str, tuple[int, int] | None] = {}
    json_payloads: dict[str, bytes] = {}

    def h_webp(stem: str, payload: bytes) -> None:
        webp_dims[stem] = webp_header_size(payload)

    def h_json(stem: str, payload: bytes) -> None:
        json_payloads[stem] = payload

    n_webp, n_json = st.walk_shard(feed.get, len(data), 8192, h_webp, h_json)
    assert n_webp == 3
    assert n_json == 3
    # every json payload byte-identical to the original member
    for sid in ("101", "102", "103"):
        orig = {n: p for n, p in members}[f"danbooru/5.9/1_2024/{sid}.json"]
        assert json_payloads[sid] == orig, sid
        assert _parse_meta(json_payloads[sid]) is not None
    # webp dims keyed by basename stem, order-independent
    assert webp_dims["101"] == (1200, 800)
    assert webp_dims["102"] == (1410, 2048)
    assert webp_dims["103"] == (1600, 1200)
    # each member costs at least its header read; the windowed buffer may
    # cover several members per fetch, so total requests <= n_members (+1 for
    # the big-json top-up) and >= the members that must have their own fetch
    n_members = 7
    assert 2 <= feed.n_req <= n_members + 1


def test_walk_shard_tiny_window_degrades_but_stays_correct() -> None:
    members: list[tuple[str, bytes]] = [
        ("danbooru/5.9/1_2024/201.webp", _vp8x_header(900, 700)),
        ("danbooru/5.9/1_2024/201.json", _doc("201")),
        ("danbooru/5.9/1_2024/202.webp", _vp8x_header(5000, 4000)),
        ("danbooru/5.9/1_2024/202.json", _doc("202", w=5000, h=4000)),
    ]
    data = _make_tar_bytes(members)
    feed = FakeFeed(data)
    webp_dims: dict[str, tuple[int, int] | None] = {}
    payloads: list[bytes] = []

    n_webp, n_json = st.walk_shard(
        feed.get,
        len(data),
        600,  # smaller than header+payload for the json -> forces top-ups
        lambda stem, p: webp_dims.__setitem__(stem, webp_header_size(p)),
        lambda stem, p: payloads.append(p),
    )
    assert (n_webp, n_json) == (2, 2)
    assert webp_dims["201"] == (900, 700)
    assert webp_dims["202"] == (5000, 4000)
    assert all(_parse_meta(p) is not None for p in payloads)
    assert feed.n_req >= 4  # tiny window => every member needs its own fetch(es)


def test_walk_shard_empty_tar() -> None:
    data = _make_tar_bytes([])  # only the two zero blocks
    feed = FakeFeed(data)
    n_webp, n_json = st.walk_shard(feed.get, len(data), 8192, lambda s, p: None, lambda s, p: None)
    assert (n_webp, n_json) == (0, 0)
    assert feed.n_req >= 1  # read the zero block, then stop
