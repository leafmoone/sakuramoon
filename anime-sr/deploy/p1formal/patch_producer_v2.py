#!/usr/bin/env python3.11
"""Surgical port of the verified process-pool producer (p1 latent_flow
md5 0f9740d8, canary #6 evidence base) into the p1formal V2 tree.

Targets (pre-patch md5, asserted):
  src/anime_sr/train/latent_flow.py  3af6c35f
  src/anime_sr/config/schema.py      ec5369af

Edits are strict literal replacements (each anchor must occur exactly
once); the module block carries the P1-PRODUCER-PORT-V2 marker, so the
script is idempotent. Post-patch: py_compile both files under DTK python.
"""
import hashlib
import pathlib
import subprocess
import sys

ROOT = pathlib.Path("/root/anime-sr-p1formal")
LF = ROOT / "src/anime_sr/train/latent_flow.py"
SCHEMA = ROOT / "src/anime_sr/config/schema.py"
DTK_PY = "/usr/local/bin/python3.11"
MARKER = "P1-PRODUCER-PORT-V2"
EXPECT = {"latent_flow.py": "3af6c35f", "schema.py": "ec5369af"}


def md5(p: pathlib.Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()[:8]


def apply(path: pathlib.Path, edits: list[tuple[str, str]]) -> None:
    txt = path.read_text()
    if MARKER in txt and "producer" in txt:
        # idempotency: only latent_flow carries the marker; schema uses its own
        pass
    for old, new in edits:
        c = txt.count(old)
        assert c == 1, (
            f"{path.name}: anchor occurs {c} times (need exactly 1):\n"
            f"{old[:160]!r}"
        )
        txt = txt.replace(old, new)
    path.write_text(txt)
    print(f"[ok] {path.name}: {len(edits)} edit(s) applied")


MODULE_BLOCK = '''# P1-PRODUCER-PORT-V2 ----------------------------------------------------------------
# Process-pool producer (lf.producer="process"): the worker body is a
# MODULE-LEVEL function so the pickled task payload is just (slot, step);
# the heavy dataset/store/cfg context is inherited copy-on-write through
# the fork and lives in _PRODUCER_CTX. The body is line-for-line the same
# per-stage work as the in-loop _fetch closure -> the §11.5 stream is
# unchanged (only the transport differs).
# ---------------------------------------------------------------------------
_PRODUCER_CTX: dict[str, Any] | None = None


def _pp_worker_init() -> None:
    """Pool initializer (fork start method, runs once per worker).

    The producer context is inherited COW — nothing is pickled. The parent
    may pin torch intra-op threads (or not); a worker re-running small CPU
    torch ops (degrade convs) with the inherited intra-op pool leaves its
    cores idle (or oversubscribes), so re-tune from OMP_NUM_THREADS (the
    launch env, inherited through the fork)."""
    import os

    try:
        n = int(os.environ.get("OMP_NUM_THREADS", "1"))
    except ValueError:
        n = 1
    if n > 1 and torch.get_num_threads() != n:
        torch.set_num_threads(n)


def _pp_fetch(args: tuple[int, int]) -> tuple:
    """Process-worker body: returns (hr | None, lq, z_hr | None, meta, st).

    ``hr`` is None in store mode (the consumer never consumes it there;
    skipping it halves the pipe payload); ``z_hr`` is None in on-fly mode
    (the consumer encodes it from the hr crop)."""
    ctx = _PRODUCER_CTX
    assert ctx is not None, "producer context unset (pool must fork before use)"
    slot, step = args
    ds = ctx["ds"]
    cfg = ctx["cfg"]
    store = ctx["store"]
    j = ctx["order"][slot % ctx["n"]]
    meta = ds.samples[j]
    st: dict[str, float] = {}
    hr_full, dec = ds.decode_hr_timed(meta)  # shard/decode stage split
    st["shard"] = dec["shard"]
    st["decode"] = dec["decode"]
    t_c0 = time.perf_counter()
    x, y = ds.crop(meta, 0, 0)  # pinned (0,0) box — matches the pre-encoded z_hr
    bucket_hr = int(ctx["bucket_hr"])
    hr_crop = hr_full[..., y : y + bucket_hr, x : x + bucket_hr].contiguous()
    st["crop"] = time.perf_counter() - t_c0
    t_d0 = time.perf_counter()
    lq, _ = degrade_hr(
        hr_crop,
        cfg,
        global_seed=ctx["global_seed"],
        sample_id=meta.sample_id,
        data_cycle=step // int(ctx["exposure_per_cycle"]),
        exposure_index=step % int(ctx["exposure_per_cycle"]),
    )
    st["degradation"] = time.perf_counter() - t_d0
    t_z0 = time.perf_counter()
    if store is not None:
        z_hr_s = store.read(meta.sample_id)  # fp16 CPU
    else:
        z_hr_s = None  # P1 ④ on-fly: the consumer encodes z_hr
    st["z_hr"] = time.perf_counter() - t_z0
    return (None if store is not None else hr_crop, lq, z_hr_s, meta, st)


def _make_process_pool(n_workers: int) -> _MPPool:
    """Fork n_workers producer workers (Linux start method only)."""
    ctx = multiprocessing.get_context("fork")
    return ctx.Pool(processes=n_workers, initializer=_pp_worker_init)


def _fut_done(f: Future | _MPAsyncResult) -> bool:
    """Done-check across both producer backends (duck-identical APIs)."""
    if isinstance(f, _MPAsyncResult):
        return f.ready()
    return cast("Future", f).done()


def _fut_result(f: Future | _MPAsyncResult, timeout: float | None = None) -> Any:
    """Blocking collect across both producer backends."""
    if isinstance(f, _MPAsyncResult):
        return f.get(timeout)
    return cast("Future", f).result(timeout)


'''

LF_EDITS = [
    # E1: stdlib import — multiprocessing (get_context)
    (
        "import json\nimport time\n",
        "import json\nimport multiprocessing\nimport time\n",
    ),
    # E2: pool aliases (AsyncResult/Pool — duck-identical dispatch below)
    (
        "from concurrent.futures import Future, ThreadPoolExecutor\n"
        "from contextlib import nullcontext\n",
        "from concurrent.futures import Future, ThreadPoolExecutor\n"
        "from contextlib import nullcontext\n"
        "from multiprocessing.pool import AsyncResult as _MPAsyncResult\n"
        "from multiprocessing.pool import Pool as _MPPool\n",
    ),
    # E3: module-level producer block (marker-carrying) before run_latent_flow
    (
        "def run_latent_flow(\n",
        MODULE_BLOCK + "def run_latent_flow(\n",
    ),
    # E4: fork BEFORE any HCU context (anchor: the VAE load line)
    (
        "    vae = load_frozen_vae(vae_path or cfg.vae.path, device, dtype=dtype)\n",
        "    # P1-PRODUCER-PORT-V2: the process-pool producer must fork BEFORE any\n"
        "    # HCU context exists (workers are CPU-only and never touch the HCU;\n"
        "    # forking after the device init would inherit accelerator runtime\n"
        "    # state). Thread mode creates its pool at the classic spot below.\n"
        "    pool: ThreadPoolExecutor | _MPPool | None = None\n"
        "    if lf.producer == \"process\":\n"
        "        global _PRODUCER_CTX\n"
        "        _PRODUCER_CTX = {\n"
        "            \"ds\": ds,\n"
        "            \"order\": order,\n"
        "            \"n\": n,\n"
        "            \"cfg\": cfg,\n"
        "            \"store\": store,\n"
        "            \"global_seed\": ds.global_seed,\n"
        "            \"bucket_hr\": bucket_hr,\n"
        "            \"exposure_per_cycle\": _EXPOSURE_PER_CYCLE,\n"
        "        }\n"
        "        n_pp = max(1, (lf.prefetch_depth or 1) * lf.batch_size)\n"
        "        pool = _make_process_pool(n_pp)\n"
        "        if rank == 0:\n"
        "            print(\n"
        "                f\"[latent] producer=process: {n_pp} forked workers \"\n"
        "                f\"(intra-op re-tuned from OMP_NUM_THREADS)\",\n"
        "                flush=True,\n"
        "            )\n"
        "\n"
        "    vae = load_frozen_vae(vae_path or cfg.vae.path, device, dtype=dtype)\n",
    ),
    # E5: guard the classic thread-pool creation site
    (
        "    depth = max(0, lf.prefetch_depth)\n"
        "    pool = ThreadPoolExecutor(\n"
        "        max_workers=max(1, (depth or 1) * bs), thread_name_prefix=\"lfetch\"\n"
        "    )\n",
        "    depth = max(0, lf.prefetch_depth)\n"
        "    if lf.producer == \"thread\":\n"
        "        pool = ThreadPoolExecutor(\n"
        "            max_workers=max(1, (depth or 1) * bs), thread_name_prefix=\"lfetch\"\n"
        "        )\n"
        "    assert pool is not None, (\n"
        "        \"producer pool must be created (thread here, process above)\"\n"
        "    )\n"
        "\n"
        "    _is_proc = lf.producer == \"process\"\n",
    ),
    # E6: _submit_batch dual dispatch (process pool apply_async packing)
    (
        "    def _submit_batch(step: int) -> list[Future]:\n"
        "        return [\n"
        "            pool.submit(\n"
        "                _fetch, latent_sample_index(step, rank, i, bs, world_size, n), step\n"
        "            )\n"
        "            for i in range(bs)\n"
        "        ]\n",
        "    def _submit_batch(step: int) -> list[Future | _MPAsyncResult]:\n"
        "        pairs = [\n"
        "            (latent_sample_index(step, rank, i, bs, world_size, n), step)\n"
        "            for i in range(bs)\n"
        "        ]\n"
        "        if _is_proc:\n"
        "            ppool = cast(\"_MPPool\", pool)\n"
        "            return [ppool.apply_async(_pp_fetch, ((slot, st),)) for slot, st in pairs]\n"
        "        tpool = cast(\"ThreadPoolExecutor\", pool)\n"
        "        return [tpool.submit(_fetch, slot, st) for slot, st in pairs]\n",
    ),
    # E7: ready-queue type admits both future kinds
    (
        "    ready: deque[list[Future]] = deque()\n",
        "    ready: deque[list[Future | _MPAsyncResult]] = deque()\n",
    ),
    # E8: ready-queue telemetry across both backends
    (
        "            ready_occ_sum += sum(1 for q in ready if all(f.done() for f in q))\n"
        "            if not all(f.done() for f in futs):\n",
        "            ready_occ_sum += sum(1 for q in ready if all(_fut_done(f) for f in q))\n"
        "            if not all(_fut_done(f) for f in futs):\n",
    ),
    # E9: collect path (process pool returns bare tuples -> rewrap)
    (
        "        prepared = [f.result() for f in futs]\n",
        "        if _is_proc:\n"
        "            # process pool hands back the (hr, lq, z_hr, meta, stages) tuple;\n"
        "            # rewrap into _Prepared (hr is None in store mode)\n"
        "            prepared = [\n"
        "                _Prepared(hr=r[0], lq=r[1], z_hr=r[2], meta=r[3], stages=r[4])\n"
        "                for r in (_fut_result(f) for f in futs)\n"
        "            ]\n"
        "        else:\n"
        "            prepared = [_fut_result(f) for f in futs]\n",
    ),
    # E10: _Prepared.hr is None in process-pool store mode
    (
        "        hr: torch.Tensor\n"
        "        lq: torch.Tensor\n"
        "        z_hr: torch.Tensor | None  # None in P1 ④ on-fly mode (consumer encodes)\n",
        "        hr: torch.Tensor | None  # None: process-pool store mode (never read)\n"
        "        lq: torch.Tensor\n"
        "        z_hr: torch.Tensor | None  # None in P1 ④ on-fly mode (consumer encodes)\n",
    ),
]

SCHEMA_EDITS = [
    (
        "    zhr_source: Literal[\"store\", \"onfly\"] = \"store\"\n",
        "    zhr_source: Literal[\"store\", \"onfly\"] = \"store\"\n"
        "    # Producer backend for the §11.5 prefetch stream: \"thread\"\n"
        "    # (in-process ThreadPoolExecutor, default) or \"process\" (forked CPU\n"
        "    # worker pool — forks before any HCU context exists; verified in the\n"
        "    # sakrua10 process-pool canary #6: data_wait ~4%, 12/12 bit-exact).\n"
        "    producer: Literal[\"thread\", \"process\"] = \"thread\"\n",
    ),
]


def main() -> int:
    # pre-state assertions (md5 registry)
    for name, want in EXPECT.items():
        p = LF if name == "latent_flow.py" else SCHEMA
        got = md5(p)
        assert got == want, f"{name}: pre md5 {got} != expected {want}"
        print(f"[pre] {name} md5={got} OK")

    lf_txt = LF.read_text()
    if MARKER in lf_txt:
        print(f"[skip] latent_flow.py already patched ({MARKER})")
    else:
        apply(LF, LF_EDITS)

    sc_txt = SCHEMA.read_text()
    if "producer: Literal[\"thread\", \"process\"]" in sc_txt:
        print("[skip] schema.py already patched (producer field present)")
    else:
        apply(SCHEMA, SCHEMA_EDITS)

    # syntax gate under the runtime interpreter (DTK python 3.11)
    for p in (LF, SCHEMA):
        subprocess.run(
            [DTK_PY, "-m", "py_compile", str(p)],
            check=True,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/root"},
        )
        print(f"[compile] {p.name} OK")

    for p in (LF, SCHEMA):
        print(f"[post] {p.name} md5={md5(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
