#!/usr/bin/env python3
"""P1-WEDGE-FIX r3: fork the producer pool BEFORE dist.init_process_group.

Root-cause chain established empirically (2-rank smoke, sakrua10):
  * 2-rank + OMP_NUM_THREADS=4  -> forked producer workers SIGSEGV in their
    first heavy CPU op (torch.randn in degradation._noise_like); crash loop.
  * 2-rank + OMP_NUM_THREADS=1  -> zero crashes, 300 clean steps (the OMP
    thread pool is a required factor).
  * The fork happens AFTER the CLI's dist.init_process_group("nccl"), so
    every worker inherits NCCL/HCU runtime state; a worker whose CPU ops
    re-enter the inherited (fork-corrupted) OMP pool segfaults.

Fix: call prepare_producer_prefork() from the CLI before NCCL init: it
builds the CPU-only producer ctx (dataset/store/order) and forks the pool
while the parent's address space is still pre-accelerator.
run_latent_flow reuses the pool via the module global _PRE_FORK_POOL.
The r2 crash-recovery (detection + resubmit + crash-loop guard) stays as a
defense layer.

Targets:
  latent_flow.py  (pre md5 98a3e2ba*): E1 new function + global, E2 pool reuse
  cli/train_latent_flow.py (pre md5 c79abfe4*): E3 pre-fork call + import
"""
import hashlib
import subprocess
import sys
import time

TARGET = "/root/anime-sr-p1formal/src/anime_sr/train/latent_flow.py"
CLI = "/root/anime-sr-p1formal/src/anime_sr/cli/train_latent_flow.py"
PRE_MD5 = "98a3e2ba"
PRE_CLI_MD5 = "c79abfe4"
PYTHON = "/usr/local/bin/python3.11"
ENV_SETUP = "source /opt/dtk-26.04/env.sh"


def md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main() -> None:
    pre = md5(TARGET)
    pre_cli = md5(CLI)
    if not pre.startswith(PRE_MD5):
        sys.exit(f"abort: latent_flow md5 {pre} != {PRE_MD5}* (expected r2 state)")
    if not pre_cli.startswith(PRE_CLI_MD5):
        sys.exit(f"abort: cli md5 {pre_cli} != {PRE_CLI_MD5}*")
    if "prepare_producer_prefork" in open(TARGET, encoding="utf-8").read():
        print("P1-WEDGE-FIX marker present; already applied (idempotent no-op)")
        print(f"md5 after = {md5(TARGET)} / {md5(CLI)}")
        return
    src = open(TARGET, encoding="utf-8").read()
    cli = open(CLI, encoding="utf-8").read()

    # E1: module global + prepare_producer_prefork, inserted before run_latent_flow
    e1_old = '''def run_latent_flow(
    cfg: Config,
    *,
    index_dir: str | Path,'''
    e1_new = '''_PRE_FORK_POOL: _MPPool | None = None


def prepare_producer_prefork(
    cfg: Config,
    *,
    index_dir: str | Path,
    webp_dir: str | Path,
    latent_dir: str | Path | None,
    bucket_hr: int,
    rank: int,
) -> None:
    """P1-WEDGE-FIX: build the producer ctx and fork the worker pool BEFORE
    ``dist.init_process_group`` (call from the CLI).

    A forked worker that inherits NCCL/HCU runtime state SIGSEGVs on its
    first heavy CPU op (observed: torch.randn in the degradation path,
    2-rank smoke; the inherited OMP pool is a required factor — OMP=1 runs
    are crash-free).  The dataset/store/order construction is CPU-only, so
    it can run before any accelerator initialization; the forked pool then
    carries a clean (pre-accelerator) address space.  ``run_latent_flow``
    reuses the pool via ``_PRE_FORK_POOL``; its ``_PRODUCER_CTX`` is the
    very dict the workers inherited at fork time."""
    global _PRE_FORK_POOL, _PRODUCER_CTX
    lf = cfg.latent_flow
    onfly = lf.zhr_source == "onfly"
    store: LatentStore | None = None
    sids: list[str] = []
    if not onfly:
        store = LatentStore(latent_dir, bucket_hr)
        doc = read_index(latent_dir)
        sids = sorted(doc["samples"].keys())
    clean_cache: CleanScoreCache | None = None
    if cfg.filter.clean_score_stage == "lazy" and cfg.filter.clean_score_cache:
        clean_cache = CleanScoreCache(index_dir)
    ds = SRDataset(
        index_dir, webp_dir, cfg, bucket_hr=bucket_hr, split="train",
        clean_score_cache=clean_cache,
    )
    if onfly:
        order = list(range(len(ds.samples)))
    else:
        sid_to_idx = {m.sample_id: i for i, m in enumerate(ds.samples)}
        missing = [s for s in sids if s not in sid_to_idx]
        if missing:
            raise RuntimeError(
                f"{len(missing)} latent sample ids missing from the train index "
                f"(e.g. {missing[:3]}); rebuild the latent store from this index"
            )
        order = [sid_to_idx[s] for s in sids]
    n = len(order)
    _PRODUCER_CTX = {
        "ds": ds,
        "order": order,
        "n": n,
        "cfg": cfg,
        "store": store,
        "global_seed": ds.global_seed,
        "bucket_hr": bucket_hr,
        "exposure_per_cycle": _EXPOSURE_PER_CYCLE,
    }
    n_pp = max(1, (lf.prefetch_depth or 1) * lf.batch_size)
    _PRE_FORK_POOL = _make_process_pool(n_pp)
    if rank == 0:
        print(
            f"[latent] producer=process: {n_pp} forked workers BEFORE "
            f"NCCL/HCU init (intra-op re-tuned from OMP_NUM_THREADS)",
            flush=True,
        )


def run_latent_flow(
    cfg: Config,
    *,
    index_dir: str | Path,'''
    if src.count(e1_old) != 1:
        sys.exit("abort: E1 anchor not unique")
    src = src.replace(e1_old, e1_new)

    # E2: reuse the pre-fork pool when present, else the classic spot
    e2_old = '''    pool: ThreadPoolExecutor | _MPPool | None = None
    if lf.producer == "process":
        global _PRODUCER_CTX
        _PRODUCER_CTX = {
            "ds": ds,
            "order": order,
            "n": n,
            "cfg": cfg,
            "store": store,
            "global_seed": ds.global_seed,
            "bucket_hr": bucket_hr,
            "exposure_per_cycle": _EXPOSURE_PER_CYCLE,
        }
        n_pp = max(1, (lf.prefetch_depth or 1) * lf.batch_size)
        pool = _make_process_pool(n_pp)
        if rank == 0:
            print(
                f"[latent] producer=process: {n_pp} forked workers "
                f"(intra-op re-tuned from OMP_NUM_THREADS)",
                flush=True,
            )'''
    e2_new = '''    pool: ThreadPoolExecutor | _MPPool | None = None
    if lf.producer == "process":
        global _PRODUCER_CTX, _PRE_FORK_POOL
        if _PRE_FORK_POOL is not None:
            # P1-WEDGE-FIX: the pool (and its inherited ctx) was created in
            # prepare_producer_prefork() BEFORE dist.init_process_group, so
            # the workers carry no NCCL/HCU runtime state.
            pool = _PRE_FORK_POOL
            if rank == 0:
                print(
                    "[latent] producer=process: reusing pre-NCCL pool "
                    "(fork-before-init_process_group; ctx inherited at fork)",
                    flush=True,
                )
        else:
            _PRODUCER_CTX = {
                "ds": ds,
                "order": order,
                "n": n,
                "cfg": cfg,
                "store": store,
                "global_seed": ds.global_seed,
                "bucket_hr": bucket_hr,
                "exposure_per_cycle": _EXPOSURE_PER_CYCLE,
            }
            n_pp = max(1, (lf.prefetch_depth or 1) * lf.batch_size)
            pool = _make_process_pool(n_pp)
            if rank == 0:
                print(
                    f"[latent] producer=process: {n_pp} forked workers "
                    f"(intra-op re-tuned from OMP_NUM_THREADS)",
                    flush=True,
                )'''
    if src.count(e2_old) != 1:
        sys.exit("abort: E2 anchor not unique")
    src = src.replace(e2_old, e2_new)

    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(src)

    # E3: CLI — pre-fork call before init_process_group + import
    e3a_old = "from anime_sr.train.latent_flow import run_latent_flow"
    e3a_new = ("from anime_sr.train.latent_flow import "
               "prepare_producer_prefork, run_latent_flow")
    if cli.count(e3a_old) != 1:
        sys.exit("abort: E3a anchor not unique")
    cli = cli.replace(e3a_old, e3a_new)

    e3b_old = '''    if world_size > 1 and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")'''
    e3b_new = '''    if world_size > 1 and dist.is_available() and not dist.is_initialized():
        # P1-WEDGE-FIX: fork the CPU producer pool BEFORE NCCL/HCU init so
        # workers never inherit accelerator runtime state (inheriting it
        # makes forked workers SIGSEGV on their first CPU op; observed in
        # the 2-rank smoke: SIGSEGV in torch.randn inside _noise_like).
        if cfg.latent_flow.producer == "process":
            prepare_producer_prefork(
                cfg,
                index_dir=args.index_dir,
                webp_dir=args.webp_dir,
                latent_dir=args.latent_dir,
                bucket_hr=args.bucket_hr,
                rank=rank,
            )
        dist.init_process_group(backend="nccl")'''
    if cli.count(e3b_old) != 1:
        sys.exit("abort: E3b anchor not unique")
    cli = cli.replace(e3b_old, e3b_new)

    with open(CLI, "w", encoding="utf-8") as fh:
        fh.write(cli)

    for path in (TARGET, CLI):
        r = subprocess.run(
            f"{ENV_SETUP} && {PYTHON} -m py_compile {path}",
            shell=True, capture_output=True, text=True,
        )
        if r.returncode != 0:
            sys.exit(f"abort: py_compile failed for {path}:\n{r.stderr}")
    print("P1-WEDGE-FIX r3 applied (E1 pre-fork fn/E2 pool reuse/E3 CLI call)")
    print(f"latent_flow md5 before = {pre}")
    print(f"latent_flow md5 after  = {md5(TARGET)}")
    print(f"cli md5 before = {pre_cli}")
    print(f"cli md5 after  = {md5(CLI)}")
    print(f"py_compile OK ({time.strftime('%H:%M:%S')})")


if __name__ == "__main__":
    main()
