#!/usr/bin/env python3
"""P1-WORKER-RECOVERY: make the process-pool producer crash-resilient.

Observed failure (p1formal 2-rank smoke, sakrua10 HCU):
  A forked pool worker segfaulted inside a CPU op on a task
  (SIGSEGV in the degradation path, silent without PYTHONFAULTHANDLER).
  The lost in-flight task's AsyncResult never completes; the consumer
  blocks on it forever -> the "2-rank wedge" (no step lines for 10+ min).
  1-rank runs only survived by luck (no worker crash in 300 steps).

Fix (minimal, process-mode only, thread backend untouched):
  - track in-flight tasks (future -> (slot, step)) at submit time;
  - poll liveness of pool workers while waiting for the front batch;
  - on worker death, resubmit every incomplete in-flight task (deterministic
    recompute from the exposure seed; in-place future replacement keeps batch
    order and exactly-once consumption), the Pool's own _handle_workers
    thread has already restarted the dead worker;
  - abort loudly (RuntimeError) after 4 cumulative worker deaths
    (crash-loop guard) with a hint to relaunch under PYTHONFAULTHANDLER=1.

Applied on top of P1-PRODUCER-PORT-V2 (latent_flow.py md5 7f113225).
Idempotent via the P1-WORKER-RECOVERY marker. Backup is written by the
operator (see runbook) before execution; this script verifies pre/post md5.
"""
import hashlib
import py_compile
import subprocess
import sys
import time

TARGET = "/root/anime-sr-p1formal/src/anime_sr/train/latent_flow.py"
PRE_MD5 = "7f113225"          # after P1-PRODUCER-PORT-V2
MARKER = "P1-WORKER-RECOVERY"
PYTHON = "/usr/local/bin/python3.11"
ENV_SETUP = "source /opt/dtk-26.04/env.sh"


def md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main() -> None:
    pre = md5(TARGET)
    if not pre.startswith(PRE_MD5):
        sys.exit(f"abort: pre md5 {pre} != {PRE_MD5}* (expected post-PORT-V2 state)")
    with open(TARGET, encoding="utf-8") as fh:
        src = fh.read()
    if MARKER in src:
        print(f"{MARKER} marker present; patch already applied (idempotent no-op)")
        print(f"md5 after = {md5(TARGET)}")
        return

    edits: list[tuple[str, str, str]] = []

    # E1: recovery helper + constants, after _fut_result
    e1_old = '''def _fut_result(f: Future | _MPAsyncResult, timeout: float | None = None) -> Any:
    """Blocking collect across both producer backends."""
    if isinstance(f, _MPAsyncResult):
        return f.get(timeout)
    return cast("Future", f).result(timeout)
'''
    e1_new = e1_old + '''
# P1-WORKER-RECOVERY ---------------------------------------------------------
# A forked producer worker can die (observed on sakrua10 HCU: SIGSEGV inside
# a CPU op on a task, silent without PYTHONFAULTHANDLER). The lost in-flight
# task's AsyncResult never completes and the consumer would block forever.
# CPython's Pool auto-restarts dead workers (_handle_workers repopulates), so
# a lost task is recovered by resubmitting the same (slot, step) payload:
# _pp_fetch is deterministic (seeded from the exposure index), the restarted
# worker recomputes the identical tensor, and in-place future replacement
# keeps batch order with exactly-once consumption (the dead future is
# dropped; only the resubmitted future is ever collected).
_PP_RECOVER_POLL_S = 0.5  # liveness poll interval while waiting for a batch
_PP_MAX_WORKER_CRASHES = 4  # then abort loudly instead of crash-looping


def _pp_recover_lost_tasks(
    ppool: "_MPPool",
    fs: list[Future | _MPAsyncResult],
    ready: "deque[list[Future | _MPAsyncResult]]",
    inflight: dict[Any, tuple[int, int]],
    crash_state: list[int],
) -> None:
    """Resubmit in-flight tasks after a worker death (in-place replacement).

    No-op while every worker is alive. ``crash_state[0]`` accumulates worker
    deaths seen by this rank; at ``_PP_MAX_WORKER_CRASHES`` the pool is
    crash-looping and the run is aborted (fail loud, not hang)."""
    dead = [w for w in ppool._pool if w is not None and not w.is_alive()]
    if not dead:
        return
    crash_state[0] += len(dead)
    codes = ", ".join(str(w.exitcode) for w in dead)
    print(
        f"[latent] producer: {len(dead)} pool worker(s) died "
        f"(exit codes: {codes}); resubmitting in-flight tasks to the "
        f"restarted workers (total crashes: {crash_state[0]}/"
        f"{_PP_MAX_WORKER_CRASHES})",
        flush=True,
    )
    if crash_state[0] >= _PP_MAX_WORKER_CRASHES:
        raise RuntimeError(
            f"producer worker crash-loop: {crash_state[0]} worker deaths "
            f"(last exit codes: {codes}); in-flight tasks are not recoverable. "
            "Relaunch with PYTHONFAULTHANDLER=1 to capture the crashing "
            "worker's stack (last observed crash: SIGSEGV in a CPU op inside "
            "_pp_fetch's degradation path)"
        )
    replaced: dict[Future | _MPAsyncResult, _MPAsyncResult] = {}
    for f, task in list(inflight.items()):
        if _fut_done(f):
            continue
        nf = ppool.apply_async(_pp_fetch, (task,))
        replaced[f] = nf
        inflight[nf] = task
    for f in replaced:
        inflight.pop(f, None)
    for q in ready:
        for i in range(len(q)):
            if q[i] in replaced:
                q[i] = replaced[q[i]]
    fs[:] = [replaced.get(f, f) for f in fs]
'''
    edits.append(("E1", e1_old, e1_new))

    # E2: track in-flight tasks at submit time (proc backend only)
    e2_old = '''        if _is_proc:
            ppool = cast("_MPPool", pool)
            return [ppool.apply_async(_pp_fetch, ((slot, st),)) for slot, st in pairs]'''
    e2_new = '''        if _is_proc:
            ppool = cast("_MPPool", pool)
            futs = [ppool.apply_async(_pp_fetch, ((slot, st),)) for slot, st in pairs]
            for f, task in zip(futs, pairs):
                inflight[f] = task  # P1-WORKER-RECOVERY
            return futs'''
    edits.append(("E2", e2_old, e2_new))

    # E3: state for recovery, before the prefill loop
    e3_old = '''    ready: deque[list[Future | _MPAsyncResult]] = deque()
    for k in range(depth):'''
    e3_new = '''    ready: deque[list[Future | _MPAsyncResult]] = deque()
    inflight: dict[Any, tuple[int, int]] = {}  # P1-WORKER-RECOVERY: future -> (slot, step)
    crash_state = [0]  # P1-WORKER-RECOVERY: worker deaths seen by this rank
    for k in range(depth):'''
    edits.append(("E3", e3_old, e3_new))

    # E4: collect with liveness polling + resubmission (proc backend)
    e4_old = '''        tf = time.perf_counter()
        if _is_proc:
            # process pool hands back the (hr, lq, z_hr, meta, stages) tuple;
            # rewrap into _Prepared (hr is None in store mode)
            prepared = [
                _Prepared(hr=r[0], lq=r[1], z_hr=r[2], meta=r[3], stages=r[4])
                for r in (_fut_result(f) for f in futs)
            ]
        else:
            prepared = [_fut_result(f) for f in futs]
        n_produced += bs'''
    e4_new = '''        tf = time.perf_counter()
        if _is_proc:
            # P1-WORKER-RECOVERY: poll with worker-liveness checks so a dead
            # worker (lost in-flight task) is detected and its task
            # resubmitted instead of blocking forever.
            while not all(_fut_done(f) for f in futs):
                _pp_recover_lost_tasks(
                    cast("_MPPool", pool), futs, ready, inflight, crash_state
                )
                time.sleep(_PP_RECOVER_POLL_S)
            # process pool hands back the (hr, lq, z_hr, meta, stages) tuple;
            # rewrap into _Prepared (hr is None in store mode)
            prepared = [
                _Prepared(hr=r[0], lq=r[1], z_hr=r[2], meta=r[3], stages=r[4])
                for r in (_fut_result(f) for f in futs)
            ]
            for f in futs:
                inflight.pop(f, None)
        else:
            prepared = [_fut_result(f) for f in futs]
        n_produced += bs'''
    edits.append(("E4", e4_old, e4_new))

    for name, old, new in edits:
        cnt = src.count(old)
        if cnt != 1:
            sys.exit(f"abort: {name} anchor found {cnt} times (need exactly 1)")
        src = src.replace(old, new)

    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(src)
    post = md5(TARGET)
    # DTK python py_compile gate
    r = subprocess.run(
        f"{ENV_SETUP} && {PYTHON} -m py_compile {TARGET}",
        shell=True, capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"abort: py_compile failed:\n{r.stderr}")
    print(f"{MARKER} applied (4 edits: E1 helper/E2 submit-track/E3 state/E4 poll-collect)")
    print(f"md5 before = {pre}")
    print(f"md5 after  = {post}")
    print(f"py_compile OK ({time.strftime('%H:%M:%S')})")


if __name__ == "__main__":
    main()
