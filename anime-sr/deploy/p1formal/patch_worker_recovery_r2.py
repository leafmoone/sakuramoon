#!/usr/bin/env python3
"""P1-WORKER-RECOVERY r2: fix dead-worker detection (membership diff).

r1's liveness check scanned ``ppool._pool`` for non-alive members, but the
CPython Pool's maintenance thread reaps dead workers OUT of that very list
(_join_exited_workers does ``del pool[i]``; _repopulate_pool_static appends
replacements), so a dead worker is never observable there -> detection
never fired (verified: 2-rank smoke r2 segfaulted in _noise_like again and
the recovery line was never printed).

r2 detects worker deaths by membership diff: the rank snapshots the set of
worker Process objects in ``pool._pool`` on every poll; any worker that
disappeared was reaped (dead), its replacements are new members. On a
disappearance the incomplete in-flight tasks are resubmitted (deterministic
recompute, in-place future replacement) and the death is counted toward the
crash-loop guard.

Applied on top of P1-WORKER-RECOVERY r1 (latent_flow.py md5 90079ad5).
"""
import hashlib
import subprocess
import sys
import time

TARGET = "/root/anime-sr-p1formal/src/anime_sr/train/latent_flow.py"
PRE_MD5 = "90079ad5"
MARKER = "P1-WORKER-RECOVERY"
PYTHON = "/usr/local/bin/python3.11"
ENV_SETUP = "source /opt/dtk-26.04/env.sh"


def md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main() -> None:
    pre = md5(TARGET)
    if not pre.startswith(PRE_MD5):
        sys.exit(f"abort: pre md5 {pre} != {PRE_MD5}* (expected r1 state)")
    with open(TARGET, encoding="utf-8") as fh:
        src = fh.read()
    if "seen_workers" in src:
        print(f"{MARKER} r2 marker present; already applied (idempotent no-op)")
        print(f"md5 after = {md5(TARGET)}")
        return

    edits: list[tuple[str, str, str]] = []

    # A: replace the whole r1 recovery function with the membership-diff version
    a_old = '''def _pp_recover_lost_tasks(
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
    a_new = '''def _pp_recover_lost_tasks(
    ppool: "_MPPool",
    fs: list[Future | _MPAsyncResult],
    ready: "deque[list[Future | _MPAsyncResult]]",
    inflight: dict[Any, tuple[int, int]],
    crash_state: list[int],
    seen_workers: set,
) -> None:
    """Resubmit in-flight tasks after a worker death (in-place replacement).

    The Pool's maintenance thread reaps dead workers OUT of ``ppool._pool``
    (``_join_exited_workers`` does ``del pool[i]``; ``_repopulate_pool_static``
    appends the replacements), so a dead worker is never observable by a
    liveness scan of the list. Deaths are therefore detected by membership
    diff against ``seen_workers`` (the previously observed set of Process
    objects): anything that disappeared was reaped. No-op while the set is
    unchanged. ``crash_state[0]`` accumulates worker deaths seen by this
    rank; at ``_PP_MAX_WORKER_CRASHES`` the pool is crash-looping and the
    run is aborted (fail loud, not hang)."""
    current = set(ppool._pool)
    gone = [w for w in seen_workers if w not in current]
    if not gone:
        seen_workers.update(current)
        return
    seen_workers.update(current)
    crash_state[0] += len(gone)
    codes = ", ".join(str(w.exitcode) for w in gone)
    print(
        f"[latent] producer: {len(gone)} pool worker(s) died "
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
    edits.append(("A", a_old, a_new))

    # B: snapshot the initial worker set (process mode only)
    b_old = '''    crash_state = [0]  # P1-WORKER-RECOVERY: worker deaths seen by this rank'''
    b_new = '''    crash_state = [0]  # P1-WORKER-RECOVERY: worker deaths seen by this rank
    seen_workers: set = set()  # P1-WORKER-RECOVERY: live worker Process objects
    if _is_proc:
        seen_workers.update(cast("_MPPool", pool)._pool)'''
    edits.append(("B", b_old, b_new))

    # C: pass the snapshot through at the poll site
    c_old = '''                _pp_recover_lost_tasks(
                    cast("_MPPool", pool), futs, ready, inflight, crash_state
                )'''
    c_new = '''                _pp_recover_lost_tasks(
                    cast("_MPPool", pool), futs, ready, inflight, crash_state,
                    seen_workers,
                )'''
    edits.append(("C", c_old, c_new))

    for name, old, new in edits:
        cnt = src.count(old)
        if cnt != 1:
            sys.exit(f"abort: {name} anchor found {cnt} times (need exactly 1)")
        src = src.replace(old, new)

    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(src)
    post = md5(TARGET)
    r = subprocess.run(
        f"{ENV_SETUP} && {PYTHON} -m py_compile {TARGET}",
        shell=True, capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"abort: py_compile failed:\n{r.stderr}")
    print(f"{MARKER} r2 applied (A recovery-fn membership diff/B seen_workers/C call)")
    print(f"md5 before = {pre}")
    print(f"md5 after  = {post}")
    print(f"py_compile OK ({time.strftime('%H:%M:%S')})")


if __name__ == "__main__":
    main()
