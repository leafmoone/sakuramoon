# Camera MBS Data-Readiness & Queue-Canonicalization Fix (Phase C)

- Date: 2026-09-08 (SCNet, come7)
- Branch: `camera-v2-mbs-data-readiness-queue-fix-review`
- Base: `b78ba023f781535a9d8fd98b0fc543925e5f04a6` (last reviewed SHA, Phase A HEAD)
- Training this round: **NONE** (fix + tests + static gates only; rerun U118101-118200 awaits later explicit GO after external review)

## 1. Why this round exists

The corrected MBS 100U treatment rerun (U118100 -> U118200, window U118101-118200,
host come7, attempt 1) failed with **0 successful updates** (0U FAIL #2, 2026-09-08
16:45-16:49 local). The supervisor correctly FAILed the run and stopped the stack
(no auto-restart, per spec). Post-mortem established two independent
infrastructure defects, neither MBS-related:

1. **Readiness-contract bug**: `training_stack.sh` treats Unix-socket existence
   as "data service ready", but the server bound the socket *before* its
   16-shard warmup barrier completed. On a cold cache the barrier took ~4
   minutes; the trainer's health requests sat in an empty backlog and expired
   against the 30s client timeout, killing both ranks in production preflight.
2. **Queue cross-process non-determinism**: `_QueueStore.new` started its seeded
   shuffle from `list(frozenset)`. frozenset iteration order is per-process
   (string hash randomization), so "fresh cycle-0 + same manifest" was only
   deterministic *within one process*. Across processes (control arm, Phase B,
   rerun) the cycle-0 sample sequences differed.

User decision (2026-09-08, verbatim block in §8): **Option B (code fix) = GO**,
Option A (prewarm retry) = NO-GO for now (a prewarm success on the now-warm
cache would "accidentally succeed" and hide the defects again), Option C = NO.

## 2. 0U FAIL #2 timeline and root cause (evidence)

Log root: `/sakuramoon-runtime/logs/mbs-corrected-treatment-rerun-118100-118200/`

| Time (local) | Event |
|---|---|
| 16:45:24 | Supervisor 70269 START, WAITING_FOR_START (self-test-stop-env exit 0, `training_stack.sh validate` PASS, 8/8 pre-launch assertions PASS) |
| 16:45:4x | `training_stack.sh start`: data 70359 / publisher 70488 / train 70617 (2/2 ranks); data service resumed 13 partial downloads (11 GB) from Phase B leftovers and started 8 fresh 2GB window shards (~30 MiB/s) |
| 16:45:4x | Data service bound the socket and logged "监听 … PID=70359" -> stack printed `data service ready` on socket-exists alone; trainer ranks launched |
| ~16:47-16:49:2x | Both ranks (after ~3.5 min model load) sent health requests; the server's accept loop was still parked in `wait_until_ready()` (waiting for 16 ready shards); connections queued in the kernel backlog; 6 retries disconnected while the server was sending responses (broken pipe), 2 final health recvs timed out at 30s |
| 16:49:26 | Both ranks: `TimeoutError` -> `DataServiceUnavailable: data service is unavailable` -> `ProductionPreflightError: production single-GPU lifecycle failed` |
| 16:49:30 | Supervisor detected fatal `Traceback` pattern: RESULT WRITTEN status=FAIL (last_successful_update=118100, all cumulative counters 0, early_exposure_gate=not_reached) -> invoked stack stop |
| 16:49:38 | Stack stop rc=0 (train/data/publisher all stopped, no data socket, no checkpoint written); 0 live processes |

Code chain (base b78ba02):

- `scripts/training_stack.sh:360-363` — `wait_for_data_service()` readiness =
  `[[ -S "${DATA_SOCKET}" ]]` + process alive (no protocol probe), gated by the
  generic `START_TIMEOUT_SECONDS=180`.
- `src/sakuramoon/data/service.py` (base) — `serve()` order: `service.start()` ->
  socket-occupancy precheck -> `mkdir`/`bind`/`listen` (**socket file exists**)
  -> `wait_until_ready()` (waits for `min(worker_count, pending)` = **16 ready
  shards**) -> accept loop.
- Cold window: 16 x 2 GB shards at 8-way concurrency ~30 MiB/s => barrier
  ~2.5-4 min. Client `DataServiceClient.__init__` health uses
  `request_timeout_seconds=30.0` (`config/train_g1.toml`) => guaranteed timeout.
- Phase B (13:22) survived the *same* race: its cycle-0 window had more cached
  shards (v1-era migration cache) and favorable timing. The race is
  distribution-dependent luck, not a one-off.

State damage: **none** (training state, source, and control all verified intact;
see §7).

## 3. Defect 1: queue cross-process non-determinism

`src/sakuramoon/data/service.py`, `_QueueStore` (base b78ba02):

```python
self.paths = frozenset(item.path for item in manifest.shards if ...)
...
def new(self, cycle: int) -> _QueueState:
    paths = list(self.paths)                      # <- per-process iteration order
    seed = (0x9E3779B97F4A7C15 * (cycle + 1)) & 0xFFFFFFFFFFFFFFFF
    random.Random(seed).shuffle(paths)
```

The shuffle is a pure function of `cycle` *given a fixed input arrangement*, but
`list(frozenset[str])` is randomized per interpreter process (string hash
randomization; `PYTHONHASHSEED` is not pinned in the training env). The source
comment claimed "Deterministic per-cycle order" — true only within one process.

**Observed evidence** (same manifest, 2963 shards, both "fresh cycle-0"):

| | first 8 queue entries |
|---|---|
| Phase B (come7, 13:22) | 2_2026.1/337, 1_2025.9/291, a2_filter/402, 2_2026.1/479, 2026.7/345, 2026.7/336, 2_2026.1/480, a2_image/89 |
| Rerun attempt (come7, 16:45) | 1_2024/92, a2_filter/259, a2_filter/189, 1_2024/662, 2_2026.1/292, a2_filter/87, a2_image/142, 2_2026.1/392 |

Consequence: the U118200 camera-only control (v1 canary queue on come3), Phase B,
and the treatment rerun consumed **different sample sequences**. The frozen
Phase-B "deterministic matched-data reset" was only same-process deterministic.

## 4. Fixes (exactly three, no other behavior changes)

### 4.1 Queue canonicalization — `src/sakuramoon/data/service.py`

```python
    def new(self, cycle: int) -> _QueueState:
        # Canonical input order: self.paths is a frozenset whose iteration
        # order is per-process (string hash randomization). The shuffle must
        # start from sorted(paths) — the only input arrangement that is
        # bit-identical across processes, restarts, and experiment arms —
        # otherwise two fresh cycle-0 runs consume different sample
        # sequences despite identical manifest and seed.
        paths = sorted(self.paths)
        # Deterministic per-cycle order: ... (unchanged rationale)
        seed = (0x9E3779B97F4A7C15 * (cycle + 1)) & 0xFFFFFFFFFFFFFFFF
        random.Random(seed).shuffle(paths)
```

- `sorted(paths)` makes the shuffle input bit-identical across processes,
  restarts, and experiment arms. No `PYTHONHASHSEED` dependency in production
  (it remains a valid diagnostic tool only).
- Cycle-shuffle seed and all downstream lookahead/rollover logic unchanged.

### 4.2 Readiness contract — `src/sakuramoon/data/service.py` `serve()`

The socket file is now created **only after** the warmup barrier passes:

```
service.start()
  -> socket-occupancy precheck (unchanged, still before bind)
  -> wait_until_ready()          <- moved before bind
  -> mkdir parent / bind / chmod / listen
  -> ready_callback
  -> accept loop
```

Socket existence now encodes "protocol servable", which makes the stack's
existing `[[ -S "${DATA_SOCKET}" ]]` check a *correct* readiness contract again.
Stop semantics unchanged: `wait_until_ready()` returns False on stop (the
`_ServiceStopping` path) and `serve()` exits without ever creating the socket —
teardown mid-warmup leaves no stale socket behind.

### 4.3 Dedicated cold-data startup timeout — `scripts/training_stack.sh`

```bash
DATA_READY_TIMEOUT_SECONDS="${DATA_READY_TIMEOUT_SECONDS:-900}"
```

- Used **only** by `wait_for_data_service()` (loop bound + failure message).
- `START_TIMEOUT_SECONDS=180` for every other component is untouched; no
  unrelated startup gate was weakened.
- `validate_integer DATA_READY_TIMEOUT_SECONDS` added to the validation block.

## 5. Tests (5 new, 2 files)

`tests/unit/data/test_queue_order_canonicalization.py`:
1. `test_cycle_order_bit_identical_across_hash_seeds` — same manifest (64
   training shards + validation) and cycle 0, run in **child interpreters**
   with `PYTHONHASHSEED=1`, `2`, `12345`; asserts the full row order is
   bit-identical across seeds, row count/duplicates correct, and the shuffle is
   not a no-op (order != sorted order).
2. `test_same_process_same_cycle_is_stable_and_cycle_dependent` — same cycle
   stable within a process; different cycles produce different orders.

`tests/unit/data/test_service_readiness_contract.py` (all three construct a
blocking `wait_until_ready()` via a `threading.Event` — **no shard, cache, or
download is involved, so a warm production cache cannot make them pass**):
3. `test_socket_hidden_until_warmup_barrier_released` — while the barrier is
   held the socket stays absent and the endpoint is unconnectable (1.5s stable
   poll); after release the socket appears as a 0600 unix socket and a **real
   protocol health round-trip** via `DataServiceClient` succeeds; clean stop
   unlinks the socket.
4. `test_stop_during_warmup_returns_without_socket` — the 0U FAIL #2 teardown
   shape: stop during warmup exits `serve()` and leaves **no** socket file.
5. `test_wait_until_ready_false_means_no_socket_created` — `wait_until_ready()
   == False` skips socket creation entirely.

**Regression capture proof** (new tests run against base b78ba02 with the same
test files): 3 FAILED on base (`test_cycle_order_bit_identical_across_hash_seeds`,
`test_socket_hidden_until_warmup_barrier_released`,
`test_stop_during_warmup_returns_without_socket`), 2 passed on base (the two
invariants that already held); on the fixed tree 5/5 pass.

## 6. Corrected scientific framing (MBS experiment)

The old U118200 camera-only control checkpoint **remains a valid control**
(it was trained and archived correctly). It can no longer be called an
"exact matched-data control". Correct label:

> **same-distribution, independent-sequence camera-only continuation control**

The treatment (MBS 100U) and the control did not share a per-sample training
sequence. The experiment tree is therefore:

```
                  PRE U118100
                     |
       +-------------+-------------+
       |                           |
 camera-only 100U (100U)       MBS 100U (100U)
 historical queue (v1 canary)  canonical queue (post-fix)
       |                           |
       v                           v
     CONTROL                   TREATMENT
 (same-distribution, independent-sequence continuation control)
```

**Conditional control plan (user decision):** do **not** rerun a control now
(算力最缺 = compute is the scarcest resource). First fix the infrastructure and
produce the real MBS 100U run. Only if the later three-way audit is *borderline*
(minimal improvement, CI grazing, weak CONTROL/TREATMENT separation) will an
extra 100U be spent on a **canonical camera-only control** that shares the
post-fix deterministic cycle-0 sequence with the treatment. If the three-way
result is strong and directional (CONTROL bottom continues to fall; TREATMENT
bottom clearly recovers; TOP-BOTTOM gap narrows substantially; clean bootstrap
CI), the sequence confound cannot by itself explain such a directional change,
and the evidence stands as valuable — with the corrected (non-"exact
matched-data causal") wording.

## 7. State integrity after 0U FAIL #2 (all verified)

- Source manifest sha `821a8c12546a094865972fef0c23152e9754e01f82eed8353f7248b92076b892` — EXACT
- Control manifest sha `f014827fb441aca8780a5bb710eeb2631a40aac112ed3aa6b07b6e3b65d657f7` — EXACT
- v2 treatment checkpoint root: empty (0 entries); no checkpoint, no metrics.jsonl, no trainer state written
- 0U-fail #1 archive intact at `/sakuramoon-runtime/failed-attempts/corrected-mbs-0u-e6d7b103/` (metrics.jsonl 0B sha e3b0c442… intact)
- Queue state after the failed attempt: sha `797e8164f7e74724332cd37f790063d6cf6f3e4e46b36edb7ab659cf28c7b2c3` (advanced ~4 min of data-service activity); the frozen §6 reset procedure (backup + delete -> fresh cycle-0) applies again before any rerun
- Rerun evidence worktree clean at b78ba02; no commit/push from it (failure report commits are out of scope for that authorization)

## 8. User decision block (verbatim, 2026-09-08)

```
0U FAIL #2
TRAINING STATE DAMAGE             = NONE
SOURCE DAMAGE                     = NONE
CONTROL DAMAGE                    = NONE

STARTUP RACE ROOT CAUSE           = ACCEPTED
QUEUE CROSS-PROCESS NONDETERMINISM= CONFIRMED

OPTION A PREWARM RETRY            = NO-GO FOR NOW
OPTION B CODE FIX                 = GO
OPTION C HOLD                      = NO

FIX SCOPE:
1. queue canonicalization   sorted(paths) -> seeded shuffle
2. readiness contract       socket must not advertise READY before warmup barrier completes
3. dedicated cold-data startup timeout (do not weaken unrelated startup gates)
4. regression tests: different PYTHONHASHSEED -> identical queue order
5. cold-start readiness test: before warmup -> socket/READY unavailable; after -> protocol health succeeds
6. NO camera/MBS/model/optimizer changes
```

Additional user mandates honored: fix validation must construct/simulate a
blocking `wait_until_ready()` (not rely on the now-warm cache); new fix-review
branch from the last reviewed `b78ba02…` (not the rerun evidence worktree).

## 9. Verification

| Gate | Result |
|---|---|
| Focused: 2 new test files (fix tree) | 5/5 pass |
| Regression capture: same tests on base b78ba02 | 3 fail / 2 pass (both defects captured) |
| Broad unit suite (tests/unit, both deepghs env ignores) fix vs base | base: 4 failed / 1008 passed (272.9s); fix: 4 failed / 1013 passed (286.2s); failure sets IDENTICAL (4 pre-existing `test_cmuon_fp32_forensic` failures); +5 passed = the 5 new tests; 0 new failures |
| `ruff check` changed scope | clean |
| `python -m py_compile` changed py files | OK |
| `bash -n scripts/training_stack.sh` | OK |
| `git diff --check` | clean |
| Config/camera/MBS/model/optimizer diff | NONE (diff scope = 2 modified + 2 new test files only) |
| `request_timeout_seconds` | unchanged (30.0) |

## 10. Next steps

1. External review of this branch (2 commits: fix + docs).
2. On GO: rerun the corrected MBS 100U treatment (U118100 -> U118200) with the
   reviewed SHA, fresh worktree, fresh operational paths, frozen §6 queue reset
   (new backup of the current 797e8164… state), tracked supervisor mandatory,
   HARD STOP at U118200, no auto-restart, no U118201.
3. Rerun note: the post-fix canonical cycle-0 order is now computable offline
   (sorted input + seeded shuffle), so the cold-window pre-warm question is no
   longer an accident — the exact first-window shard set can be listed before
   launch if desired (not required: the readiness contract + 900s data-ready
   gate now tolerate a cold barrier by design).
