# D010-D026 DATA package Infra/performance rereview

Reviewer: independent agent `/root/data_d025_d026_rereview`

Review head: `4143a605d70491ab39365b8b2e3fc2017862ed4a`

Overall verdict: **PASS for the committed CPU and previously recorded bounded 1GPU
implementation scope**. The D025 governed-issuance and stable two-worker evidence
fixes in `4143a60`, plus the D010 fixture/readiness fixes in `da4c479`, close all four
rereview findings. Production data, cold-cache performance, formal stage and
multi-GPU gates remain pending and are not inferred from this result.

## Remediation rereview

### Resolved High: D025 ungoverned construction

The historical direct-construction reproduction accepted arbitrary validation IDs and
factory identity while the configured validation manifest was absent. Commit
`4143a60` disables direct dataclass initialization and registers only the exact object
allocated after `from_config()` completes canonical manifest validation.

Issuance checks both exact weak-registry object identity and owner PID. Targeted tests
reject direct construction, copied-slot `object.__new__` forgery and pickle copies. A
fresh child also rejected a deserialized valid parent factory with distinct actual
PIDs. The accepted-stream authority can no longer originate from the previously
demonstrated bypass.

### Resolved Medium: D010 shared-schema fixtures

The historical `6 failed, 1 passed` result came from two unresolved D026 storage
placeholders. Commit `da4c479` adds explicit governed synthetic mount source and free
space values to this fixture only. All seven CLI cases now exercise and pass their
original local/remote/build/drift/no-clobber/redaction assertions; production storage
preflight remains strict.

### Resolved Medium: D025 exact-worker evidence

The PID marker now occupies the first fixed system-prefix token for every sample,
independent of body dropout and persisted `mainset` order. The real AF_UNIX integration
observed exactly two non-parent spawned worker PIDs in the full selector plus four
isolated runs, three of them concurrent. The observation no longer depends on a
non-empty body.

### Resolved Medium: D010 SIGKILL readiness

The historical five-second readiness barrier expired during fresh NFS imports. Commit
`da4c479` changes only this fault test's bounded deadline to 30 seconds, within the
driver's existing 60-second maximum. The actual SIGKILL/partial/restart-from-zero path
passed independently in 21.17 seconds. No production retry, publication or storage
timeout changed.

## Infra conclusions

- The previous unsafe default-`fork` launch is remediated. Command queues, completion
  queue and DataLoader workers share `mp.get_context("spawn")`; the live-parent-thread
  test completed, two worker IDs/PIDs progressed, and no 120-second hang recurred.
- Worker inputs are capacity one each; ready work is the exact divisible configured
  budget; completion capacity equals worker count. State/cache mutation remains in the
  parent/service owner. Worker death and parent close leave active leases for restart.
- D024 owns bounded download concurrency, verified lookahead, lease/ACK admission,
  mainset state and cache reservations. The real AF_UNIX/service/fault selectors pass
  in CPU scope. Existing bounded RTX 5090 evidence was reviewed but not rerun here.
- D026 correctly keeps durable data on the approved NFSv3 mount and moves only the
  AF_UNIX endpoint and singleton `flock` file to non-NFS `/run/sakuramoon`. NFS is not
  rejected for cache/state merely because it is NFS; the governed checks instead bind
  exact source/version/hard-mount identity, probe file/directory fsync plus replace and
  readback, and account explicit cache high watermark, three measured checkpoints and
  reserve. Host-local IPC avoids placing AF_UNIX namespace and host ownership locks on
  a shared network filesystem.

## Per-task verdicts

| Task | Infra/performance verdict | Current boundary |
|---|---|---|
| D010 | PASS (implemented CPU scope) | All manifest CLI cases and the bounded real-SIGKILL restart pass after `da4c479`; live network performance remains pending. |
| D011 | PASS (implemented CPU scope) | Deterministic selection and streaming bundle publication pass; approximately 11M-row memory/time remains pending. |
| D012 | PASS (implemented CPU scope) | Explicit quota, protected LRU and durable state pass; legacy 300-500 GiB assumption is superseded by D026. Production capacity/replay audit remains pending. |
| D013 | PASS (implemented CPU scope) | Fixed counters/histogram remain bounded; production scan/decode throughput remains pending. |
| D014 | PASS (implemented CPU scope) | Work remains bounded by 512 condition tokens with no model/network/cache path. |
| D015 | PASS (implemented CPU scope) | Ready budget and bucket fragments remain bounded; production RSS/pinned/ready-wait evidence remains pending. |
| D016 | PASS (implemented CPU scope) | Strict startup-only schema has no material hot-path cost. |
| D017 | PASS (implemented CPU scope) | Atomic replace publication and cleanup contracts remain bounded. |
| D018 | PASS (implemented CPU scope) | The 10,001-bin retention histogram is constant-memory. |
| D019 | PASS (implemented CPU scope) | Validation remains bounded by existing candidate IDs. |
| D020 | PASS (implemented CPU scope) | Twelve fixed counters remain bounded through collate. |
| D021 | PASS (implemented CPU/recorded 1GPU scope) | Trusted mapping adds bounded per-sample CPU work; no new throughput claim. |
| D022 | PASS (implemented CPU scope) | Active cardinality and cache protection are bounded by exact worker count. |
| D023 | PASS (implemented CPU scope) | Explicit spawn closes the default-fork hang; input/ready/completion paths and shutdown/replay behavior pass. |
| D024 | PASS (implemented CPU/existing bounded 1GPU scope) | Service/mainset/AF_UNIX consumer/fault paths pass; production cold-cache and fully-cached comparison remain pending. |
| D025 | PASS (implemented CPU scope) | Exact governed-object/PID issuance and direct/object-new/pickle/cross-PID rejection pass; dropout-independent exact-two-worker evidence is stable across repeated real spawn runs. |
| D026 | PASS (implemented CPU scope) | Exact NFS identity, capacity, atomic probes and host-local IPC pass; the shared-schema consumer regression is closed by `da4c479`. |

## Independent validation

- Historical failing baseline at `249bfe4`: `383 passed, 8 failed, 5 warnings in
  89.08s`, comprising the four findings above.
- D025 targeted assembly/service/fault file: `9 passed in 27.38s`.
- Exact-two-worker real AF_UNIX/spawn evidence: five total passes; four isolated runs
  completed in 7.21, 8.59, 8.67 and 8.53 seconds.
- D010 seven manifest CLI cases plus real SIGKILL/restart: `8 passed, 17 warnings in
  22.94s`; isolated SIGKILL passed in 21.17 seconds.
- A fresh child process rejected a deserialized parent-issued factory across actual
  PIDs.
- Trace contracts: `40 passed in 76.81s`; direct verifier `237/237`, zero errors.
- Ruff passed; strict Pyright reported `0 errors, 0 warnings`.

All pytest basetemps were isolated below repository `cache/` and removed after the
review. No production network, GPU, long run, formal stage, DDP/NCCL, multi-GPU,
1,000-step, two-hour cold-cache, or full dataset scan was run.

## Pending performance and hardware gates

- A governed immutable production manifest and the complete approximately 11M-ID,
  validation-exclusion, bucket and caption scans.
- Exact governed parser plus real local Qwen/Mage on one RTX 5090.
- Cold/warm-cache overlap and continuous two-hour cold supply at `>=12 samples/s`,
  ready wait `<2%`, no swap/unbounded RSS/quota breach, with fully-cached same-backend
  p50/p95/p99 control.
- Benchmark-driven 1/2/3 worker, ready-depth, download concurrency, lookahead and small
  cache watermark selection; no current number is locked by this review.
- Formal stage, long-run, 1,000-step and every four-GPU/DDP/NCCL gate remain pending or
  blocked and cannot be closed by CPU or one-GPU evidence.

---

## Historical review retained verbatim

# D010-D026 DATA package Infra/performance rereview

Reviewer: independent agent `/root/data_d025_d026_rereview`

Review head: `249bfe4062aa7666bc0d036da3b0ec08295f5510`

Overall verdict: **CHANGES REQUIRED**. D025's public factory constructor bypasses the
governed validation-manifest path and can issue accepted stream authority from
caller-supplied validation IDs and identity. The explicit-spawn worker runtime and
D026 NFS/host-local IPC implementation pass their bounded CPU scope, but the committed
regression/fault suite is also not stable enough for an independent package PASS.

## Findings

### High: D025 permits ungoverned factory construction and accepted-stream issuance

The public dataclass at `src/sakuramoon/data/production.py:358` accepts precomputed
`validation_ids` and `factory_identity`. Its `__post_init__` verifies only local shape,
cardinality and serialization. Only `from_config()` loads the configured validation
JSONL and checks SHA-256/canonical identity, but direct construction is public and is
not distinguished from governed construction when `batches()` issues an accepted
stream.

A read-only reproduction used a valid config whose synthetic validation manifest was
absent, supplied arbitrary IDs `9000001..9002000` and `"f" * 64`, and obtained a stream
that passed `require_accepted_production_batch_stream()`. The observed result was
`configured_manifest_exists=False`, `accepted_arbitrary_ids=True`, and
`stream_issued=AcceptedProductionBatchStream`. The boundary therefore cannot establish
that validation exclusion came from the configured immutable artifact. D025 must make
the governed constructor non-bypassable and cover rejection of unvalidated direct
construction.

### Medium: D026's shared schema change breaks the D010 CLI regression suite

At `tests/unit/data/test_manifest_cli.py:33`, the synthetic replacement table still
starts with data fields and leaves both required D026 storage placeholders from the
all-options example unresolved. Six of seven manifest-CLI tests now exit with status 2
and `configuration_invalid`, before testing local/remote verification, build
publication, drift, no-clobber, or redaction. Isolated result: `6 failed, 1 passed in
10.88s`.

The D026 targeted selector is green, but it did not include this shared-schema
consumer. The package regression surface must be restored before D026 is accepted.

### Medium: D025 exact-worker integration evidence is nondeterministic

The integration check at `tests/unit/config/test_d025_data_assembly.py:578` derives
worker PIDs from body tokens, while the tokenizer at line 60 emits no PID for an empty
body. Approved component dropout can empty the only body, and the persisted random
mainset can assign that sample as one worker's only observed work. The broad selector
therefore reported one PID; the isolated rerun passed with two. This does not show a
runtime fallback, because the direct D023 test independently observed two distinct
spawned PIDs, but it makes the D025 integration artifact non-reproducible.

### Medium: the D010 real-SIGKILL contract cannot reach its five-second barrier

`tests/fault_injection/test_data_failures.py:118` imports the broad
`sakuramoon.fault_injection` package inside the timed child, and line 134 permits only
five seconds from process creation through imports and partial-write readiness. The
test failed both in the full selector and in isolation with `fault process did not
reach its readiness barrier`; isolated wall time was 15.53 seconds including cleanup.
This matches the already-recorded D023 observation that fresh imports on this NFS
environment need longer than five seconds. The failure does not contradict the D023 or
D024 real worker-exit/restart tests, but it leaves the D010 interruption evidence
unreliable on the approved server-backed deployment.

## Infra conclusions

- The previous unsafe default-`fork` launch is remediated. Command queues, completion
  queue and DataLoader workers share `mp.get_context("spawn")`; the live-parent-thread
  test completed, two worker IDs/PIDs progressed, and no 120-second hang recurred.
- Worker inputs are capacity one each; ready work is the exact divisible configured
  budget; completion capacity equals worker count. State/cache mutation remains in the
  parent/service owner. Worker death and parent close leave active leases for restart.
- D024 owns bounded download concurrency, verified lookahead, lease/ACK admission,
  mainset state and cache reservations. The real AF_UNIX/service/fault selectors pass
  in CPU scope. Existing bounded RTX 5090 evidence was reviewed but not rerun here.
- D026 correctly keeps durable data on the approved NFSv3 mount and moves only the
  AF_UNIX endpoint and singleton `flock` file to non-NFS `/run/sakuramoon`. NFS is not
  rejected for cache/state merely because it is NFS; the governed checks instead bind
  exact source/version/hard-mount identity, probe file/directory fsync plus replace and
  readback, and account explicit cache high watermark, three measured checkpoints and
  reserve. Host-local IPC avoids placing AF_UNIX namespace and host ownership locks on
  a shared network filesystem.

## Per-task verdicts

| Task | Infra/performance verdict | Current boundary |
|---|---|---|
| D010 | CHANGES REQUIRED (evidence) | Streaming/verification code retains its prior bounded design, but CLI regressions and the five-second SIGKILL barrier fail. Live network performance remains pending. |
| D011 | PASS (implemented CPU scope) | Deterministic selection and streaming bundle publication pass; approximately 11M-row memory/time remains pending. |
| D012 | PASS (implemented CPU scope) | Explicit quota, protected LRU and durable state pass; legacy 300-500 GiB assumption is superseded by D026. Production capacity/replay audit remains pending. |
| D013 | PASS (implemented CPU scope) | Fixed counters/histogram remain bounded; production scan/decode throughput remains pending. |
| D014 | PASS (implemented CPU scope) | Work remains bounded by 512 condition tokens with no model/network/cache path. |
| D015 | PASS (implemented CPU scope) | Ready budget and bucket fragments remain bounded; production RSS/pinned/ready-wait evidence remains pending. |
| D016 | PASS (implemented CPU scope) | Strict startup-only schema has no material hot-path cost. |
| D017 | PASS (implemented CPU scope) | Atomic replace publication and cleanup contracts remain bounded. |
| D018 | PASS (implemented CPU scope) | The 10,001-bin retention histogram is constant-memory. |
| D019 | PASS (implemented CPU scope) | Validation remains bounded by existing candidate IDs. |
| D020 | PASS (implemented CPU scope) | Twelve fixed counters remain bounded through collate. |
| D021 | PASS (implemented CPU/recorded 1GPU scope) | Trusted mapping adds bounded per-sample CPU work; no new throughput claim. |
| D022 | PASS (implemented CPU scope) | Active cardinality and cache protection are bounded by exact worker count. |
| D023 | PASS (implemented CPU scope) | Explicit spawn closes the default-fork hang; input/ready/completion paths and shutdown/replay behavior pass. |
| D024 | PASS (implemented CPU/existing bounded 1GPU scope) | Service/mainset/AF_UNIX consumer/fault paths pass; production cold-cache and fully-cached comparison remain pending. |
| D025 | FAIL / CHANGES REQUIRED | Direct construction bypasses validation-manifest verification and issues accepted authority with caller-selected validation IDs/identity. Exact-worker PID evidence is also nondeterministic. |
| D026 | IMPLEMENTATION PASS / PACKAGE CHANGES REQUIRED | Exact NFS identity, capacity, atomic probes and host-local IPC pass; shared-schema consumers must be restored. |

## Independent validation

- Broad bounded selector: `383 passed, 8 failed, 5 warnings in 89.08s` under a
  240-second outer timeout; it progressed through the former D023 hang boundary.
- D023 state/worker/fault plus D024 consumer and D025 factory: `37 passed in 51.79s`
  under a 150-second timeout.
- D026 config/storage/service/preflight/fault: `52 passed, 3 warnings in 19.14s`.
- D010 manifest CLI isolated: `6 failed, 1 passed in 10.88s`.
- D010 download SIGKILL isolated: `1 failed in 15.53s` at the five-second readiness
  barrier.
- D025 direct-construction reproduction: absent configured validation manifest,
  arbitrary 2,000-ID set and caller-selected identity were accepted, and an
  `AcceptedProductionBatchStream` was issued.
- Trace contracts: `40 passed in 80.63s`; direct verifier `237/237`, zero errors.
- Ruff passed; strict Pyright reported `0 errors, 0 warnings`; diff check passed before
  review evidence was added.

All pytest basetemps were isolated below repository `cache/` and removed after the
review. No production network, GPU, long run, formal stage, DDP/NCCL, multi-GPU,
1,000-step, two-hour cold-cache, or full dataset scan was run.

## Pending performance and hardware gates

- A governed immutable production manifest and the complete approximately 11M-ID,
  validation-exclusion, bucket and caption scans.
- Exact governed parser plus real local Qwen/Mage on one RTX 5090.
- Cold/warm-cache overlap and continuous two-hour cold supply at `>=12 samples/s`,
  ready wait `<2%`, no swap/unbounded RSS/quota breach, with fully-cached same-backend
  p50/p95/p99 control.
- Benchmark-driven 1/2/3 worker, ready-depth, download concurrency, lookahead and small
  cache watermark selection; no current number is locked by this review.
- Formal stage, long-run, 1,000-step and every four-GPU/DDP/NCCL gate remain pending or
  blocked and cannot be closed by CPU or one-GPU evidence.
