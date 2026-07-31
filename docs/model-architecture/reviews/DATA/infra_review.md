# Data package Infra/performance rereview

Reviewer: independent agent `/root/data_package_review`

Scope: committed `D010-D023` durability, boundedness, multiprocessing failure behavior,
and current engineering evidence through HEAD
`1b4fd87e19a5c1c9e21a502742ba5a38a6ad354`. Uncommitted T024 model changes were
excluded. Overall verdict: **PASS in the implemented task scope**. Production
throughput/capacity acceptance remains **PENDING**.

## Findings

No blocking Infra implementation finding remains after D017 and D022/D023.

1. D010 shard publication now fsyncs the rollback unlink after a post-replace parent
   fsync failure. D011 removes a renamed validation bundle on publication failure.
   D012 preserves or restores an unambiguous state namespace across initial/update
   publication failure and distinguishes the committed-state cleanup error. D017 uses
   the repository's regenerable-report replacement protocol.
2. D018's retention evidence is constant-memory in manifest size. D013 scan state is
   bounded by the fixed bucket family, D014 serialization by 512 condition tokens,
   and D020 telemetry by twelve integer counters per batch.
3. D023 bounds each worker's shard input queue at one command, DataLoader ready work at
   the exact divisible budget, and completion messages at `worker_count`. The targeted
   two-worker configuration observed capacities `2/2/2`, prefetch factor one, two
   distinct worker PIDs, and PID reuse across four shards. A blocked idle worker does
   not head-of-line block the other because result delivery is unordered and every
   message carries worker/shard identity.
4. Parent interruption and worker death never publish shard completion. The parent
   shuts workers down and leaves every already-prepared shard active for replay. The
   subsequent coordinator re-prepares all recovered active shards before activating a
   new one and protects the full active set from eviction on each fetch.

The Python 3.12 test run emitted multiprocessing warnings because PyTorch's default
Linux DataLoader context uses `fork` from a process that already has library threads.
The real fault and persistence tests completed without a hang, so this is not a current
failure. It remains part of the required long-lived one-GPU data/consumer engineering
smoke; the CPU result alone does not establish CUDA-parent or endurance reliability.

## Per-task verdicts

| Task | Infra/performance verdict | Evidence boundary |
|---|---|---|
| D010 | PASS | Listing is page-bounded, shard bodies are streamed, retries/timeouts are explicit, credentials are stripped on cross-host redirect, and failed publication rollback is durably fsynced. Live access and network throughput remain pending. |
| D011 | PASS | Selection/bundle contracts are deterministic and tar writing is one-sample-at-a-time. The in-memory approximately 11M-row scan still requires production memory/time evidence. |
| D012 | PASS | Explicit watermarks, protected LRU, verified fetch, manifest-bound atomic state, and failure rollback pass. Production 300-500 GiB sizing, concurrent-rank behavior, disk-full, and NVMe evidence remain pending. |
| D013 | PASS | Assignment/full-scan counters and fixed decode counters are bounded; report construction validates totals. Production decode/resize throughput remains pending. |
| D014 | PASS | Caption work is bounded by the fixed 512-token contract and adds no model forward, network, or cross-batch cache. Production tokenizer/truncation timing remains pending. |
| D015 | PASS | Ready work is explicit, bucket fragments are bounded by the finite image/text key space, and no embedding/latent/activation cache exists. Production ready-wait, RSS/pinned memory, and throughput remain pending. |
| D016 | PASS | Strict startup-only schema binding has no material hot-path cost. |
| D017 | PASS | Sibling temporary, file fsync, atomic replace, previous-report preservation, and cleanup pass. |
| D018 | PASS | The 10,001-bin histogram has constant memory and deterministic quantized output. |
| D019 | PASS | Candidate validation is bounded by the already-materialized set and adds no I/O/model work. |
| D020 | PASS | Fixed-key boolean/count transfer is bounded and compatible with worker pickle transfer. Production 100k execution cost remains pending. |
| D021 | PASS | Trusted record lookup and explicit metadata adaptation add bounded per-sample CPU work; the real RTX 5090 smoke remains engineering rather than throughput evidence. |
| D022 | PASS | Active cardinality is bounded by `worker_count`; state publication and cache protection pass targeted rollback/fault contracts. State cost is per shard, not per sample. |
| D023 | PASS | Exact persistent-worker topology, capacity-one inputs, bounded ready/completion paths, parent-only mutation, independent completion, worker exit, parent close, and restart replay pass. |

## Independent validation

- CPU/data/config/fault rereview: `315 passed, 20 warnings in 27.43s`.
- Ruff: passed.
- Pyright: `0 errors, 0 warnings`.
- Trace verifier: `222` requirements, `222` source nodes, `0` errors.

The test-local basetemp under `reviews/DATA/` was removed after validation. This review
made no network or GPU call and did not run a production NVMe sweep, long training,
DDP/NCCL, multi-GPU validation, 1,000-step canary, or formal stage.

## Remaining performance gates

- Cold-cache continuous two-hour supply at `>=12 samples/s`, ready wait `<2%`, no host
  swap/unbounded RSS, and no cache-quota violation.
- Fair 1/2/3-worker, queue-depth, download concurrency, Range worker, and 300-500 GiB
  watermark sweep before locking the smallest stable production values.
- Phase timing for cache/tar/JSON/caption/tokenize/decode/EXIF/resize/crop/bucket/H2D,
  plus real rejection counts, pinned/host memory, and production replay audit.
- One-GPU consumer integration with the persistent two-worker path; no current CPU or
  one-GPU result may be extrapolated to four-GPU DDP/NCCL behavior.

## D024 remediation rereview (`35e7e6f`)

本节追加于历史 D010-D023 package report 之后；历史结论与生产 pending gates 保持不变。D024 的独立 Infra/性能结论为 **PASS（已实现 CPU/有界 1GPU scope）**。service ownership lock、winner socket preservation 和真实进程边界 CUDA consumer 三项 remediation finding 均已关闭；bounded channels、active eviction protection 与 whole-shard replay 保持成立。不可变复审记录见 `d024_infra_rereview.md`。
