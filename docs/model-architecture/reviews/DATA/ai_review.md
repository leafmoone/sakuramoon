# Data package AI/model correctness rereview

Reviewer: independent agent `/root/data_package_review`

Scope: committed `D010-D023` data implementation, current decisions, task pages,
traceability mappings, per-task evidence, and targeted CPU/one-GPU evidence through
HEAD `1b4fd87e19a5c1c9e21a502742ba5a38a6ad354`. Uncommitted T024 model changes in the
shared worktree were excluded. Overall verdict: **PASS in the implemented task scope**.
The production Data milestone remains **PENDING** on the explicit scans, immutable
inventory, and performance gates listed below.

## Findings

No blocking AI/model correctness finding remains in the D010-D023 implementation.
The earlier package findings are closed by D017-D023 without changing the governing
caption, image, validation, or replay semantics:

1. D021 binds every local shard to one immutable D010 `ShardRecord`, resolves each
   WebDataset `__url__` through that trusted mapping, and takes `release` only from the
   record. Explicit field mapping and the nested adapter precede D011 parsing, while
   the unmodified JSON remains the caption parser input. Validation exclusion occurs
   before caption planning, tokenization, image decode, and crop. Trusted release is
   retained in `PipelineSample` and `TrainingBatch`.
2. D022 schema v3 binds state to the canonical manifest digest and exact positive
   `worker_count`, bounds sorted unique active shards by that count, and rejects v1/v2
   and topology drift. Activation is durably published before cache fetch; every fetch
   protects the complete active set. Completion removes only the named shard, and
   restart accounts every recovered active shard/sample once per recovery attempt.
3. D023 launches the exact configured DataLoader worker count and the targeted contract
   observes two distinct persistent PIDs reused across four shards. Workers receive
   only a parent-prepared local path and trusted record; no state store or cache enters
   the worker dataset. The parent completes a shard only after its ordered done marker
   and normal bounded completion message. Real `os._exit(23)` and parent-generator
   close leave all prepared shards active, after which restart re-prepares and replays
   both from shard start.
4. D017/D018/D019/D020 close the remaining image/caption findings: replace-publication
   of regenerable image reports, fixed-memory retention P01/P50/P99, strict candidate
   deletion IDs, and source-independent twelve-key dropout hit telemetry through
   collate. D016's current confirmed probabilities govern; stale pre-D016 wording in
   older D014/D015 evidence does not reopen those values.

## Per-task verdicts

| Task | AI/model verdict | Evidence boundary |
|---|---|---|
| D010 | PASS | Fixed source/commit schema, canonical explicit shard facts, exact remote inventory comparison, streamed byte/SHA verification, and redacted credential boundary pass. The immutable production revision/inventory and live enumeration remain pending. |
| D011 | PASS | Explicit field mapping, trusted shard release, duplicate detection, deterministic exact-2,000 stratification, bundle identity, and exclusion contracts pass. Production field mapping, approximately 11M-ID uniqueness, real bundle, and full zero-leak dry run remain pending. |
| D012 | PASS | Manifest-bound cache/state and at-least-once semantics pass; D022/D023 supply the later multi-active/two-worker remediation. Production quota, concurrent-rank, and replay audit evidence remain pending. |
| D013 | PASS | EXIF/RGB, exact 17-shape families, no-upscale, nearest aspect, cover crop, inclusive retention, scan accounting, and hard dimension threshold pass. Full manifest and real 100k decode scans remain pending. |
| D014 | PASS | Fixed category/separator/framing rules, deterministic dropout/shuffle, whole-boundary truncation, Artist placement, and structured indices pass. D016 supplies the current fixed values; production component/truncation distribution remains pending. |
| D015 | PASS | Validation-before-processing, one decode/serialization path, deterministic RNG identity, framing-owned padding, homogeneous typed collate, and expected image-rejection behavior pass. D021-D023 close its trusted metadata and durable topology findings. |
| D016 | PASS | All twelve approved dropout values are required strict TOML floats with no runtime defaults or drift. |
| D017 | PASS | Canonical image scan report bytes are preserved while publication follows the required sibling-temp, file-fsync, `os.replace` protocol. |
| D018 | PASS | Accepted-crop P01/P50/P99 use deterministic nearest-rank semantics over a fixed 10,001-bin histogram; all-rejected input reports null. |
| D019 | PASS | Candidate deletion IDs require an exact immutable set of non-empty trim-stable strings and cannot silently evade canonical matching. |
| D020 | PASS | All twelve independent hit decisions survive plan, serialization, worker transfer, and batch aggregation with T051-compatible keys. The production 100k run and final empty-body rate remain pending. |
| D021 | PASS | Trusted `ShardRecord`, explicit mapping/adapter, validation exclusion, caption/image contracts, and batch release propagation pass on CPU plus the recorded real-shard/local-model RTX 5090 smoke. The smoke is not an immutable production manifest claim. |
| D022 | PASS | Exact schema-v3 topology, bounded active set, persist-before-fetch, full active protection, independent completion, replay accounting, and recovered-active barrier pass. |
| D023 | PASS | Two persistent workers, parent-only coordination, bounded channels, prepare-before-handoff, normal completion, real worker exit, parent interruption, whole-shard restart replay, and recovery ordering pass. |

## Independent validation

The rereview CPU suite passed `315` tests in `27.43s` with `20` dependency/fork
deprecation warnings. It covered all data unit tests, data/text contracts, strict
dropout config contracts, D010-D012 fault paths, and D023 worker-exit/restart faults.
Targeted Ruff passed; targeted Pyright reported `0 errors, 0 warnings`. The direct
trace verifier passed with `222/222` requirements/source nodes and no errors.

No network, GPU, production dataset/NVMe scan, long training, DDP, NCCL, multi-GPU,
1,000-step canary, or formal stage was run by this reviewer. Existing D015/D021
one-GPU evidence is accepted only for its stated local pipeline/Qwen/Mage engineering
scope and does not close a throughput, quality, production inventory, or four-GPU gate.

## Remaining milestone gates

- Immutable production manifest and source access evidence; approximately 11M global
  ID uniqueness; production metadata/caption-availability mapping; exact 2,000-ID
  validation bundle; complete training zero-leak audit.
- Full 256/512 bucket scan with production retention/rejection distributions and the
  real 100,000-image post-EXIF dimension check.
- Production 100k caption dry run reporting every component hit, final empty-body
  rate, and `all_condition=10% +/- 0.5pp` acceptance.
- VAE reconstruction/latent quality evidence belongs to T020 and remains outside this
  CPU Data package pass.
