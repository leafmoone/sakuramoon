# D010-D026 DATA package AI/model correctness rereview

Reviewer: independent agent `/root/data_d025_d026_rereview`

Review head: `249bfe4062aa7666bc0d036da3b0ec08295f5510`

Scope: committed D010-D026 implementation, with focused source review of the D023
explicit-spawn remediation, D024 AF_UNIX consumer and lease/ACK path, D025 governed
factory/process-local accepted stream, and D026 server-backed NFS/host-local IPC
boundary. Existing review evidence was retained as history. No production code, task,
current decision, trace registry, or commit was changed by this reviewer.

Overall verdict: **CHANGES REQUIRED**. D025 has a production-boundary bypass: its
public dataclass constructor can issue an accepted stream from caller-supplied
validation IDs and factory identity without loading or validating the configured
validation manifest. The former D023 default-`fork` hang is closed, but the package
also cannot receive an independent PASS because D026 invalidated six D010
manifest-CLI contracts and the D025 exact-two-worker integration assertion is
nondeterministic.

## Findings

### High: D025's public factory constructor bypasses governed validation issuance

`src/sakuramoon/data/production.py:358` exposes
`ProductionPipelineFactory` as a directly constructible public dataclass.
`__post_init__` checks only the caller-supplied ID set's type, cardinality and positive
integer shape plus a caller-supplied 64-hex identity. The configured validation JSONL
path, SHA-256, canonical records and IDs are loaded only by the optional `from_config`
path at line 391. Nothing makes that path authoritative.

A read-only reproduction built a valid `RuntimeConfig` whose configured synthetic
validation manifest does not exist, passed the direct constructor an arbitrary set of
2,000 IDs (`9000001..9002000`) and `"f" * 64`, and then called `batches()` against a
healthy matching service identity. Construction succeeded and
`require_accepted_production_batch_stream()` accepted the resulting stream. Observed:
`configured_manifest_exists=False`, `accepted_arbitrary_ids=True`, and
`stream_issued=AcceptedProductionBatchStream`.

This violates D025's governed-factory claim and lets a caller replace validation
exclusion identity while retaining factory-issued stream authority. D025 must make
governed construction non-bypassable and add a contract rejecting direct or otherwise
unvalidated construction. This fix belongs strictly to D025.

### Medium: D026 leaves the D010 manifest CLI contracts configuration-invalid

`tests/unit/data/test_manifest_cli.py:33` derives test configs from the all-options
example but does not replace D026's required `storage.shared_mount_source` and
`storage.minimum_free_gib` placeholders. All local, remote, build, drift, no-clobber,
and redaction cases therefore stop at `configuration_invalid` instead of exercising
their D010 assertions. Isolated reproduction: `6 failed, 1 passed in 10.88s`.

This is not evidence that the production manifest implementation is wrong. It is a
cross-task regression in the current committed verification surface, so D010 and D026
cannot jointly receive a package PASS until the governed synthetic storage values are
provided and the intended CLI assertions pass again.

### Medium: D025's exact-two-worker evidence is dropout-sensitive and flaky

`tests/unit/config/test_d025_data_assembly.py:60` records a worker PID only when the
tokenized body is non-empty, while `tests/unit/config/test_d025_data_assembly.py:578`
infers exact worker participation only from those PID tokens. The production dropout
contract remains enabled and sample 3001 deterministically drops its only `nsfw` body.
With a random mainset order, the other worker can process only that empty-body sample,
so the full rereview observed one PID and failed at line 583. The identical test passed
when rerun alone in 7.69 seconds.

The direct D023 worker contract independently observed both spawned worker IDs and two
distinct persistent PIDs. This finding is therefore an evidence defect rather than a
demonstrated silent fallback to one worker. D025 still needs a dropout-independent
worker observation before its integration evidence is stable.

## Focused correctness conclusions

- D023 uses one explicit `spawn` context for command queues, completion queue, and
  DataLoader workers. Exact schema-v3 topology, parent-only state/cache mutation,
  prepare-before-handoff, normal-exhaustion completion, worker exit, parent close,
  full-active replay, and recovered-active reprepare ordering passed the bounded
  rereview selector. The prior default-`fork` hang did not recur.
- D024 keeps service ownership of manifest order, network/download/verification,
  cache, mainset, active/completed/replay state, and lease identity. The trainer-side
  consumer only handles service descriptors and completion ACKs. D021 trusted
  `ShardRecord`, explicit metadata mapping, validation-before-processing, caption,
  image, and collate contracts remain intact.
- D025's loader controls, manifest/service topology checks and non-serializable,
  process-local stream handle work after factory creation, but factory creation itself
  is not governed: direct construction bypasses validation-manifest loading and allows
  caller-selected validation IDs and factory identity. T050 must still require and
  close the accepted handle on every path, but downstream enforcement cannot repair an
  invalid authority issuer.
- D026 correctly makes persistent state/cache paths server-backed while keeping the
  AF_UNIX socket and singleton lock on host-local `/run/sakuramoon`. Its storage schema,
  exact mount identity, hard NFSv3 check, atomic probe, three-checkpoint capacity math,
  and no-fallback failure semantics passed their targeted tests.

## Per-task verdicts

| Task | AI/model verdict | Current boundary |
|---|---|---|
| D010 | CHANGES REQUIRED (evidence) | Manifest and verified publication code retain the prior semantic PASS, but six CLI contracts are configuration-invalid after D026. Immutable production revision/inventory and live source evidence remain pending. |
| D011 | PASS (implemented CPU scope) | Explicit mapping, trusted release, exact-2,000 selection and exclusion contracts pass; approximately 11M uniqueness, real bundle and full zero-leak scan remain pending. |
| D012 | PASS (implemented CPU scope) | Manifest-bound state/cache semantics pass; D022-D024 govern later multi-active/service ownership. Production quota/replay audit remains pending. |
| D013 | PASS (implemented CPU scope) | EXIF/RGB, 17 buckets, no-upscale, crop/retention and bounded scan semantics pass; full metadata and real 100k decode scans remain pending. |
| D014 | PASS (implemented CPU scope) | Caption order, separators, deletion, Artist placement, structured indices and boundary truncation pass under D016 values; production distribution remains pending. |
| D015 | PASS (implemented CPU scope) | Validation-before-processing, typed homogeneous collate and deterministic sample identity remain preserved by D021-D025. |
| D016 | PASS (implemented CPU scope) | All twelve approved probabilities remain required exact TOML floats with no runtime default. |
| D017 | PASS (implemented CPU scope) | Regenerable image report publication retains sibling-temp, file-fsync and atomic replace semantics. |
| D018 | PASS (implemented CPU scope) | Fixed-memory deterministic P01/P50/P99 retention evidence remains correct. |
| D019 | PASS (implemented CPU scope) | Candidate deletion IDs remain immutable, non-empty and trim-stable. |
| D020 | PASS (implemented CPU scope) | Twelve independent dropout hits survive plan, serialization and collate; production 100k distribution remains pending. |
| D021 | PASS (implemented CPU/recorded 1GPU scope) | Trusted record, explicit adapter/mapping, exclusion and batch release propagation remain intact; exact immutable production mapping remains pending. |
| D022 | PASS (implemented CPU scope) | Schema v3 topology, bounded active set, persist-before-fetch, active protection, per-shard completion and recovery barrier pass. |
| D023 | PASS (implemented CPU scope) | Explicit-spawn persistent two-worker remediation closes the previous hang and preserves replay/completion semantics. |
| D024 | PASS (implemented CPU/existing bounded 1GPU scope) | Process-isolated service, mainset, consumer, ACK/replay and rotation semantics pass; production network/storage overlap remains pending. |
| D025 | FAIL / CHANGES REQUIRED | Public direct construction bypasses configured validation-manifest verification and can issue an accepted stream with arbitrary validation IDs and factory identity. Exact-two-worker PID evidence is also flaky; exact governed real-shard GPU rerun remains pending. |
| D026 | CODE SEMANTICS PASS / PACKAGE CHANGES REQUIRED | Server-backed NFS and host-local IPC contracts pass targeted validation; D010 CLI regression must be repaired before package acceptance. |

## Independent validation

- Bounded combined CPU/data/config/fault selector: `383 passed, 8 failed, 5 warnings
  in 89.08s`. Failures were the six D010 manifest-CLI fixture regressions, one flaky
  D025 PID assertion, and the separate fixed-five-second D010 SIGKILL readiness test.
- D023/D024-consumer/D025 spawn and service isolation selector: `37 passed in 51.79s`.
- D025 real service factory test, isolated: `1 passed in 7.69s`.
- D025 direct-construction reproduction: missing configured validation manifest,
  arbitrary 2,000-ID set and caller-selected identity were accepted, and an
  `AcceptedProductionBatchStream` was issued.
- D026 storage/config/service/preflight/fault selector: `52 passed, 3 warnings in
  19.14s`.
- Trace contracts: `40 passed in 80.63s`; direct trace verification passed `237/237`
  requirements/source nodes with zero errors.
- Targeted Ruff passed. Targeted strict Pyright reported `0 errors, 0 warnings`.

No network, GPU, production manifest, approximately 11M scan, 100k production scan,
two-hour cold-cache run, long training, 1,000-step canary, formal stage, DDP/NCCL, or
multi-GPU validation was run by this reviewer.

## Remaining milestone gates

- Immutable production manifest/source access, approximately 11M globally unique IDs,
  exact production metadata/caption-availability mapping, exact 2,000-ID validation
  bundle, and complete training zero-leak evidence.
- Full 256/512 17-bucket scan, real 100,000-image post-EXIF dimension check, production
  crop-retention/rejection distribution, and production caption/dropout/truncation
  distribution including final empty-body rate.
- Exact governed parser on a real shard with local Qwen/Mage and one RTX 5090 consumer.
- Cold/warm-cache service overlap, two-hour cold-cache throughput, ready wait, bounded
  RSS/no swap/quota, worker/queue/concurrency sweep and fully-cached control.
- Formal stage, long-run, 1,000-step and every four-GPU/DDP/NCCL gate remain explicitly
  pending or blocked; CPU or one-GPU evidence does not close them.
