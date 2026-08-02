# S000 Infra/performance self-review

Reviewer authority: main agent, under the user's no-agent direction. This is not an
independent review.

## Verdict

PASS for bounded process, resource, publication, and fresh-recovery mechanics only.

## Findings

No blocking Infra issue remains in the implemented engineering scope.

- The service, two DataLoader workers, and fresh-resume worker are spawned processes
  with bounded startup/request/shutdown/fresh-process timeouts. The final run left no
  matching child process or AF_UNIX socket; the GPU returned to 1 MiB used and idle.
- The runner refuses an occupied socket and an existing output root. Runtime evidence,
  raw checkpoint, and fresh result use no-clobber publication. The expected zero-byte
  ownership lock remains as a non-active filesystem identity.
- The raw checkpoint contains 5,143,061,370 payload bytes and exact `COMPLETE` content.
  The fresh service recorded two replayed shards/two replayed samples before update 2.
- Parent model/optimizer references are deleted before garbage collection and CUDA
  cache release, avoiding retention of the initial model allocator cache while the
  fresh process runs.

## Performance boundary

The observed peak allocation of 11,981,415,936 bytes is not a capacity result. There
was no warmup/measured benchmark window, max-batch search, throughput statistic,
same-backend repeat distribution, cold-cache run, long run, profiler claim, or four-GPU
measurement. No `perf_baseline.json` or `perf_after.json` is warranted. Production
throughput, memory, quality, formal S000/S001, and all multi-GPU gates remain blocked.

## Independent Infra/performance review (2026-08-02)

Reviewer authority: fresh independent Infra reviewer. The reviewer did not modify the
implementation, run GPU workloads, stage files, or create commits.

### Verdict

FAIL for formal S000 production readiness pending the two evaluator remediations below
and resolution of the governed launch blockers. The existing bounded CPU/1GPU results
remain valid engineering evidence only; they are not formal evaluator, capacity, S001,
or four-GPU evidence.

### Findings

1. **[HIGH] `S000-INFRA-001`: evaluator execution lacks governed storage preflight.**
   `preflight_evaluator()` only resolves the artifact root and rejects symlink or
   pre-existing run paths (`src/sakuramoon/eval/runner.py:390-423`). It does not verify
   the configured NFS filesystem/source/version/hard-mount identity, run the required
   same-directory atomic-publication probe, or prove a governed free-space budget before
   GPU work. `run_evaluator()` can construct the CUDA extractor before it constructs the
   publisher (`runner.py:538-540`), while the publisher first creates staging at
   `src/sakuramoon/eval/publisher.py:69-94` and discovers some durability failures only
   during commit (`publisher.py:138-149`). A formal 10k/50k, 99-NFE job can therefore
   consume GPU time and pause training before an invalid or insufficient persistent
   target is rejected. Add evaluator-specific, fail-closed storage identity and atomic
   probe checks before model/extractor construction. The required output-space budget
   must remain an explicit blocker until it is governed; this review does not invent one.

2. **[MEDIUM] `S000-INFRA-002`: evaluator cost accounting stops before work it claims
   to cover.** Per-checkpoint CUDA and wall timing is frozen at
   `src/sakuramoon/eval/runner.py:630-649`, before FID/IS finalization and scalar/manual
   artifact publication at `runner.py:650-694`; the checkpoint generator remains alive
   until `runner.py:695`. Consequently `wall_seconds` and, when training is paused,
   `training_pause_seconds` omit metric reduction, artifact writes, and model teardown.
   Overall timing is likewise frozen at `runner.py:739-762` before the summary,
   `COMPLETE`, tree fsync, rename, and parent-directory fsync performed by
   `publisher.commit()` at `runner.py:763-780`. Record checkpoint costs only after the
   governed work is complete, release GPU-resident checkpoint state before CPU-heavy
   metric finalization where valid, and include final publication latency in the run
   cost or expose it as a separate explicit duration.

### Independent checks

- Current S000/config/evaluator CPU selector: 281 passed, 1 skipped, 3 warnings.
- T054 archive-free CPU regression selector: 161 passed, 5 warnings.
- Scoped Ruff passed for S000/config/evaluator and T054 paths.
- Scoped Pyright reported 0 errors for both scopes.
- `git diff --check` passed before this report append.

### Remaining formal blockers and evidence boundary

- `config/train_s0.toml` and `config/eval.toml` each retain 59 unresolved bindings:
  42 `BENCHMARK_*` values and 17 `REQUIRED_*` identities.
- `S0_WARMUP_FUNCTION_UNRESOLVED`, `S0_PASS_INDEX_OWNERSHIP_UNRESOLVED`,
  `S0_LIVE_READY_QUEUE_DEPTH_UNBOUND`, and `S0_DIT_FLOPS_OBSERVATION_UNBOUND` remain
  explicit non-TOML blockers.
- The capacity sweep and formal evaluator have not run. S001 has not started.
- All four-GPU gates remain blocked. Synthetic and bounded single-GPU results must not
  be promoted to formal FID/IS, throughput, capacity, S001, or four-GPU conclusions.

## Independent Infra post-remediation rereview (2026-08-02)

Reviewer authority: the same independent Infra reviewer that reported
`S000-INFRA-001` and `S000-INFRA-002`. This rereview inspected the remediation and ran
CPU/static checks only. It did not modify implementation, use a GPU, stage files, or
create commits.

### Verdict

PASS for the two requested evaluator Infra remediations. `S000-INFRA-001` and
`S000-INFRA-002` are resolved. This is not a formal S000, evaluator-quality, capacity,
S001, or multi-GPU release verdict.

### Finding disposition

1. **`S000-INFRA-001` RESOLVED.** `preflight_evaluator()` now invokes
   `require_evaluation_storage()` before prompt, extractor, model, or checkpoint loading
   (`src/sakuramoon/eval/runner.py:444-451`). The storage gate validates one governed
   NFS filesystem/source/version/hard-mount identity across every persistent path, runs
   the configured same-directory atomic publication probes, verifies host-local runtime
   paths, and requires free space for minimum reserve, cache high-watermark, three
   measured raw checkpoint copies, and explicit `evaluation.output_reserve_gib`
   (`src/sakuramoon/storage.py:282-311`). The output reservation is a strict positive
   config field and remains a benchmark binding until governed rather than receiving a
   fallback. The publisher is also constructed before extractor or generator creation
   (`src/sakuramoon/eval/runner.py:908-920`). Storage failure is surfaced as the precise
   fail-closed `EVALUATION_STORAGE_INVALID` blocker.

2. **`S000-INFRA-002` RESOLVED.** Checkpoint timing now releases the generator and
   completes FID/IS finalization before freezing checkpoint wall/pause cost
   (`src/sakuramoon/eval/runner.py:1009-1038`). Job artifacts explicitly declare that
   publication is excluded from that checkpoint cost. Overall pre-commit wall timing
   covers job, artifact, generation, and comparison writes through
   `runner.py:1124-1147`; final summary/`COMPLETE`/tree fsync/rename/directory fsync time
   is measured separately as `publication_seconds`, while `total_wall_seconds` spans
   the entire call (`runner.py:1148-1180`). The committed summary declares the
   unavoidable self-accounting boundary, and the CLI emits both measured durations.

### Rereview checks

- Focused evaluator/storage/config CPU selector: 129 passed, 3 dependency warnings.
- Scoped Ruff: all checks passed.
- Scoped Pyright: 0 errors, 0 warnings, 0 informations.
- `git diff --check`: passed before this append.

### Remaining boundary

No Infra finding remains open from this two-item remediation. Unresolved governed
configuration bindings and non-TOML S0 blockers, the capacity sweep, formal evaluator,
S001, formal stage work, and all four-GPU gates remain outside this PASS and retain
their prior blocked/pending status. Synthetic or bounded evidence is still
engineering-only.

## Final independent Infra/performance review (2026-08-02)

Reviewer authority: fresh independent `s000_infra_reviewer_final`. This review inspected the
post-remediation production launcher, data-service bootstrap, checkpoint retention,
telemetry, evaluator execution and current handoff evidence. It ran CPU/static checks
only and did not modify implementation, use a GPU, stage files, or create commits.

### Verdict

BLOCKED pending two operational data-contract corrections and one evaluator publication
correction. The real bounded single-GPU lifecycle and fresh-process RAW resume evidence
remain valid engineering evidence, but the current tree does not yet satisfy its fixed
manifest/validation and atomic no-clobber contracts for a production handoff.

### Findings

1. **[HIGH] `S000-INFRA-003`: an existing operational manifest is coupled back to the
   mutable upstream listing.** `ensure_dataset_manifest()` loads an existing manifest
   and then unconditionally lists `master` and requires exact path/size/digest equality
   (`src/sakuramoon/data/modelscope.py:319-322`). An unrelated tar addition upstream
   therefore prevents every later data-service restart, despite the governing C11
   contract that an existing operational manifest is not refreshed and that no immutable
   revision is required. Existing manifests must be validated locally and used as the
   operational snapshot; remote enumeration belongs only to missing-manifest
   initialization. Per-shard size/digest verification during download remains required.

2. **[HIGH] `S000-INFRA-004`: validation selection is still algorithmic rather than the
   fixed two-tar decision.** `select_validation_shards()` ranks the whole manifest by a
   hash of seed, manifest identity and path, while `validate_selection_manifest()`
   requires every persisted selection to equal that recomputation
   (`src/sakuramoon/data/validation.py:246-283`). The canonical decision fixes
   `shard-000509.tar` and `shard-000060.tar`; rebuilding a missing manifest/selection
   after upstream `master` changes can silently choose another pair or reject the fixed
   pair. Selection construction must fetch exactly those two records from the operational
   manifest and fail explicitly if either path is absent.

3. **[MEDIUM] `S000-INFRA-005`: evaluator final publication does not provide atomic
   no-clobber at the rename edge.** `AtomicEvaluationPublisher.commit()` checks the final
   path and then calls ordinary `os.rename()` in a separate operation
   (`src/sakuramoon/eval/publisher.py:144-147`). A destination empty directory created
   between those operations can be replaced by POSIX rename, so the promised no-clobber
   property is not atomic. Use a no-replace primitive for the final directory transition
   and add a race-injection test; summary, `COMPLETE`, tree fsync and parent-directory
   fsync ordering should remain unchanged.

### Confirmed behavior

- W&B and ModelScope environment-variable presence is checked by strict config loading
  before CUDA selection or model construction; no credential value was read or logged.
- The production lifecycle accepts only fresh start or an exact canonical absolute RAW
  `COMPLETE` path. RAW restore and full binding precede the data-service connection, and
  the configured S0 batch contract is local batch 2, accumulation 4, global batch 8.
- The real first-update and fresh-process N-to-N+1 evidence covers data service, local
  Qwen/VAE, 16-layer DiT, loss, backward, FP32 clip, TorchAO update, telemetry and RAW
  publication. It remains explicitly bounded engineering evidence.
- Evaluator storage preflight, checkpoint/overall timing, separate publication timing,
  reference Heun-50/99-NFE metadata, extractor/real-stat fail-closed blockers and
  engineering-only classification remain consistent after the earlier remediations.

### Independent checks

- Evaluator-focused CPU selector: 60 passed.
- Manifest/validation/train/checkpoint CPU selector: 115 passed.
- Additional T054/training/fault CPU selector: 66 passed.
- Scoped Ruff passed; affected-path Pyright reported 0 errors.
- No traceability verifier, GPU workload, formal evaluator, S001 stage, DDP/NCCL or long
  run was executed.

### Evidence boundary

Formal FID/IS remains blocked by intentionally absent extractor, preprocess, real-stat
and sample-count bindings. The capacity rows and single-update timings are engineering
selection evidence, not steady-state throughput or a maximum batch result. S001 has not
started, and every multi-GPU gate remains pending.

## Final Infra remediation rereview (2026-08-02)

Reviewer authority: the same independent `s000_infra_reviewer_final` that reported
`S000-INFRA-003`, `S000-INFRA-004`, and `S000-INFRA-005`. This append-only rereview
inspected the targeted remediation and ran CPU/static checks only. It did not modify
implementation, use a GPU, stage files, create commits, or change the T054 review.

### Verdict

PASS for all three requested Infra remediations. `S000-INFRA-003`,
`S000-INFRA-004`, and `S000-INFRA-005` are RESOLVED. No new Infra finding was
identified in the remediated scope.

### Finding disposition

1. **`S000-INFRA-003` RESOLVED.** Existing operational manifests are now loaded and
   strictly validated only from the local file, with no call to the mutable `master`
   listing (`src/sakuramoon/data/modelscope.py:305-320`). Remote enumeration occurs only
   when the manifest is absent; a concurrent initialization winner is also adopted by
   strict local load rather than by relisting (`modelscope.py:321-331`). The explicit
   manifest CLI retains separate local and remote-validation modes. Tests bind zero
   listing calls and unchanged bytes when `initialize` encounters an existing snapshot
   whose remote listing has drifted.

2. **`S000-INFRA-004` RESOLVED.** The validation contract now fixes the complete paths
   `data/2_2026.1/shard-000509.tar` and
   `data/2_2026.1/shard-000060.tar` in canonical order
   (`src/sakuramoon/data/validation.py:26-31`). Construction fetches exactly those two
   manifest records and hard-fails when either is absent (`validation.py:250-271`), while
   the selection type and reload validation reject any different paths, records, order,
   manifest identity, or seed (`validation.py:59-72,274-288`). This removes dependence
   on whole-manifest hash ranking while preserving the internal selection identity and
   per-shard download integrity fields.

3. **`S000-INFRA-005` RESOLVED.** The actual NFSv3 target rejects
   `renameat2(RENAME_NOREPLACE)` with `EINVAL`, so the final implementation uses an
   executable same-filesystem commit-marker protocol instead of falling back to an
   overwrite-capable rename. It first atomically reserves the final directory with
   no-clobber `mkdir`, hard-links staged payloads into it without replacement, fsyncs
   every destination directory, removes staged payload names, and hard-links the already
   fsynced `COMPLETE` file last (`src/sakuramoon/eval/publisher.py:67-115,189-197`). A
   concurrent final owner is preserved and receives no writes. Any failure before the
   final hard link leaves a visible but permanently uncommitted directory with no
   `COMPLETE`; consumers must treat `COMPLETE`, not final-directory existence, as the
   atomic publication boundary.

### Independent checks

- Complete data unit suite: 268 passed, 1 multiprocessing warning.
- Complete evaluator unit suite: 144 passed, 17 dependency warnings.
- Scoped Ruff passed; scoped Pyright reported 0 errors, 0 warnings, 0 informations.
- `git diff --check HEAD` passed before this append.
- An independent probe on the configured NFSv3 mount verified nested payload and
  `COMPLETE` durability, successful staging cleanup, no `COMPLETE` after an injected
  pre-commit failure, and preservation of a concurrently owned final directory. Its
  temporary directory was automatically removed.

### Evidence boundary

This PASS closes only the three implementation findings. The retained single-update,
resume, capacity-row and bounded evaluator results remain engineering-only. Formal
FID/IS still requires its explicit extractor, preprocess, real-stat and sample bindings;
S001 has not started, and formal-stage, long-run and multi-GPU gates remain pending.
