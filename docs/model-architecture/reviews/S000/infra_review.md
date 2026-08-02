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
