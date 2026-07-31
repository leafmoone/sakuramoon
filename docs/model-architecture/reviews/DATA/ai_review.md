# Data package AI/model correctness review

Reviewer: `/root/roadmap_inventory` (independent package reviewer)

Scope: `D010-D016` current CPU implementation, contracts, task evidence, and the
existing bounded one-GPU engineering evidence. Overall verdict: **CHANGES_REQUIRED**.

## Findings

1. **D015 bypasses D011's trusted shard metadata contract.**
   `src/sakuramoon/data/pipeline.py:249-252` parses each sample with
   `parse_metadata(raw_metadata)`, so `release` is accepted from sample JSON. D011's
   trusted boundary is `parse_shard_metadata(..., shard=..., fields=...)` at
   `src/sakuramoon/data/metadata.py:102-136`, which takes `release` only from the
   immutable D010 `ShardRecord`. The production pipeline can therefore stratify or
   audit a sample under an untrusted release value. D015 is **CHANGES_REQUIRED**.

2. **D012/D015 do not implement the locked initial two-worker durable contract.**
   `src/sakuramoon/data/collate.py:285-300` rejects every durable invocation whose
   `worker_count` is not one, and `tests/unit/data/test_pipeline.py:272-324` explicitly
   preserves that rejection. This conflicts with the initial two persistent workers
   per GPU in `current/confirmed-decisions.md:151-158` and the D015 contract in
   `progress/IMPLEMENTATION_ROADMAP.md:305-313`. The earlier 1/2/3-worker mechanics
   sweep did not use D012 lease state and cannot establish shard-level durable resume.
   D012 and D015 are **CHANGES_REQUIRED**; production durable multi-worker use remains
   blocked rather than silently reduced to one worker.

3. **D013 cannot produce the required retention percentiles.**
   `src/sakuramoon/data/buckets.py:252-301` retains assigned counts and rejection
   counts only; each accepted `crop_retention` value is discarded. The complete scan
   requirement at `current/open-items.md:40-45` requires retention quantiles as well
   as eligible/rejection/bucket counts. D013 is **CHANGES_REQUIRED** until a bounded,
   deterministic quantile/report contract and golden test exist.

4. **D014 does not validate candidate deletion IDs.**
   `CaptionFields.candidate_tags` is an unchecked `frozenset[str]` at
   `src/sakuramoon/data/caption.py:66-74`; deletion is exact string membership at
   `caption.py:183-189`. A focused probe accepted `" character:alice "` and did not
   delete the canonical `character:alice` tag. The existing boundary test at
   `tests/unit/data/test_caption.py:187-190` validates `Tag.canonical`, not candidate
   IDs. D014 is **CHANGES_REQUIRED**.

5. **D014 discards the component dropout decisions required by its 100k report.**
   `CaptionPlan` exposes selected content and `all_condition_dropped` only at
   `src/sakuramoon/data/caption.py:123-132`; the category, Artist, candidate-source,
   and per-NL hit decisions made at `caption.py:173-201` are not retained. The current
   contract requires the final empty-body rate and every component hit rate
   (`current/open-items.md:21-26`). Inferring hits from final content is not equivalent
   when a source is empty or several dropouts overlap. D014 remains
   **CHANGES_REQUIRED** until explicit telemetry and the production 100k distribution
   run exist.

6. **D016 is correct in its config-only scope, but older status text is stale.**
   `src/sakuramoon/config/schema.py:202-231` and
   `config/examples/all_options.example.toml:95-110` bind all approved values with no
   runtime defaults. D016 is **PASS**. It does not satisfy D014's still-pending 100k
   component-distribution validation. The old undecided-value statements in
   `progress/tasks/D014.md:3-5,59-62`, `progress/tasks/D015.md:36-38`, and
   `progress/traceability.toml:948-952` are evidence/status drift and must not be used
   to reopen or reinterpret the confirmed values.

## Per-task verdicts

| Task | AI/model verdict | Evidence boundary |
|---|---|---|
| D010 | PASS | Canonical builder/transport contracts pass; real immutable manifest and production inventory remain pending evidence. |
| D011 | PASS | Trusted shard-release parser, selection, and exclusion contracts pass; approximately 11M-ID uniqueness, real 2,000-ID bundle, and zero-leak dry run remain pending. |
| D012 | CHANGES_REQUIRED | Single-worker state semantics are covered, but the locked durable two-worker contract is absent. |
| D013 | CHANGES_REQUIRED | Bucket/crop goldens pass; required retention percentiles and production full/100k scans are absent. |
| D014 | CHANGES_REQUIRED | Serializer goldens pass; candidate IDs and component-hit telemetry are incomplete. |
| D015 | CHANGES_REQUIRED | Pipeline accepts untrusted sample `release` and lacks the locked durable two-worker path. |
| D016 | PASS | Fixed strict TOML values are correct; the production 100k distribution run belongs to D014/Data milestone evidence. |

## Validation and boundaries

The independent CPU package suite passed:

```text
285 passed, 8 warnings in 23.47s
```

Command scope: `tests/unit/data`, `tests/contracts/data`,
`tests/contracts/text_protocol`, targeted config contracts, and
`tests/fault_injection/test_data_failures.py`, run with `CUDA_VISIBLE_DEVICES=` and
`uv run --frozen`. Ruff passed. Pyright reported 0 errors and 0 warnings. The warnings
were dependency deprecations and multiprocessing fork warnings, not test failures.

Passing tests do not close the findings above because several tests currently encode
the incomplete behavior. This review did not read `.env` or `reference/`, use the
network or GPU, run long training, or exercise DDP, NCCL, multi-GPU, a 1,000-step
canary, or a formal stage. No upstream four-column algorithm comparison is applicable
to this Data package: its governing contracts are the current SakuraMoon decisions and
task specifications, not runtime use of an upstream algorithm implementation.
