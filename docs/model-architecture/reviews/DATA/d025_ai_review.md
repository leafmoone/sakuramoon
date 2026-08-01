# D010-D025 DATA package AI/model correctness review

Reviewer: independent agent `/root/data_package_review`

Scope: committed Data implementation through D025 at
`016b5263a0add818a1a9bd5efae5b3cdbf406e35`. The unrelated user roadmap edit and
uncommitted T050 work were excluded. Existing D010-D024 review evidence was retained
as history and reread; this file records the post-D025 package review.

Overall AI/model verdict: **PASS for the implemented deterministic data semantics, but
the DATA package is not accepted until the blocking persistent-worker Infra finding in
`d025_infra_review.md` is fixed and rereviewed**.

## Findings

No new caption, image, trusted-metadata, validation-exclusion, state, replay, or
completion semantic defect was found in D025. In particular:

- `ProductionPipelineFactory` derives batch size, exact two-worker topology, ready
  budget, pinning, and drop-last only from the resolved strict TOML object and rejects
  service topology drift.
- The production metadata adapter does not project an untrusted release. The pipeline
  continues to bind release to the service-issued D010 `ShardRecord`, excludes the
  governed validation IDs before caption/image work, and passes the original JSON to
  the caption parser.
- The validation loader requires the configured SHA-256, exactly 2,000 canonical,
  sorted, globally unique positive IDs, exact fields/types, UTF-8 canonical JSONL, and
  rejects symlinks before worker construction.
- D025 does not import ModelScope transport, cache, eviction, partial cleanup,
  `mainset`, or service state implementation.

The exact governed production parser was not rerun against the real ModelScope tar on
GPU during D025. D021's earlier RTX 5090 run used the equivalent test-local helper;
changing that test to import the governed function is source coverage, not a new
execution result. The exact real-parser/Qwen/Mage run therefore remains pending rather
than being inferred.

## Per-task verdicts

| Task | AI/model verdict | Boundary |
|---|---|---|
| D010 | PASS | Immutable source/manifest and verified publication contracts pass; production fixed revision and live inventory remain pending. |
| D011 | PASS | Trusted release, explicit mapping, exact-2,000 selection, bundle, and exclusion semantics pass; approximately 11M-ID and full zero-leak evidence remain pending. |
| D012 | PASS | Manifest-bound at-least-once cache/state semantics pass; later multi-active behavior is governed by D022. |
| D013 | PASS | EXIF/RGB, 17 buckets, no-upscale, crop, retention, and bounded scan semantics pass; production scans remain pending. |
| D014 | PASS | Caption ordering, separators, deletion, structured indices, Artist placement, and whole-boundary truncation pass under D016 values. |
| D015 | PASS | Validation-before-processing, typed homogeneous collate, deterministic sample identity, and image rejection semantics pass. |
| D016 | PASS | All twelve dropout probabilities are strict required TOML floats at the approved values. |
| D017 | PASS | Regenerable report content is unchanged and uses the governed atomic replacement protocol. |
| D018 | PASS | Fixed-memory P01/P50/P99 accepted-retention semantics pass. |
| D019 | PASS | Candidate deletion IDs are immutable, non-empty, and trim-stable. |
| D020 | PASS | Twelve independent dropout-hit decisions survive plan, serialization, and collate; production distribution evidence remains pending. |
| D021 | PASS | Trusted `ShardRecord`, explicit adapter/mapping, validation exclusion, and release propagation pass. |
| D022 | PASS | Schema v3 topology, bounded active set, persist-before-fetch, full active protection, per-shard completion, replay accounting, and recovery barrier pass. |
| D023 | SEMANTICS PASS / PACKAGE BLOCKED | Parent-only state/cache mutation, prepare-before-handoff, completion/ACK, and replay contracts are coherent; the worker runtime can hang under the inherited default-fork path. |
| D024 | SERVICE SEMANTICS PASS / PACKAGE BLOCKED | Independent ownership, `mainset`, lease/ACK identity, replay, rotation, and bounded service state pass; its DataLoader consumer inherits the D023 worker finding. |
| D025 | CPU SEMANTICS PASS / PACKAGE BLOCKED | Governed parser, strict exclusion manifest, resolved loader controls, and topology identity pass; exact real-parser GPU evidence is pending and the factory inherits the D023 worker finding. |

## Independent validation

- D025 production/config selector: `8 passed in 3.04s`.
- D022 state tests completed `21 passed` before the adjacent D023 worker test hung.
- D024 service/mainset/fault tests excluding DataLoader workers: `14 passed in 20.90s`.
- Targeted Ruff: passed.
- Targeted Pyright: `0 errors, 0 warnings`.
- Trace verifier: `235/235` requirements/source nodes, `0` errors.

No network, GPU, production dataset/NVMe scan, long run, DDP/NCCL, multi-GPU,
1,000-step canary, or formal stage was run by this reviewer.

## Remaining semantic evidence gates

- Immutable production manifest, approximately 11M global-ID uniqueness, exact
  production metadata/caption-availability mapping, validation bundle, and complete
  zero-leak audit.
- Full 256/512 bucket scan, real 100,000-image post-EXIF dimension check, and
  production caption/dropout/truncation distribution.
- Exact governed production parser on a real shard with local Qwen/Mage and one-GPU
  consumer, after the persistent-worker fix.

