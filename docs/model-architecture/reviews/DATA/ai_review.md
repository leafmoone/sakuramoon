# Data package AI/model correctness audit

Scope: D010-D015 current implementations and their targeted tests. This audit did not
read `.env`, access `reference/`, call the production dataset, invent dropout values,
or run training.

Initial result: FAIL with one bounded finding per task.

- D010: identical duplicate remote listing entries were hidden by set equality.
- D011: non-string or whitespace-only aspect bucket keys were not rejected explicitly.
- D012: persisted completed/active state was not bound to the immutable manifest.
- D013: a non-finite crop-retention threshold could fail open.
- D014: an oversized first Artist could block later complete Artist sources.
- D015: no-upscale/retention rejection raised out of the pipeline instead of skipping
  the sample.

Production metadata mapping, 11M uniqueness/zero-leak scans, unresolved non-global
dropout values, full rejection distributions, and quality gates remain pending. Each
finding is remediated and accepted in its owning task commit; direct independent
re-review startup was unavailable, so final per-task conclusions are main-agent
acceptance rather than fabricated independent passes.

A later independent D015 follow-up returned CHANGES_REQUIRED for durable-path bypass,
unobservable rejection reasons, framing-unbound padding and strict inputs. Those code
findings now have main-agent remediation and tests; independent post-fix rereview remains
pending. D012's single active shard conflicts with shard-level multi-worker splitting,
so durable 2/3-worker production use remains blocked rather than silently reduced.
