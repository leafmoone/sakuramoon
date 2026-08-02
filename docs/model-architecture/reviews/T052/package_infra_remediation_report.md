# T052 package Infra finding remediation

Date: 2026-08-02

The independent Training Utilities Infra review found that the immutable scalar
artifact boundary accepted impossible metric values even though the governed metric
implementations produce valid ranges. `EvaluationArtifact` now rejects `FID < 0` and
`IS < 1` before publication. The focused negative contract covers both metric kinds
and proves that no artifact path is created.

This is a fail-closed artifact validation change. It does not add an extractor,
checkpoint, prompt identity, real-stat identity, evaluator resource plan, numeric
quality threshold, or production execution. The prior synthetic RTX 5090 metric
plumbing evidence remains unchanged and is not rerun or promoted.

Main-agent targeted validation passed 17 focused artifact/metric tests and 182 T052
evaluation/config/runtime tests. Ruff and strict Pyright passed. Independent Infra
rereview remains required before the Training Utilities package gate can close.
