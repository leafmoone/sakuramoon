# Dense Model package Infra/performance review

Reviewer: independent agent `/root/dense_package_review`

Date: 2026-07-31

Scope: M030-M034 frozen CPU/one-GPU implementation and evidence. The reviewer did not edit or commit repository files.

## Verdicts

| Task | Verdict | Basis |
|---|---|---|
| M030 | PASS | The conditioning hot path introduces no tensor-to-host scalar synchronization. |
| M031 | PASS | Dense SDPA remains an explicit correctness reference; no production throughput claim is inferred from it. |
| M032 | PASS | Single-GPU capacity evidence is correctly limited and does not claim full 24-layer/512 training performance. |
| M033 | PASS | Objective and CFG tensor paths avoid per-sample host synchronization; DDP aggregation remains explicitly outside scope. |
| M034 | PASS | Euler/Heun loops have no per-NFE host synchronization; the only finite-state synchronization is after integration. Profile dispatch does not add dynamic fallback or unvalidated combinations. |

## Independent validation

- Dense targeted CPU: 53 passed.
- Sampling/config/eval: 86 passed with three existing TorchAO deprecation warnings.
- Trace contracts: 40 passed after the registry was frozen.
- Ruff, full strict Pyright, `uv lock --check`, and `git diff --check`: passed.
- The first trace failures occurred only while C08-007 was being written concurrently and were not reproduced on the frozen tree.

## Remaining boundaries

- No production DiT sampling throughput, decoder cost, formal quality cost, four-GPU behavior, DDP, or NCCL claim is closed by this package review.
- K001 retains production attention performance ownership; T041 retains four-GPU behavior; T050 retains full-chain training integration.
- The environment uses the mounted host 580.105.08 NVIDIA driver libraries. No 610 userspace library change is part of M034.

No Infra/performance finding blocks the M034 atomic commit.
