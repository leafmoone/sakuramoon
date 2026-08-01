# Encoders/Conditioning T020-T024 remediation Infra/performance rereview

Reviewer: independent agent `/root/encoders_remediation_rereview`

Scope: T020-T024 durability, process/device boundaries, synchronization, memory
boundedness, and one-GPU engineering evidence at repository HEAD `4143a60`. The
historical package Infra review is retained unchanged. Concurrent DATA, K001, and M037
work was excluded.

Overall CPU/one-GPU Infra verdict: **PASS**.

## Per-task verdicts

| Task | Verdict | Infra/performance conclusion |
|---|---|---|
| T020 | **PASS** | The immutable upstream lock adds no runtime dependency, automatic download, local-model hash layer, or `reference/` execution path. Local strict loading and bounded real encode/decode pass. Reconstruction evaluation retains at most 2,000 scalar observations and uses same-directory temporary publication, file fsync, `os.replace`, cleanup, and prior-report preservation. Production metric and large-scan costs remain unmeasured. |
| T021 | **PASS** | Raw block-24 capture adds a temporary hook to the existing single Qwen forward; it does not perform a second forward, enable cache, or add D2H. The hook is removed in `finally`. The real local test passes on RTX 5090; prior eight-shape latency/memory evidence remains engineering evidence rather than a production throughput gate. |
| T022 | **PASS** | The valid CUDA hot path contains only fixed-shape predicates, `_assert_async`, safe gather, tensor mixing, and masked output. Static inspection found no `.item()`, `.tolist()`, CPU move, dynamic boolean compression, or Python branch on a CUDA tensor value. The production-shape test passes with CUDA synchronization debug mode `error`. |
| T023 | **PASS** | CPU collate builds a bounded active-sample plan. GPU routing uses fixed-size `index_select`/`index_copy`; the only Python branch reads the host-known tensor `numel` metadata. Null samples skip Artist projection, while fixed-shape device assertions preserve fail-closed behavior. The mixed active/null production-shape test passes synchronization debug mode. |
| T024 | **PASS** | Public offsets are checked exactly once at packed entry and discarded after private offsets are rematerialized from canonical host lengths. Sample IDs and FA4 boundaries share that host identity. The accepted handle is reused across 16 blocks with one acceptance call per packed forward; there is no per-block D2H, boundary `.item()`, or `.tolist()`. All queues/caches are outside this task, and packing constructs only the flat varlen tensor rather than a production dense batch. The recorded 50-call entry p50 `0.030010 ms`/p95 `0.031951 ms` is bounded engineering evidence, not formal K001 performance acceptance. |

No Infra/performance finding requires remediation within T020-T024.

## Commands and results

- `uv run --frozen pytest -q --basetemp=cache/.pytest-encoders-rereview` over
  T020-T024 unit contracts plus collate, FA4, PackedDiT, and train-step adjacency:
  `134 passed, 17 warnings in 14.92s`.
- The same frozen environment over targeted real GPU pipeline/Qwen/conditioning/
  packing/FA4/PackedDiT tests: `23 passed, 18 warnings in 37.32s` on one RTX 5090,
  driver 580.105.08.
- `uv run --frozen ruff check ...`: passed.
- `uv run --frozen pyright ...`: `0 errors, 0 warnings, 0 informations`.
- A bounded real Mage encode/decode command passed with finite BF16
  `[1,128,2,2] -> [1,3,32,32]` tensors and no gradient/training state.

The trace suite's five failures and direct-verifier failure were caused solely by
concurrent uncommitted K001 evidence/status/history edits, not encoder/conditioning
code or trace mappings. Live trace must be rerun from the settled worktree before the
main agent commits package closure.

## Performance boundaries

The retained per-component timing and memory observations are suitable for bounded
engineering evidence only. The T020 2,000-image evaluation and 50k-100k latent scan,
formal K001 FA4 provenance/benchmark, end-to-end batch/throughput sweep, 1,000-step
canary, multi-GPU DDP/NCCL, and formal stage measurements remain pending. Current
one-GPU results must not be extrapolated to the four-GPU production gates.
