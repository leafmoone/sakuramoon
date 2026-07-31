# Encoders/Conditioning package Infra/performance review

Reviewer: independent agent `/root/encoders_package_review`

Scope: `T020-T024` CPU/one-GPU durability, synchronization, boundedness, and current
engineering evidence at `c7b2e90a293aaaddb39015d0ab4c86f6b7c9af39`.
Overall verdict: **CHANGES_REQUIRED**.

## Finding

1. **T024's no-D2H remediation leaves two inconsistent runtime sources of sequence
   boundaries.** The host tuple owns `batch_size`, `max_seqlen`, and total-token checks,
   while a separately mutable CUDA tensor owns `PackedDiT._sample_indices()` and the
   native FA4 boundaries. `src/sakuramoon/model/attention.py:61-75` validates only
   static tensor metadata. A correct-shape, contiguous int32 CUDA value mismatch is
   therefore invisible, and `model/dit.py:496-507` can assign a sample's tokens to a
   different condition before the same corrupted values reach FA4.

   The focused RTX 5090 probe observed host `(1,1)`, CUDA `[0,2,2]`, derived sample IDs
   `[0,0]`, and one native callable invocation. Existing normal-input FA4 versus dense
   alignment and cross-sample-isolation tests still pass; they prove the kernel when
   boundaries are correct, not integrity of the production boundary handoff. T024 is
   **CHANGES_REQUIRED**.

   Remediation must retain the current no-host-synchronization goal. Use one canonical
   host-derived identity for routing and native offsets, add the well-shaped CUDA
   mismatch and post-construction mutation tests, rerun real FA4 isolation, and measure
   any extra boundary materialization/check launch in a multi-block small-shape smoke.
   A per-layer `.item()`, `.tolist()`, or unconditional D2H validation is not an
   acceptable fix.

## Per-task verdicts

| Task | Infra/performance verdict | Evidence boundary |
|---|---|---|
| T020 | PASS | Local-only strict load and bounded one-GPU encode/decode pass. Evaluation-time finite checks and metric D2H extraction are outside the training hot path; the 2,000 scalar observations are bounded. Same-directory temp, file fsync, `os.replace`, cleanup, and prior-report preservation satisfy the regenerable-report protocol. Production metrics, 2,000-image cost, and 50k-100k latent scan remain pending. |
| T021 | PASS | The frozen local model uses one forward, no cache, no visual module, no download/fallback, and a bool mask without conversion. Existing batch-1 evidence reports 34.7-38.2 ms steady forward and 3,700.4 MiB peak allocation. Any T021 correctness remediation must remeasure selected-state retention but does not invalidate the current runtime-boundary result. |
| T022 | PASS | Fixed-shape range predicates, asynchronous CUDA assertion, safe gather, and masked output avoid tensor-to-host scalar sync. Production-shape BF16 forward/backward passed synchronization debug mode; undecided constructor values are not performance defaults. |
| T023 | PASS | Host-derived active sample IDs drive fixed-shape `index_select`/`index_copy`; null samples skip projection without reading a CUDA value into Python. Mixed active/null BF16 forward/backward and synchronization-debug evidence pass. |
| T024 | CHANGES_REQUIRED | Normal packing/RoPE and real FA4 are bounded and avoid D2H, but mutable device contents can disagree with host lengths and break routing/isolation. |

## Independent validation and boundaries

- Targeted CPU package plus collate/composite boundary: `122 passed` in `16.47s`.
- Targeted RTX 5090 suite: `9 passed` in `22.88s`, including real Qwen, text/style
  forward-backward, CUDA packing/RoPE, and real FA4 dense alignment/isolation.
- Real Mage 32x32 BF16 encode/decode completed with finite outputs and no gradients.
- Targeted Ruff passed; targeted strict Pyright reported 0 errors and 0 warnings.
- The T024 negative probes used a recorded native callable for safety; they do not
  substitute for the separate real-FA4 correctness run.

The retained all-17-shape evidence was not rerun. No production metric/latent scan,
training long run, DDP, NCCL, multi-GPU test, 1,000-step canary, or formal stage canary
was run. Current one-GPU timing and memory evidence must not be extrapolated to four
GPUs or used to close production stage gates.
