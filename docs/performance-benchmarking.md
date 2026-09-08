# SakuraMoon canonical training-performance benchmarking (P0 observatory)

Status: **P0 — measurement infrastructure only.** This document defines how
the canonical logical-update performance baseline is produced on the
current production code. It is not a claim that the current system is
optimal; it is the fixed reference against which any later (P1)
optimization work may be compared.

## What is measured

The primary unit is **one logical update** of the production training
loop: `train.accumulation` microbatches of `train.local_batch` samples,
one backward/optimizer cycle — exactly the production
`SingleGpuTrainingLoop` update. One measured run is

- `--warmup-updates 5` warmup updates (model load, `torch.compile`
  compilation, first-allocation, lazy optimizer-state initialization are
  all excluded), followed by
- `--measure-updates 10` measured updates (the window that is reported).

The benchmark reuses the production components with zero production code
changes:

| component | source |
| --- | --- |
| DiT composite (config-driven, growth slots) | `config.assembly.build_trainable_composite_from_config` |
| objective + timestep sampling + encoders in the measure path | `train.runtime.SingleGpuBatchRuntime.measure` |
| update/step state machine (clip, optimizer, nonfinite handling) | `train.step.SingleGpuStep` / `train.loop.SingleGpuTrainingLoop` |
| optimizer (hybrid CMuon) + LR schedule | `train.production._build_optimizer` / `_SuccessfulUpdateLrScheduler` |
| DDP wrapping (2-GPU run) | `accelerate.Accelerator.prepare` with the production `DistributedDataParallelKwargs(find_unused_parameters=True)` |
| Qwen text encoder (2B, frozen) | `encoders.qwen.load_local_qwen` |
| Mage VAE (frozen) | `encoders.mage_vae.load_local_mage_vae` |

Data is the only non-production input: deterministic in-memory synthetic
batches built through the **real** caption serialization
(`data.serialize.serialize_caption`, real Qwen tokenizer, 34/5 framing
contract) and the **real** collator (`data.collate.collate_samples`), with
the canonical 256×256 bucket (256 image tokens per sample, the dominant
G1 bucket) and deterministic caption-density variation so multiple Qwen
dense-length buckets are exercised.

## Benchmark assumptions (documented, not hidden)

1. **Steady growth state.** `growth_alpha` is held at the ramp-complete
   steady value **1.0** for every update. The measured run represents a
   mature G1 run (growth finished), not a fresh ramp.
2. **In-memory data.** Batches are pregenerated before timing; the
   `data` phase measures an in-memory batch handoff (≈ 0 s). Production
   shard download/decode/prefetch is **not** measured and must not be
   inferred from these numbers.
3. **Per-rank determinism.** Rank `r` uses seed `seed + r * 1_000_003`
   for images, captions and the JLT noise generator
   (`torch.cuda.default_generators[local_index]`), so two runs on the
   same machine are bit-identical in inputs and the comparison is
   variance-only.
4. **1-GPU mode is a rank-local compute baseline.** The in-memory config
   copy derives `distributed = native/1` and
   `global_batch = local_batch * accumulation`. It measures what one
   rank computes; it is NOT the same effective training batch as the
   2-GPU run and must be labeled as such in every report.
5. **No persistence.** The loop runs with `cadence=None` and a no-op
   checkpoint callback: no model, state, metrics, W&B or publisher I/O
   ever happens.

## Running the benchmark

Environment (DTK/DCU host):

```bash
source /opt/dtk/env.sh
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
export FA_SO_PATH=<venv site-packages>/flash_attn_2_cuda*.so
PY=<venv>/bin/python
```

`FA_SO_PATH` must point at the DAS flash-attn 2 shared object (the DTK
runtime does not scan site-packages). Run the benchmark **from a
scratch directory**, e.g. `/sakuramoon-runtime/infra-bench/p0/<sha>/<timestamp>/` —
benchmark artifacts are never committed.

### RUN A — 1-GPU rank-local compute baseline

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/benchmark_training_perf.py \
    --config train_g1_cmuon_production.toml \
    --config-root config \
    --repository-root <worktree> \
    --output-root /sakuramoon-runtime/infra-bench/p0/<sha>/<ts>/run-a \
    --gpus 1 --label baseline-1gpu
```

### RUN B — 2-GPU real DDP (the canonical distributed baseline)

```bash
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 \
    --mixed_precision no --dynamo_backend no --main_process_port 29511 \
    scripts/benchmark_training_perf.py \
    --config train_g1_cmuon_production.toml \
    --config-root config \
    --repository-root <worktree> \
    --output-root /sakuramoon-runtime/infra-bench/p0/<sha>/<ts>/run-b \
    --gpus 2 --label baseline-2gpu
```

### RUN C — profiler snapshot (opt-in kernel view)

Add `--profiler` to either command (2-GPU preferred). The canonical
timing baseline (warmup 5 / measure 10) ALWAYS runs without a
torch.profiler context. The profiler is a SEPARATE scratch stage that
starts AFTER the normal measured baseline finishes and captures exactly
`--profiler-updates` logical updates (default **1**; 0 is rejected). It
does not append to the measured baseline and may advance scratch
model/optimizer state (no persistence anyway).

A cheap device-event probe runs before any capture and must yield at
least one real CUDA device event with positive device time; importing
torch.profiler alone never counts as available. In distributed mode,
device availability is probed ONCE per rank, reduced to one all-rank
decision (all_reduce MIN), and capture trusts that decision
(`availability_verified=True`) — it does not re-probe, so no rank can
independently skip the DDP workload after the consensus. If the
(single) decision reports unavailable, NO extra workload is executed
(a prior ungated full-workload capture in that state was observed to
hang the DTK runtime — CPU spin, device idle) and the record is
`device_trace_available = false` with empty operator rows and
`trace_path = null`; no kernel rows are invented.

## Outputs

```
<output-root>/
  samples-<label>-rank<r>.json     # strict per-update samples per rank
  <label>.json                     # aggregated summary (rank 0)
  <label>.md                       # human report
  runtime-fingerprint.json         # exact machine/software identity
  profiler-summary.json            # RUN C only
  profiler-trace-rank0.json        # RUN C, trace export (rank 0)
  diagnostics-rank<r>/             # loop diagnostics (empty unless a fault)
```

Summary conventions (exact, schema 2):

- **GLOBAL STEP** — `global_step_seconds` = the canonical mean / p50 /
  p90 / p95 / min / max / stddev over the **aligned per-update global
  walls**: all ranks are aligned by logical-update id (missing, duplicate
  or mismatched update identities fail closed), and
  `global_wall[u] = max over ranks of rank.wall[u]` (the slowest rank
  bounds the distributed logical update). This is NOT a pooled
  all-rank wall distribution and NOT max-of-means.
- **PER-RANK STEP** — `per_rank_step_seconds` = rank-local distributions
  (rank-local component cost, in aligned update order).
- **THROUGHPUT** — `global_samples_per_second`,
  `image_tokens_per_second`, `text_tokens_per_second` = actual summed
  cross-rank volumes over the measured window divided by the summed
  aligned global walls:
  `sum_u(global_volume[u]) / sum_u(global_wall[u])`. Ranks are NOT
  assumed to carry equal per-update sample/image/text counts.
- `rank_step_skew_pct` = `(slowest_rank_mean - fastest_rank_mean) /
  slowest_rank_mean * 100` over rank-local means.
- `phase_seconds[name]` = pooled rank-local phase statistics;
  `share_of_step` is defined only when the phase was measured in
  **every** sample (otherwise `null`; absent phases are never reported
  as fake 0.0 s) and its denominator is the **mean aligned global
  logical-update wall**. Phase means are pooled rank-local component
  costs — they are NOT claimed to decompose one exact global critical
  path when overlap exists.
- `dit_forward_matmul_tflops_per_second` = the exact DiT **forward
  matmul** FLOP counter (`ActualDitFlopCounter`) divided by the measured
  `dit_forward` phase seconds. It is NOT MFU and NOT a whole-step
  training FLOPS model. No other "TFLOPS"/"MFU" figure may be derived
  from this observatory.
- **MEMORY** — per rank: `peak_allocated_bytes` / `peak_reserved_bytes`
  = the max over measured updates of the TRUE per-update
  `torch.cuda.max_memory_allocated/reserved` counters. Each measured
  update's peak window is reset in the loop's `update_started` hook
  (immediately before that update) and read at the existing
  update-finalization boundary (no extra synchronization), so each
  sample's peak is the peak of exactly that logical update and warmup
  memory never enters measured peaks. `final_allocated_bytes` /
  `final_reserved_bytes` are the CURRENT post-update counters of the
  last measured update and are never called peaks.
  `max_across_ranks` reports the cross-rank maxima of the true peaks.

## BASE vs CANDIDATE comparison protocol

1. Same machine, same fingerprint fields (git sha, torch/DTK/flash
   versions, device, visible devices). Record the fingerprint delta.
2. Same config, seed, warmup/measure sizes, same run label layout.
3. Compare `global_step_seconds.p50` (primary) and `p95` (stability),
   per-phase `share_of_step` (where a candidate shifts time between
   phases), `rank_step_skew_pct`, peak memory, and the DiT forward
   matmul TFLOP/s.
4. A candidate claim requires the full summary pair (JSON), not a
   single-number delta.

## P0 boundary (what this observatory does NOT do)

- No optimization work, no custom kernels, no math changes to model /
  objective / optimizer / CMuon / DDP.
- No production data service, no real checkpoint, no W&B, no FID/eval.
- No "communication bottleneck" conclusions: with `find_unused_parameters`
  DDP overlap is not measured or characterized here
  (`overlap_claim = NOT_EVALUATED_IN_P0`).
- No committed artifacts: scratch roots only.

## Tests

- `tests/unit/perf/` — fingerprint, sample/summary semantics (aligned
  distributed global step, cross-rank volume sums, true peak memory,
  schema 2 round-trip), profiler probe/validator semantics, batch-shape
  override / sweep math (CPU).
- `tests/gpu/perf/test_benchmark_harness.py` — small-model, real-encoder
  two-stage harness smoke test, peak-vs-current memory proof,
  profiler-scratch-stage isolation, and an alternate batch shape (4x1
  vs 2x2) proving the same logical sample population through the
  production loop (skips when no DCU / no local assets).

## P1-R1A batch-shape sweep (microbatch / accumulation)

The 2-GPU runner accepts an IN-MEMORY benchmark-only batch shape:

```
--local-batch N --accumulation M --expected-global-batch 800
```

`benchmark_batch_shape_config()` derives the shape with
`model_copy` (world size, model, resolution, optimizer and LR rule
untouched); `global_batch` is re-derived as
`local_batch * accumulation * world_size` and the run FAILS CLOSED when
it differs from `--expected-global-batch`. No TOML is written and no
canonical config file is modified.

Identity guarantee: the synthetic population per logical update is
shape-invariant — update `u` always covers the contiguous identity range
`[u * 400, (u + 1) * 400)` per rank for every factor pair with the same
product (see `logical_update_identities()` and the unit tests).
Throughput is compared at the SAME effective global batch (800);
bitwise gradient equivalence across GEMM batch shapes is NOT claimed.

`scripts/benchmark_batch_shape_sweep.py` runs the canonical matrix
(A0 20x20, S 16x25, C1 25x16, C2 40x10 gated from C1's measured peaks at
<= 48 GiB allocated / <= 56 GiB reserved, A1 20x20 repeat) with EVERY
candidate in its own fresh process group (accelerate launch, bounded
timeout, per-shape scratch root).  The A0/A1 bracket (p50 drift <= 2%)
is the stability gate; the baseline reference is the MEDIAN of A0/A1,
never the faster of the two.  Outputs: `sweep-summary.json` +
`sweep-summary.md` with per-candidate status (PASS / OOM /
SKIPPED_SAFETY_GATE / TIMEOUT / ERROR — failures are never hidden),
whole-step p50/p95, global samples/s, phase means, true per-rank memory
peaks, rank skew, speed class (NEUTRAL < 2% <= USEFUL < 5% <= STRONG)
and memory class (SAFE <= 54 GiB allocated / <= 60 GiB reserved,
TIGHT beyond, UNSAFE on OOM / instability).  The sweep is a benchmark
recommendation only: it changes no production configuration.

Canonical P0 baseline it is compared against (separate session, DO NOT
replace with sweep runs): 1-GPU p50 15.777 s / 25.46 samples/s /
peak allocated 32.25 GiB; 2-GPU p50 15.636 s / 51.26 samples/s /
peak allocated 36.12 GiB per rank.
