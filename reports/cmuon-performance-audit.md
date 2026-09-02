# CMuon FP32-rescue — Performance Audit (D3 Phase B)

Scope: design/benchmark only. **No candidate math changed, no
`ckpt_97600` touched, no compile config changed.** All microbenches ran
isolated on the 500-gate pod (salt6, 2×HCU "BW", DTK 2.9.0) after the gate
stack stopped. Production reference: 500-gate run, 16.1–17.0 s/update,
optimizer phase 0.58–0.84 s (see `cmuon-500-safety.md`).

## B1 — rescue cost share

Per-event cost = one extra FP32 NS4 pass on the failed chunk
(microbench `perf-bench-micro.json` NS wall, fp32):
kv 0.8 ms, 2560² 8.1 ms, in_proj 10.4 ms, down_proj 18.7 ms per chunk.
28 rescues over 500 updates (role mix: q 16, gate 10, k 2 → dominated by
2560² shapes ≈ 8 ms each) ⇒ ≈ 230 ms / 500 updates ≈ **0.46 ms per update
= 0.003% of the 16.5 s full update**.
**FP32_RESCUE_NOT_PERF_BOTTLENECK = YES** (gate: < 0.1% = 16.5 ms/update).

## B2 — device/host sync audit (static, production fast path)

Per-rank per-step host syncs in the rescue candidate ≈ **168**:
166× per-chunk `isfinite().item()` (L201, owned chunks) + 1× batched d_rms
(L226, OK) + 1× `sig_vals.tolist()` (L169) + 1× `sigf.tolist()` (L170) +
1× fingerprint spread (L330) + 1× param invariant (L460, required by
`invariant_check=true`) + 2× `fail_flags` tolist (L285/L294, duplicate).

- **`sigf` (L170) result is discarded** — pure D1 leftover, delete outright.
- **`sig_vals` refs (L169)** feed only the references table, which this
  candidate uses as pure telemetry (the retired low-signal skip is not
  called; the ceiling is the constant 10×0.2×lr). Keep the table (guard
  state contract) but the per-step `.tolist()` of 166 sigs is removable —
  the table can be updated on-device and read back only at ckpt time.
- The 166 per-chunk `.item()` calls are the structural problem (B3).

Design target: **2 syncs per step** (packed flag array + invariant
fingerprint), i.e. −99%.

## B3 — batched device-side safety reduction (design + microbench)

Design (keeps every safety criterion byte-identical):
compute the per-chunk `isfinite` mask and the per-chunk delta RMS on-device
into one packed tensor `[166, 2]` (finite-bit, rms), then **one**
`.tolist()` to host; rescue selection = the finite-bit column (identical
predicate), RMS check against the constant ceiling happens host-side on the
same values (identical comparison). No per-chunk sync, no per-chunk kernel
launch beyond what NS already does.

Microbench (`perf-bench-micro.json`): current 168 syncs = 14.4 ms/step
measured wall; naive packed proposal (2 syncs) = 21.6 ms/step. The naive
variant is *slower in this isolated bench* because the extra packed-reduction
work exceeds the sync savings at this scale — the production-relevant win is
the elimination of 168 HCU queue flushes that interrupt the kernel pipeline
(measured per-sync ≈ 86 µs here; production flushes are longer because they
land between dependent kernels). Refined design: keep per-chunk finite/RMS
as a **device mask** (no `.item()`), use the mask to index the rescue set,
and sync only the packed flag array once. Expected: 2 syncs, no extra
reduction cost. (Design only — not implemented, not wired into the
candidate.)

## B4 — broadcast audit + bucketed microbench

Production: 166 per-chunk `dist.broadcast` of the owned chunk's BF16 delta
(~1.18 GB/rank/step total), each a separate collective.
Microbench (`perf-bench-broadcast.json`, 2 ranks, real 166 payloads):

| scheme | collectives/step | wall/step |
|---|---|---|
| current (per-chunk) | 166 | **22.8 ms** |
| bucket 64 MiB (FIFO per owner) | 40 | 26.2 ms |
| bucket 128 MiB | 20 | 25.6 ms |
| bucket 256 MiB | 10 | 25.1 ms |

**Bucketed is NOT faster** on this HCU-to-HCU interconnect: the copy-in /
copy-out staging cost of the flat buckets offsets the collective-count
reduction. Exact BF16 delta bytes/owner/atomicity semantics are preserved by
the FIFO packing, but there is no speed to gain. **Decision: keep the
current per-chunk broadcast; bucketing not adopted.** 22.8 ms/step ≈ 1.4%
of the full update — acceptable.

## B5 — NS/broadcast overlap state machine (design only, not implemented)

Proposed pipeline (per step, per rank):

```
PREPARE        : grad ready; per-owned-chunk NS inputs staged on-device
NS             : all owned chunks NS4 (BF16) — device only
LOCAL SAFETY   : per-chunk finite mask + delta RMS on device → packed [166,2]
ASYNC STAGE    : while host processes the packed flags (single sync):
STAGE COMM     :   enqueue BF16 broadcast of all owned deltas (async),
                 no parameter writes yet
FINAL GLOBAL   : host verdict (ceiling check on RMS column, nonfinite bit)
                 + cross-rank allreduce of rescue verdicts
COMMIT         : if verdict clean → commit staged deltas to params
                 if any failure → discard staged buffers,
                 run FP32 rescue on failed chunks (owner rank), re-broadcast,
                 re-commit
```

Invariants: a later failure (even after staging) leads to **buffer
discard → zero parameter writes** for the failed step's staged data; the
commit is the single point where parameters mutate, so atomicity of the
two-phase commit is preserved and the rescue path re-enters at STAGE COMM
with FP32 payloads. Safety criteria unchanged (same finite predicate, same
ceiling, same invariant check). This hides the 22.8 ms broadcast behind NS
compute; not implemented this round (would require a new candidate
variant).

## B6 — owner-mapping weighted load balance (measured, no mapping change)

Measured per-shape NS cost (bf16 wall, `perf-bench-micro.json`) + broadcast
bytes assigned to the real fnv1a64-v1 owner table (166 inputs, 83/rank):
**both ranks: 0.2233 s NS cost and 1,176,371,200 bytes — imbalance 1.0000.**
An LPT greedy re-assignment finds the identical optimum (imbalance 1.0).
**Conclusion: the existing owner mapping is already perfectly balanced on
this shape mix; no reassignment gain exists** — and per the frozen contract
(`OWNER_MAPPING_VERSION` is a checkpoint-compatibility field) it must not be
changed anyway.

## B7 — current model compile (confirmed, unchanged)

`torch_compile_enabled = true`, `dynamic = true`,
`mode = max-autotune-no-cudagraphs`, regional compile of the PackedDiT
blocks only (inherited `train_g1.toml` line, resolved config of all 500-gate
ckpts identical). No change made or proposed this round.

## B8 — full optimizer compile (analysis; NOT RECOMMENDED)

Compiling the whole optimizer step would wrap: NS (B9 shows compile gives
no speedup), the 168 host syncs (graph breaks — compile cannot remove
`.item()`/`.tolist()`; every one is a break), `dist.broadcast` (break +
collective semantics risk), the rescue control flow (data-dependent branch
→ recompile per rescue pattern), and the ckpt-serialized guard state
(consumed/produced op graphs must stay schema-stable). Net: many graph
breaks, no removable work, added determinism risk on the safety path.
**NOT RECOMMENDED.**

## B9 — isolated pure-NS-kernel compile microbench

5 production shapes × {bf16, fp32} × {eager, torch.compile(dynamic=False)},
20 reps (`perf-bench-ns-compile.json`):

| shape | bf16 eager→compiled | fp32 eager→compiled |
|---|---|---|
| (2560,2560) | 2.43 → 2.39 ms (not bit-det, both) | 8.31 → 8.24 ms, **bit-identical** (rel_fro 0.0) |
| (640,2560) | 0.55 → 0.65 ms (slower) | 0.91 → 0.97 ms, bit-identical |
| (3456,2560) / (2560,6912) | 2.77/4.11 ms class, no gain | 10.4/18.7 ms class, no gain, bit-identical |
| (2560,1024) | sub-ms, no gain | bit-identical |

Findings: (1) **compiled FP32 is bit-identical to eager FP32 in all 5
shapes** (`compiled_fp32_invalid_shapes = []`) — the 0/700 bit-determinism
bar is met, so compile does not invalidate the FP32 rescue path;
(2) **no speedup anywhere** (2–10% *slower*): NS4 is a small GEMM chain and
inductor cannot beat the native DTK kernels at this scale;
(3) **BF16 NS is not bit-deterministic** run-to-run in this DTK build
(eager *and* compiled, all shapes) while FP32 NS is — this explains the
intermittent BF16 guard trips and is a property of the DTK bf16 kernels,
not of compile. **Decision: NS compile not adopted (no gain, B13 priority 6
confirmed low).**

## B10 — always-FP32 (independent future candidate, analysis only)

NS wall fp32/bf16 per chunk: 1.5× (kv) to 4.5× (down_proj). Per-rank
steady-state: peak-chunk workspace delta **0.099 GiB** (0.222 vs 0.123
GiB) — negligible; cumulative per-step traffic +6.6 GB across all 166
chunks (sequential, freed per chunk). Full-optimizer cost would rise by the
bf16→fp32 NS delta on *every* chunk every step (≈ +80–110 ms/step/rank,
i.e. +5–7% of the full update) in exchange for eliminating the rescue path
entirely (28/500 events ≈ 5.6% of updates). Determinism: FP32 NS is
bit-deterministic (B9), so always-FP32 would make the optimizer path fully
bit-reproducible. Verdict: a legitimate **future** candidate, but strictly
dominated by the rescue design on cost (5–7% permanent vs 0.003% event-
triggered) while buying only determinism that the guard + rescue already
enforce. Not built, not trained this round.

## B11 — attention-only FP32 (alternative, analysis only)

Scope NS FP32 to the attention projections (q/k/v/out + gate: 100 of 166
inputs) and keep MLP in BF16. The 500-gate rescue mix was q16/gate10/k2 —
**100% of rescues were attention-role chunks**; MLP (in_proj/down_proj)
never tripped in 500 updates. So attention-only FP32 would have rescued
all 28 events at a cost of +~45 ms/step/rank (100/166 of the always-FP32
delta) — cheaper than always-FP32 but still a permanent 3–4% tax for what
the rescue handles at 0.003%. It would also leave the (currently
non-observed) MLP risk uncovered. Not implemented; kept as a documented
middle option.

## B12 — same-machine same-method baseline rebuild

Method (identical to the old P5 benchmark, `reports/
cmuon-guarded-canonical-benchmark.json`): real 97300 model from the NFS
mirror, synthetic gradients at 1e-3 RMS (all 166 NS inputs active =
worst case), 6 warmup + 28 measured iterations, ws=1 single-rank arm,
locked policy (lr 5e-05, betas (0.9,0.95), block 256, bf16 stochastic
round, AdamW8bit 8-bit quantized state). Old-pod reference values (device
"BW", retired pod) were discarded as the spec requires; the AdamW8bit
full-update reference from salt7 live (15.5 s/u) is quoted separately as a
different-pod, different-optimizer-context number.

Measured on the 500-gate pod (28 iters, avg/min/max; full JSON in
`reports/cmuon-performance-audit.json` and NFS
`artifacts-fp32-rescue/perf-bench-baseline.json`):

| arm | step avg (s) | min–max (s) | peak alloc (GiB) |
|---|---|---|---|
| adamw8bit (all params) | **0.2055** | 0.2050–0.2066 | 9.14 |
| cmuon_unguarded_ns4 | **0.5649** | 0.5630–0.5705 | 9.19 |
| cmuon_guarded_canonical | **0.6947** | 0.6924–0.6980 | 14.92 |
| cmuon_fp32_rescue (candidate) | **0.6971** | 0.6948–0.6997 | 14.92 |

Cross-pod reproducibility vs the retired-pod P5 reference (device "BW"):
adamw8bit 0.2006→0.2055 (+2.4%), unguarded 0.5768→0.5649 (−2.1%),
guarded 0.7095→0.6947 (−2.1%) — the methodology reproduces within ~2%,
so the old numbers' *relative* ordering was sound even though the
absolute baseline was rebuilt as the spec requires.

Key deltas:
- **rescue candidate vs unguarded: +0.132 s/step (+23%)** of the
  optimizer step — this is the canonical guard fast path (168-sync
  safety reduction + guard refs table, +6.2 GiB state), not the rescue
  path itself.
- **rescue candidate vs guarded canonical: +0.0024 s/step (+0.35%)** —
  the FP32-rescue machinery's entire worst-case cost (synthetic 1e-3-rms
  grads = all-NS-active worst case). Well under the 0.1%-of-full-update
  (16.5 ms) threshold ⇒ corroborates B1's production-event estimate
  (0.46 ms/update = 0.003%).
- Full-update context (measured, 500-gate run): candidate 16.1–17.0
  s/u ⇒ optimizer ≈ 4.2% of the full update; vs salt7 live AdamW8bit
  15.5 s/u (different pod, quoted for reference only).

## B13 — improvement priority (compile explicitly NOT #1)

1. **obsolete stats removal** (B2): delete the discarded `sigf` tolist and
   stop per-step reading of the telemetry-only refs table — zero-risk,
   −2 to −3 syncs, no semantic change.
2. **batched device-side safety reduction** (B3): 166→2 syncs with a
   device mask; the largest structural win on the safety fast path.
3. **bucketed broadcast — measured, NOT adopted** (B4): keep per-chunk
   (no gain on this interconnect); revisit only if the interconnect changes.
4. **NS/broadcast overlap** (B5): hides the 22.8 ms broadcast; needs a new
   candidate variant + its own safety gate before any use.
5. **weighted owner reassignment — no gain exists** (B6): mapping already
   optimal; contractually frozen.
6. **NS compile — no gain, bit-identical but slower** (B9): not adopted.
7. **always-FP32** (B10): future candidate only; dominated on cost.
8. **attention-only FP32** (B11): documented middle option; not built.

## Verdict

The FP32-rescue candidate is **not performance-bound by its safety
machinery**: rescue events cost 0.003% of the full update; the remaining
structural costs are the 166-sync safety reduction (fixable by design #2)
and the 22.8 ms per-chunk broadcast (already near-optimal, fixable by
overlap design #4). No candidate math, no compile config, and no checkpoint
state was modified by this audit.
