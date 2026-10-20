# CMuon missing-gradient P1 — native retest

## Verdict and boundary

**The original missing-gradient index defect is repaired in the tested native scopes.** Healthy full-gradient numerical regression is bit-exact, inconsistent rank masks fail closed, real FP32 rescue is exercised, and artificial checkpoint restore/next-update parity passes. Production full-depth models/assets, production sharded resume/publication, cross-node training and a production performance SLA remain unverified. This report is not deployment approval.

Source baseline: `31d51cc225d77c1df9d8293b8450f7de2e92b6b8`. Candidate source is the ordinary commit containing this report on `fix/cmuon-missing-gradient-index`; final commit SHA, its own GitHub CI result, generated `P1_FIX_DIFF.patch`, and `P1_VERDICT.json` are external delivery records to avoid self-referential commit hashes. No dev/main merge, forced push, automatic merge, deployment, production train stop/restart or production checkpoint access.

## Native environment and isolation

Independent last45: `crdnotebook-2112366733908836354-last45-32851`, two BW Hygon DCUs, gfx936, 65520MiB/card. DTK26.04; driver6.3.31-V1.5.3.beta; Python3.11.9; PyTorch package2.9.0+das.opt1.dtk2604 (import2.9.0); HIP6.3.26093; RCCL2.22.3; TorchAO0.16.0; native Triton3.3.0+das.opt1.dtk2604.torch290. Earlier approved compatible venv was reused, without installing/upgrading dependencies. GPU-free preflight, memory/disk budget and process/device clients checked before each independent workload. Per-process allocator cap8GiB and thread count2; no production host was used for compute.

Exact commands, raw stdout/tracebacks and JSON are outside the source tree at `tmp/dcu-native-acceptance-31d51cc/p1-fix-evidence/` locally and `/root/dcu-native-acceptance-31d51cc/p1-fix-evidence/` remotely. Scripts contain no credentials. All synthetic artifacts and compiler caches stay in the isolated acceptance root, not production model/data paths.

## Original reproduction and activity matrix

`ORIGINAL_RECHECK_COMMAND.sh` reruns the untouched original repro against original source. Both rank reports again show K.grad=None, Q4096 mapped incorrectly to K1024 and V1024 mapped incorrectly to Q4096. Expected safety assertion occurs before the dangerous step. A first shell command had a quoting error before execution; preserved as bring-up history, not counted as source failure.

`NATIVE_SINGLE_MATRIX_COMMAND.sh` and `NATIVE_TWO_MATRIX_STRESS_COMMAND.sh` run actual native builders. Per optimizer variant, each rank executes 11 successive real updates:

1. all gradients;
2. first K missing;
3. middle Q missing (real differentiable Q/K/V projection paths);
4. last six-chunk AdaLN missing;
5. consecutive K/Q missing;
6. two-chunk FFN active with K/down projection missing;
7. no CMuon gradients, AdamW active;
8. no gradients at all;
9–11. changing nonempty masks and return to all gradients.

Both guarded canonical and canonical FP32-rescue variants pass on one DCU and two RCCL ranks. Per-step evidence compares active FQN/chunk/shape/owner to independent full-routing expectations; delegates real broadcast and records actual dtype/shape/numel; checks native NS calls/ownership, weights/momentum/references for missing parameters, quantized AdamW moment codes/qmap/scale, training RNG isolation, SR advancement, replicated state and owner-local counter conservation. No CPU NS, pass-through SR or zero-gradient injection is used.

Two-rank mismatch cases independently drop K or condition AdamW on rank1. Both ranks raise the same `RuntimeError`, exactly one fixed-shape int32 presence SUM is observed, and model/optimizer/momentum/reference/counters/RNG are byte-identical to pre-step. No delta broadcast, NS, AdamW or SR step is reached; no hang/timeouts are used as success criteria.

## Real BF16 and FP32 NS / rescue

`native-two-stress-mask/` runs a real finite extreme-state canary: only Q/V have CMuon gradients, K and other CMuon parameters are naturally absent from the loss. Finite BF16 maximum gradients and finite FP32 maximum momentum produce finite FP32 kernel inputs, but their BF16 cast and FP32 norm overflow. Production BF16 NS fails; real FP32 NS produces a safe **zero delta** accepted by the unchanged soft-rescue policy.

- Initial step: two BF16 safety failures, two actual `cmuon_zeroth_power_fp32` calls and two persisted rescues, one per owner (Q rank0, V rank1); zero hard failures.
- Native FP32 NS input/output dtype, active shapes/FQNs, rescued-delta BF16 broadcast element counts and counter conservation recorded; missing K momentum/reference remain unchanged.
- Post-rescue zero_grad/save/load/fresh native objects/normal backward/actual next-step comparison is byte-exact. Construction uses original finite calibration refs, then loads persisted overflow state unchanged; this extreme fixture can persist +inf signal references.
- Status is explicitly `PASS_REAL_DEGENERATE_SOFT_RESCUE`; **useful nonzero FP32 rescue remains UNVERIFIED**. This does not establish normal-magnitude numerical recovery or fitness of those extreme references.

The earlier `native-two-stress/` and archived harness remain unmodified: they exercise only leading K on owner0, one zero-delta rescue, with no post-rescue restore. Independent review caught the initial report's incorrect three-rescue/missing-K description; the final report binds the fresh missing-K run above instead of relabeling the old evidence.

This is genuine native kernel/branch execution, not monkeypatch NS, but degenerate zero recovery is a strict limitation. CPU fault-injection tests separately cover hard-failure artifact attribution and do not count as native rescue evidence. Full-depth production extreme states are not tested.

## Partition clipping, independent LR, Guard Window, restore

`NATIVE_CLIPPING_EXISTING_COMMAND.sh`: 12 additional real steps per variant/rank for allgrad/KNone/multi-chunk-other-missing with independent clipping enabled, both thresholds disabled, and each partition disabled in turn. Positive thresholds are measured condition norm/2 and rest norm/3; checks exact dtype-cast scaling, pre/post/coefficient identities, natural None preservation and unchanged native checked-step assertions. Condition LR factor0.37 is an acceptance fixture, not a training recommendation; physical AdamW order and locked group policy remain unchanged.

Each matrix variant also runs real Guard Window recording before/after an artificial checkpoint, merges actual rank files offline, verifies owner/completeness/counters, saves native model/optimizer/momentum/refs/quantized states/SR/training RNG, reconstructs real objects, and compares uninterrupted/restored next real update exactly. This is not a production sharded artifact/publisher or production model resume claim.

Unmodified applicable GPU regression tests rerun on the repaired source: **10 passed / 0 skipped**, 38.69s, including native 1000-step mixed-SR canary and next-step optimizer-state round-trip. TorchAO deprecation/C++ extension compatibility messages and existing RCCL NUMA/iommu warnings remain in logs; no sysctl/kernel or production environment changes were made to silence them.

## Healthy-path baseline versus repair

`NATIVE_HEALTHY_PAIR_COMMAND.sh` uses separate processes/import roots for old and new, identical generated inputs/model/optimizer/guard/RNG, both variants, world1 and2. Per run5 warmup +20 measured updates, no instrumentation in timing; identical explicit end fences. Untimed probes delegate real kernels/collectives solely to count and record calls. Comparison loads trusted saved physical-state snapshots: initial, measured and profiled states are **bit-exact** for every pair/rank, including quantized moments, momentum, references, counters and RNG. Checkpoint key/schema/format is unchanged; serialized file container bytes need not be identical.

Optimizer step wall ms (old -> repaired), individual rank results retained rather than hiding the slower rank:

| variant / ranks / rank | median ms | p90 ms | median change |
| --- | ---: | ---: | ---: |
| guarded /1/0 | 11.688 -> 11.676 | 11.754 -> 11.952 | -0.10% |
| rescue /1/0 | 13.134 -> 13.448 | 13.184 -> 13.735 | +2.39% |
| guarded /2/0 | 12.513 -> 13.015 | 13.484 -> 14.073 | +4.01% |
| guarded /2/1 | 12.687 -> 13.394 | 15.290 -> 14.540 | +5.57% |
| rescue /2/0 | 13.014 -> 13.166 | 13.888 -> 13.287 | +1.17% |
| rescue /2/1 | 13.026 -> 13.442 | 13.794 -> 13.632 | +3.19% |

Peak allocated bytes are exactly unchanged within every pair: world1 guarded67697152/rescue67691520, world2 both68017664. Healthy NS counts remain12/world1 and owner7+5/world2; SR and AdamW calls remain1. World1 adds0 collectives/0 readbacks. World2 adds exactly **one presence SUM and one Tensor.tolist entry** per step; 10 parameters make40bytes in this fixture. Full generated-update wall raw median/p90 and all samples are retained alongside optimizer timing. No per-chunk redundant consensus was added. GPU-to-CPU readback is necessary to raise synchronously before any mutation; it is not advertised as zero overhead. Python-entry fence counts are not a census of every C++/HIP implicit synchronization.

Results are sequential small-model measurements with noise, not a production performance promise. The distributed median penalty in these runs is0.15–0.71ms, not zero. No operator fusion, fallback, relaxed tolerance or alternate optimizer was introduced to improve the numbers.

## CPU, Gloo and static validation

GPU visibility masked before every local process. Same CPU environment and baseline/candidate deselections:

- baseline default executable CPU suite: **1905 passed /122 skipped /48 deselected**;
- final candidate: **1965 passed /122 skipped /48 deselected**;
- baseline CI unit collection:393 passed; final candidate: **451 passed** (all prior393 retained, new50 index/failure +8 real2-rank Gloo cases);
- two legacy distributed-adapter tests additionally cover restored full-routing fingerprint behavior; no legacy assertions relaxed;
- full tracked Python Ruff, config checks and shell syntax: PASS;
- changed-scope Pyright **496 ->467**, full-message/path/rule/severity multiset **0 new /29 removed**; new test files0 errors. This is historical typing debt, not zero total errors. CI uses a same-environment archived baseline and uploads both reports.

The complete suite still excludes the same real-teacher asset file, accelerator-required iREPA cases and historical discovery tree as earlier acceptance; it is not a claim that every file in the repository executes on CPU. Historical non-default-tree failures remain outside this repair; no unrelated fixes or new skip/xfail were added.

## Reproduce

After an idle-device preflight on last45, source `/opt/dtk/env.sh`; set thread counts2, `PYTHONPATH=<candidate>/src`, offline variables and dedicated TMPDIR/cache, select physical devices0 or0,1. Use the existing `native-venv/bin/python`:

```sh
# Existing exact commands/env/timeout wrappers are delivered externally.
python native_p1_retest.py --mode matrix --source fixed --source-root "$CANDIDATE" --out "$NEW_OUT"
python -m torch.distributed.run --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29743 --max-restarts=0 \
  native_p1_retest.py --ranks 2 --mode matrix --source fixed --source-root "$CANDIDATE" --out "$NEW_OUT"
# Same invocation with --mode stress or --mode clipping.
# Healthy baseline/fixed separately, then --mode compare; see delivered command.
```

Pause after this branch's normal push and its own GitHub CI. No auto-merge or deployment.
