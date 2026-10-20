# CMuon missing-gradient P1: root cause and minimal repair

Baseline: `31d51cc225d77c1df9d8293b8450f7de2e92b6b8`. Remote dev was fetched and verified equal to this SHA before branching. Repair branch: `fix/cmuon-missing-gradient-index`. Production checkout, running training, checkpoints and dependencies were not modified.

## Frozen native reproduction

Original external evidence: `tmp/dcu-native-acceptance-31d51cc/p1-fix-evidence/frozen-original/`, containing exact command/log, both original rank reports, actual grad-present masks, full routing/chunks/owners and staged mappings. The original `missing-grad-loopback/` and `MISSING_GRAD_LOOPBACK_COMMAND.sh` are retained unchanged. `original-recheck/` independently reproduced the same contract assertion on last45 before testing the repair.

A real forward/backward has `K.grad is None`, with Q/V gradients present and finite:

| active staged index | actual parameter/chunk | actual shape | wrong baseline helper identity/shape |
| --- | --- | --- | --- |
| 0 | Q / 0 | 64 x 64 (4096 elements) | K / 0, 16 x 64 (1024 elements) |
| 1 | V / 0 | 16 x 64 (1024 elements) | Q / 0, 64 x 64 (4096 elements) |

Both ranks and both native builders (`build_guarded_canonical`, `build_fp32_rescue`) observe the error. The repro **asserts before calling the unsafe optimizer step**. It does not execute known-mismatched broadcasts, claim a hardware hang, or count earlier rendezvous timeouts as reproduction.

## Cause

Prepare skips each `grad=None` parameter, so staged indices are compressed. `_input_key` and `_chunk_shape` previously enumerated all routed chunks. A compressed index therefore referred to a different full-routing chunk. Different shapes can make the owner send and nonowner receive incompatible RCCL element counts; equal shapes still misattribute safety/reference identities. FP32 rescue inherited the same helpers.

## Repair

1. Shared base `_ActiveInput` is an ephemeral record of the original `CMuonChunkSpec`, chunk index, exact shape, canonical owner and `(FQN, chunk)` key. `_active_inputs()` filters CMuon parameters once per step in unchanged routing/chunk order. Both optimizer variants call this same method.
2. Every staged input has the corresponding activity record. Prepare, safety messages, BF16/FP32 rescue attribution, soft-rescue logging, forensic/capsule metadata and exact inputs, receiver allocation, phase-2 commit, and guard reference updates use that record or pass the same list to `_input_key`/`_chunk_shape`.
3. No-grad parameters have no staged update. Their parameter, momentum and reference remain unchanged; no zero gradient is injected. The table is neither cached between steps nor serialized.
4. The optional **parameter fingerprint still covers the entire stable routing**, including missing-gradient parameters and all-None steps. Active delta fingerprints cover precisely the shared active sequence. Repair does not weaken the original parameter invariant.
5. Empty active-CMuon steps skip only the empty NS failure collective; AdamW and the existing successful-step/observation semantics remain intact. All-None is tested separately from AdamW-only.

Stable owner hashing (`fnv1a64-v1`), complete routing identities, reference keys, aliases, parameter/SR traversal order, NS4 math and rescue limits, SR implementation, Guard Window semantics and checkpoint schema are unchanged. Full-gradient old/new physical-state comparison verifies this, rather than assuming it from the diff.

## Different rank masks: minimal fail-closed consensus

Without agreement, rank-dependent activity lists still cause different collective schedules even after local indexing is fixed. Both variants now call shared `_validate_gradient_presence()` before prepare can mutate momentum, counters, references or SR:

- one fixed-length **int32 SUM all_reduce**, one element per original CMuon parameter followed by original AdamW parameter;
- a count must be either 0 or world_size; any mixed count raises the same descriptive `RuntimeError` on every rank;
- AdamW is included because rank-dependent AdamW/SR advancement would also break replicated state;
- one `tolist()` readback after that collective; no per-chunk all_gather, second consensus, hashes/collision assumptions or GPU zero-gradient fabrication;
- world_size=1 has **zero extra collective/readback**; an empty complete routing needs none.

The fixed-length vector assumes the same full routing on all ranks, as required by the existing DDP/optimizer contract. It does not solve different model topologies, rank crashes, or arbitrary rank-local failures before entering `step()`.

Actual two-rank tests give both consistent masks (step succeeds) and inconsistent CMuon/condition-AdamW masks (both ranks raise after exactly one presence collective, before NS/broadcast/SR/state mutation). Performance costs are measured in `P1_NATIVE_RETEST.md`; distributed safety is not claimed free.

## Regression coverage and CI

New focused tests cover first/middle/last and consecutive missing parameters, Q/K/V, multi-chunk FFN/AdaLN, no CMuon/no gradients, changing masks, stable owner/reference/checkpoint identities, exact missing-state preservation and two-rank consistency. CPU/Gloo tests explicitly use CPU AdamW/SR boundaries; native acceptance never substitutes NS or SR.

Additional CPU-only fault injection checks exact exception identity, active FQN/chunk/shape, BF16/FP32 nonfinite and ceiling outcomes, soft rescue, legacy records, owner capsule metadata/SHA256/exact inputs, and fail-closed parameter/AdamW/SR/reference/observation behavior after missing/multichunk prefixes. Momentum is checked against prepared EMA, not falsely described as rolled back after kernel safety failure.

Existing test doubles were corrected to model SUM semantics and the shared distributed helper; production assertions were not relaxed. CI retains all 393 prior unit cases and adds the 58 new focused cases. Ruff remains required. Changed-file Pyright has pre-existing errors, so CI archives a pinned merge-base and compares full-message/path/rule/severity multisets: any new occurrence fails, including new files; raw reports and multiplicity are retained. This is **not a zero-error claim**. The pre-existing origin/dev merge-base behavior on direct dev pushes is not expanded into branch protection by this fix; dev/main are not pushed here.
