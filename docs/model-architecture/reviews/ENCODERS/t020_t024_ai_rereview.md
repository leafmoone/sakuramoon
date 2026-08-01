# Encoders/Conditioning T020-T024 remediation AI/model rereview

Reviewer: independent agent `/root/encoders_remediation_rereview`

Scope: the committed T020-T024 implementation and remediation at repository HEAD
`4143a60`, including remediation commits `1d7b51a`, `1b4fd87`, `cff29ac`,
`cdbbca1`, `80525f2`, and `249e87d`. The earlier package report remains the
historical record of the findings that triggered these changes. Uncommitted DATA,
K001, and M037 work from other sessions was excluded.

Overall CPU/one-GPU implementation verdict: **PASS**.

## Remediation conclusions

| Task | Verdict | Independent conclusion |
|---|---|---|
| T020 | **PASS** | The governed lock identifies the official Microsoft Mage repository at immutable commit `8c94a0ac905167f40b05b09332b78752b7f9fbef`, Git tree `73288529688298fc2934707d6b8bb39071810dc1`, source SHA-256 `64f4d7041003e416bc2f4fac5bbf8aabf2e7c798ad106682c34332ba347b0ef9`, and MIT license SHA-256 `275b4dd619de4e16a017b10d0beec72abbbbf14ee8a2fc68f8bdb398e821f623`. Independent official-remote checks matched every value. Static comparison covers checkpoint prefixes, posterior-mean selection, zero-timestep encode/decode, replicate-padded attention, and 128-channel H/16 x W/16 geometry. Commit `1d7b51a` did not modify R001 task or historical evidence. Current real BF16 encode/decode was finite, frozen, detached, and shape-correct. |
| T021 | **PASS** | `FrozenQwenEncoder.forward()` installs one temporary hook on decoder block 24 around the same single model forward, retains tuple indices 2/4/8/12/16/20, substitutes the raw block-24 return as state seven, and removes the hook in `finally`. The real RTX 5090 contract proves state seven equals final RMSNorm input, is not final RMSNorm output, and differs from that output by max abs `25.15625`. Boolean-mask, no-cache, frozen, local-only, no-visual-path, and fast-kernel hard-failure contracts remain intact. |
| T022 | **PASS** | Structured main-token gather, seven independent FP32-accumulating norms, one shared 2048-to-1024 projection, per-token eight-group seven-layer softmax mixing, deepest residual anchor, and non-causal attention-only refinement match the current decision. CPU collate validates routing values; CUDA uses fixed-shape predicates plus a device-side asynchronous assertion and performs no tensor-to-host scalar read. Masked placeholders are sanitized before gather and padded outputs remain zero. |
| T023 | **PASS** | Only host-planned active Artist samples reach gather/projection. The CUDA path validates that fixed-shape plan asynchronously and uses `index_select`/`index_copy`; null-routed samples receive four learned null tokens without dynamic CUDA boolean compression or Python tensor-value branches. The same Qwen states are detached, four slots remain valid and independently query-conditioned, and all missing/dropout routes are correct. |
| T024 | **PASS** | Modality, packing, and RoPE remain separate. Packing removes text padding, preserves `[text|4 style|image]`, emits exact spans and host-derived lengths, and checks the CUDA prefix mask without dynamic boolean indexing. PackedDiT performs exactly one content comparison between the public CUDA offsets and canonical host lengths at entry, rejects well-shaped forged `[0,3,4]` and post-construction mutation, then rematerializes private offsets. Sample routing is derived from the same accepted host-length tuple used to create FA4 offsets. Every block reuses that accepted handle; no block performs D2H. Real FA4 dense alignment, cross-sample isolation, all-parameter gradients/update comparison, and 16-block PackedDiT forward/backward pass. |

No new AI/model correctness finding requires code changes.

## Independent validation

- Targeted CPU package and adjacent collate/composite/model contracts:
  `134 passed, 17 warnings in 14.92s`.
- Targeted real RTX 5090/driver 580.105.08 suite: `23 passed, 18 warnings in
  37.32s`. It covered a real local Qwen/Mage pipeline batch, raw block 24,
  T022/T023 forward/backward under CUDA synchronization debug mode, CUDA packing/RoPE,
  forged and mutated boundary rejection, real FA4, and PackedDiT.
- Independent real Mage BF16 round trip: latent `[1,128,2,2]`, reconstruction
  `[1,3,32,32]`, finite, detached, and eval-only.
- Targeted Ruff passed. Targeted strict Pyright reported `0 errors, 0 warnings`.
- Static production scan found the one intentional packed-entry `.to(device="cpu")`
  and no boundary `.item()`/`.tolist()` or block/attention D2H reuse.
- Official remote Mage commit/tree/source/license checks matched the governed lock;
  no upstream or `reference/` code was imported or executed.

The live trace contract run reached `35 passed, 5 failed in 77.13s`; every failure was
caused by concurrent, uncommitted K001 provenance edits outside this review's allowed
paths (`C05-003`, `C06-006`, `OPEN-043`, and registry history revision 100). The direct
verifier reported only the same K001 draft errors and no T020-T024 mapping error. The
main agent must rerun live trace after that concurrent task settles before publishing
package closure.

## Remaining boundaries

T020's production LPIPS/SSIM engine identity, actual 2,000-image metric/manual
acceptance, and 50k-100k latent statistics remain pending. T022/T023 unresolved
constructor choices remain explicit and production resolved-config binding remains
T050 scope. T024's accepted-entry timing is engineering evidence only; K001 owns formal
FA4 provenance/performance. No long run, 1,000-step canary, formal stage, DDP/NCCL, or
multi-GPU validation was executed, and this one-GPU PASS closes none of those gates.
