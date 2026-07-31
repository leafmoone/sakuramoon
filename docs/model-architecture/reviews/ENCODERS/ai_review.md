# Encoders/Conditioning package AI/model correctness review

Reviewer: independent agent `/root/encoders_package_review`

Scope: `T020-T024` CPU/one-GPU implementation, contracts, task evidence, and the
committed model/conditioning code at `c7b2e90a293aaaddb39015d0ab4c86f6b7c9af39`.
Unrelated in-progress Data and K001 review changes in the shared worktree were excluded.
Overall verdict: **CHANGES_REQUIRED**.

## Findings

1. **T024 does not bind the CUDA boundary values to the validated host lengths.**
   `ValidatedCuSeqlens` derives both fields together at
   `src/sakuramoon/conditioning/packing.py:25-56`, but its public `tensor` remains a
   mutable tensor. `fa4_varlen_attention()` checks host metadata plus only the device
   tensor's shape, dtype, device, and contiguity at
   `src/sakuramoon/model/attention.py:61-75`; it never checks the actual offsets.
   `PackedDiT._sample_indices()` independently trusts those device values at
   `src/sakuramoon/model/dit.py:496-507`. The four adversarial tests at
   `tests/gpu/fa4/test_varlen_attention.py:255-297` cover bad dtype, shape,
   contiguity, or host metadata, but not correct static metadata paired with wrong
   int32 CUDA contents.

   A focused one-GPU probe used host lengths `(1,1)` and a well-shaped contiguous
   device tensor `[0,2,2]`. The derived sample IDs were `[0,0]`, and a recorded native
   callable was invoked with `[0,2,2]`. A simpler `(2,)` / `[0,1]` probe also reached
   the callable. The same corruption can be produced by mutating the public tensor of
   a normally constructed handle, so this is not limited to `object.__new__`.
   Incorrect offsets can relabel conditioning tokens and remove the boundary between
   samples, violating `C05-002` and `C05-008`. T024 is **CHANGES_REQUIRED**.

   Minimum remediation: derive every value consumed by sample routing and FA4 from
   one canonical host-length/offset identity and do not trust an externally reachable
   mutable tensor as a second source of truth. Preserve a stream-ordered, no-D2H hot
   path. Add a CUDA negative contract with host `(1,1)` and device `[0,2,2]` that
   proves no native callable is reached and no token is assigned to the wrong sample;
   also cover post-construction tensor mutation.

2. **T021's seventh state is final-RMSNorm output, not the raw output after block 24.**
   The current decision and task require states after blocks `2/4/8/12/16/20/24`, but
   `src/sakuramoon/encoders/qwen.py:21-30,94-102` selects `hidden_states[24]`.
   In pinned `transformers==5.14.1`, decoder layers return their residual output at
   `modeling_qwen3_5.py:773-813`, the text model then applies final RMSNorm at
   `modeling_qwen3_5.py:1211-1225`, and `capture_outputs()` replaces the final captured
   layer state with `last_hidden_state` at `output_capturing.py:215-277`.

   A real local-Qwen RTX 5090 hook confirmed that SakuraMoon's selected seventh state
   equals the final-normalized output exactly, does not equal block 24's returned
   tensor, and has maximum absolute difference `25.15625` from that raw block output.
   The fake-state unit test at `tests/unit/encoders/test_qwen.py:14-45` fixes numeric
   tuple indices but cannot detect this upstream capture semantic. T021 is
   **CHANGES_REQUIRED**.

   Minimum remediation: either capture the actual block-24 return in the same forward
   and add a real semantic contract, or explicitly revise the current decision and
   trace identity to require the final-normalized state. The latter is an architecture
   decision, not a silent implementation interpretation.

3. **T020's claimed Mage source is not present in the R001 reference lock.**
   `src/sakuramoon/encoders/mage_vae.py:1-5` claims Microsoft Mage commit
   `8c94a0ac905167f40b05b09332b78752b7f9fbef`, while
   `docs/model-architecture/reviews/R001/reference_manifest.json:6-55` locks only HDM,
   JLT, and krea-2. The T020 task names the Microsoft checkpoint structure but records
   no repository/commit binding. Strict local weight loading and a real encode/decode
   prove compatibility with the prepared file, but they cannot supply the required
   fixed-upstream formula/implementation audit. T020 is **CHANGES_REQUIRED** for
   upstream evidence; this finding does not request importing or executing reference
   code.

   Minimum remediation: establish a governed, immutable Mage source/commit/license
   lock and statically compare encoder posterior-mean selection, zero-timestep
   one-step encode/decode, normalization, attention padding, and checkpoint key
   mapping. If the claimed commit is not the intended source, correct the claim rather
   than auditing an unverified identifier.

## Per-task verdicts

| Task | AI/model verdict | Evidence boundary |
|---|---|---|
| T020 | CHANGES_REQUIRED | Local-only frozen posterior-mean encode/decode and the evaluator contracts pass, but the explicit Mage upstream audit is not reproducible. The production LPIPS/SSIM engine, 2,000-image acceptance, manual labels, and 50k-100k latent statistics remain pending. |
| T021 | CHANGES_REQUIRED | Local-only load, one forward, bool mask, freeze, and seven outputs pass, but state 24 has final-norm rather than the specified after-block semantics. |
| T022 | PASS | Structured main gather, seven independent FP32 norms, shared projection, per-token eight-group layer mixing, deepest anchor, bidirectional padding isolation, and detached Qwen boundary are correct. Undecided heads/init/bias values remain explicit and unbound. |
| T023 | PASS | Artist-only active gather, seven-layer construction, four queries, residual SwiGLU, active-only projection, and all null routes are correct. Resolved-TOML composite binding remains T050 scope. |
| T024 | CHANGES_REQUIRED | Packing order, coordinate formula, Q/K normalization, `32/48/48` shared-frequency RoPE, and native KV heads pass, but host/device boundary disagreement can corrupt sample routing and FA4 isolation. |

## Upstream algorithm/contract comparison

| Planned formula or contract | Fixed upstream source and lines | SakuraMoon implementation | Golden or contract test |
|---|---|---|---|
| Mage posterior mean, 128 channels at H/16 x W/16 | **Blocked:** no Mage entry exists in the R001 lock; only an unverified commit claim appears in the local docstring | `mage_vae.py:441-504` selects the first 128 moment channels; frozen shape enforcement is at `mage_vae.py:529-557` | `tests/unit/encoders/test_mage_vae.py`; real RTX 5090 BF16 `[1,3,32,32] -> [1,128,2,2]` probe |
| Mage decoder round trip without extra DiT patchification | **Blocked:** the claimed Mage commit cannot be statically read through the governed reference lock | `mage_vae.py:506-520` reconstructs Hx16/Wx16 directly from latent input | Frozen wrapper unit contract and real RTX 5090 BF16 `[1,128,2,2] -> [1,3,32,32]` probe |
| Qwen states after blocks 2/4/8/12/16/20/24 | Pinned dependency source: `modeling_qwen3_5.py:773-813,1211-1225`; output capture replacement: `output_capturing.py:215-277` | `qwen.py:21-30,94-102` selects indices `2/4/8/12/16/20/24`; index 24 resolves to final RMSNorm | Existing fake tuple-index test passes; real local-Qwen hook disproves the block-24 semantic |
| T022/T023/T024 conditioning, packing, and RoPE contracts | No R001-locked external algorithm source is assigned; current SakuraMoon decisions are authoritative | `conditioning/text_mixer.py`, `style_resampler.py`, `packing.py`, and `rope.py` | Targeted CPU/GPU conditioning, packing/RoPE, and FA4 dense-alignment contracts |

## Independent validation and boundaries

- Targeted CPU package plus collate/composite boundary: `122 passed` in `16.47s`.
- Targeted RTX 5090 Qwen, text/style forward-backward, packing/RoPE, real FA4 dense
  alignment/isolation, and existing forged-static-metadata cases: `9 passed` in
  `22.88s`.
- Real Mage RTX 5090 BF16 encode/decode: finite `[1,128,2,2]` latent and finite
  `[1,3,32,32]` reconstruction, both detached and eval-only.
- Targeted Ruff passed; targeted strict Pyright reported 0 errors and 0 warnings.
- Static production/test search found no import or execution of `reference/`.

No 2,000-image quality run, 50k-100k latent scan, 17x8 milestone rerun, training long
run, DDP, NCCL, multi-GPU validation, or formal stage canary was executed. One-GPU
evidence does not close any four-GPU gate.
