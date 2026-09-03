# iREPA Phase 3 — Gate4 Final Environment Parity (evidence addendum)

Status: ADDENDUM — evidence only. The Phase 3 functional commit is unchanged and remains immutable.
Date: 2026-09-03
Host: salt10 (2x BW DCU, DTK 26.04, system torch 2.9.0+das.opt1.dtk2604, Python 3.11.9)

## Verdict

```
IPREA_P3_TEST_PARITY = PASS
iREPA P3 = ZERO NEW FAILURES vs dev (iprea-only failures = 0)
```

## Gate4 final numbers (full 8-stage gate, both branches on the same salt10 venv)

| branch | passed | failed | skipped |
|--------|--------|--------|---------|
| iprea  | 926    | 2      | 2       |
| dev (BASE) | 786 | 2      | 2       |

- iprea-only failures = 0, base-only failures = 0
- Progression of the environment fixes: Gate2 41/39 failed → Gate3 36/34 → Gate4 2/2.

### Remaining failures (2, identical on both branches — pre-existing on dev, non-blocking)

1. `tests/gpu/data/test_pipeline_encoders.py::test_real_pipeline_qwen_and_mage_encode_one_batch`
   - Qwen model weights missing on salt10 (asset gap, not a code defect).
   - Decision: migrate the `model/qwen_3.5_2B/` asset from salt11 (G1 CONTROL); follow-up, non-blocking.

2. `tests/gpu/fa4/test_varlen_attention.py::test_forged_boundary_handle_fails_before_native_kernel[host_metadata]`
   - dev's own exception-contract bug: `src/sakuramoon/conditioning/packing.py` raises
     `ValueError("validated boundaries contain inconsistent host metadata")` where the test expects
     `TypeError` (capability rejection). Deterministic, reproduced on both branches.
   - Classification: **DEV BUG / NON-BLOCKING**. Fix separately on `fix/fa4-host-metadata-exception`
     branched from dev after Phase 3 acceptance and freeze; before the change, verify what the other
     packing API invalid-input cases contractually require (TypeError vs ValueError) and fix the
     correct side only — do not change the exception type just to satisfy the test.
     Then merge to dev normally; a future iREPA merge carries it in.

## Environment corrections (environment work only; no src/, config, or test changes)

The salt10 venv (`include-system-site-packages = true`, system DTK torch) had been rebuilt after a
pod rebuild. Two packages needed the DAS/DTK 26.04 builds instead of the PyPI versions:

1. flash_attn — installed the DAS build (missing entirely before):

   ```
   flash_attn-2.8.3+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl
   ```

   Source: Biren official mirror, category 4 (AI生态包) / `flash_attn/DAS1.8`.
   Effect: the 5 fa4 varlen kernel tests pass on both branches.

2. triton — replaced PyPI `triton 3.3.0` with the DAS build:

   ```
   triton-3.3.0+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl
   ```

   Source: Biren official mirror, category 4 (AI生态包) / `triton/DAS1.8`.
   Effect: the 32 torch.compile/Inductor optimizer tests pass on both branches
   (6 previously failing test families, 89 tests, all green in one run).

## Root cause (triton)

- The rebuilt venv inherited PyPI `triton 3.3.0` (pulled as a transitive dependency, e.g. via
  torchao). PyPI triton's AMD backend cannot codegen for the Biren DCU gfx target:
  Inductor precompilation crashed in the MLIR pass `ConvertTritonAMDGPUToLLVM`
  (`GPUTarget(backend='hip', arch='gfx936')`), so every `torch.compile(..., fullgraph=True)`
  optimizer test failed with `InductorError`.
- Reference environment (salt11, G1 CONTROL, torch.compile healthy) has no PyPI triton/torchao;
  the DTK torch Inductor works with the DAS triton build.
- Fix: swap in the DAS triton build matching `dtk2604 + torch290 + cp311`.
  Verified with a `torch.compile(fullgraph=True)` DCU smoke test and the full 6-family rerun.

## Standing rule for HCU (DCU) environment recovery (user-confirmed 2026-09-03)

- On DTK/DCU machines, `flash_attn` and `triton` must come from the DAS/DTK-matched builds
  (mirror category 4: `/flash_attn/<DAS version>/`, `/triton/<DAS version>/`;
  wheel tags encode `dtk<YYMM>.torch<XXX>-cp3XX`).
- PyPI triton must never be mixed into the venv: it silently hijacks Inductor codegen and breaks
  all compiled optimizer/training paths on DCU.
- When a rebuild pulls PyPI triton as a transitive dependency (torchao is the known source),
  uninstall it and install the DAS build for the matching DTK/torch/python triple.

## Environment backup (recovery artifacts, verified uploaded)

- NFS: `/root/private_data/sakuramoon/salt10-env-20260903/`
  (venv tar with DAS triton, both DAS wheels, profile.d patches, manifest + restore runbook)
- ModelScope: `leafmoone/docker_tmp` → `salt10-env-20260903/` (private repo)
- Verified: no API token material is contained in any backup artifact.
