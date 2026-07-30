# T053 Infra/performance review

Reviewer: `/root/t053_infra_rereview_2`

Verdict: PASS for the CPU harness/control plane and short 1GPU mechanics scope.

The review confirms:

- Candidate windows enforce exactly 100 warmup plus 500 measured updates; final
  windows require at least 1,000 measured updates.
- Runtime sample-ID and shape streams are hashed across warmup and measured updates
  and must match the workload identity, preventing iterator or shape drift.
- DiT forward, loss, backward, clip, optimizer, zero-grad, data, and checkpoint timing
  use real boundaries. CUDA phases use deferred events without per-phase synchronize.
- CUDA memory uses a measured-window peak interval. Host accounting covers the process
  tree using RSS high-water plus bounded pinned/swap sampling.
- The PyTorch profiler runs inside the measured loop; trace metrics and exact
  measured/successful-update ranges are derived from its Chrome trace.
- Any measured compile, recompile, or fallback increment fails before trace publication.
- Regional compile remains schema-disabled. Future enablement requires an explicit
  compile feature transition, a 4GPU workload, and hash-bound 4GPU correctness, DDP,
  and resume evidence.
- The measured window includes the update-1000 checkpoint cadence and requires a
  timed nonempty artifact. Off-cadence checkpoint claims are rejected.
- Publication uses unique temporary files, fsync, hard-link no-clobber publication,
  and streaming SHA verification.
- Nsight collector runs return unbound smoke artifacts only. Formal Nsys/NCU indexing
  remains fail-closed until a marker/range/kernel importer exists.

Verification: 49 targeted CPU/phase tests, 96 config/manifest regressions, and two
RTX 5090 short harness tests passed. Ruff, strict Pyright, traceability verification,
and `git diff --check` passed. A real installed Nsys smoke generated a 102,640-byte
`.nsys-rep`. NCU remains blocked by `ERR_NVGPUCTRPERM`; no report is claimed.

Real data/Qwen/VAE/DiT 100+500, final 24L/512 1,000 measured updates, formal Nsight
imports, 4GPU DDP/NCCL, and long-run conclusions remain pending or blocked.
