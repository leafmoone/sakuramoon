# S000 production single-GPU readiness integration

This extension adds the fail-closed production lifecycle without changing the
historical bounded engineering-smoke evidence. The public train CLI now loads the
strict resolved S0 config, accepts only fresh start or one exact absolute raw
`COMPLETE` directory, completes static topology/storage/asset/logging/checkpoint/
evaluator-identity preflight, and only then resolves the remaining governed runtime
bindings. It never selects `latest`, PMA, model-only state, a different topology, or a
configuration fallback.

The accepted lifecycle connects the existing data-service client and production batch
factory to local Qwen, Mage-VAE, the 16-layer DiT composite, strict JLT loss,
backward, FP32 clipping, TorchAO AdamW8bit, successful-update scheduling, T051
telemetry, and T044 raw checkpoint publication. Checkpoint resume restores the exact
raw training/optimizer/RNG/config identity before the data-service connection. Update
wall time is frozen at the optimizer boundary so scheduler, checkpoint and telemetry
publication are excluded from the update metric.

The storage preflight now binds capacity to the larger of the restored raw payload and
the directly evidenced 5,143,061,370-byte S000 raw checkpoint. This prevents a smaller
resume artifact from reducing the governed three-checkpoint reservation.

## Bounded acceptance

Coordinator-only RTX 5090 validation executed the real Qwen/VAE/16L DiT/loss/
backward/clip/update chain. A second bounded test ran the real data-service and
multi-process DataLoader, mandatory preflight, update 1, raw `COMPLETE` publication,
and a fresh spawned process that restored update 1 and completed update 2. The latter
artifact is retained under
`artifacts/s000-gpu-main-20260802-003/`; its fresh-result SHA-256 is
`fe30ae860580b53b6426c40d415b333cf9ff86b8f74ed2b6405f3a243093ed85`.

This remains synthetic bounded engineering evidence. It is not S001, a capacity
sweep, a throughput result, a quality result, or a formal stage canary.

## Readiness blockers

`config/train_s0.toml` and `config/eval.toml` each still contain 60 unresolved
bindings: 43 benchmark bindings and 17 required identities. No current source directly
supports values for the manifest identities, S0 batch/capacity matrix, W&B identity,
evaluator extractor/preprocess/real-stat identities, evaluator output reservation,
evaluation resource plan, or stage budgets.

Five non-TOML production contracts also remain unresolved:

- `S0_WARMUP_FUNCTION_UNRESOLVED`
- `S0_PASS_INDEX_OWNERSHIP_UNRESOLVED`
- `S0_FORMAL_PROMPT_CONDITION_CONTRACT_UNRESOLVED`
- `S0_LIVE_READY_QUEUE_DEPTH_UNBOUND`
- `S0_DIT_FLOPS_OBSERVATION_UNBOUND`

The train CLI therefore exits with structured blockers before dependency identity,
checkpoint bootstrap, CUDA selection, or data-service connection. Production
zero-update, capacity sweep and S001 were not run. Manual launch commands are only
handoff templates until these bindings are governed and the complete static preflight
passes.
