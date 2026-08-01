# C002 implementation report

## Result

C002 supplies the canonical base, eight stage overlays, evaluator config, sampler
config, exact model assembly, and local-first training telemetry assembly. The files
remain intentionally non-runnable until S000 and immutable external bindings replace
every explicit sentinel with measured or qualified production values.

## Configuration and stage contracts

- The current C002 session decision supplies the exact Text constructor values
  `attention_heads=16`, `mix_gate_init=0.0`, `layer_scale_init=1.0`, and
  `projection_bias=false`, and the exact Style values `attention_heads=16`,
  `init_std=0.02`, and `projection_bias=false`. C002 records these as current
  task-local supplemental decisions; it does not promote or infer them from the
  earlier T022/T023 smoke-only evidence.
- `base.toml` contains the complete confirmed architecture, dropout, data, storage,
  optimizer, objective, telemetry, evaluation, and failure policy surface. Every
  production entry merges through the governed `extends` loader.
- Stage overlays bind the approved S0/S1/G1/S2/G2/S3 sequence and expose measured
  batch, accumulation, checkpoint, budget, and wall-time fields without defaults.
  Exact validators enforce `global_batch = local_batch * accumulation * world_size`
  and `planned_valid_samples = global_batch * planned_updates`.
- H1/H2 are disabled template intents. Eval and sample are distinct non-training
  intents. None can pass the single-GPU training boundary.
- The loader rejects every unresolved production entry. The eight resolved hashes in
  the stage report use synthetic validation substitutions only; S000 production
  budgets and hashes remain pending.
- `timing.phases` is the exact ordered T051 vocabulary: 12 core phases followed by 13
  detailed phases. Schema rejects missing, unknown, reordered, or drifted values, and
  telemetry assembly rechecks config against T051 before any run or sink is created.

## Model assembly

`trainable_composite_spec` maps strict Text, Style, DiT, RoPE, condition, head, dtype,
initialization, active-slot, and attention-backend fields to the exact
`TrainableComposite` constructor document. `build_trainable_composite_from_config`
constructs on the requested device and exports the module back to the same document;
any drift fails. The production CLI evaluates this binding before reporting the real
downstream lifecycle gate (`T052-T054` and `S000`).

## Telemetry assembly

- Local JSONL, W&B retry JSONL, observer queue, remote queue, event timeout, fsync
  cadence, run directory, W&B identity, resolved hash, and resume policy all come from
  the validated config or explicit assembly arguments.
- W&B uses stable `id=run_id`, `name=run_id`, and `resume="allow"`. Existing retry
  records replay before the new asynchronous sink accepts submissions.
- W&B initialization communication failure selects an explicit retry-only remote.
  Both generic `ConnectionError` and W&B `CommError` take that path; authentication
  and other non-communication initialization failures remain fatal.
  Replay communication failure leaves the old queue intact and permits local startup.
  Runtime upload spills only explicit communication failures. Authentication,
  protocol/non-communication errors, malformed payloads, symlinks, non-regular files,
  and retry files not using exact mode 0600 remain fatal and surface through health or
  close. Queue-full local spill remains explicit and unchanged.
- Normal close order is observer, remote sink, managed run, then local sink. Construction
  and close paths attempt all owned cleanup and preserve primary plus cleanup errors.

## Correctness and infrastructure self-check

- Confirmed dropout and Text/Style values are strict literals; no historical candidate
  or code default was used. Unknown or missing constructor fields fail validation.
- Stage differences are restricted to metadata/budgets plus the approved adjacent
  topology, depth, resolution, or growth axis. Unsupported activation checkpointing
  does not silently fall back.
- All queues are bounded. W&B failures do not mutate training, checkpoint, batch,
  backend, world-size, LR, token, or feature controls. Retry files are private before
  read and local durable output remains first.
- C002 performs startup-only CPU assembly and validation. It does not claim GPU kernel,
  throughput, multi-card, or formal stage evidence, and creates no performance
  baseline/after placeholder.

## Review state

Independent final review found the incomplete timing vocabulary; Infra additionally
found runtime authentication was incorrectly retryable. Both blockers are remediated.
Focused post-remediation CPU, strict Pyright, trace smoke, and live traceability pass;
final Ruff/diff evidence is recorded in `test_report.json`. A current
post-remediation full CPU run passed all 898 unit/contract tests. Final PASS/FAIL
was independently reviewed: AI/model correctness and Infra/performance both returned
PASS for the declared C002 CPU scope.
