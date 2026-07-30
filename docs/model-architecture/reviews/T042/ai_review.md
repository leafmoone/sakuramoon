# T042 AI review

Status: prior raw/model-only PASS remains valid. Fresh independent review of the
PMA/release/policy expansion is pending.

The independent review confirmed that raw and model-only artifacts use distinct kinds and publication names. Raw resume requires `CheckpointKind.RAW`, while the model-only loader requires `CheckpointKind.MODEL_ONLY`; PMA and release remain non-resumable artifact kinds. Model tensors are deterministically sharded by sorted canonical FQN, and load rejects any FQN, dtype, shape, declared-size, architecture, parameter-schema or full checkpoint-identity mismatch.

Raw restore validates the `COMPLETE` marker, exact payload file set, every payload size and SHA-256, kind and identity, model metadata, optimizer canonical groups, the safely loaded TorchAO state schema, trainer/data/growth state, rank RNG and isolated optimizer-SR RNG before applying model, optimizer or global RNG state. The optimizer sidecar is loaded with `torch.load(..., weights_only=True)` and explicitly accepts TorchAO `OptimState8bit` only with the expected block, signedness, tensor and per-parameter step schema. Lazy and lagging per-parameter state remains valid without weakening the canonical parameter boundary.

The checkpoint boundary is exactly the unwrapped `TrainableComposite` with `dit`, `text` and `style` children, all trainable; Qwen and Mage-VAE cannot enter the model tree, optimizer coverage or checkpoint FQNs. The copied `model/` directory has its own strict manifest and self-describing architecture, so it loads independently without optimizer, trainer, data, growth or RNG sidecars. The existing RTX 5090 evidence demonstrates safe TorchAO restore, all RNG-stream restoration, fresh-process next-update equality and the full 16-layer 239-FQN/two-shard artifact; that GPU evidence was inspected but not rerun during this CPU-only review.

The reviewer ran 81 CPU tests covering checkpoint plus the directly affected conditioning, model, optimizer and train contracts. Ruff, strict Pyright, traceability verification and `git diff --check` all passed.

Four-rank sharding/barriers and all-rank state equality remain pending until four RTX 5090 GPUs are available. Growth migration remains T043. The new PMA/release/cadence/retention code and its CPU contracts were added after this review and therefore are not covered by this PASS until a fresh reviewer signs them off.
