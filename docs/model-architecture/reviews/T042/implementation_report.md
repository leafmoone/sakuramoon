# T042 implementation report

The writer publishes only raw and model-only artifacts. It saves the unwrapped trainable module by sorted state-dict FQN into deterministic Safetensors shards and records the standard weight map, reconstructable architecture, and locked `prediction_type=x` and `out_channels=128`. Raw checkpoints add the underlying TorchAO state, exact canonical parameter groups, immutable trainer/data/growth state and separate rank/SR RNG tensors.

Every payload file is recorded with size and SHA-256 in the checkpoint manifest. The loader requires the exact payload file set and `COMPLETE`, validates checksums, kind, identity, model FQN/shape/dtype, optimizer schema, all JSON sidecars and RNG tensors before applying model or optimizer state. TorchAO deserialization always uses `weights_only=True`.

Temporary publication directories are task-owned and removed on failure; a final directory appears only through atomic rename after fsync. The raw writer remains free of scheduling policy. `checkpoint.policy` separately resolves the fixed update/wall cadence and forced reasons, advances only after a matching successful publication, and produces a revalidated retention plan that keeps two rolling raws plus all accepted raws. Growth migration, Qwen/VAE ownership and distributed behavior remain outside this module.

`checkpoint.pma` validates exactly ten complete raw sources, exact required raw sidecars, strict update order, identity hashes, architecture/shard layout, completed growth alpha and stage/world/resolution/slot topology. It averages one shard at a time in FP32 and casts back to the source dtype. PMA and explicit manual release artifacts carry immutable source records and no trainer/optimizer/RNG continuation state.

The retention apply boundary now accepts the trusted accepted-checkpoint set again,
replans against the same resolved root and compares the complete plan plus checkpoint
identities before deleting anything. This prevents a caller-constructed plan from
moving an accepted raw into the deletion set. Retention validates canonical names,
strict manifests, the exact physical file tree, symlink absence and payload sizes, but
deliberately does not recalculate every multi-GiB payload checksum; full load, PMA and
resume validation continue to calculate and enforce those checksums.

AI/model self-check: same-topology next-update equality covers model parameters, TorchAO state progression, isolated SR state and Python/NumPy/Torch RNG; copied raw `model/` content also loads without training sidecars. Synthetic CPU contracts establish exact arithmetic and kind separation, while a real CUDA composite now establishes that a ten-raw PMA fresh-loads through the public inference artifact boundary with the exact expected mean. This remains a mechanics contract, not a production PMA quality result. Infra self-check: model and PMA shards are streamed with bounded per-shard CPU state, retention selection is metadata-only, physical full-composite shard sizes remain below 2 GiB, and no timing is represented as formal NVMe performance.

Main-agent remediation review found that the initial expansion allowed a forged
`RawRetentionPlan` to nominate an accepted raw for deletion and lacked a real-composite
PMA fresh-load contract. Both findings are fixed and targeted CPU/one-GPU suites pass.
Two direct attempts to start a fresh independent reviewer failed with
`agent thread limit reached`; per user direction work continued without agents. The
expansion therefore has no falsely claimed independent rereview.
