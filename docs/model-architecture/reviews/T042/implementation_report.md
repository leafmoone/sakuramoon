# T042 implementation report

The writer publishes only raw and model-only artifacts. It saves the unwrapped trainable module by sorted state-dict FQN into deterministic Safetensors shards and records the standard weight map, reconstructable architecture, and locked `prediction_type=x` and `out_channels=128`. Raw checkpoints add the underlying TorchAO state, exact canonical parameter groups, immutable trainer/data/growth state and separate rank/SR RNG tensors.

Every payload file is recorded with size and SHA-256 in the checkpoint manifest. The loader requires the exact payload file set and `COMPLETE`, validates checksums, kind, identity, model FQN/shape/dtype, optimizer schema, all JSON sidecars and RNG tensors before applying model or optimizer state. TorchAO deserialization always uses `weights_only=True`.

Temporary publication directories are task-owned and removed on failure; a final directory appears only through atomic rename after fsync. The checkpoint module contains no schedule, retention, PMA averaging, growth migration, Qwen/VAE ownership or distributed path.

AI/model self-check: same-topology next-update equality covers model parameters, TorchAO state progression, isolated SR state and Python/NumPy/Torch RNG; copied raw `model/` content also loads without training sidecars. Infra self-check: model shards are streamed through bounded CPU copies, physical full-composite shard sizes remain below 2 GiB, and no timing is represented as formal NVMe performance.
