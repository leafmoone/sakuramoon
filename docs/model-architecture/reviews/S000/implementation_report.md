# S000 bounded engineering-smoke implementation report

The dedicated CLI accepts only a config path, config root, and repository root. The
strict standalone schema rejects unknown, missing, wrong-type, out-of-range, sentinel,
distributed, multi-GPU, FA4, production-claim, or output-escape inputs. Every training
semantic used by the runner, including fixed growth alpha and the loss observation
boundary, comes from the checked TOML. The resolved document is persisted inside the
raw checkpoint and bound by its SHA-256 identity.

The runner generates four tiny local tar shards solely as synthetic input, but passes
them through the real D024 service process, AF_UNIX client, lease/ACK protocol, shard
cache, WebDataset pipeline, and two persistent spawned workers. It loads the actual
prepared local Qwen tokenizer/encoder and Mage-VAE. The trainable module is the native
16-layer `DenseDiT` plus the production text and style conditioning modules; the shared
training boundary now dispatches exact `DenseDiT` instances through their homogeneous
dense signature while retaining the existing packed path.

Each update records H2D, Qwen, VAE, conditioning, DiT forward, JLT loss, backward,
clip, optimizer, and zero-grad phases. The update uses FP32 global clipping and the
real TorchAO AdamW8bit state. After update 1, the existing raw-checkpoint publisher
writes sharded Safetensors model state, optimizer/trainer/growth/RNG state, resolved
config, manifest/checksums, and `COMPLETE`. Parent model references are deleted before
garbage collection and CUDA cache release.

A spawned fresh process loads local frozen assets, assembles the same dense model,
verifies the checkpoint/config identity, restores the raw state, reconnects to the
restarted service, and performs update 2. The service state records replay rather than
embedding a data cursor in the checkpoint. Output publication is no-clobber; failed
trees are never overwritten.

The final run used resolved config SHA-256
`09458f37c567b38f2d26fa6b1d4f4ab17c6c504e0014178e1d9b64f92846cede`.
Its report records a peak allocated value of 11,981,415,936 bytes, but this is a single
synthetic local-batch observation and is not a maximum-batch or production-capacity
claim. No throughput or quality threshold was evaluated.

The production training CLI was not modified or invoked. Formal configuration,
production checkpoint, evaluator/extractor/preprocess/real-stat identities, approved
long-run resources, and four GPUs remain absent, so formal S000 and `P060-P067` remain
blocked.
