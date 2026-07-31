# T024 implementation report

The implementation emits a flat varlen tensor and int32 boundaries without constructing a production dense batch. A block-diagonal dense mask exists only as an explicit correctness-test helper. Real RTX 5090 BF16 validation preserved native 20-query/5-key heads and finite forward/backward behavior at two 1,024-token image grids.

Packing now rejects cross-device or cross-dtype text/style/image inputs and a mask on a
different device. RoPE rejects Q/K dtype differences, non-FP32 coordinates, and any
query/key/coordinate/frequency device split before normalization or rotation, so
PyTorch cannot silently promote these contracts.

The packer validates positive host-side sequence lengths, derives cumulative int32
boundaries and total/max/batch metadata together, and stores them in an untrusted
public handoff. The earlier raw `.to(cpu).tolist()` validator and scalar boundary reads
in the dense reference helper were removed. Targeted CUDA packing/RoPE, FA4 dense
alignment, and PackedDiT block alignment passed on driver 580.105.08; the 17-shape
milestone matrix was not rerun.

Independent review then showed that the dataclass could still be directly forged and
that CUDA boolean text indexing required a data-dependent output shape. The boundary
constructor now accepts only validated host lengths and creates its own int32 tensor;
FA4 also checks host-length consistency, dtype, rank, shape, batch length, and
contiguity before native-kernel import. Collate and the trainable composite now carry
host text lengths, and packing verifies a fixed-shape prefix mask before using static
slices. Four adversarial forged handles failed before the kernel, while real FA4 and
PackedDiT short GPU tests passed. No 17-shape, multi-GPU, or long-run evidence was
rerun; package rereview is pending.

A later independent review demonstrated that correct static metadata could still be
paired with mutated CUDA values such as host lengths `(2,2)` and offsets `[0,3,4]`.
The final remediation therefore treats the public handoff as untrusted. PackedDiT now
performs exactly one host/CUDA content comparison at the packed-batch entry, rejects a
forged or post-construction-mutated handoff before native-kernel import, and creates a
capability-protected accepted handle with private CUDA offsets rematerialized from the
canonical host lengths. Sample routing is derived from the same accepted host tuple,
and only that accepted handle reaches all blocks and FA4. Mutation of the discarded
public tensor cannot alter the private offsets.

The entry check is the only intentional production D2H operation; no block calls
`.item()`, `.tolist()`, or moves boundaries to CPU. A 50-call engineering sample at
lengths `(1028,1540)` measured entry wall p50 `0.030010 ms` and p95 `0.031951 ms`.
Targeted CPU contracts passed 26 tests, adjacent conditioning/model regression passed
75, real RTX 5090 FA4 contracts passed 16, and PackedDiT isolation plus the 16-block
three-stage forward/backward passed 2. The existing Encoders/Conditioning package
rereview remains pending, and this evidence does not close K001 provenance/performance,
multi-GPU, formal-stage, or long-run gates.
