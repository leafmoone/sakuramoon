# T024 implementation report

The implementation emits a flat varlen tensor and int32 boundaries without constructing a production dense batch. A block-diagonal dense mask exists only as an explicit correctness-test helper. Real RTX 5090 BF16 validation preserved native 20-query/5-key heads and finite forward/backward behavior at two 1,024-token image grids.

Packing now rejects cross-device or cross-dtype text/style/image inputs and a mask on a
different device. RoPE rejects Q/K dtype differences, non-FP32 coordinates, and any
query/key/coordinate/frequency device split before normalization or rotation, so
PyTorch cannot silently promote these contracts.

The packer validates positive host-side sequence lengths, derives cumulative int32
boundaries and total/max/batch metadata together, and stores them in one boundary
handle. PackedDiT and FA4 reuse that handle directly. The former raw CUDA validator and
its `.to(cpu).tolist()` path were removed, as were scalar boundary reads in the dense
reference helper. Targeted CUDA packing/RoPE, FA4 dense alignment, and PackedDiT block
alignment passed on driver 580.105.08. Encoders/Conditioning package review is pending;
the 17-shape milestone matrix was not rerun.
