# T024 implementation report

The implementation emits a flat varlen tensor and int32 boundaries without constructing a production dense batch. A block-diagonal dense mask exists only as an explicit correctness-test helper. Real RTX 5090 BF16 validation preserved native 20-query/5-key heads and finite forward/backward behavior at two 1,024-token image grids.
