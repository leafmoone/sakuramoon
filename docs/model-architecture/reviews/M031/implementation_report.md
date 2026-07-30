# M031 implementation report

The dense reference block uses PyTorch SDPA with `enable_gqa=True` and keeps K/V at five heads. It applies the approved content and condition gates without LayerScale or a learned growth control. All block projections are bias-free and dropout is fixed to zero. The boolean dense mask covers both query and key axes, and padding is explicitly cleared after residual updates.
