# T021 implementation report

The wrapper loads the multimodal checkpoint through its text configuration, retains only `Qwen3_5TextModel`, and returns `[B,L,7,2048]` from one inference forward. Real CPU state loading found no missing or unexpected keys and no visual tower. Real RTX 5090 BF16 execution passed all eight approved dense lengths after removing an unnecessary `device_map` dependency on Accelerate.

The wrapper now accepts only a `torch.bool` attention mask, where `True` means a valid
token. Integer and floating masks, including long values `2` and `-1`, fail before the
backend model is called. A valid mask is passed through and returned by identity, so the
strict contract adds no tensor conversion, CUDA value scan, allocation, or D2H sync.
Fresh local-only RTX 5090 validation passed a 98-token BF16 forward on driver
580.105.08. Encoders/Conditioning package review remains pending.

Transformers 5.14.1 replaces the last captured decoder-layer state with
`last_hidden_state` after final RMSNorm. The wrapper now installs a temporary hook on
decoder layer 24, performs the same single model forward, substitutes that raw return
as the seventh selected state, and removes the hook in `finally`. The first six states,
frozen boundary, boolean mask preflight, local-only loading, and no-cache behavior are
unchanged. A real RTX 5090 contract captured final RMSNorm input and output in the same
forward: SakuraMoon's seventh state equals the input and has max absolute difference
`25.15625` from the normalized output. No second Qwen forward was used.
