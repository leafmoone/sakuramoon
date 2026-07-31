# T020 Mage upstream implementation comparison

Governed source: Microsoft Mage commit
`8c94a0ac905167f40b05b09332b78752b7f9fbef`, file
`mage_flow/models/modules/mage_vae.py`. The repository, Git tree, source digest,
and MIT license digest are fixed in `mage_upstream_lock.json`.

This is implementation provenance only. It is not a local model identity manifest,
does not hash or inspect the prepared model files, does not create a runtime dependency,
and does not import or execute `reference/` or upstream code.

| Contract | Fixed upstream implementation | SakuraMoon implementation | Validation | Result |
|---|---|---|---|---|
| Checkpoint key mapping | Lines 559-595 select `student.dconv_encoder.*` and `pipeline.*`. | `mage_vae.py:454-479` uses the same prefixes and requires strict complete local module state. | Existing real checkpoint load proved 321 encoder and 365 decoder state entries; the governed-lock unit contract pins both prefixes. | PASS |
| Posterior mean encode | Lines 597-623 construct zero `z_t`, use `t=0`, split packed mean/logvar, and return mean when posterior sampling is disabled. | `mage_vae.py:487-504` constructs zero latent and timestep, then returns the first 128 moment channels unconditionally. | Frozen-wrapper contracts plus prior real RTX 5090 encode show detached `[B,128,H/16,W/16]`. | PASS |
| One-step decode | Lines 625-633 derive CoD conditioning, create zero RGB noise, use `t=0`, and run the denoiser. | `mage_vae.py:506-520` follows the same conditioning, zero-image, zero-timestep path. | Frozen-wrapper round trip and prior real RTX 5090 decode are finite with exact output shape. | PASS |
| Attention padding | Lines 303-335 replicate-pad Q/K/V to 32x32 patch multiples, perform patch attention, and crop back. | `mage_vae.py:265-313` preserves the same replicate padding, patch layout, attention scaling, and crop. | Static lock contract binds the upstream source digest; strict checkpoint load covers parameter layout. | PASS |
| Latent and image geometry | Lines 1-10 and 535-542 state 128 channels at H/16 x W/16 with no latent patch packing; decode restores Hx16/Wx16. | `mage_vae.py:441-445,487-520` enforces the same channel and spatial contract without extra patchification. | CPU shape contracts and real 512x512 RTX 5090 smoke passed. | PASS |

The upstream wrapper permits additional checkpoint formats and optional posterior
sampling. SakuraMoon intentionally narrows that interface to the already prepared local
safetensors file and posterior mean only. This is a fail-closed project constraint, not
an algorithm divergence.
