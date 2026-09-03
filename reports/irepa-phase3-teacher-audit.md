# iREPA Phase 3 — frozen PE-Spatial teacher HCU audit

- device: `BW` (torch 2.9.0)
- generated: 2026-09-03T00:02:20.603473+00:00
- deterministic (bitwise, repeated forward): True
- max viable batch @512x512: 64
- compile recommendation: n/a

| shape | grid | batch | median ms | p95 ms | peak alloc | peak reserved | tokens |
|---|---|---|---|---|---|---|---|
| 256x256 | 16x16 | 4 | 19.113 | 19.719 | 255.2 MiB | 276.0 MiB | 1024 |
| 256x256 | 16x16 | 16 | 29.916 | 30.127 | 408.8 MiB | 444.0 MiB | 4096 |
| 512x512 | 32x32 | 4 | 46.732 | 46.821 | 696.7 MiB | 732.0 MiB | 4096 |
| 512x512 | 32x32 | 16 | 165.099 | 165.315 | 2178.2 MiB | 3006.0 MiB | 16384 |
| 256x1024 | 16x64 | 4 | 47.254 | 47.504 | 714.7 MiB | 3006.0 MiB | 4096 |
| 256x1024 | 16x64 | 16 | 165.472 | 165.656 | 2178.2 MiB | 2988.0 MiB | 16384 |
| 1024x256 | 64x16 | 4 | 47.797 | 48.681 | 714.7 MiB | 2988.0 MiB | 4096 |
| 1024x256 | 64x16 | 16 | 165.288 | 165.38 | 2178.2 MiB | 2988.0 MiB | 16384 |
