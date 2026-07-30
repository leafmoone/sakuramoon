# T040 Infra review

Status: PASS after independent remediation rereview; no blocking findings.

Initial findings that remain applicable were incomplete full-composite FQN/state bytes evidence, unguarded TorchAO state fallback, missing held-out validation EMA and missing serialized next-step comparison. The earlier concern about DiT content-gate dtype was withdrawn after confirming that the content-gate projection is a BF16 decay matrix.

Independent rereview reran all 35 affected tests, Ruff, strict Pyright and `git diff --check`. It reproduced the 239-FQN full composite, 152 quantized and 87 regular state entries, 2,568,392,844 optimizer-state bytes, held-out EMA ratios and serialized bitwise next-step equality. A warm-cache rerun measured 7,555.6 ms cold state initialization, 16.077 ms steady clip and 101.994 ms steady optimizer, consistent with the recorded profile.

Four-rank state equality and strict global mean remain pending T041. End-to-end optimizer share remains pending T050 and is not inferred from the isolated zero-gradient profile.
