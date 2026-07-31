# M037 upstream algorithm matrix

Status: implementation comparison complete; independent Dense remediation review
pending.

R001 locks JLT to commit
`aca236efa97aab3b7d865fd3d99a270431cf6ae5`. The reference was inspected only
through static `git show`/`git grep` reads. It was not imported, executed, called,
or exposed to production code, tests, preflight, or runtime.

| Planned formula or contract | Formula and file lines in locked reference commit | SakuraMoon local implementation | Golden or contract test |
|---|---|---|---|
| Forward path and strict x-pred objective remain `z_t=t*x+(1-t)*epsilon`, `d=max(1-t,0.05)`, `v_target=(x-z_t)/d`, `v_pred=(x_pred-z_t)/d`, and feature-then-sample MSE reduction | `denoiser.py:211-220,233-241`; `main_jit.py:93-96` | `src/sakuramoon/objective/flow.py:72-81,151-185`; M037 changes only the masks used to summarize the already-computed `per_sample` tensor | `tests/unit/objective/test_flow.py:84-177` keeps exact-x zero loss, shared clamp, inverse-square equivalence, sample-first reduction, and the 400 cap |
| Observation-only boundary is high noise `t<0.95`, low noise `t>=0.95`; bucket sums/counts do not modify full-batch loss | The locked upstream objective at `denoiser.py:239-241` computes only one loss and contains no high/low observation bucket or `0.95` threshold; static grep finds no such contract | `src/sakuramoon/objective/flow.py:151-185` validates the explicit fixed boundary, computes masks after `per_sample`, and still returns `loss=per_sample.mean()` | `tests/unit/objective/test_flow.py:180-200` covers `t=0.94`, the half-open `t=0.95` boundary, and unchanged full-batch mean; `tests/unit/objective/test_flow.py:203-251` rejects the superseded `0.5` value |
| The observation boundary is explicit configuration identity and cannot drift | Upstream has no corresponding configuration field; this is an intentional SakuraMoon user decision | `src/sakuramoon/config/schema.py:53-66,633-637` fixes the field to exact float `0.95`; `config/examples/all_options.example.toml:315-319` lists it under `[logging]` | `tests/unit/config/test_schema.py:116-140` rejects `0.5`; the complete example parse proves the required field is present |
| `t=0.95` is near the clean endpoint under the noise-to-clean time direction; naming the `t>=0.95` bucket low noise does not change endpoint weighting | `denoiser.py:211-220` makes increasing `t` progressively cleaner and applies the clamp independently of any observation label | `src/sakuramoon/objective/flow.py:79-81,171-185` keeps the `0.05` clamp and maximum inverse-square weight 400 while classifying `t>=0.95` as low noise | `tests/unit/objective/test_flow.py:103-118,163-199` separately covers the clamp branch, 400 cap, and observation boundary |

## Result and evidence boundary

The upstream JLT repository does not define `0.95` as a high-noise loss-weighting
threshold. M037 implements a SakuraMoon observability decision only. It neither adds
loss weights nor changes timestep sampling, x-prediction, target conversion, clamp,
reduction, CFG, or solver behavior.

The one-GPU test remains component evidence using a standalone parameter and SGD. It
does not close T050 full-chain training integration, T041 DDP/global-mean work, any
multi-GPU gate, a long run, or a formal stage canary.
