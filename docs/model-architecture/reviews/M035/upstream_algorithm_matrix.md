# M035 upstream algorithm matrix

Status: implementation-agent static comparison complete; independent review pending.

## Reference lock and method

R001 locks the JLT repository to commit
`aca236efa97aab3b7d865fd3d99a270431cf6ae5`. This comparison used only static
`git show`/`git grep` reads of that commit. No reference module was imported, executed,
or exposed to production code, tests, preflight, or runtime.

The current confirmed decisions remain authoritative where they intentionally differ
from the upstream experiment defaults.

| Planned formula or contract | Formula and file lines in locked reference commit | SakuraMoon local implementation | Golden or contract test |
|---|---|---|---|
| JLT samples `t=sigmoid(N(-0.8,0.8))`; unit noise scale | `main_jit.py:73-75`; `denoiser.py:57-60,99-101,211` | `src/sakuramoon/objective/flow.py:83-121` requires the locked floats and an explicit generator | `tests/unit/objective/test_flow.py:18-67` fixes deterministic timestep and noise goldens and rejects drift |
| Forward direction is `z_t=t*x+(1-t)*epsilon`; `t=0` is noise and `t=1` is clean | `denoiser.py:211-213`; comments at `denoiser.py:61-63` use noise-to-clean direction | `src/sakuramoon/objective/flow.py:124-134` | `tests/unit/objective/test_flow.py:70-81` checks both endpoints |
| The model predicts clean `x`, not direct velocity | `README.md:72-99,193-206`; `denoiser.py:233-237` treats network output as x-pred unless the separate velocity baseline flag is enabled | `src/sakuramoon/config/schema.py:359-366`; `src/sakuramoon/objective/flow.py:150-178` | `tests/unit/config/test_schema.py:119-126`; `tests/unit/objective/test_flow.py:84-116` |
| Shared strict clamp: `d=max(1-t,0.05)`, `v_target=(x-z_t)/d`, `v_pred=(x_pred-z_t)/d` | `main_jit.py:93-96`; `denoiser.py:214-220,233-237` | `src/sakuramoon/objective/flow.py:71-80,161-165` converts clean and prediction through the same FP32 helper | `tests/unit/objective/test_flow.py:102-116,137-157` checks exact clean prediction at `t=0.99` and the weighted identity |
| `MSE(v_pred,v_target)=MSE(x_pred,x)/d^2`; feature mean per sample, then global sample mean; endpoint weight capped at 400 | `denoiser.py:239-241` performs feature mean then batch mean. The inverse-square form follows algebraically from its two shared-clamp conversions | `src/sakuramoon/objective/flow.py:165-178` | `tests/unit/objective/test_flow.py:119-172` checks reduction order, inverse-square weighting, and the 400 cap |
| Velocity conversion, squared error, and reduction are FP32 | Upstream `denoiser.py:209-241` inherits the model/input dtype and does not independently guarantee FP32 objective arithmetic | `src/sakuramoon/objective/flow.py:78-80,165-170` explicitly promotes both conversions and therefore the loss to FP32 | `tests/unit/objective/test_flow.py:98-99`; `tests/gpu/objective/test_flow_sampling.py:24-44` verifies BF16 inputs with FP32 loss/velocity. This is an intentional current-decision override |
| Observation only: high noise `t<0.5`, low noise `t>=0.5`; buckets do not alter the full-batch loss | The locked upstream objective has no corresponding observation-bucket contract | `src/sakuramoon/objective/flow.py:167-177` emits sums and counts while retaining `per_sample.mean()` | `tests/unit/objective/test_flow.py:175-193`; this is an intentional SakuraMoon observability addition |
| Convert conditional and unconditional x-predictions independently, then apply `v_uncond+2.9*(v_cond-v_uncond)` over the full interval with no rescale | `denoiser.py:273-289` converts both branches before CFG. `main_jit.py:113-117` leaves scale and interval configurable, so its defaults are not copied | `src/sakuramoon/objective/flow.py:181-212`; `src/sakuramoon/config/schema.py:462-466` locks scale/order/full interval/no rescale | `tests/unit/objective/test_flow.py:273-303`; `tests/unit/config/test_schema.py:127` rejects scale drift. Fixed 2.9/full interval is an intentional current-decision override |
| `t=0.99` is in the near-clean, low-noise region | Upstream interpolation at `denoiser.py:213` makes increasing `t` progressively cleaner | `src/sakuramoon/objective/flow.py:167-168` places every `t>=0.5` sample in the low-noise bucket | `tests/unit/objective/test_flow.py:102-116,175-193` covers the clamp and bucket boundary without calling `t=0.99` high noise |

## Evidence qualification

The current M033 CUDA test is valid component evidence for BF16 inputs, FP32 strict
JLT arithmetic, backward, one SGD update, velocity CFG, and Heun execution on one RTX
5090. It constructs a standalone `torch.nn.Parameter`, `torch.optim.SGD`, and a
synthetic velocity function in `tests/gpu/objective/test_flow_sampling.py:19-79`.

It does not invoke `PackedDiT`, `TrainableComposite`, `SingleGpuTrainingLoop`, real
data, Qwen, Mage-VAE, the production optimizer, or checkpoint publication. It must not
be described as full one-GPU training integration. T050 retains the real one-to-ten
step data/Qwen/VAE/DiT/loss/checkpoint engineering smoke; T041 retains DDP global-mean
equivalence and every multi-GPU conclusion.

## Result

The post-remediation SakuraMoon JLT/CFG implementation agrees with the authoritative
plan. The original pre-remediation `clean-noise` target and its `t=0.99` golden were
incorrect in the clamp region, but commit
`7a36605fb17235580262b76cab07181076aad459` corrected both. No remaining JLT/CFG
runtime mismatch was found in this static comparison.
