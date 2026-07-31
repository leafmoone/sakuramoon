# M036 upstream algorithm matrix

Status: implementation comparison complete; independent Dense remediation review
pending.

R001 locks JLT to commit
`aca236efa97aab3b7d865fd3d99a270431cf6ae5`. The reference was read statically; it
is not imported, executed, or exposed to production code, tests, preflight, or runtime.

| Planned formula or contract | Formula and file lines in locked reference commit | SakuraMoon local implementation | Golden or contract test |
|---|---|---|---|
| Heun predictor evaluates `v(z_i,t_i)`, corrector evaluates `v(z_predict,t_{i+1})`, then averages both slopes | `denoiser.py:252-269,292-305` | `src/sakuramoon/sampling/heun.py:80-104` | `tests/unit/sampling/test_heun.py::test_time_dependent_velocity_uses_next_timestep_for_heun_corrector` uses `dz/dt=t`, which fails if the corrector reuses `t_i` |
| The last interval is Euler, so `N` linear intervals use `2N-1` NFE and never evaluate the model at `t=1` | `denoiser.py:263-268,292-305`; the locked reference skips the second evaluation on its final interval | `src/sakuramoon/sampling/heun.py:91-100` | The CPU golden checks 99 NFE, `max(t)<1`, and `z(1)=1/2-1/(2N^2)=0.4998` for `N=50`; `tests/gpu/sampling/test_profiles.py::test_time_dependent_heun_golden_executes_on_cuda` repeats the analytic result on CUDA |
| Solver state and returned velocity are FP32 | The upstream code inherits its sampling dtype rather than independently locking FP32 | `src/sakuramoon/sampling/heun.py:25-43` promotes state and rejects non-FP32 velocity | Both M036 goldens start from BF16 noise and assert the FP32 analytic result; this is an intentional current-decision override |

The existing implementation already matches the locked predictor/corrector ordering.
M036 adds regression evidence only; it does not change solver code or the M034 profile
registry.
