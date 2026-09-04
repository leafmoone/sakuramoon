# CMuon F3 — SOFT BELOW-FLOOR FP32 RESCUE（实现与验证报告）

- 分支：`cmuon-soft-below-floor-f3`（基线 `bb41292`，F2 forensic 收口 commit）
- 工作树：`sakuramoon-cmuon-f3`（隔离 worktree，未 push/merge/deploy）
- 日期：2026-09-04（salt11 HCU 2×BW 全部权威执行；本机 Windows 仅 CPU 类基线）
- 机器红线遵守：仅 salt11 的 /tmp 暂存区与只读生产资产（capsule/ckpt），无任何生产写入

## 0. 语义变更（唯一）

FP32 rescue verdict（`src/sakuramoon/optim/fp32_rescue.py`）：

| 输入 | F2（bb41292） | F3 |
|---|---|---|
| FP32 nonfinite | HARD FAIL（CMuonSafetyError + forensic + capsule） | 同 F2 |
| FP32 finite 且 > ceiling（10×0.2·lr） | HARD FAIL | 同 F2 |
| FP32 finite 且 < floor（0.05×0.2·lr） | HARD FAIL | **接受原 FP32 delta + 诊断遥测**（`below_floor_soft_rescue`；exact-zero → `zero_delta_soft_rescue`） |
| FP32 finite 带内 | 接受 | 同 F2 |

- `rescue_floor` 数值保持 `0.05×target_delta_rms`，角色从 SAFETY BOUNDARY 降为 DIAGNOSTIC BOUNDARY（单测 J 钉住）。
- 无 clamp/rescale/normalize；无 AdamW/SVD fallback；NS 系数、floor/ceiling 数值、alpha、checkpoint schema 均不变。
- F2 fast-capture 仅在 nonfinite/above_ceiling 触发。

## 1. 变更面

- 运行时 diff：`src/sakuramoon/optim/fp32_rescue.py`（+83/−19）
  - verdict 条件 `if not finite32 or rms32 > ceiling:`（去掉 `or rms32 < rescue_floor`）
  - 接受分支后置 `if rms32 < rescue_floor:` 诊断块：reason 选择、`fp32_low_delta_rescues` / `fp32_low_delta_by_role`（**进程局部，不持久化**）、`stats_logger` 单行遥测（reason/fqn/role/rms/target/floor/比值/u_t_rms）
  - 两处 telemetry JSON（per-rescue obs + 周期 stats）追加两个新计数器
- 新增：`tests/unit/optim/test_fp32_rescue_f3.py`（A–K 15 例）、`tests/gpu/optim/fp32_rescue_f3_2rank.py`（5 场景）、`dev-tools/cmuon_f3_parity.py`、`dev-tools/cmuon_f3_r2_exact_replay.py`、`dev-tools/cmuon_f3_ckpt_compat_audit.py`
- 未修改：F2 forensic 测试、既有 rescue 测试、其他优化器路径、模型、目标、iREPA、生产配置。

## 2. 验证证据（salt11 HCU，权威）

### 2.1 单元测试（tests/unit/optim/test_fp32_rescue_f3.py）

**15/15 PASS**（73.7s）：A zero→soft（exact-zero reason）、B 带内 2×floor 非 soft、C 1.01×floor 非 soft、D 0.99×floor soft + 提交值参考逐位校验（原始 fp32 delta 单轮 bf16 rounding，无修正）、E 2×ceiling hard fail + capsule 本地/镜像计数与镜像一致性、F inf hard fail + capsule reason=nonfinite、G×6 真实 FP32 NS 矩阵（zero/constant/rank-1/rank-8/weak-low-rank/full-rank，BF16 全毒化 evil const 1e9 强制 rescue 路径）、I checkpoint schema（guard 块恰 6 键、round-trip、父类无块加载、low-delta 计数器进程局部复位）、J floor 数值 0.05、K hard fail 无 soft 遥测。

### 2.2 §6/§14 R2 精确输入回放（obs-611-rank0 capsule，1c782a46…fbb5541）

- 输入 sha256 与 capsule 元数据一致；input_rms 复核一致；alpha/lr/floor/ceiling 从元数据复算一致（|Δ|<1e-18）。
- **BF16 生产 addmm ×20**：全部 finite 且全部 above_ceiling（rms 2.05e11–3.01e11，中位 2.35e11）→ BF16 路径对该输入恒为 hard-fail 类；非确定性记录在案（§2.7）。
- **BF16 确定性 oracle ×3**（matmul+mul+add，TEST ONLY）：非位确定（rms 2.07e11–2.52e11）→ salt11 上该病态输入连 matmul 级也跨次非确定（环境事实，见 §2.7；不影响任何生产判定）。
- **FP32 生产路径 ×3**：**位精确**，rms = 1.377804551339068e-06 **逐位等于 capsule 记录值**（readback_matches_capsule=true）。
- staged bf16 delta sha `65422097…`（原 fp32 delta 单轮 bf16 rounding，既有提交路径）。
- verdict：**F2 参考 = HARD_FAIL/below_floor；F3 = SUCCESSFUL_RESCUE/below_floor_soft_rescue**；`f3_semantic_change=true`。

### 2.3 §8–§11 跨树 parity（F2 树 vs F3 树，同 seed 20260904）

- **safe-oracle 10 步**（TEST-ONLY 确定性 bf16 NS）：params/momenta/refs/sr_rng_state/CMuon 计数器 **逐位一致**；唯一差异 = mock `dit.input_projection`（AdamW 路由参数）的 8-bit AdamW `exp_avg/exp_avg_sq` —— **同树双进程对照（f2a vs f2b、f3a vs f3b）证实为 HCU 环境固有非确定性**，与 F3 补丁无关。
- **normal 3 步**（毒化 bf16 + 真实 FP32 NS）：同上，逐位一致（modulo 同一环境非确定键）。
- **hardfail-nonfinite / hardfail-ceiling**：两树 outcome=CMuonSafetyError、异常消息逐字符一致、final state 一致、零提交。
- **below-floor（§10 故意 delta）**：F2 = CMuonSafetyError（0 步、16 失败、零提交）；F3 = SUCCESS（20 步、320/320 soft rescue，by_role 分解完整、spread 0）。
- **safe-real 10 步**（真实生产内核，无强制）：无 BF16 失败、无 FP32 attempt、无 rescue —— 健康路径保真。

### 2.4 §12 2-rank HCU 5 场景（torchrun 2 rank，真实 NCCL）

**PASS**：healthy（spread 0）/ soft_rescue（owner FP32 恰 1 次、非 owner 0 次、soft 计数器 +1、spread/fingerprint 全 0.0）/ mixed_step（双 chunk rescue、两 rank 参数逐位一致）/ above_ceiling（两 rank CMuonSafetyError + 零提交）/ nonfinite（同）。

### 2.5 §17 性能

20 步对比：normal-constant 133.00 ms/step vs below-floor 132.50 ms/step → soft 分支开销 **−0.5 ms/step ≈ 噪声**。

### 2.6 §18 ckpt_112100 只读兼容审计

`_load_hybrid_optimizer_state`（生产 resume 的 exact-key loader）接受 3.1GB F2 `optimizer.pt`：顶层键/schema 版本全过；`fp32_rescue` 块 **恰为 F3 loader 期望的 6 键**（bf16_attempts 49800 / bf16_safety_failures 170 / fp32_attempts 170 / fp32_rescues 170 / fp32_rescue_failures 0 / rescue_by_role {q:84, content_gate:58, k:28}）；F3 进程局部键不在文件中；**无需任何迁移**。反向兼容由单测 I（F3 state_dict 与 F2 字节级 schema 相同）保证。

### 2.7 §13 非确定性记录（记录不修复）

- BF16 addmm 跨次调用非确定（S18 既有结论，R2 回放再次确认）。
- **新环境事实（salt11 HCU，R2 病态输入）**：bf16 `matmul`（A@B）亦跨次非确定（5 次 sha 全异；探针确认 norm 确定性、FP32 matmul 确定性）——S18（salt13）曾测 matmul 位确定，说明该属性**机器/构建相关**；病态近零信号输入（NS 收敛边界）放大约数舍入混沌（与 R2 的 ±17% bf16 非确定及 trace iter-3 失稳一致）。健康 O(1) 输入上 oracle 路径跨进程位一致（§2.3 safe-oracle params 逐位一致可证）。
- 生产不受影响：owner-rank 每 update 只算一次 NS 并广播；FP32 路径（rescue 实际依赖的路径）在 salt11 位确定。

### 2.8 §19 质量门

| 门 | 结果 |
|---|---|
| ruff（5 个 F3 文件） | 0 error |
| pyright（统一 uv3.12 环境） | fp32_rescue.py 246=246 FLAT；test_fp32_rescue.py 274=274 |
| F3 单测（HCU） | 15/15 |
| F2 回归（tests/unit/optim，HCU） | 129 PASS + 4 预期语义差（§2.9） |
| 2-rank HCU 5 场景 | PASS |
| R2 精确回放 | PASS（§2.2） |
| ckpt_112100 兼容 | PASS（§2.6） |
| 全量 HCU 单测 | **完整树（含 config/）权威终判：756 passed / 4 failed / 0 errors（15:58），4 failed = §2.9 的 4 个预期 F3 语义差，无其他失败**。中间一轮（staging 缺 `config/`）729P/14F/17E，13F+17E 全部为 `FileNotFoundError: /tmp/f3-src/config`（staging 产物，补 config/ 后同批 46/46 PASS 复核） |
| 本机全量 pytest | 59F/710P/3S/3E（Windows）= 46 既有 Windows 基线类（3 config symlink、2 subprocess watchdog、41 TritonMissing CUDA）+ 13 新 F3 CUDA 测试（同一 TritonMissing 已知类；F3 文件本地 13/15 TritonMissing，2 个 CPU 用例本地亦 PASS，HCU 15/15）；passed 与基线完全一致，无新增非已知类失败 |
| 环境缺口记录 | 2 个 deepghs 数据管道测试文件（`test_deepghs_quality_pipeline.py`/`test_run_deepghs_quality_pipeline.py`）在 DTK venv 不可收集（脚本需 onnxruntime，venv 未装）——既有环境缺口，与 F3 无关，任何树同态排除 |

### 2.9 预期语义差（既有 forensic 测试中 below_floor 驱动的用例，按规格不修改既有测试）

`tests/unit/optim/test_cmuon_fp32_forensic.py` 4 个用例以 `below_floor` FP32 结果驱动 hard-fail 故障注入路径，F3 下该类输入改为成功 soft rescue，故障注入路径不再触发：

1. `test_bcd_hard_fail_captures_fp32_reason[below_floor-below_floor]`
2. `test_e_writer_failure_still_raises_original_error`
3. `test_f_serialization_failure_still_raises_and_leaves_no_capsule`
4. `test_m2_mirror_failure_keeps_local_capsule_and_original_error`

hard-fail 捕获机制本身由仍通过的 above_ceiling/nonfinite 用例与 F3 单测 E/F 完整覆盖；below_floor 新语义由 test_fp32_rescue_f3.py A/D + §2.2 R2 回放覆盖。**分类为预期 F3 语义差，非回归。**

## 3. 汇总判定

- 唯一语义差 = §0 表第三行；其余路径 vs bb41292 **位精确**（§2.3）或同态 hard-fail（§2.3）。
- checkpoint 双向兼容（§2.6 + 单测 I）；生产 salt11 无写入、无重启、无 push/merge/deploy。
- **F3 VERDICT：IMPLEMENT + VERIFY 完成，等待用户拍板部署。**
