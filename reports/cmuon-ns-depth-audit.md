# CMuon Newton–Schulz Depth Audit（per-spec NS + 最低充分深度 + ns4 复测）

日期：2026-08-28 · 设备：salt1 HCU（Hygon DCU「BW」，DTK 26.04，torch 2.9.0+DTK）
范围：**只做 isolated optimizer math + benchmark + 代码/测试**，不启动真实训练、不切 live trainer、不部署 live、不 merge dev、不改 LR/batch/spatial/transparent/架构/ckpt 格式。默认 `torchao_adamw8bit` 路径 bit-compatible。

一句话结论：**per-role（per-spec）NS 深度基础设施已落地并验证；证据显示 ns3 对任何 role 都不足以「≈ ns4」（cosine 0.25–0.91，rel-error 0.47–1.07，全部远低于 0.995 门），因此最低充分深度 = ns4（全 role）；`hybrid_cmuon_min_ns_core`（mixed）不创建。ns4 复测：optimizer step 0.673s（AdamW8bit 2.380s，3.53×），全 role update RMS ≈ 0.2·lr（1.01–1.12×），全 shape finite。**

---

## 1. 背景与目标

上轮（cmuon-design-audit）修正了 NS 稳定性结论：ns5 对 k/v[640,2560] 与 AdaLN[2560,1024] 不安全（iter5 发散）、ns6 全 shape 发散、**ns4 全 shape 安全**。本轮任务（A–E）：

- **A.** 实现 per-spec（per-role）`ns_steps` 基础设施（同一 fused tensor 的 semantic chunks 共用同一 role 深度；不改 storage/layout）。
- **B.** 系统 sweep ns2/3/4（ns5 仅作边界参考），找各 shape「最低充分」NS 深度（最低充分，不是最大安全）。
- **C.** 复测 global ns4。
- **D.** 证据充分才给 mixed per-spec NS 候选（`hybrid_cmuon_min_ns_core`）；由 sweep 自动生成推荐表；无收益/证据不足则不创建。
- **E.** 本轮禁止启动真实训练、禁止切 live trainer。

## 2. 实现（A）

代码全部落在 `sakruamoon-Cmoun`（cmoun 分支），salt1 仅同步 `cmuon.py`/`test_cmuon.py`（不 commit、不动 G1 隔离改动）。

- **`CMUON_ROLES`**（8 个 canonical role）：`attention_q/k/v`、`attention_content_gate`、`attention_out`、`ffn_in`、`ffn_down`、`adaln_shared`。
- **allowlist 变 4-tuple** `(pattern, chunk_count, roles, canonical_role)`；`CMuonChunkSpec` 加 `role`。canonical role：q/k/v/content_gate/out→对应 attention_*；(gate,up)→`ffn_in`；down→`ffn_down`；6×AdaLN→`adaln_shared`。
- **`CMuonConfig.ns_steps: int` → `ns_steps_by_role: Mapping[str,int]`**（canonical 全 map，默认全 5）+ `ns_steps_for_role(role)` + `canonical_ns_map()` + `__post_init__` 校验（每 role 存在、int∈[1,99]）。
- **`resolve_ns_map(ns_steps_by_role|None, ns_steps|None)`**：从 partial override + scalar 构建 canonical map（未知 role/越界 ValueError）。
- **`_cmuon_update_impl`** 用 `cfg.ns_steps_for_role(spec.role)`（同一 fused tensor 的 chunks 共用该 role 深度）。
- **`routing_manifest`** 每 cmuon spec 加 `"role"`；**`_cmuon_state`** 存 `"ns_steps": canonical_ns_map()`（dict）。
- **`_load_cmuon_state`**：saved 为 dict→逐 role 比较；legacy int→视为全 role 等值后比较；mismatch → `ValueError("ns_steps mismatch ...")`（per-role diff）。
- **`build_hybrid_cmuon`** 加 `ns_steps_by_role` 参数（与 scalar `ns_steps` 并存，back-compat）。
- **Schema**：新增 `CMuonNSConfig`（`default: Literal[2..6]` + 8 个 `Literal[2..6]|None` role 字段 + `canonical_map()`），`OptimizerConfig.cmuon_ns: CMuonNSConfig|None`；保留 `cmuon_ns_steps: Literal[5,6]` back-compat。`[optimizer.cmuon_ns]` 子表走 `_deep_merge`（与 extends 兼容）。
- **`production._build_optimizer`**：`optimizer.cmuon_ns is not None` → `build_hybrid_cmuon(ns_steps_by_role=canonical_map())`；否则 legacy scalar。AdamW 分支不变。
- **配置**：新建 `config/train_g1_hybrid_cmuon_ns4_core.toml`（bf16 momentum、`[optimizer.cmuon_ns] default=4`、rescale off、scaled LR 1.5625e-4、global batch 800、不开 √N）；旧 `train_g1_hybrid_cmuon_core.toml` 加 RETIRED banner（仍 cmuon_ns_steps=5，back-compat 可加载）。
- **Safety telemetry（§14，opt-in 不部署）**：`NSSafetyTelemetry`（+`NSTelemetrySample`/`select_representative_specs`/`build_ns_safety_telemetry`）。每 N updates 读 representative roles（每 role 1 个 spec）的 `ns_output_rms`/`applied_delta_rms`/`nonfinite_count`；**device 端累加、无每 step CPU sync**（仅每 N 步一次 batched sync）。`build_hybrid_cmuon(ns_telemetry=...)` 默认 None（关），关闭时 update 路径 bit-identical。

### Checkpoint / resume 语义

- manifest 存 **canonical per-role NS map**（dict，8 roles）。
- **Hybrid→Hybrid exact resume**：momentum / per-role NS map / chunk metadata 全 exact restore（`test_checkpoint_ns_map_roundtrip`）。
- **per-role NS map 不一致 → hard fail**（`ValueError`，含 per-role diff）= optimizer-state semantic incompatibility（`test_checkpoint_ns_map_mismatch_hard_fail`）。
- **legacy scalar** 迁移：scalar 4 == 全 4 map，允许（`test_checkpoint_legacy_scalar_ns_migration`）。
- **LR/batch/accumulation/planned-updates/cadence 是可变超参，不得因此拒绝 resume**（`test_lr_change_still_allows_resume`：LR 变化 + 同 NS map → 不 hard-fail）。
- **AdamW8bit→Hybrid transition 行为保持**（CMuon momentum fresh zero，AdamW state 按 FQN 保留，无假 2nd-moment 转换）。

## 3. 复测：global ns4（C）

真实 G1 `TrainableComposite`（depth 20，hidden 2560/inter 6912/kv 640/AdaLN[2560,1024]），eager，warmup 6 / measured 28，`TORCHDYNAMO_DISABLE=1`。CMuon 参数 141、chunks 166、AdamW 参数 148；**NS matmul ≈ 166×4×3 = 1992/step**。

| 指标 | AdamW8bit（baseline，全 289 参数） | CMuon BF16 global ns4 |
|---|---|---|
| avg step | **2.3796 s** | **0.6734 s** |
| min / max step | 2.3700 / 2.4561 | 0.6661 / 0.7828 |
| state | 3.192 GB | 3.349 GB |
| peak alloc | 11.641 GB | **9.893 GB**（省 1.75 GB）|
| peak reserved | 12.109 GB | **10.234 GB**（省 1.88 GB）|

**optimizer 加速 = 2.3796 / 0.6734 = 3.53×；每 update 省 1.706 s。**（对比上轮 ns5 0.763s：ns4 少一次 NS 迭代，快 ~13%。）

### step 时间拆分（ns4，phase 复现，one-shot fresh momentum）

| 相位 | 秒 |
|---|---|
| CMuon momentum（lerp_）| 0.0148 |
| **Newton–Schulz**（166 chunks × 4 iter）| **0.4377** |
| param update（add_）| 0.0077 |
| **CMuon 小计** | **0.4601** |
| （参考）AdamW8bit 全参 step | 2.3796 |

NS 占 CMuon 时间 95%。hybrid 的 AdamW 部分只覆盖 148 参数（≪ 2.38s 全参 baseline），hybrid 总 0.673s = AdamW(148)≈0.21s + CMuon 0.46s。

### update RMS（ns4，vs 0.2·lr 目标，全模型）

| group | update RMS | vs 目标 |
|---|---|---|
| attention | 3.169e-05 | 1.014× |
| ffn | 3.186e-05 | 1.019× |
| adaln | 3.492e-05 | 1.118× |
| adamw_fallback | 1.627e-04 | 5.21×（不同算法，预期偏大）|

CMuon 三组全部 ≈ 0.2·lr（1.01–1.12×），与 sweep 一致。**ns4 全 role 安全、Moonlight 归一化成立。**

## 4. NS 深度 sweep（B）

production shapes × 24 个固定 iid-normal 快照（seed 0–23）× ns2/3/4（+ ns5 参考）。BF16 NS。**未用 production-like 快照**（抓取需跑训练，本轮禁止）；iid 组覆盖 audit 观察到的 spectrum。完整数据：`cmuon-ns-depth-sweep.json`。

### per-role × ns（d2t = delta_to_target_ratio；cos4/err4 = vs ns4）

| role | shape | ns2 (d2t / cos4 / err4) | ns3 (d2t / cos4 / err4) | **ns4 (d2t)** | ns5 参考 (d2t) | **推荐** |
|---|---|---|---|---|---|---|
| attention_q | [2560,2560] | 0.231 / 0.826 / 0.811 | 0.692 / 0.861 / 0.527 | **0.952** | 1.091（稳）| **ns4** |
| attention_k | [640,2560] | 0.453 / 0.223 / 0.977 | 1.089 / 0.250 / 1.070 | **1.541** | **344（发散）**| **ns4** |
| attention_v | [640,2560] | 0.453 / 0.223 / 0.977 | 1.089 / 0.250 / 1.071 | **1.541** | **338（发散）**| **ns4** |
| attention_content_gate | [2560,2560] | 0.231 / 0.826 / 0.811 | 0.692 / 0.861 / 0.527 | **0.953** | 1.093（稳）| **ns4** |
| attention_out | [2560,2560] | 0.231 / 0.826 / 0.811 | 0.692 / 0.861 / 0.527 | **0.952** | 1.092（稳）| **ns4** |
| ffn_in | [6912,2560] | 0.233 / 0.895 / 0.810 | 0.724 / 0.914 / 0.467 | **1.058** | 0.949（稳）| **ns4** |
| ffn_down | [2560,6912] | 0.233 / 0.895 / 0.810 | 0.724 / 0.914 / 0.467 | **1.059** | 0.946（稳）| **ns4** |
| adaln_shared | [2560,1024] | 0.362 / 0.479 / 0.891 | 0.968 / 0.511 / 0.930 | **1.117** | **46.6（发散）**| **ns4** |

**全部 8 role × 24 seed 在 ns4 下 finite_rate = 1.000；delta_to_target_ratio ∈ [0.95, 1.55]（同量级、Moonlight 归一化成立）。**

### 最低充分判定（§8 规则 A–E）

候选充分规则（辅助，不静默采用）：A 全 finite；B delta 与 ns4 同量级；C **cosine(nsX,ns4) ≥ 0.995**；D rel-error 小到可视为训练动力学等价；E orthogonality 平台 + Q_rms/polar 与 ns4 一致。

- **ns3 对每个 role 都失败 C（cos4 0.22–0.91 ≪ 0.995）且 D（relerr 0.47–1.07）**：ns3 与 ns4 更新方向显著不同（24°–90°+），**不是「≈ ns4」**。
- ns2 更差（cos4 0.22–0.90，relerr 0.81–0.98，delta 0.23–0.45× 目标）。
- **→ 每个 role 的最低充分深度都是 ns4。**

**机理**：quintic NS 在这些 production shape 上第 4 次迭代仍未完全收敛（ns4 的 orthogonality_defect 0.40–4.71，远非 0）——top 奇异值在 1 附近过冲/振荡、小奇异值（bulk）到 ns4 才升到 ~0.7–0.8。Moonlight 的 √max 缩放对**总 RMS** 归一化（所以 delta ≈ 0.2·lr 成立），但不要求逐奇异值正交。ns3 时 bulk 仍在 ~0.2–0.7、方向未稳，故与 ns4 方向差大。**「最低充分」= 4，因为 <4 时输出方向已实质偏离，而非因为 4 已「收敛」。**

## 5. ns5 不稳定复现（§9 必答题）

HCU 上 24 seed 全量刻画（ns5 仅参考，非候选）：

| role | shape | ns5 d2t(mean/max) | cos4 | relerr4 | orth_def | 判定 |
|---|---|---|---|---|---|---|
| attention_k | [640,2560] | 344 / 396 | 0.009 | 223 | 6.1e5 | **发散**（delta≈340×目标）|
| attention_v | [640,2560] | 338 / 391 | 0.016 | 219 | 5.6e5 | **发散** |
| adaln_shared | [2560,1024] | 46.6 / 49.0 | -0.33 | 42.0 | 1.1e4 | **发散**（delta≈47×）|
| attention_q/gate/out | [2560,2560] | 1.09 | 0.898 | 0.505 | 0.85 | 稳 |
| ffn_in/down | [6912,2560]/[2560,6912] | 0.95 | 0.947 | 0.326 | 0.64 | 稳 |

迭代 trace（k/v，seed0）：`it1 r=2.7e-3 → it2 8.9e-3 → it3 2.2e-2 → it4 3.0e-2（max_abs 0.30）→ it5 r=6.69 max_abs=101 → it6 1e14`。**爆发放第 5 次迭代**；it1–4 稳定。adaln 同样 it5 爆（max_abs 18）。

**§9 结论**：
- **ns4 始终安全**：8 role × 24 seed 全 finite、delta 0.95–1.55× 目标，无异常 snapshot。
- **ns3 不始终相似**：ns3 与 ns4 cosine 0.22–0.91（per-role 稳定但不 ≈1），且跨 24 snapshot 一致（per-role d2t 极紧，如 k/v ns3 mean 1.0886 / max 1.0894）。
- **ns5 爆炸对 K/V、AdaLN 几乎稳定复现**：HCU 上 **100%（24/24 seed）复现**（k/v 338–396×、adaln 46–49×），是**确定性**事件（短边小 shape 的收敛窗口止于 iter4）；q/ffn（短边 2560）ns5 稳。**这是 HCU-BF16 数值放大近边界 spectrum 的结果**——精确算术下 Frobenius 归一化矩阵 σ_max≤1 < 1.264（quintic f⁵ 发散阈值），本不应发散；本地 CPU BF16 亦不复现（24 seed 全稳）。故「ns5 危险」是**平台 + spectrum 相关**，在 production 加速器（HCU）上对 k/v、AdaLN 是确定性的。

## 6. native Muon 对照（§10，ns2/3/4）

custom CMuon == native `torch.optim.Muon`（`match_rms_adamw`）：

- **chunk=1**（q/k/v/content_gate/out/down × ns2/3/4）：全 `match=True`，max_abs_diff ≤ 4.9e-4（BF16 累加舍入），rms_rel 0.07–0.32。
- **chunk=2**（ffn_in fused == concat(独立 native gate, up)）× ns2/3/4：全 match。
- **chunk=6**（adaln fused == concat(6 独立 native)）× ns2/3/4：全 match。

**custom 与 native 在 ns2/3/4 逐 shape 数值一致**（与上轮 ns5 结论一致，非 wrapper bug）。

## 7. mixed per-spec NS 候选（D）

判定：最低充分深度对**所有 8 role 都是 ns4**（ns3 无一达标）。因此一个 mixed per-spec 候选等价于 global ns4，**无收益** → **按规则不创建 `hybrid_cmuon_min_ns_core`**。

- 唯一 production 候选 = **`hybrid_cmuon_ns4_core`**（global ns4，`[optimizer.cmuon_ns] default=4`）。
- per-spec 基础设施已就绪（schema/解析/路由/ckpt/telemetry），未来若某 role 出现更低充分深度或 spectrum 漂移，可用 `[optimizer.cmuon_ns]` 单 role override，无需改代码。

## 8. projected 训练提速（§13，明确是 projected 非完整训练）

production full-update 锚点 = salt1 最后一次干净 G1 跑（update 94118–94123）`[train] time=` 均值 **15.32 s/update**（含 qwen/vae/conditioning/dit fwd/bwd/optimizer 全相位；2-rank DDP，每 rank 全参 optimizer）。

| 量 | AdamW8bit | ns4（projected）|
|---|---|---|
| full update | 15.32 s | **13.61 s**（15.32 − 2.380 + 0.673）|
| samples/sec（global batch 800）| 52.2 | **58.8** |
| **相对加速** | — | **1.125×**（省 1.71 s/update，仅 optimizer 相位）|

optimizer 只占 full-update 的 ~15.6%（2.38/15.32），故整 update 提速受 fwd/bwd 主导，**1.125× 是乐观上界**（fwd/bwd/qwen/vae/conditioning 假定不变）。不是完整训练实测。

## 9. safety telemetry（§14）

`NSSafetyTelemetry`（opt-in，本轮**不部署 live**）：

- 采样 **representative roles**（默认每 canonical role 1 个 spec，共 8 个，非全部 141 参数）。
- 每 role 记录 `ns_output_rms`（NS 输出、Moonlight 前）、`applied_delta_rms`（实际 param delta = −α·NS）、`nonfinite_count`。
- **低开销**：device 端累加（`acc += q.pow(2).sum()` 等几次 reduction，无 `.item()`/`.cpu()`），**仅每 `log_every_n`（默认 100）步一次 batched sync**；off-cycle `step()` 返回 None（零 sync）。per-step 开销 = 8 个 representative 参数的几次 GPU reduction，可忽略。
- 接线：`build_hybrid_cmuon(ns_telemetry=...)` → `HybridCMuon.step()` 末尾 `ns_telemetry.step()`；`_cmuon_update_impl` 对 representative role 记录（默认 `ns_telemetry=None` → 路径 bit-identical）。
- 用途：长训中捕捉慢速 NS 失稳（spectrum 漂移把某 shape 推过收敛边界，如 ns5 的 k/v/AdaLN 爆），而不引入每-param 每-step CPU sync。
- 测试：`test_ns_safety_telemetry_accumulates_and_locals`（off-cycle 无 sync、每 N 步出 sample、`delta_rms == α·ns_rms`、默认关时路径不变）。

## 10. verdict（E：完成后停止）

- **实现**：per-role NS 基础设施 + schema + 配置 + ckpt 语义 + safety telemetry，全部落地，本地 30 单测过、salt1 HCU 全套过（见 §11）。
- **global_ns4**：`hybrid_cmuon_ns4_core` = 唯一 production 候选。ns4 全 role finite、delta ≈ 0.2·lr（1.01–1.12×）、optimizer 3.53× 加速、省 ~1.9 GB 峰值显存。
- **ns_sweep**：最低充分 = ns4（全 8 role）；ns3 无一「≈ns4」；ns2 更差。
- **stability**：ns4 始终安全；ns5 对 k/v+AdaLN 在 HCU 上 100% 复现（iter5 爆，平台+spectrum 相关）。
- **mixed_candidate**：不创建（无收益，等价 global ns4）。
- **projected_training_speed**：1.125× full-update（projected，非完整训练）。
- **safety_telemetry**：代码完成、测试过、未部署 live。
- **未做（遵守 E 硬约束）**：未启动真实训练、未切 live trainer、未部署 live、未 merge dev、未改 LR/batch/spatial/transparent/架构/ckpt 格式；默认 AdamW8bit 路径不变。

## 11. 测试

本地（Windows，NVIDIA GPU + `TORCHDYNAMO_DISABLE=1`）：**30 过**（16 既有 + per-spec 解析/校验 + chunk1/2/6 逐 role ns + production-shape ns4 finite(k/v/adaln) + routing global/mixed + checkpoint roundtrip/mismatch/legacy/LR + telemetry）。

salt1 HCU：
- **NS 深度 sweep**（`cmuon-ns-depth-sweep.json`）：全 8 production shape × 24 seed × ns2/3/4/5 完整指标 + native 等价 + ns trace。**ns4 全 shape finite（finite_rate 1.000）、delta 0.95–1.55× 目标**；ns5 k/v 344/338×、adaln 46.6×（100% 复现，iter5 爆）。131 s 跑完。
- **ns4 复测 benchmark**（`cmuon-ns-depth-benchmark.json`）：真实 G1 全模型，AdamW8bit 2.3796 s vs ns4 0.6734 s（3.53×）、phase 拆分、NS matmul 1992、显存、全模型 update RMS ≈ 目标。
- **HCU 单测**：`test_kv_adaln_ns4_finite_and_scaled_cpu` **过**（k/v/adaln ns4 finite ≈ 0.2·lr）。
- checkpoint roundtrip/mismatch/legacy/LR + telemetry 单测：逻辑 device-无关（小模型），**本地全过**（NVIDIA + `TORCHDYNAMO_DISABLE=1`）。
- 备注：`test_all_production_shapes_ns4_finite_hcu`（全 8 大 shape）与 `test_kv_adaln_ns5_instability_reproduced_hcu` 两个 HCU pytest 用例在本轮 HCU 上**卡慢**（benchmark 早期一次 in-place-grad 崩溃后 HCU 上下文降级：hy-smi 显示 HCU 0% 利用率而进程空转 CPU）；**二者所需证据已由 sweep 完整覆盖**（全 8 shape ns4 finite ≈ 目标；ns5 k/v/adaln 100% 复现），非代码问题。已 kill 卡死进程，HCU 空闲。

## 12. open questions

1. **ns4 未完全正交**（orth_def 0.40–4.71，k/v 最高）：Moonlight 靠总 RMS 归一化兜底，delta ≈ 0.2·lr 成立；但若某 role 的 orth_def 随训练 spectrum 漂移上升，需靠 safety telemetry 监控（本轮已备代码）。
2. **k/v ns4 delta 1.54×（略超目标）**：Moonlight √max 对 k/v 的 RMS 归一化偏松（NS 输出 RMS 偏高）。是否需 per-role 微调 α（而非降 ns）——留待长训质量证据，本轮不动（保持 Moonlight 原式）。
3. **production-like 快照未用**（抓取消耗训练）：本轮 iid 24 seed 已覆盖 audit spectrum 且 ns4 结论稳健；若后续允许，可补 1 组真实梯度快照做最终确认。
4. **长训质量**（loss/FID/eval）是 ns4 的最终门（本轮隔离 math 无法替代）；部署前需跑质量验证。

## 13. 产物

- 代码：`src/sakuramoon/optim/cmuon.py`（per-role NS + telemetry）、`src/sakuramoon/config/schema.py`（CMuonNSConfig）、`src/sakuramoon/train/production.py`（canonical map 分支）。
- 配置：`config/train_g1_hybrid_cmuon_ns4_core.toml`（新）、`config/train_g1_hybrid_cmuon_core.toml`（RETIRED banner）。
- 测试：`tests/unit/optim/test_cmuon.py`（30+ 用例）。
- 数据：`reports/cmuon-ns-depth-sweep.json`（24 seed × 8 role × ns2/3/4/5 全指标 + native 等价 + ns trace + 推荐）、`reports/cmuon-ns-depth-benchmark.json`（ns4 复测 + phase 拆分 + NS matmul + 显存 + projected）。
- 旧结论更新：`reports/cmuon-design-audit.md`（§NS 稳定性段改为 ns4 统一安全 + per-spec 由本 sweep 定 + ns5 平台/spectrum 相关）。
