# Guarded Canonical Hybrid CMuon v1 — 设计与验证计划

分支：`cmoun-guarded`（基于 `cmoun-forensic`@9e87cdf）。候选名：
**`hybrid_cmuon_guarded_canonical_ns4`**。
原候选 `hybrid_cmuon_ns4_core` **永久退役**（ORIGINAL_CANDIDATE_RETIRED=YES，
任何后续运行/评测不得再使用该名）。

本文件是"新候选"开发轮的设计定稿。guard 数值（ratio / reference_decay /
min_reference / numerical_floor / warmup）由 P3 shadow-gradient 校准
（salt1，ckpt_97100，≤100 fwd/bwd，无参数更新）决定后填入 §5.4；
**若校准无干净双峰分离，结论为 INCONCLUSIVE，不采用任意阈值**。

## 0. 根因基线（已结案，单位已统一）

根因（`reports/cmuon-root-cause.md`，cmoun-forensic@9e87cdf，08-31 单位审计
修正版）：NS 输入是 **Nesterov 矩阵** u = (1−μ)g + μm（μ=0.95），不是原始
梯度。近零信号 spec（slot_08.q_proj：nesterov rms 1.88e-9 / fro 4.82e-6 =
78×eps，**1e-7 clamp 未触发**，分母=真范数 4.828e-6）→ Frobenius 归一化
×2.0766e5（设计行为，抹掉幅度）→ 归一化弱信号矩阵（top-1 σ² 占 88.9%，
秩-1 主导 + 薄尾）落在 NS4 收敛边界：前 3 轮稳定（CPU fro 1→1.37→3.78→
8.80），**第 4 轮对 Gram 比特混沌敏感**——同一输入比特 5 次 HCU 实现 →
fro {125235, 16.3, 15.9, 15.9, 15.9}（~20% 灾难分支）。灾难不是 eps 地板、
不是输入表示噪声（输入跨 rank 逐位相同；nesterov 中位数离零 ~400 ulp），
而是 NS4 边界混沌 × GEMM 5–8% 噪声选分支。旧报告「fro≈1.5e-7 略高于 eps」
「×66k 噪声放大」两处量纲错误已修正（总放大 nesterov→delta ×1.675e5 =
2.0766e5 × 510.1 × 1.5811e-3）。

**推论（本候选的设计依据）**：输出侧（NS 后）幅度无法判别好坏
（141-spec 横截面：active 与 near-zero 簇的 ns_rms 分布几乎完全重叠）；
**唯一可操作的判别量是 pre-NS 信号幅度**。guard 因此作用在 NS 之前的
Nesterov 矩阵上；且由于分支概率与输入比特相关，"幅度低于参考" 的 spec
属于**混沌敏感类**（rank-1 弱信号 + 薄尾），整体跳过是该类的保守消除，
不是逐事件预测（spec17 g_rms 6.5e-9 落安全分支、spec52 4.7e-6 落灾难
分支的反例证明：幅度是类判别器，不是事件判别器）。

## 1. 不变约束（本轮硬性）

- 不动 live AdamW8bit 主训练（salt3 dst p50）、不部署、不 merge dev、
  不改模型架构 / NS 深度(=4) / 基础 LR / batch(=800) / Moonlight scaling /
  FFN chunk(=2) / AdaLN chunk(=6) / LR recipe；
- 无 sqrt(N_chunk)、无 ns3/5/6、无 FP32 momentum、无 FP32 NS "修复"、
  无手动 K/V alpha 削减、无 delta-clamp-continue、无 restart-until-luck、
  安全违规后不继续；
- 唯一允许的 guard 动作：low signal → **CMuon 参数 delta 置零 + momentum
  按定义继续更新**；数值异常 → fail closed + 停。v1 **禁止** low-signal
  → fallback AdamW（会改变参数更新语义与 checkpoint 迁移面）。

## 2. Guard 定义（pre-NS）

对每个 CMuon **NS 输入**（= (FQN, chunk)；chunk 划分与 NS 实际执行完全
一致：`spec.chunk_dim` / `spec.chunk_count`；单 chunk spec 即 (FQN, 0)）：

```
u_t   = (1-μ)·g_t + μ·m_t          # Nesterov 矩阵（生产同式，bf16）
sig   = rms(u_t)                   # 主判别量（FP32 归约）
sig_f = fro(u_t)                   # 辅助绝对量
```

**低信号判定**（两条独立成立即跳过；均 rank 一致）：

```
low_signal ⟺  sig < guard_ratio · ref          (相对，主)
        ∨     sig_f < numerical_floor          (绝对，次)
```

- **ref**：per-(FQN, chunk) FP32 `signal_reference`（见 §5.3）。
- 无固定绝对 1e-X 作主判据；`numerical_floor` 只是数值地板（防 ref 本身
  塌缩到 bf16 噪声带的边界情况），量级由校准确定。

**low_signal 动作**：跳过该 NS 输入（不跑 NS、delta=0、参数不变），
**momentum EMA 照常更新**（`m ← m.lerp_(g, 1−μ)`；这是 optimizer state，
不是参数）；skip 计数 +1（total / per-role / per-slot）。
active 动作：正常 NS4 → Moonlight → 安全检查 → 提交（§4）。

**rank 一致性**：guard 决策输入 = 全 reduce 后的 grad（跨 rank 逐位相同）
+ 本地 momentum（bf16 lerp，元素级确定性；P3 校准逐 update 用 device
MIN/MAX all-reduce 实证 rank 差 = 0，非零即 hard fail）。因此
low_signal 决策天然跨 rank 一致，无需额外通信。

## 3. Canonical owner-rank NS（消除 rank 间 NS 非确定）

- **owner 映射**：`owner((FQN, chunk)) = stable_hash(f"{fqn}#chunk{i}") %
  world_size`（FNV-1a 64，稳定、与进程/启动顺序无关）。同一 ckpt/config/
  world_size 下映射不变；**映射规则 + 版本号写入 checkpoint manifest**
  （`owner_mapping_version`）；world_size 变化 ⇒ 映射变化 ⇒ 视为新的
  canonical 配置（ckpt 兼容性检查拒绝跨 world_size 直续，需显式迁移）。
- **执行流**（v1 按 spec 顺序，正确性优先，不做 overlap）：
  1. 所有 rank：grad 校验 → momentum 候选 → guard 决策（§2）；
  2. 对每个 active NS 输入：owner rank 执行 chunked NS4（生产同式）+
     Moonlight alpha + 安全校验（§6）→ **broadcast canonical delta
     (bf16, 该 chunk 的 (-alpha·NS) 整块)** → 所有 rank 应用同一 delta；
  3. 非 owner rank 不执行 NS（省算力 + 从根上消除 rank 间 NS 分岔）。
- **跨 rank 验证**：broadcast 后所有 rank 对 delta 做指纹
  （rms/max/probe-dot），任何不一致 → fail closed。
- **momentum**：所有 rank 各自本地 EMA（与生产位级一致），每 N update
  做一次一致性检查（per-spec rms / max / probe dot 的 MIN/MAX all-reduce）；
  超出 HCU 元素级噪声容差（**预期 = 精确 0；P3 提供实证**）→ hard fail。
  bf16 lerp rank-精确的证据记录在 benchmark/safety 报告中。
- **通信成本**（v1 顺序模式，P6 实测）：每 spec 一次 broadcast
  （chunk 字节 = numel×2）；记录 comm_s / ns_s / guard_s / 总 optimizer_s。

## 4. 两阶段原子提交

**PHASE 1 — PREPARE（全部 141 spec，先算全量、后决定）**：

1. 校验：全部 grad 有限（与生产 `_validate_finite_gradients` 同式）；
2. momentum 候选：`m_cand = m.lerp_(g_md, 1−μ)` 的**候选值**
   （实现：先算 `m_cand = (1−(1−μ))·m + (1−μ)·g` 到临时缓冲，或保留旧值
   备回滚——二选一在 P4 实现时定，语义必须等价：未通过则 m 不变）；
3. guard 决策（§2，用 m_cand 算 nesterov）；
4. owner NS4 + Moonlight（仅 active）→ delta 候选；
5. 安全校验全部 delta（§6）；
6. broadcast + rank 一致性验证（§3）。

**任一 spec 失败 → 本 update 一个参数都不更新：CMuon delta 不应用、
AdamW 不 step、momentum 不提交**；抛 `CMuonSafetyError`（带 spec/原因/
统计），rank0 dump（复用 forensic dump 目录布局），**停**。

**PHASE 2 — COMMIT（全部 PASS 才进入）**：

1. AdamW 部分 step（生产路径，SR RNG 下）；
2. 全部 CMuon delta 应用（`param.add_(delta)`，bf16，生产同式）；
3. momentum 状态提交（候选→正式）；
4. 参数指纹验证（§8）：提交后再取一次 per-spec 指纹并比对。

## 5. signal_reference 与 bootstrap

### 5.1 语义

ref 表示"该 NS 输入处于 active 模式时的特征信号尺度"。guard 用
`sig < guard_ratio · ref` 判定"远低于自身 active 尺度"= 近零/混沌敏感类。

### 5.2 更新规则

```
active:    ref_t = max(sig_t, ref_{t-1} · reference_decay)
inactive:  ref_t = ref_{t-1}                 # 不衰减到 0
```

`reference_decay ∈ (0,1]` 接近 1（慢松弛）；长期 active 的 spec ref 跟踪
其信号尺度；inactive streak 期间 ref 恒定（**绝不衰减到 0**）；回 active
立即再入 Muon（ref 若已被慢松弛压低，由 `max(sig_t, ...)` 立即抬回）。
`ref ≥ min_reference` 恒成立（构造时夹底）。

### 5.3 存储

per-(FQN, chunk) FP32 标量，随 checkpoint 保存（§10）；141 spec 合计
< 240 个标量。

### 5.4 数值（P3 校准后填入）

来源：salt1 校准 JSONL（≤100 update，per-(FQN, chunk) nesterov_rms 时序）：
- 每 spec 的 inactive/active 双峰分离度（谷底 vs 两簇）；
- `min_reference` = 全 spec inactive 簇 p99 与 active 簇 p1 之间的谷底
  附近（取能最大间隔处）；
- `guard_ratio` = 谷底/该 spec active p10 的比值带内取值（报告选定逻辑）；
- `reference_decay` / `warmup_observations` = 由 ref 收敛速度定（报告）。
**若无干净双峰分离 → INCONCLUSIVE，不采用任意阈值，报告说明**。

## 6. 写前安全检查（fail closed，无 clamp）

对每个 active NS 输入的 delta（PHASE 1 内、任何参数写之前）：

- NS 输出有限（逐元素 `isfinite`，bf16 域）；
- delta 有限；
- `delta_rms ≤ 10 × target_delta_rms`，其中
  `target_delta_rms = 0.2 × lr`（正常 NS4 审计值 0.95–1.55× target）；
  超界 → `CMuonSafetyError`（**不 clamp、不继续**）。

记录 `guarded_skip_count/rate`（total + per-role + per-slot），
每 10 update 汇总（S1/S2 输出）。

## 7. 内存方案（先报方案，无静默降级）

- **staged flat bf16 delta 缓冲**：全部 active delta 暂存为一个 flat
  bf16 张量（或 per-spec 视图），PHASE 1 全量安全校验通过后 PHASE 2
  一次性应用。预算：CMuon 参数总字节 × 2（bf16）——P6 实测
  `staged_delta_bytes / peak_allocated / peak_reserved`；预算上限 ~3GB。
- 若无法一次 staged（实测超出）：**分块 staging，但原子性不降级**——
  分块只在"全部块已通过安全校验"后按块应用；任何块失败 ⇒ 整个 update
  回滚语义（未应用块不应用 + 已应用块**不部分回滚**而是按 §4 规则：
  PHASE 2 之前不应用任何块 ⇒ 分块应用发生在全部校验通过之后，失败面
  只剩设备级异常，此时按生产 failure 语义停）。方案在 P6 benchmark 前
  写入报告，不做静默 partial-commit。

## 8. rank 不变量（S1/S2 每步）

- 每个成功 update 后：全部 141 CMuon spec 的跨 rank 参数指纹
  （RMS / max / 固定种子 probe dot，FP32 归约 + 小 all-gather），
  输出 `first_param_rank_diff` / `max_param_rank_diff`（相对差）；
  **目标：0 持续漂移**；任何 >0 → 立即停（S1/S2 规则，不 restart）。
- 抽样 AdamW 参数同法（每 10 update，per-slot 抽样）。

## 9. 等价性保证（数学检查）

对 **active** 信号：GuardedCanonicalCMuon 的更新 == 原核心 CMuon NS4
更新（差异仅为 owner broadcast 带来的"同一值"）：

- guard 不改变 active 路径的任何算式（momentum / nesterov / NS /
  Moonlight / alpha / add 全部生产同式）；
- owner 的 NS 输出在确定性机器上 == 任意 rank 自行计算的输出（HCU 上
  这正是 canonical 化的目的：只保留一份实现结果）；
- 测试（P5）：
  A. zero grad → skip（无 NS、delta 0、momentum 更新）；
  B. 近零 bf16 噪声 grad → skip；
  C. 正常 active → NS4 更新（与原核心候选逐元素一致，CPU 确定性下
     bit-exact 比对）；
  D. ref 在长 inactive streak 后稳定（不衰减到 0）；
  E. inactive 期间 momentum 照常更新；
  F. active 恢复 → 重新进入 Muon；
  G. rank0/rank1 同信号 → 同决策（2×HCU）；
  H. checkpoint save/load 后 ref/计数/momentum 精确恢复。

## 10. checkpoint 语义

新增 manifest 字段（guard schema v1）：

- `guard_schema_version`；
- per-(FQN, chunk) `signal_reference`（FP32）；
- guard config（ratio / reference_decay / min_reference /
  numerical_floor / warmup_observations）；
- observation/bootstrap 状态（bootstrap 完成标志 + 观察计数）；
- skip counters（total / per-role / per-slot）；
- `owner_mapping_version`（stable_hash 规则版本）；
- canonical NS 模式标志 + per-role NS map（= 全 4，显式存储）；
- momentum 状态（bf16 buffers，生产同布局）+ AdamW 状态（原样保留）。

**恢复规则**：

- guarded → guarded：精确恢复（ref/计数/momentum/owner 映射版本必须
  一致；world_size 不一致 ⇒ owner 映射变化 ⇒ 拒绝直续，需显式迁移）；
- **旧无 guard CMuon ckpt → 新候选：不得直接 resume**——必须显式
  optimizer transition + guard reference 重新 bootstrap（记录
  transition artifact，同 AdamW→CMuon transition 语义）；
- AdamW → guarded CMuon（S1 场景）：AdamW state 按 FQN 保留，CMuon
  momentum 全新（零初始化，生产同），guard reference 走
  **校准 bootstrap（方案 A，首选）**：直接用 P3 校准产物初始化 ref
  （每 spec 的 active 模式 p50/p90，见 §5.4），并记录为
  "optimizer transition bootstrap"（**不是 LR warmup**）；方案 B
  （观测 bootstrap：前 `warmup_observations` 次观察建立 ref）作为
  后备，若采用必须在 manifest 标注。

## 11. 命名与 W&B

- 原候选 `hybrid_cmuon_ns4_core` / run `g1_cmuon_ns4_from_97100`
  永久退役，绝不复用名称；
- 新候选所有运行：run_id 前缀 `g1_cmuon_guarded_canonical_*`
  （校准：`g1_cmuon_guard_calibration`；S1：`..._s1`；S2：`..._s2`；
  benchmark：`..._bench`），独立 W&B run，绝不覆盖原候选 run。

## 12. 验证阶段（P5–P8）

- **P5 测试**：§9 A–H（CPU 确定性 + 2×HCU 双 rank）；2×HCU 对照：
  旧候选允许出现 rank-local NS 差（控制组），新候选必须
  `delta_rank_diff = 0` / `param_rank_diff = 0`（尽管 HCU GEMM 非确定）；
  DTK 非确定性记录：重复 GEMM 变差（rel_rms 5–8% 基线）+ 平台 NS
  非确定**不得**转化为 DDP 参数非确定；salt1 全量测试套件绿。
- **P6 benchmark**（真实 1.57B，正确性之后）：AdamW8bit / 旧无 guard
  CMuon（仅历史参考）/ 新候选；分项秒数（momentum / guard / NS /
  broadcast / staging-commit / 总 optimizer）+ state/staged/peak 内存；
  **不得引用旧 "0.67s optimizer" 代表新候选**。
- **P7 S1**（200u 安全门）：从 salt1 健康 AdamW COMPLETE ckpt
  （`ckpt_97100_raw-97100-update-cadence`）独立分支，≤200 成功 update，
  2 rank，同生产 model/LR/batch/数据策略；每步全 rank 参数一致性（§8）；
  每 10 update guard + NS 统计（skip rate per-role/slot、delta_rms 分布）；
  **任何** rank 漂移 / 非有限 / delta 超限 / 灾难性 loss 跳变 → 立即停，
  不 restart。
- **P8 S2**（500u）：S1 PASS 后 500 成功 update 安全运行；500 稳定 ⇒
  `READY_FOR_1K_QUALITY_GATE = YES`（本轮不做 5k A/B）。

**S1 PASS 判据**：0 nonfinite；0 rank drift；0 灾难性 NS delta；guard
生效且 skip 集中于真实近零 spec；active spec 仍走 CMuon；无单步 4×
loss 跳变；grad norm 合理；≥1 个可精确 resume 的 guarded hybrid ckpt。

## 13. 交付物

- 本报告（设计）：`reports/cmuon-guarded-canonical-design.md`（本文件，
  P3 后更新 §5.4 校准数值）；
- `reports/cmuon-guarded-canonical-benchmark.json`（P6）；
- `reports/cmuon-guarded-canonical-safety.json`（P7/P8 汇总）；
- `=== COPY TO CHATGPT: GUARDED CANONICAL CMUON ===` 终报块（P9）。

## 14. 开放问题（随阶段更新）

1. （P3）近零簇与 active 簇的分离度是否足以定阈值？spec 级 vs role 级
   参考（per-spec ref 样本少时是否回退 role 级 p90）？
2. （P4）momentum 候选的"暂存-回滚"实现：临时缓冲 vs 旧值快照
   （显存成本 141×numel×2B ≈ 与 delta 缓冲同量级；是否复用 staged 缓冲）。
3. （P6）顺序 broadcast 的通信占比 → 是否需要 v2 overlap（本轮不做）。
4. （P7）skip rate 过高（>50% specs 长期 skip）⇒ guard 过紧 ⇒ 回到
   校准重定阈值（不在线调参）。
