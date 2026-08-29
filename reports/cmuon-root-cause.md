# CMuon Root Cause Report — G1 Hybrid CMuon 候选数值不稳定（结案）

**日期**: 2026-08-29 · **宿主**: salt1 (172.31.73.15, 2× HCU, DTK 26.04)
**候选身份**: 远端 `Cmoun` @ f27fde9（代码 3967ade），salt1 原始树字节级验证未动
**取证分支**: `cmoun-forensic` @ bff14ab（插桩独立分支，候选数学按位不变）
**判定**: **ORIGINAL_CANDIDATE_SAFE = NO（机制级确认）**

---

## 1. 结论摘要

1. **根因机制（已实证，单位已于 08-31 从存档张量重算修正）**：候选的 NS 路径对
   **近零信号**参数存在灾难性放大。`cmuon_zeroth_power` 的输入归一化
   `ortho = nesterov / nesterov.norm().clamp(min=1e-7)`（输入是 **Nesterov 矩阵**
   u=(1-μ)g+μm，不是 raw grad）。F1 事件张量（slot_08.q_proj, 2560×2560,
   numel 6,553,600）实测：nesterov element_rms=1.8821e-9、**Frobenius
   norm=4.8155e-6**（= 78× eps，**未触发 clamp**，归一化分母=真范数
   4.8280e-6（bf16 归约））。归一化按设计抹掉幅度（×2.0766e5）——对健康
   梯度这是无害的方向归一化；但近零信号的归一化矩阵（top-1 奇异值占能量
   88.9%、细长尾到 7e-12）落在 **NS4 五次迭代的收敛边界**：前 3 轮迭代
   fro 1.0→1.37→3.78→8.80（CPU 确定性复现），**第 4 轮对 Gram 矩阵的
   bf16 比特敏感（混沌分岔）**——同一输入比特在 3 个 GEMM 实现下分岔为
   ×1.95（fro 17.2，CPU）/×1.98（fro 17.6，HCU rank1）/×57.9（fro
   510.4，HCU rank0），由 HCU bf16 GEMM 5–8% 非确定性噪声决定 → 两 rank
   得到 29× 不相关的更新（delta_rms 3.15e-4=10× Moonlight target vs
   1.09e-5；单元素 delta 0.4395=1395× delta_rms，对 ~0.02 量级权重为
   一步数千% 扰动）。**修正旧表述**：原报告「Frobenius≈1.5e-7 仅略高于
   eps」「放大 ×~66,000」两处量纲错误——真实分母是 4.8e-6（无 clamp），
   放大链为 ×2.0766e5（归一化）× ×510.1（NS4 坏分支）× 1.5811e-3
   （alpha）= nesterov→delta 总 ×1.675e5；灾难性放大来自 **NS4 边界
   混沌分支**，不是 eps 地板、也不是输入侧 bf16 表示噪声（输入比特跨
   rank 逐位相同；nesterov 元素中位数离零 ~400 ulp，是有结构的弱信号）。
2. **硬件放大器**：DTK/HCU 的 bf16 GEMM **非确定**（实测：同设备、同输入、
   硬同步重复 2560×2560 matmul，rel_rms 5–8%，~2.4% 元素不同；norm 归约确定）。
   候选按设计在每个 rank 独立跑 NS 且**更新后无跨 rank 同步** → DDP「两 rank
   同模型」不变量从第一个 update 起被破坏，rank 参数漂移持续积累（已逐 update
   记录 `rank_drift`）。
3. **与 3 次崩溃的对应**：崩溃 = 非确定噪声实现 × 非确定 kernel 选中某个
   「坏轨迹」（幅度大到触发 global gradient norm nonfinite / loss 突变）。
   3/~165u ≈ 1.8% 崩溃率、MTT~55u、clip.py:48 同崩点、rank1 为主，与该机制
   一致（静默写入垃圾 delta 若干步后，漂移+后续事件累积到不可恢复）。
4. **F2（安全路由 K/V+adaln_shared→AdamW）不足以排除该缺陷**：F2 在 fork 后
   第一个 update 即触发 fail-closed，故障 spec = `slot_08.attention.q_proj`
   （**Q 角色**，F2 未排除），且为**双 rank** 真实 ceiling 违规（非插桩误报）。
   机制是**角色无关**的（任何走到 CMuon 的近零梯度 spec 都会中招）。
   按决策树 **F3 不执行**（F3 仅在 F2 稳定时拆）。
5. **纪律**：报完即停 —— 不部署、不做线上切换、不设计新候选（用户 08-29 拍板）。
   候选**不可用**；修复方向见 §8（仅记录，未实施）。

---

## 2. 背景：崩溃史与 H1/H2 鉴别

### 2.1 三次崩溃（同一候选、同一 fork 点 ckpt_97100）

| # | 宿主 | update | 形态 | 细节 |
|---|------|--------|------|------|
| 1 | salt1 | 97131 | 突发型 | `FloatingPointError: global gradient norm nonfinite` @ clip.py:48 ← step.py:403 `finish_update`，rank1；97131 两次独立通过=非确定性 |
| 2 | salt1 | 97178 | 突发+per-sample inf | 同崩点，rank1 |
| 3 | salt2（全新健康宿主，实验 A） | 97147→97157 | 发散型 | 97147 loss 0.559→2.544（单步 4.5×），97148–97156 持续 2.1–2.4（10u 未恢复），~97157 同崩点 nonfinite |

- 崩溃率 3/~165u ≈ **1.8%**，MTT ~55u → 跑完 98100 安全门需 ~18 次重启、
  102100 需 ~90 次 = **CMuon 臂无法完成 A/B 窗口**（实验 A 结论）。
- **H1（候选数值不稳定）确认**；**H2（salt1 宿主问题）排除**：salt2 健康宿主
  独立复现；**数据排除**：CONTROL（salt3，AdamW8bit，同数据区 97620–97633）
  preclip 0.037–0.072 完全正常，0 崩溃 / 17,143+ updates。

### 2.2 时间线披露（不影响结论）

- **CONTROL 误杀**：实验 A 筹备期间的宽 pkill 波及 CONTROL 栈；CONTROL 于
  08-29 13:50 从 salt2 迁至 salt3 后经 `/root/start-g1.sh` 重启（dev@eabe608
  + rollover-fix），update 连续性已核对；崩溃排除结论所用的数据区对比窗口
  完整可用。
- **salt2 重建**：平台重启清盘后 salt2 以全新健康宿主身份重建为实验 A 宿主。
- **已作废的 3.53× 审计**：取证前的角色级审计曾给出 K/V 相关 3.53× 比值，
  作为「把 K/V 路由到 AdamW」（F2）的动机之一；该审计本身已被作废，且本轮
  取证实证缺陷机制**角色无关**（Q 角色同样触发），3.53× 不能作为根因指标。

---

## 3. 取证装置（cmoun-forensic @ bff14ab）

- **fail-closed 两段式守卫**：CMuon 部分拆为 phase1（纯计算、不落参、逐 spec
  记录）→ 跨 rank 比对 + 本地安全判定 → phase2（写参）。触发 → **每个 rank**
  在 all_reduce 判定后抛 `CMuonSafetyError`；rank0 dump JSON（+故障 spec 的
  phase1 张量 .pt）到 `/sakuramoon-runtime/artifacts/g1/forensic`。
  **永不 clamp-continue**。
- **遥测**：141 个 CMuon spec ×（grad/momentum/nesterov/ns/delta × rms/sum/max
  + finite/has-grad 标志）+ 每 spec NS 逐轮 norm trace + 探针点积 + 10-update
  环形缓冲 + 每步跨 rank `rank_drift`（全列最大相对差，含未超容差值）。
- **跨 rank 比对范围（最终规则，经两次实测校准）**：
  - 比较：五阶段 **rms/max**（尺度统计=真实损坏通道）+ finite/has-grad 标志
    （精确）+ 探针点积（对阶段 rms 的**绝对**容差）；
  - 仅记录不触发：**sum 列**（零居中输出的符号敏感聚合，近零梯度 spec 上
    硬件噪声的相对差无界——F2@97101 实测 ns_sum rel 347.9 而 ns_rms rel
    0.0017、delta 健康）；
  - 容差 `divergence_rel_tol=1e-1`（高于实测 HCU 噪声底 5–8%，真实缺陷为
    ≥100×，裕度充足）。
- **update 编号**：`update_offset=97100`（锚定恢复的 trainer 计数），dump
  文件名即真实 update。
- **候选数学按位不变**：phase1/phase2 与原单段 step 同算子同顺序（单测验证）；
  全程未改 LR/NS 步数/alpha/batch，未用 FP32 momentum/NS，未改 clip 阈值。
- **F1/F2/F3 同源**：全部从 ckpt_97100（COMPLETE+manifest 验证）启动。

### 部署与工具坑（披露）

| 坑 | 处置 |
|----|------|
| salt1 `/root/mig-workload-env`（崩溃 run 的环境快照）残留 `CONFIG_NAME=core toml`，栈启动时全量 `export` 覆盖继承 env → 第一次 F1 启动**实际加载了无插桩的原始配置**（resolved.toml run_id 暴露） | 从 env 文件剔除该键；此后 F1/F2 配置身份经 train 进程 environ + resolved.toml run_id 双重验证 |
| 固定 mtime 的 tar 部署 + 单字节代码改动（0.0→1.0）→ 旧 `.pyc` 被判有效，salt1 持续跑旧字节码 | 每次部署后清 `__pycache__`；部署文件 SHA 与 git blob 逐字节核对 |
| DTK 进程组**无 CPU backend** → 首版取证的 CPU 张量 collective 在 step0 崩（`No backend type associated with device type cpu`） | 全部 collective 改 device 张量 |
| 本地 Windows torch 无 triton → 所有 torchao-quantized hybrid CUDA 测试本地必败（12 个，既有环境问题）；salt1 DTK venv 为权威门 | 全量套件以 salt1 为准（43/43 PASS，94.8s） |
| 本地 pyright 环境无法 strict-clean（基线 cmuon.py 52 errors 为环境性）；结构性问题已清零 | 报告中注明环境限定 |
| Hygon CPU bf16 慢（NS4 单 shape 291s）+ cgroup 30 核但 258 线程超订 | 用户拍板 NS 测试一律走 DCU/HCU；OMP_NUM_THREADS=8 |

---

## 4. 关键发现

### 发现 1：HCU bf16 GEMM 非确定（硬件事实，盐上实测）

2560×2560 bf16 `x @ x.T`，同设备、同输入、**每次调用间硬同步**：

| 设备 | 重复对 | rel_rms | 不同元素 |
|------|--------|---------|----------|
| cuda:0 | 1v0 / 2v0 | 6.2% / 7.4% | 154,519 / 170,272（/6.5M） |
| cuda:1 | 1v0 / 2v0 | 5.1% / 8.0% | 126,355 / 172,738 |

跨设备（cuda:0 vs cuda:1）同量级（NS 全链 out rel_rms 8.7% @ 2560×2560、
27.5% @ 2560×1280）；`norm()` 归约**确定**。小矩阵（2560×8）逐位一致。
**含义**：候选的 per-rank 独立 NS 在本硬件上**每个 update 都产生 rank 间
不相关的更新** → DDP 同模型不变量破坏，漂移随 update 积累。取证以每步
`rank_drift` 量化（F1 事件步 16.8、F2b 事件步 25.2 为事件值；正常步在
噪声底 0.1–8% 量级，见 ring）。

### 发现 2：近零梯度 → NS 噪声放大（根因机制，F1/F2b 实证）

**F1 @ 97101（原始路由 + 取证）**：本地故障（rank0），spec 52
`slot_08.attention.q_proj`（role=attention_q）`delta_rms_ceiling`：

| 列 | rank0 | rank1 | 说明 |
|----|-------|-------|------|
| g_rms / g_max | 1.9294e-8 / 1.2577e-5 | 同左（逐位） | 梯度**近零**（该 batch 该 slot 几乎未激活）；fp32 SVD：top-1 σ² 占能量 88.9%，长尾到 7e-12 |
| n_rms / n_fro | 1.8821e-9 / **4.8155e-6** | 同左（逐位） | nesterov=(1-μ)g+μm（μ=0.95，fresh momentum 时 =0.0975g）；Frobenius=78×eps，**未触发 clamp** |
| ns_rms / ns_max | **0.1994 / 278** | 6.9e-3 / 0.676 | NS4 收敛边界混沌分岔：fro 510.4 vs 17.6（×29.0）；CPU 确定性复现=fro 17.2（与 rank1 同支） |
| d_rms / d_max | **3.1518e-4（超 ceiling 0.9%）/ 0.4395** | 1.09e-5 / 1.07e-3 | rank0 单元素 delta=1395× delta_rms，对 ~0.02 量级权重为一步数千% 扰动 |

**F2b @ 97101（安全路由）**：**双 rank** 本地 ceiling 违规，同一 spec
`slot_08.attention.q_proj`（数据队首同一 batch）：

| 列 | rank0 | rank1 |
|----|-------|-------|
| g_rms | 2.54e-8（两 rank 逐位相同） | |
| ns_rms / ns_max | 0.418 / 644 | 0.244 / 408 |
| d_rms / d_max | 6.60e-4（**2.1× ceiling**）/ 1.016 | 3.86e-4（**1.2× ceiling**）/ 0.644 |

**F2（首跑）@ 97101**：无本地故障；仅 spec 2 `slot_00.q_proj` 的 `ns_sum`
跨 rank 相对差 14.3%（超旧版全列严格/统一容差）→ 属 sum 列病态（见 §3 校准），
非候选故障 —— 该事件用于校准比对规则，不作为候选证据。

### 单位审计（08-31，自存档张量 `cmuon-forensic-crash-97101-...-q_proj.pt` 重算）

| 量 | grad | momentum | nesterov | ns_output | delta |
|----|------|----------|----------|-----------|-------|
| shape / numel | 2560×2560 / 6,553,600 | 同左 | 同左 | 同左 | 同左 |
| element_rms | 1.9294e-08 | 9.6489e-10 | 1.8821e-09 | 1.9937e-01 | 3.1518e-04 |
| fro_norm | 4.9365e-05 | 2.4687e-06 | 4.8155e-06 | 5.1026e+02 | 8.0672e-01 |
| max_abs | 1.2577e-05 | 6.2957e-07 | 1.2293e-06 | 2.78e+02 | 4.3945e-01 |
| zero_frac / subnormal | 0.00091 / 0 | 同左 | 同左 | 0 / 0 | 0 / 0 |
| exp_median / ulp@median / ulpcount_median | -29 / 7.28e-12 / 256 | -34 / 2.27e-13 / 410 | -33 / 4.55e-13 / 400 | -7 / 3.05e-05 / 266 | -17 / 2.98e-08 / 430 |

- 恒等式验证：nesterov = (1-μ)g+μm（μ=0.95，max_abs_diff 2.4e-9=bf16 舍入）；
  delta = -alpha·ns（alpha=1.5811e-3，repro 3.1523e-4 vs 存档 3.1518e-4，
  0.02% 吻合）。
- **normalization_denominator = 4.8280e-06**（bf16 `.norm()` 真范数；
  fp32 参考 4.8155e-06，bf16 归约误差 +0.26%；eps=1e-7 **未 clamp**）。
- 放大链：×2.0766e5（=1/denom，归一化）→ ×510.1（NS4，rank0 坏分支；
  逐轮 fro 1.0→1.43→3.92→9.43→**510.4**，CPU 确定性复现
  1.0→1.37→3.78→8.80→**17.2**）→ ×1.5811e-3（alpha）
  = nesterov_rms→delta_rms 总 ×1.675e5。
- 谱结构（fp32 SVD）：grad/nesterov top-1 σ² 占能量 88.9%、top-10 94.7%，
  σ_top 4.66e-05/4.54e-06，σ[-1] 7.0e-12/3.2e-13（~1e7 倍跨度）。
- **旧报告错误修正**：「Frobenius≈1.5e-7 仅略高于 eps=1e-7」→ 实为
  4.82e-6（78×eps，无 clamp）；「×~66,000 放大 bf16 表示噪声」→ 实为
  方向归一化 ×2.0766e5（设计行为）+ **NS4 收敛边界第 4 轮混沌分岔
  ×510**（灾难源，由 HCU GEMM 5–8% 噪声实现决定）；输入侧并非「纯噪声」
  （nesterov 元素中位数离零 ~400 ulp，秩-1 弱信号有结构）。
- **HCU 混沌分支按需复现**（08-31，salt1，F2b 事件同 spec 张量，nesterov
  fro=6.35e-6，DTK bf16 `.norm()` 精确且确定：err -0.06%、无 clamp）：
  5 次独立 HCU 实现（cuda:0 ×3 + cuda:1 ×2）→ fro
  {**125235**, 16.3, 15.9, 15.9, 15.9}：1/5 落灾难分支（比 F1 事件的 510
  高 245×），4/5 落小分支（≈16，与 CPU 确定性 17.2、F1 rank1 17.6 同支）。
  分支概率 ~20%（该矩阵）；事件幅度多模态（小~16 / 中~10²–10³ / 大~10⁵）。

**机制链条**（08-31 单位审计后修正版，全部数值来自存档张量重算）：
1. DiT 稀疏激活：部分 slot 的 q/k/v 投影在多数 batch 接收**近零信号**
   （g_rms ~1e-8 量级，为正常 spec 的 1/30–1/150；141-spec 单步横截面
   显示清晰的 active(≥1.7e-6)/near-zero(≤6.5e-8) 两簇与 ~1e-7–2e-7 山谷）；
   fresh momentum 下 nesterov=0.0975g → 同样近零 → **病态贯穿整个 run，
   非 fork 伪影**。
2. Frobenius 归一化（`nesterov / nesterov.norm().clamp(min=1e-7)`）按设计
   **抹掉信号幅度**（分母=真范数 4.8155e-6，放大 ×2.0766e5，**未触发
   eps 地板**）。对健康梯度这是无害的方向化；但近零信号的归一化矩阵
   （top-1 σ² 占 88.9% 的秩-1 主导 + 量化网格薄尾）落在 NS4 五次迭代的
   **收敛边界**。
3. **NS4 边界混沌分岔**（本机制的核心，CPU 确定性复现实证）：同一输入
   比特、同一算法，前 3 轮 fro 1.0→1.37→3.78→8.80，**第 4 轮**的 Gram
   矩阵乘法结果相差 4–7%（bf16 GEMM 非确定性）即分岔为 ×1.95（fro 17.2，
   CPU）/×1.98（fro 17.6，HCU rank1）/×57.9（fro 510.4，HCU rank0）——
   输出尺度由 GEMM 噪声实现决定，rank 间 ×29 不相关。
4. `delta = -alpha·NS`（alpha=lr·0.2·√max_dim=1.5811e-3）→ 坏分支 delta
   rms=10× Moonlight target（3.15e-4）、单元素 0.44；原候选**静默写入**并
   继续（无守卫、无跨 rank 校验）；即使两 rank 同落小分支，NS 输出仍有
   5–8% rank 差 → 参数漂移积累 → 最终非finite 梯度范数 / loss 突变
   （3 次崩溃形态）。
5. rank 特定性与非确定性由「同一输入比特 × 非确定 GEMM kernel」给出
   （崩溃多在 rank1；97131 两次独立通过 = 分掷硬币）。

### 为什么原始 run「活过」97101 而在 97131/97147/97178 崩溃

同一数据位置（队首 batch 几乎相同，见 §5 数据偏移）的首个 update 上，
垃圾 delta 已被写入（取证版以 fail-closed 拦截并留证）；单次 2200% 的局部
权重扰动对 loss 的即时影响可被后续正常更新稀释，但**漂移不可逆且随机**
（每 rank 各自漂移），叠加后续 batch 中其他近零 spec 的同类事件，在 ~31–78u
内累积到 global gradient norm nonfinite / loss 发散。取证窗口（F1 触发于
97101）未及抵达 97131 数据窗口即已完成机制级定位 —— 无需重演崩溃。

---

## 5. 数据偏移披露

- 所有取证 run 从 **ckpt@97100** 启动；数据队首为 cycle1/epoch2 第 605–606
  个分片（605=原始崩溃 run 起点；今日多次重启仅推进 ~1 分片 ≈ ±6u）。
  即 F1/F2 与原始分支 run 的数据位置**实质相同**（原始 run 的 18u 数据空档
  同样存在）。
- 全部三次取证触发均发生在 **97101（首个 update / 队首 batch）**；97131/97178
  崩溃数据窗口（~+30u 处）未再被遍历 —— 机制定位不依赖重演该窗口。

---

## 6. 决策树路径与判定

| 步 | 规格 | 结果 |
|----|------|------|
| F1（原始路由+取证，≤200u，stop-on-trigger，无 restart-continue） | 复现候选故障 | **触发 @97101**（首个 update）：delta_rms_ceiling，slot_08.q_proj，rank0；机制全量捕获（ring + rank_fp + phase1 张量 .pt） |
| F2（K/V+adaln_shared→AdamW8bit，≤500u） | 验证安全路由是否足以稳定 | **不稳定 @97101**：双 rank ceiling 违规，slot_08.q_proj（**Q 角色，F2 未排除**）；首跑事件为插桩规则校准（sum 列病态），重跑后为真实候选故障 |
| F3（F3-KV / F3-ADALN 拆分，各 ≤200u，K/V 优先） | 仅 F2 稳定时执行 | **不执行**（F2 不稳定） |

- **CASE F（不复现）不适用** —— F1 在窗口第一步即复现且机制实证；判定不是
  「未复现」意义下的 INCONCLUSIVE，而是**确定性故障定位**。
- **ORIGINAL_CANDIDATE_SAFE = NO**。
- 附加硬件发现（独立于候选）：DTK/HCU bf16 GEMM 非确定 → 任何
  per-rank 独立数值优化器计算在该平台都不可假设跨 rank 一致。

---

## 7. 证据档案

- salt1 `/sakuramoon-runtime/artifacts/g1/forensic/`：
  - `cmuon-forensic-crash-97101.json`（各 run 触发时的 rank0 dump，含全列
    rank_fp/rank_probe_fp、10 步 ring、rank_drift）
  - `cmuon-forensic-crash-97101-dit_blocks_slot_08_attention_q_proj_weight.pt`
    （65.5MB，F1/F2b 故障 spec 的 phase1 张量：grad/momentum/nesterov/ns/delta）
  - `cmuon-forensic-crash-97101-dit_blocks_slot_00_attention_content_gate_weight.pt`
    （首版严格比对误触发时的 spec 张量，保留备查）
- 本地镜像：`D:\sakruamoon\tmp\cmuon-forensic-crash-97101.json`（F1）、
  `...-f2.json`（F2 首跑，校准事件）、`...-f2b.json`（F2b 双 rank 违规）、
  `...-slot08-qproj.pt`
- 机器可读摘要：`reports/cmuon-root-cause.json`
- 代码：分支 `cmoun-forensic` @ bff14ab（65af8f6 主体 → fae53c1 NS 设备默认 →
  13f7f0b finite 标志 → 3d78583 就绪日志 → f79945e device collective →
  a1d0d29 容差+漂移 → bff14ab 比对范围校准）；**未并入 Cmoun 分支**
- 单测：salt1 全量 43/43 PASS（94.8s）；本地 31 逻辑测试 PASS + 12 个
  既有 triton 环境性失败

---

## 8. 影响与修复方向（仅记录，未实施 —— 报完即停）

候选**不可用于生产**。机制级修复方向（供下一轮候选设计，按规格不在本轮实施）：

1. **近零梯度守卫**（直接针对根因）：NS 输入范数低于「按该参数典型梯度尺度
   校准的地板」（如按 momentum 历史或角色级分位数）时**跳过该 spec 的 NS
   更新**（保留 momentum 更新），而不是用 1e-7 绝对地板归一化噪声。
2. **跨 rank 一致性**：NS 输出（或最终 delta）在写参前做跨 rank 校验/单 rank
   计算+广播；或在 DDP 语义下确认更新一致性。
3. **平台层**：与 DTK 方确认 bf16 GEMM 非确定的原因（autotune/atomics），
   评估确定性 kernel 选项（改变 kernel 选择会改变候选数学，须重新验证）。
4. 任何修复须重新走：位级一致性单测 + 健康宿主短程 + A/B 窗口。

---

## === COPY TO CHATGPT: CMUON ROOT CAUSE ===

VERDICT: ORIGINAL_CANDIDATE_SAFE = NO (mechanism confirmed, not merely reproduced).

SETUP: Candidate = remote branch Cmoun @ f27fde9 (code 3967ade), 2x HCU DDK 26.04,
DDP world=2, 141 CMuon-routed specs (pure-bf16 Muon: bf16 lerp momentum -> nesterov
-> chunked Newton-Schulz 4 iters -> Moonlight alpha = lr*0.2*sqrt(max_dim) ->
param.add_(delta) with NO pre-write check and NO post-update cross-rank sync).
Crash history: 3 crashes / ~165 updates (~1.8%), MTT ~55u, same site
(FloatingPointError global gradient norm nonfinite @ clip.py:48, rank1):
97131 & 97178 on salt1, 97147 on a pristine healthy host (salt2, diverging-loss
variant). CONTROL (AdamW8bit, same data region): 0 crashes / 17,143+ updates.

ROOT CAUSE (forensic, fail-closed instrumentation, first update after the 97100
fork; units re-audited 08-31 from the saved offending tensors, correcting two
dimensional errors in the first draft): DiT sparse activation gives some slot
q/k/v projections a NEAR-ZERO signal (F1 event, slot_08.q_proj 2560x2560,
numel 6,553,600: grad element_rms 1.9294e-8, fro 4.9365e-5; nesterov
element_rms 1.8821e-9, fro 4.8155e-6 = 78x eps, so the clamp(min=1e-7) floor
DID NOT engage - the normalization denominator was the TRUE norm 4.8280e-6,
bf16 reduction). The Frobenius normalization (design behavior) erases the
magnitude (x2.0766e5), and the normalized weak-signal matrix (rank-1
dominated: top-1 sigma^2 = 88.9% of energy, thin tail to 7e-12) sits on the
CONVERGENCE BOUNDARY of the 4-iteration quintic Newton-Schulz: iterations 1-3
track 1.0->1.37->3.78->8.80 (fro, CPU-deterministic repro) but the 4th
iteration is chaotic in the Gram-matrix bits - the SAME input bits give
x1.95 (fro 17.2, CPU) / x1.98 (fro 17.6, HCU rank1) / x57.9 (fro 510.4, HCU
rank0) depending on the 5-8% bf16 GEMM nondeterminism. Measured: rank0 NS
output ns_rms 0.199 / ns_max 278 vs rank1 0.0069 / 0.676 (29x) from
bit-identical inputs; applied delta up to 0.44-1.02 per
ELEMENT (1395x the delta rms) on ~0.02-magnitude weights,
1.2-2.1x above the 10x-normal safety ceiling on BOTH ranks in one event.
Amplification chain (exact): nesterov_rms 1.8821e-9 x 2.0766e5 (normalize)
x 510.1 (NS4 bad branch) x 1.5811e-3 (alpha) = delta_rms 3.1518e-4, total
x1.675e5. The OLD draft's "~1.5e-7 Frobenius just above eps" and "x~66,000
noise amplification" statements are both corrected by the audit above: the
catastrophe is the NS4 boundary chaos, not an eps floor and not input-side
representation noise (input bits are rank-identical; nesterov elements sit
~400 ulp above zero at the median). The
uninstrumented candidate writes these silently; rank-specific drift (plus the
platform's bf16 GEMM nondeterminism below) accumulates over ~31-78 updates until
global gradient norm goes nonfinite or loss jumps 4.5x (the observed crashes).

HARDWARE AMPLIFIER (measured on salt1, DTK 26.04): bf16 GEMM is NONDETERMINISTIC
- 2560x2560 x@x.T run repeatedly with hard syncs on ONE device differs by 5-8%
rel_rms (150-170k of 6.5M elements); norm reductions ARE deterministic. Since the
candidate computes NS independently per rank, the DDP "identical model on both
ranks" invariant is broken from update 1; per-step cross-rank drift is recorded
by the forensic ring (event steps: 16.8-25.2x; normal steps at the noise floor).

ABLATION: F2 (safe routing: attention_k/attention_v/adaln_shared -> AdamW8bit,
all else in CMuon) from the same ckpt is NOT stable: it trips the fail-closed
guard at the very first update with a REAL dual-rank delta_rms_ceiling violation
on slot_08.attention.q_proj - a Q-role spec, i.e. the mechanism is
ROLE-INDEPENDENT and the K/V/adaln routing does not remove it. F3 (role split)
not run per decision tree (only if F2 stable).

DISCLOSURES: data offset ckpt@97100 + queue head 605-606/2099 shards (same as the
original crashed run, +/-1 shard); all forensic trips at update 97101 (first
batch), the 97131/97178 data window was not re-traversed - the mechanism was
fully pinned at the first batch. Forensic instrumentation: separate branch
cmoun-forensic @ bff14ab, candidate math bit-identical (two-phase split,
verified); tolerance rules calibrated by two measured false-positive classes
(HCU noise floor; ill-conditioned sum-column statistics). Deployment pitfalls
fixed and documented (stale CONFIG_NAME in workload env snapshot; fixed-mtime
tar + 1-byte change reviving stale .pyc; DTK process group has no CPU backend so
all collectives are device-side).

IMPLICATION: the candidate cannot be used as-is. Candidate-math fix directions
(recorded only, NOT implemented - stop after report per user decision):
(a) near-zero-gradient guard - skip the NS update (keep momentum) when the NS
input norm is below a floor calibrated to the parameter's typical gradient scale
instead of an absolute 1e-7; (b) cross-rank consistency of the applied update
(pre-write check, or single-rank NS + broadcast); (c) platform-level
deterministic bf16 GEMM with the DTK vendor (changes candidate math -> full
re-validation). Any fix requires: bit-level unit tests + healthy-host short run +
A/B window, no clamp-continue, no restart-until-luck.
=== END COPY TO CHATGPT ===
