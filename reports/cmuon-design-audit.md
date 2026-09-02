# SakuraMoon Hybrid CMuon — Design & Audit

> 实验性 Hybrid CMuon optimizer backend。仅开发/验证；**不切换 live trainer**，默认路径保持 torchao AdamW8bit bit-compatible。分支 `cmoun`，不 merge dev。
> 本文档随实现推进逐步补全（审计 → 算法 → 数值 → checkpoint → 内存 → 速度 → verdict）。

> **⚠️ 08-28 NS-DEPTH 审计更新（本轮，取代下文「剩余门」与 ns 相关旧结论）**：详见 `reports/cmuon-ns-depth-audit.md`。
> ① **per-spec（per-role）`ns_steps` 基础设施已落地**（`CMUON_ROLES` 8 role + `ns_steps_by_role` + `resolve_ns_map` + `[optimizer.cmuon_ns]` schema + ckpt 存 canonical per-role map + mismatch hard-fail + LR 可变允许 resume + 低开销 safety telemetry，opt-in 未部署）。
> ② **最低充分深度 = ns4（全部 8 role）**：24 seed × 8 role sweep 显示 ns3 对任何 role 都不「≈ ns4」（cosine 0.25–0.91 ≪ 0.995 门，rel-error 0.47–1.07），ns2 更差 → **不创建 mixed `hybrid_cmuon_min_ns_core`（无收益，等价 global ns4）**。唯一候选 = **`config/train_g1_hybrid_cmuon_ns4_core.toml`**（global ns4）。
> ③ **ns4 复测（关闭旧「剩余门①」）**：optimizer step **0.6734 s**（AdamW8bit 2.3796 s，**3.53×**，省 1.71 s/update）；全模型 update RMS ≈ 0.2·lr（attention 1.014× / ffn 1.019× / adaln 1.118×）；NS 占 CMuon 时间 95%；NS matmul ≈ 1992/step；peak 省 ~1.9 GB。projected full-update 1.125×（15.32→13.61 s，**projected 非完整训练**）。
> ④ **ns5 不稳定 = 平台 + spectrum 相关**（refinement）：精确算术下 Frobenius 归一化矩阵 σ_max≤1 < 1.264（quintic f⁵ 发散阈值），本不应发散；**HCU-BF16 对 k/v[640,2560]、adaln[2560,1024]（短边小）100%（24/24 seed）在 iter5 复现爆**（k/v delta 338–396×、adaln 46–49×），本地 CPU BF16 不复现（24 seed 全稳）。故「ns5 危险」在 production 加速器（HCU）上对这两 shape 是确定性事件；q/ffn（短边 2560）ns5 稳。native/custom 在 ns2/3/4 逐 shape 一致（非 wrapper bug）。
> ⑤ 旧「剩余门」现状：门①（改 ns4 复测）**已完成**（见 ③）；门②（长训质量 loss/FID）**仍开放**（部署前必做，本轮隔离 math 不替代）。

## 0. 硬约束（不可违反）

- 默认 optimizer 行为 = torchao AdamW8bit；**任何现有配置不写 optimizer 新字段时 bit-compatible**。
- 不改变模型架构、不改变任何现有 checkpoint tensor layout（FQN/shape/dtype 不变）。
- 不改变 LR / global batch / spatial probability；不扩 24 层、不升 512。
- 不把 live trainer 切到 CMuon；不删旧 AdamW checkpoint；不 merge dev。
- 禁 CUDA-only / Triton-only 正确性依赖；HCU v1 走普通 torch matmul。
- 禁全模型 FP32 master weights（若依赖强制则立即上报）。

## 1. 环境审计（salt1，HCU 实测）

| 项 | 值 |
|---|---|
| python | 3.11.9 |
| torch | **2.9.0** |
| torchao | **0.16.0**（cpp 扩展被跳过：torch 2.9.0 与 torchao 0.16.0 cpp 不兼容 → python fallback） |
| 加速卡 | HCU/DTK，dev0 名 `BW`，capability (9,3) |
| `torch.optim.Muon` | **存在**，定义于 `torch/optim/_muon.py` |
| Muon HCU 2D step | **BF16 / FP32 均可跑，finite=True**（dev0 实测） |
| Muon 签名 | `Muon(params, lr=1e-3, weight_decay=0.1, momentum=0.95, nesterov=True, ns_coefficients=(3.4445,-4.775,2.0315), eps=1e-7, ns_steps=5, adjust_lr_fn=None)` |
| Muon 维度限制 | **仅 2D**（1D 报 `ValueError: Muon only supports 2D parameters`） |
| Muon state dtype | **momentum_buffer 的 dtype = 参数 dtype**（BF16 参数→BF16 buffer；FP32→FP32）；**不创建 FP32 master weights**（参数 step 后保持原 dtype） |
| Muon NS dtype | **恒在 BF16**（`ortho_grad = grad.bfloat16()`，与参数 dtype 无关） |

**结论**：可复用 `torch.optim.Muon` 的 NS / Moonlight / momentum 数值定义作 reference，并在项目内加 **chunk-aware wrapper**。Moonlight = `adjust_lr_fn="match_rms_adamw"` → `lr * 0.2 * sqrt(max(A,B))`（正是论文 alpha）。

### 1.1 native Muon 精确语义（reference 基准，必须逐位对齐）

来自 `torch/optim/_muon.py`（已下载精读）：
- **momentum buffer**：`buf.lerp_(grad, 1-momentum)` ⇒ `B_t = μ·B_{t-1} + (1-μ)·g_t`（梯度被 (1-μ) 缩放）。
- **Nesterov**：`update = grad.lerp(buf, momentum)` ⇒ `(1-μ)·g_t + μ·B_t`。
- **NS**（quintic，恒 BF16）：tall 矩阵先转置为 wide；`M = M / max(||M||_F, eps)`；迭代 `G = M·Mᵀ;  M ← a·M + (b·G + c·G²)·M`（`a=3.4445, b=-4.7750, c=2.0315`）；结束还原转置。
- **Moonlight**：`adjust_lr_fn="match_rms_adamw"` ⇒ `adjusted_lr = lr * 0.2 * sqrt(max(rows, cols))`。
- **weight decay**（decoupled）：`param.mul_(1 - lr·wd)`；**更新**：`param.add_(update, alpha=-adjusted_lr)`。

## 2. 当前 optimizer 路径审计（本地 cmoun 代码）

**构建**（`train/production.py::_build_optimizer` → `optim/adamw8bit.py::build_adamw8bit`）：
- `audit_trainable_parameters(module, ...)`（`optim/groups.py`）按 FQN 把每个可训练参数分入：
  - `matrix_decay`（nn.Linear weight，**BF16**）
  - `sensitive_no_decay`（1D / GlobalConditioner / FinalOutputHead / modality / condition tokens / text gate，**FP32**）
- 构造 `torchao.optim.AdamW8bit`（2 个 param group：matrix_decay / sensitive_no_decay），locked policy：`betas=(0.9,0.95)`、`eps=1e-8`、`block_size=256`、`bf16_stochastic_round=True`、`weight_decay=0`（按 group 的 decay 值）。
- 包 `IsolatedAdamW8bit`（隔离 stochastic-rounding RNG + 有限梯度校验）。

**step**（`train/step.py::SingleGpuStep`，`StepOptimizer` 协议 = `.step()` + `.zero_grad(set_to_none=True)`）：
- 有限梯度校验 → SR-RNG 隔离下 `inner.step()`。

**checkpoint**（`checkpoint/save.py` + `load.py`）：
- `train_state/optimizer.pt` = **内层** `optimizer.optimizer.state_dict()`（AdamW8bit 量化 moments，~3.2GB）。
- `train_state/optimizer_schema.json` = param groups（group_name + param_names）。
- `train_state/rng/optimizer_sr.safetensors` = SR RNG 状态；`rng/rank-0.safetensors` = 训练 RNG。
- `resolved_config.toml` = 解析后配置字节。
- `validate_optimizer_coverage`（`checkpoint/artifact.py`）校验 optimizer 覆盖全部可训练参数。

**生产装配**（`train/production.py`）：`build_trainable_composite_from_config` → `_build_optimizer` →（resume?）`_restore_checkpoint` → `accelerator.prepare(module, optimizer.optimizer)` → `optimizer.optimizer = prepared` →（可选）`compile_packed_dit_blocks`。

**接口契约**：`IsolatedAdamW8bit` 暴露 `.step() / .zero_grad(set_to_none) / .state_dict() / .load_state_dict() / .optimizer（内层 torch optimizer）/ .audit（ParameterAudit）/ .sr_rng`。preflight 有 `isinstance(optimizer, IsolatedAdamW8bit)` 检查。

## 3. 可训练参数清单 + 路由（salt1 真实模型实测，depth=20）

**Total**：289 params，**1,570,747,194** numel（1.57B），3,193,228,520 bytes（3.19GB）。

| 组 | count | numel | %numel | bytes | 说明 |
|---|---|---|---|---|---|
| **CMuon** | 141 | 1,536,163,840 | **97.80%** | 3,103,784,960 | BF16 1,520,435,200（20 块 DiT 权重）+ FP32 15,728,640（shared_block_projection） |
| **AdamW8bit** | 148 | 34,583,354 | 2.20% | 89,443,560 | BF16 24,444,928 + FP32 10,138,426 |

- **CMuon ∩ AdamW = ∅**、**CMuon ∪ AdamW = 全部 289**（脚本断言通过）。
- 注意：20 个 active block 的 slot **非连续**（slot_00-04,06-10,12-16,18-22，缺 05/11/17）→ 路由用 `slot_\d+` 正则，不硬编码 slot id。

### 3.1 CMuon allowlist（v1，按 FQN 模板）

| FQN 模板 | shape | dtype | chunk | roles |
|---|---|---|---|---|
| `dit.blocks.slot_XX.attention.q_proj.weight` | [2560,2560] | BF16 | 1 | q |
| `dit.blocks.slot_XX.attention.k_proj.weight` | [640,2560] | BF16 | 1 | k |
| `dit.blocks.slot_XX.attention.v_proj.weight` | [640,2560] | BF16 | 1 | v |
| `dit.blocks.slot_XX.attention.content_gate.weight` | [2560,2560] | BF16 | 1 | content_gate |
| `dit.blocks.slot_XX.attention.out_proj.weight` | [2560,2560] | BF16 | 1 | out_proj |
| `dit.blocks.slot_XX.mlp.in_proj.weight` | [13824,2560] | BF16 | **2** (dim0) | gate, up |
| `dit.blocks.slot_XX.mlp.down_proj.weight` | [2560,6912] | BF16 | 1 | down |
| `dit.conditioner.shared_block_projection.weight` | [15360,1024] | **FP32** | **6** (dim0) | attn_scale/attn_shift/attn_gate/mlp_scale/mlp_shift/mlp_gate |

（20 块 × 7 = 140 + shared_block_projection 1 = 141）

### 3.2 留在 AdamW8bit 的（148 个）

- 所有 norm（1D）、bias、modality embeddings（image/condition/text）。
- `dit.input_projection.weight` [2560,128]（matrix 但不在 allowlist）。
- 全部 `text.*`（TextConditioner：refinement q/k/v/out、shared_projection、output_projection、layer_norms、gate 等）。
- 全部 `condition_tokens.*`（ConditionTokenEncoder：cross_attention.in_proj/out、condition_mlp gate/up/down、queries、null_tokens 等）。
- `dit.conditioner.*` 除 shared_block_projection 外（condition_mlp、condition_global_projection、block_biases、final_projection）。
- `dit.output_head.*`（FinalOutputHead：norm、projection）。

### 3.3 GQA 特判

Q=[2560,2560]、K=[640,2560]、V=[640,2560] **物理独立且 shape 不等** → 无 QKV 融合张量，N_chunk=1，**禁止机械 sqrt(3)**。`qkv_group_rescale` 保留为 false（v1 强制 false）。

## 4. 内存审计（优化器 persistent state）

| 方案 | momentum 存储 | 每元字节 | 估算（对 CMuon 1.536B） |
|---|---|---|---|
| 当前 AdamW8bit（全 1.57B） | 8bit 量化 × 2 moment | ≈2 B/elem | **≈3.2GB**（= optimizer.pt 实测 ~3.2G） |
| CMuon **BF16** momentum | 1 buffer BF16 | 2 B/elem | **≈3.07GB**（≈AdamW8bit；Muon 只有 1 个 momentum vs AdamW 2 个） |
| CMuon **FP32** momentum | 1 buffer FP32 | 4 B/elem | **≈6.14GB**（≈2× AdamW8bit） |

- 实测 peak allocated / reserved / NS workspace 见 `cmuon-benchmark.json`（HCU 实测补全）。
- **无全模型 FP32 master weights**：native Muon 保持参数原 dtype；wrapper 同样不升精度存参数。

## 5. 设计决策（实现方案）

1. **新增模块** `src/sakuramoon/optim/cmuon.py`：
   - `cmuon_zeroth_power(chunk, ns_steps, ns_coefficients)` — 复现 native NS（恒 BF16，tall→wide 转置，Frobenius 归一，quintic）。
   - `cmuon_moonlight_alpha(rows, cols, lr, *, rescale_sqrt_n, n_chunks)` — `lr*0.2*sqrt(max(rows,cols))`，可选 `*= sqrt(n_chunks)`（独立开关，不与 chunking 融合）。
   - `cmuon_step(param, grad, buf, cfg)` — momentum→Nesterov→按 chunk split→per-chunk NS+Moonlight→concat→weight decay→update。
   - `HybridCMuon` — 与 `IsolatedAdamW8bit` 同接口（`.step/.zero_grad/.state_dict/.load_state_dict/.optimizer/.audit/.sr_rng`）：AdamW 参数走 `IsolatedAdamW8bit`（复用，bit-compatible），CMuon 参数走 `cmuon_step`。
2. **路由**：显式 FQN 正则 allowlist（§3.1）；`route_parameters(module)` 输出 (cmuon_specs, adamw_specs)，断言 disjoint+complete。
3. **momentum dtype**：`cmuon_momentum_dtype ∈ {bfloat16, float32}`，作用于所有 CMuon buffer（与参数 dtype 解耦；参数保持原 dtype）。reference 对齐用「buffer dtype = 参数 dtype」特例。
4. **chunk rescale**：`chunk_rescale_sqrt_n`（core=false）、`qkv_group_rescale`（v1 恒 false）为独立开关。
5. **checkpoint**：hybrid 额外存 `cmuon_state.pt`（momentum）+ `cmuon_schema.json`（ns_steps/dtype/scaling/rescale/routing manifest/chunk metadata）。AdamW 部分沿用原格式（`optimizer.pt` 存 AdamW8bit state）。AdamW8bit→Hybrid：权重严格恢复、AdamW state 按 FQN 保留、CMuon momentum 从 0 初始化、manifest 记录 transition。现有 AdamW8bit ckpt 原样可 resume。
6. **默认路径不变**：`[optimizer] name = "torchao_adamw8bit"`（默认）走原 `build_adamw8bit`；新名字（`hybrid_cmuon_core` / `hybrid_cmuon_paper_rescale`）才走 `build_hybrid_cmuon`。

## 6. 实验配置（2 个，不升 batch/LR）

- **A `hybrid_cmuon_core`**：同 global batch 800、同 base_lr 5e-5（JLT）、Moonlight、chunking ON、`chunk_rescale_sqrt_n=false`、`qkv_group_rescale=false`。
- **B `hybrid_cmuon_paper_rescale`**：同 A，另 `chunk_rescale_sqrt_n` 对 FFN in_proj 施加 sqrt(2)、对 shared_block_projection 施加 sqrt(6)；**无 QKV sqrt(3)**。

## 7. 数值 / checkpoint / benchmark 结果

### 7.1 §9 单元测试（A–F，salt1 HCU 环境 torch 2.9.0 实测，全部 16 通过）

| 组 | 内容 | 结果 |
|---|---|---|
| A | `cmuon_zeroth_power` finite/确定性（tall/wide/square × BF16/FP32）+ **逐位对齐 native `_zeropower_via_newtonschulz`** | PASS |
| B | chunk=1 全参 == native `torch.optim.Muon(adjust_lr_fn="match_rms_adamw")`（256²/64×256 BF16、256×128 FP32） | PASS |
| C | fused chunk=2 == `cat(independent_muon(gate), independent_muon(up))` | PASS |
| D | fused chunk=6 == 6 个独立 Muon（FP32 shared_block_projection） | PASS |
| D2 | `chunk_rescale_sqrt_n` 开关使 update RMS ≈×√2 | PASS |
| E | chunking 保 shape/FQN（chunk 1/2/6） | PASS |
| E2 | Moonlight alpha 精确值 | PASS |
| F | 路由 disjoint+complete（mock，7/block+1，in_proj c2，shared c6） | PASS |

**结论**：CMuon 更新（NS + Moonlight + chunking + momentum + Nesterov）逐位对齐 native `torch.optim.Muon` reference；chunk 语义 = 沿 chunk_dim 切块、逐块独立 NS+Moonlight、concat。

> 注：CPU 单测用小 shape（算法与 shape 无关）；生产 shape（2560²/13824×2560/15360×1024）的实测在 §7.3 HCU benchmark。

### 7.2 §8 Checkpoint 设计（save/load + transition 均已实现，HCU 单测通过）

- **现有 AdamW8bit ckpt 原样 resume**：`optimizer.pt`（AdamW8bit state_dict）格式不变；`HybridCMuon.load_state_dict` 对纯 AdamW8bit ckpt（无 `"cmuon"` 段）走 transition 路径。
- **Hybrid→Hybrid state-exact**：`state_dict` 存 `"cmuon"` 段（per-FQN momentum + ns_steps/ns_coefficients/momentum/nesterov/eps/momentum_dtype/chunk_rescale_sqrt_n/qkv_group_rescale）+ `"routing"` manifest + `"transition"` 记录 + `hybrid_cmuon_schema_version:1`；`load_state_dict` 硬校验 dtype/ns_steps/rescale 一致（不一致即 raise，不静默降级）。
- **AdamW→Hybrid transition（已实现 + 单测 §9-H 通过）**：按 FQN 把 baseline 全参 AdamW state 映射到 inner AdamW8bit 的 148 AdamW 参（直接设 live state dict、绕过 `load_state_dict` 的 per-group size 检查——baseline 全参多组 vs inner 子集）；CMuon 141 参的 AdamW moment 丢弃、CMuon momentum 保持 0；**不做假 2-moment→Muon 转换**；`state_dict["transition"]` 记录 `from_adamw8bit/preserved_adamw_params/dropped_cmuon_params`。无 update-number gate。rollback ckpt 不动。

  实现要点：baseline 的 param 顺序 == `self.audit`（full audit，与 `build_adamw8bit` 同序）；inner 顺序 == `routing.adamw_specs`（AdamW 子集）。按 FQN 建立 `name→baseline_id` 映射，逐参 `inner_state[spec.parameter] = saved_state[baseline_id]`，再 `_move_quantized_state_to_parameter_devices()` 把 8bit 量化 moment 移回参数 HCU。

### 7.3 §10-11 HCU smoke benchmark（salt1 dev0，torch 2.9.0，真实 G1 模型 1.57B，lr=1.5625e-4，warmup 5 + 25 实测，合成固定梯度）

> 说明：本 bench 只测 **optimizer step**（fwd/bwd 各 optimizer 相同、非差异项，未计入）；step 时间与生产一致（同 shape/dtype）。torch.compile 已禁用（step 为 eager，与 compile 无关）。

| 优化器 | step s (avg) | state GB | peak alloc GB | peak res GB |
|---|---|---|---|---|
| AdamW8bit (baseline，全 289 参) | 2.377 | 3.19 | 11.64 | 12.11 |
| **CMuon BF16 ns5** | **0.763** | **3.35** | **9.90** | **10.27** |
| CMuon BF16 ns6 | 0.845 | 3.35 | 9.89 | 10.36 |
| CMuon FP32 ns5 | 0.799 | 6.42 | 13.17 | 13.93 |
| CMuon FP32 ns6 | 0.892 | 6.42 | 13.23 | 13.73 |

**BF16 vs FP32（用户关注点）**
- **速度**：BF16 (0.763s) 比 FP32 (0.799s) 快 **~4.5%**（NS 恒 BF16，差异仅在 momentum 更新访存）。
- **显存**：momentum buffer 差 **3.07GB**（1.536B×2B）；总 state 3.35 vs 6.42GB；peak 9.90 vs 13.17GB。
- **精度**：NS 恒 BF16（两者相同）；momentum 累积 BF16（8-bit mantissa）略低于 FP32（23-bit），短 smoke 无影响，长训待验证。

**ns5 vs ns6（08-28 RMS audit 重审：旧结论已修正）**
- 旧结论「ns5 稳定、ns6 发散」**错误**。RMS audit 显示 quintic NS 有 **shape 相关收敛窗口**（逐迭代 RMS 追踪，polar=1/√max 为收敛目标）：
  - q [2560,2560] / ffn [6912,2560] / ffn [2560,6912]：iter5 收敛（rms≈polar），**iter6 发散**。
  - **k/v [640,2560] / adaln [2560,1024]：iter4 已收敛，但 iter5 离开收敛域发散**（adaln iter5 rms 0.0221→0.94；k iter5 rms 0.0304→6.61）。
- 因此 **ns5 对 k/v、adaln 不安全**（它们在第 5 步发散），ns6 对所有 shape 发散。
- **ns4 是全 shape 安全的选择**：ns4 的 delta RMS 对所有 shape ≈ 3.1e-5 ≈ 0.2·lr（Moonlight 不变量成立）——q 2.78e-5 / k 4.87e-5 / adaln 3.42e-5 / ffn 3.17e-5。

**update RMS 非均匀的真正根因（删除旧的「Moonlight 本征」结论）**
- 旧报告把 attention 3.9e-3 / adaln 1.46e-3 归因「Moonlight ∝√dim 本征」——**错误，已删除**。
- 论文正确解释：Moonlight 的 `sqrt(max(d_in,d_out))` 缩放用于**抵消** semi-orthogonal update 的 `1/sqrt(max(dim))` RMS 尺度，使 scaled update RMS ≈ 常量 **0.2**（与 shape 无关）。RMS audit 证实该不变量**在 NS 收敛的 shape 上成立**（q scaled_pre_lr=0.22、ffn=0.19 ≈ 0.2）。
- 非均匀的真正来源是 **NS 发散**（非 Moonlight、非 CMuon 实现 bug）：
  - attention 组 3.9e-3 = 元素加权混合了**正常的 q/content_gate/attn_out（3.3e-5）** 与 **NS 发散的 k/v [640,2560]（delta 1.1e-2，约 1000×）**（k/v 占该组 14% 元素）。
  - adaln 1.46e-3 = 6 个 [2560,1024] chunk 全部 NS 发散（delta 1.4e-3）。
  - ffn 2.8e-5 = NS 收敛，正常（≈0.2·lr）。
- **这是 quintic NS 的数值性质，native `torch.optim.Muon` 完全相同**（audit 逐 shape 验证 custom delta == native delta），**不是 CMuon 实现 bug**。

**update RMS 分组（ns5 重测，标注发散）**
| 组 | AdamW8bit | CMuon ns5 | ns4（安全） | 说明 |
|---|---|---|---|---|
| attention q/out | 1.63e-4 | 3.3e-5 | 2.8e-5 | NS 收敛，正常 |
| attention k/v | 1.63e-4 | **1.0e-2（NS 发散）** | 4.9e-5 | ns5 发散，ns4 正常 |
| adaln | 1.63e-4 | **1.4e-3（NS 发散）** | 3.4e-5 | ns5 发散，ns4 正常 |
| ffn | 1.63e-4 | 2.8e-5 | 3.2e-5 | NS 收敛，正常 |
| adamw_fallback | 1.63e-4 | 1.63e-4 | 1.63e-4 | 走 AdamW8bit，符合设计 |

> **修正结论**：Moonlight 不变量（scaled RMS≈0.2）本身正确且在 NS 收敛处成立；旧 benchmark 的 attention/adaLN 异常是 **NS 在第 5 次迭代对 k/v [640,2560]、adaln [2560,1024] 发散**所致，**非 Moonlight 本征、非实现 bug**。采用前必须改用 **ns4**（或等价稳定 NS）并复测。

### 7.4 §11 chunk-rescale bench（core vs paper_rescale，同梯度快照，ns5 BF16）

| 组 | core (rescale OFF) | paper_rescale (ON) | 比值 | 设计预期 |
|---|---|---|---|---|
| ffn | 2.8e-5 | 3.8e-5 | **1.37** | √2=1.41（in_proj chunk=2）✓ |
| adaln | 1.45e-3 | 3.55e-3 | **2.45** | √6=2.45（shared chunk=6）✓ |
| attention | 4.05e-3 | 4.11e-3 | 1.01 | 无 rescale（chunk=1）✓ |
| adamw_fallback | 1.63e-4 | 1.63e-4 | 1.0 | 不走 CMuon ✓ |

**结论**：`chunk_rescale_sqrt_n` 开关精确施加 √N_chunk（FFN √2、AdaLN √6、QKV 无 √3）——与设计逐位吻合。config A（core）与 B（paper_rescale）的唯一差异即此开关。

> ⚠️ 上表 adaln/attention 列（1.45e-3 / 4.05e-3）是 **ns5** 数值，其中 adaln/attention 的 k/v 分量含 **NS 发散**（见 §7.5）；rescale 比值本身（√2/√6）仍精确，因为发散在 core 与 paper 两列同向抵消。

### 7.5 UPDATE-RMS correctness audit（08-28，blocker audit，salt1 HCU BF16，真实生产 shape，固定梯度快照）

**动机**：§7.3 旧报告把 attention 3.9e-3 / adaln 1.46e-3 的 update RMS 非均匀归因「Moonlight ∝√dim 本征」。论文附录推导 `U_muon = 0.2·√max(m,n)·Orth(M)`、`RMS(U_muon)≈0.2`，故 applied delta RMS 应 ≈ `0.2·lr = 3.125e-5`（与 shape 无关）——24×/47× 差异不应出现。本 audit 重验。

**1. 度量定义（统一）**：旧 benchmark 的 `update_rms_per_group` = **单 step 实际 parameter delta RMS**（`RMS(p_after − p_before)`，momentum 从 0 起第 1 步，组内元素加权 `sqrt(Σδ²/Σn)`）。定义本身一致、正确；**非 relative RMS、非 momentum RMS、非混用**。

**2. 完整数值链（ns5，BF16，逐 shape；polar=1/√max 为 NS 收敛目标）**：

| shape | ns_output_rms | scaled_pre_lr_rms | expected=0.2·lr | actual_delta_rms | native_muon_delta | custom==native |
|---|---|---|---|---|---|---|
| q [2560,2560] | 2.18e-2 | **0.22** | 3.125e-5 | 3.36e-5 | 3.35e-5 | ✓ |
| k [640,2560] | **6.86（爆）** | **69.4** | 3.125e-5 | 1.09e-2 | 1.22e-2 | ≈（同向发散） |
| v [640,2560] | **7.23（爆）** | **73.2** | 3.125e-5 | 1.14e-2 | 1.12e-2 | ≈（同向发散） |
| content_gate [2560,2560] | 2.16e-2 | **0.22** | 3.125e-5 | 3.31e-5 | 3.32e-5 | ✓ |
| attn_out [2560,2560] | 2.15e-2 | **0.22** | 3.125e-5 | 3.31e-5 | 3.27e-5 | ✓ |
| ffn_gate [6912,2560] | 1.15e-2 | **0.19** | 3.125e-5 | 2.81e-5 | 2.80e-5 | ✓ |
| ffn_up [6912,2560] | 1.12e-2 | **0.19** | 3.125e-5 | 2.72e-5 | 2.83e-5 | ≈ |
| ffn_down [2560,6912] | 1.14e-2 | **0.19** | 3.125e-5 | 2.75e-5 | 2.76e-5 | ✓ |
| adaln_i [2560,1024] ×6 | **~0.91（爆）** | **~9.2** | 3.125e-5 | ~1.42e-3 | ~1.42e-3 | ✓（同向发散） |

**3. Moonlight 不变量（scaled_update_rms_before_lr ≈ 0.2）**：**NS 收敛的 shape 全部 ≈0.2**（q/content_gate/attn_out 0.22、ffn 0.19）→ **数学正确**。NS 发散的 shape（k/v、adaln）scaled_pre_lr 达 69/73/9.2（因 NS 输出未正交化）→ 不变量被 NS 发散破坏，**非 Moonlight 缩放本身问题**。

**4. native Muon 对照（同 grad/momentum/lr/ns_steps，`adjust_lr_fn="match_rms_adamw"`）**：
   - **chunk=1 逐 shape**：custom delta **逐 shape == native Muon delta**（收敛 shape RMS 差 <5%；发散 shape 同向发散、幅度差 <10%，均为 BF16 matmul 非确定性）。
   - **chunked-concat（FFN/AdaLN 完整分块 vs 多个 independent native Muon concat）**：FFN in_proj [13824,2560] chunk=2、AdaLN shared [15360,1024] chunk=6、FFN down [2560,6912] chunk=1。custom 分块 NS+concat 的 delta RMS 与「对每个 chunk 独立跑 native Muon 再 concat」的 delta RMS **相差 < 2%**（ns4：ffn_in_proj 3.177e-5 vs 3.192e-5、adaln 3.406e-5 vs 3.397e-5、ffn_down 3.181e-5 vs 3.188e-5）——与单 shape 对比的 BF16 非确定性同量级，**证明 chunking 逻辑正确**（split→per-chunk NS→concat 等价于逐块独立 NS）。个别 outlier 元素 max_abs_diff≈2.4e-4 是 NS 迭代对 BF16 matmul 非确定性的放大，非 chunking bug。
   - **结论：非 CMuon 实现 bug**（custom 与 native 在 chunk=1 与 chunked-concat 两种粒度下均一致）。

**5. ns5/ns6 逐迭代追踪（RMS；polar 为目标）**：

| shape | iter4 | iter5 | iter6 | 发散点 |
|---|---|---|---|---|
| q [2560,2560] | 1.88e-2 | 2.16e-2(≈polar) | 1.52e-1 | iter6 |
| k [640,2560] | 3.04e-2 | **6.61（爆）** | 1.18e14 | **iter5** |
| adaln [2560,1024] | 2.21e-2(≈polar) | **9.28e-1（爆）** | 9.94e9 | **iter5** |
| ffn_gate [6912,2560] | 1.27e-2 | 1.13e-2(≈polar) | 8.05e-2 | iter6 |
| ffn_down [2560,6912] | 1.28e-2 | 1.15e-2(≈polar) | 2.03e-1 | iter6 |

**delta RMS per ns_steps**（目标 ≈3.125e-5）：

| shape | ns3 | ns4 | ns5 | ns6 |
|---|---|---|---|---|
| q | 1.82e-5 | **2.78e-5** | 3.31e-5 | 3.33e-4 |
| k | 3.30e-5 | **4.87e-5** | 1.04e-2 | 1.93e11 |
| adaln | 2.85e-5 | **3.42e-5** | 1.43e-3 | 1.57e7 |
| ffn_gate | 1.93e-5 | **3.17e-5** | 2.79e-5 | 3.07e-4 |
| ffn_down | 1.93e-5 | **3.16e-5** | 2.85e-5 | 2.82e-4 |

**根因判定（task 5 A/B）**：属 **A 类——native 与 custom 同向发散**（quintic NS 在 BF16、特定 aspect ratio 下的数值现象），**非 B 类 CMuon 实现 bug**。quintic NS 的收敛窗口随 shape（更精确说随 Frobenius 归一化后最大奇异值 ≈ 1/√c + 1/√r）变化：短边更小（k/v 短边 640、adaln 短边 1024）→ 最大奇异值更大（>~0.045）→ 第 5 次迭代离开收敛域。**ns4 落在所有 shape 的收敛窗口内**（delta ≈0.2·lr），ns5 对 k/v、adaln 出窗，ns6 全部出窗。

**结论**：① CMuon 数学正确（Moonlight 不变量在 NS 收敛处成立、custom==native）；② 旧「Moonlight 本征非均匀」结论**删除**；③ 非均匀真因 = NS shape 相关发散；④ **采用前必须改用 ns4 并复测**（本轮仅定位，不改 toml/代码）。详见 `reports/cmuon-rms-audit.json`（完整链 + trace）。

## 8. Verdict / 推荐 / Open questions

### 8.1 推荐

**推荐 CMuon BF16 momentum + ns4（config A `hybrid_cmuon_core`，需把 `cmuon_ns_steps` 从 5 改为 4）**：

> ⚠️ **RMS audit（08-28）修正**：原推荐 ns5 不安全——quintic NS 对 k/v [640,2560]、adaln [2560,1024] 在第 5 次迭代发散（delta 放大 46×~1000×），会破坏训练稳定性。**ns4 是全 shape 安全的选择**（delta RMS 对所有 shape ≈0.2·lr，Moonlight 不变量成立）。**A/B 测试前必须把 toml 的 `cmuon_ns_steps` 改为 4 并复测**（本轮 audit 不改代码/toml，仅定位）。

| 维度 | CMuon BF16 ns4 | AdamW8bit (baseline) | 结论 |
|---|---|---|---|
| optimizer step | ~0.76s（ns5 实测 0.763s，ns4 略快） | 2.377s | **快 ~3×**；总 update 预计快 ~10%（step 占 ~15%） |
| state 显存 | 3.35GB | 3.19GB | 相当（+0.16GB） |
| peak 显存 | 9.90GB | 11.64GB | **省 1.74GB** |
| 稳定性 | **ns4 全 shape 收敛** | — | ns5 对 k/v、adaln 发散；ns6 全 shape 发散 |
| update 尺度 | 各组 delta RMS ≈0.2·lr（均匀，Moonlight 不变量成立） | ≈lr | ns4 下尺度一致；ns5 下 k/v、adaln 异常放大 |
| 精度 | NS 恒 BF16 | 8bit moment | 均低精度；长训待验证 |

**FP32 momentum 不推荐**（v1）：速度仅快 4.5%（相对 BF16），但显存 ×2（3.07GB→6.14GB momentum），无净收益。若长训发现 BF16 momentum 累积漂移，再升 FP32。

**理由**：CMuon step 比 AdamW8bit 快 ~3×（NS 走 HCU matmul，AdamW8bit 的 8bit 随机舍入 + 2 moment 访存更重）；BF16 momentum 显存与 AdamW8bit 相当、peak 更省；**ns4 下所有 shape 的 update 尺度一致（≈0.2·lr）**，学习动力学可控。**剩余门**：① 改用 ns4 后复测 update RMS（确认全 shape ≈0.2·lr）；② 长训验证生成质量（loss/FID）。本 smoke/audit 只验证速度/显存/数值对齐/尺度，不验证质量。

### 8.2 落地状态（代码已就绪，**不部署**；剩余门 = ① 改 ns4 复测 ② 长训质量验证）

1. **生产接线（已完成）**：`config.schema.OptimizerConfig.name` 扩展为 `Literal["torchao_adamw8bit","hybrid_cmuon"]` + 可选 `cmuon_ns_steps/cmuon_momentum_dtype/cmuon_chunk_rescale_sqrt_n`（默认保持 AdamW8bit bit-compatible，已验证现有 G1 配置不受影响）；`train.production._build_optimizer` 按 name 分支（`hybrid_cmuon`→`build_hybrid_cmuon`）；`train.preflight.optimizer_parameters` 接受 `IsolatedAdamW8bit | HybridCMuon`。
2. **2 个实验 toml（已完成，但 ns_steps 待改）**：`config/train_g1_hybrid_cmuon_core.toml`（A，rescale OFF）+ `config/train_g1_hybrid_cmuon_paper_rescale.toml`（B，rescale ON）。两者 extends G1 crop_800_s0，仅 `[optimizer]` 不同。**注意：两 toml 当前 `cmuon_ns_steps=5`，RMS audit 证明 ns5 对 k/v、adaln 不安全，A/B 前须改为 4**（本轮 audit 未改 toml）。
3. **AdamW→Hybrid transition loader（已完成 + 单测 §9-H 通过）**：见 §7.2。
4. **§12 低风险提示**：naive per-chunk NS 已是 matmul-bound（HCU 上 0.76s/step，无瓶颈）；batched-NS 优化**非必要**（report-only，不实施）。
5. **RMS correctness audit（已完成，08-28）**：见 §7.3/§7.5。**关键修正**：① 旧「update RMS 非均匀是 Moonlight ∝√dim 本征」结论**错误**，真正根因是 quintic NS 的 shape 相关收敛窗口（k/v [640,2560]、adaln [2560,1024] 在 ns5 第 5 步发散）；② Moonlight 不变量（scaled RMS≈0.2）在 NS 收敛处成立、数学正确；③ custom CMuon 与 native Muon 逐 shape 一致（非实现 bug）；④ **ns4 是全 shape 安全选择**。详见 §7.5 + `reports/cmuon-rms-audit.json`。
6. **剩余门（本任务不做）**：① 改 ns4 后复测 update RMS（确认全 shape ≈0.2·lr）；② 长训验证生成质量（loss/FID）。**本任务到此停止，不切 live trainer、不部署、不改 toml/代码。**

### 8.3 Open questions

- BF16 momentum 长训（>50k updates）是否出现累积漂移？（短 smoke 未见）
- ns4 复测后，各组 update 尺度是否全 ≈0.2·lr？（预期是，需确认）
- 能否找到比「改 ns_steps」更稳健的 NS（如 spectral-norm 预归一化、或更稳定的多项式），使 ns5/ns6 对所有 shape 安全？（v1 用 ns4 绕过，不深究）
- CMuon（ns4，尺度 ≈0.2·lr 均匀）对生成质量的影响？需长训 FID/人工评估。
- DDP 下 CMuon 参数梯度 allreduce 已验证（grad sync 在 model 上、与 optimizer 无关）；多 rank smoke（§9-I）未跑（单卡 bench 已覆盖数值）。

---

## === COPY TO CHATGPT ===

**Environment**
- salt1 HCU (Hygon DCU, DTK 26.04), torch 2.9.0, torchao 0.16.0 (cpp ext skipped → python fallback), python 3.11.9
- Real G1 TrainableComposite: 1.57B params, depth=20, global_batch=800, lr=1.5625e-4 (JLT-scaled from base 5e-5 × 800/256)
- Benchmark = optimizer-step only (fwd/bwd identical across optimizers, not timed); torch.compile disabled (step is eager); synthetic fixed grads; warmup 5 + 25 measured

**Routing (v1 allowlist, 141/289 params = 97.80% numel to CMuon, 2D only)**
- CMuon: 20 blocks × {q,k,v,content_gate,out_proj (chunk=1), mlp.in_proj (chunk=2 gate/up), mlp.down_proj (chunk=1)} + dit.conditioner.shared_block_projection (chunk=6, FP32)
- AdamW8bit (unchanged): all other 148 params (biases, norms, text/condition_tokens, input_projection, output_head)
- Disjoint+complete asserted. GQA: Q/K/V physically separate, no fused QKV tensor, no sqrt(3).

**Algorithm (bit-aligned to native torch.optim.Muon, verified)**
- raw grad → momentum B=μB+(1-μ)g → Nesterov (1-μ)g+μB → split along chunk_dim → per-chunk Newton-Schulz (quintic a=3.4445 b=-4.7750 c=2.0315, ALWAYS BF16, tall→wide, Frobenius-normalized) + per-chunk Moonlight alpha=lr·0.2·√max(d_out,d_in)·(√N_chunk if rescale) → concat → decoupled WD → update
- 16 unit tests pass on salt1 HCU (NS/ Moonlight / chunk=1/2/6 / rescale switch / routing all bit-match native Muon reference)

**Numerics (HCU measured, optimizer step)**
| optimizer | step s | state GB | peak alloc GB |
|---|---|---|---|
| AdamW8bit | 2.377 | 3.19 | 11.64 |
| CMuon BF16 ns5 | 0.763 | 3.35 | 9.90 |
| CMuon BF16 ns6 | 0.845 | 3.35 | 9.89 |
| CMuon FP32 ns5 | 0.799 | 6.42 | 13.17 |
| CMuon FP32 ns6 | 0.892 | 6.42 | 13.23 |

- BF16 vs FP32: BF16 faster ~4.5% (0.763 vs 0.799s), momentum buffer 3.07GB less (NS always BF16 so precision of NS identical; only momentum-accumulation precision differs)
- NS stability (08-28 audit, CORRECTED): the quintic NS has a SHAPE-DEPENDENT convergence window. ns5 DIVERGES for k/v [640,2560] and adaln [2560,1024] (5th iteration leaves the basin); ns6 DIVERGES for ALL shapes; **ns4 converges for ALL shapes (delta RMS ≈0.2·lr)**. Do NOT use ns5/ns6; use ns4.
- chunk-rescale: paper_rescale gives FFN ×1.37 (≈√2), AdaLN ×2.45 (≈√6), attention ×1.01 (no rescale) — exact match to design
- CMuon step is 3.1× faster than AdamW8bit (NS = HCU matmul-bound; AdamW8bit 8-bit stochastic rounding + 2 moments is heavier)

**Checkpoint**
- Existing AdamW8bit ckpt resumes unchanged (optimizer.pt format unchanged)
- Hybrid saves: AdamW state + CMuon per-FQN momentum + chunk metadata + ns_steps + dtype + scaling mode + rescale + routing manifest (hybrid_cmuon_schema_version=1)
- Hybrid→Hybrid: state-exact (hard-fails on dtype/ns_steps/rescale mismatch)
- AdamW→Hybrid: weights strict, AdamW state per-FQN preserved, CMuon momentum from zero, NO fake 2nd-moment→Muon conversion, transition recorded in manifest (loader DONE + unit-tested, §9-H)

**Memory (no FP32 master weights; param dtype preserved)**
- CMuon BF16: 3.35GB state / 9.90GB peak (saves 1.74GB peak vs AdamW8bit 11.64GB)
- CMuon FP32: 6.42GB state / 13.17GB peak (+1.53GB peak vs AdamW8bit)

**Speed (projected total-update)**
- optimizer step: CMuon BF16 (ns4, ≈ns5) = 0.763s vs AdamW8bit 2.377s (~3× faster)
- step is ~15% of the ~16s/update → total update projected ~10% faster
- fwd/bwd identical (same model), not the differentiator

**Update RMS (08-28 audit — CORRECTS the old "Moonlight intrinsic" claim)**
- METRIC (unified): update_rms_per_group = single-step ACTUAL parameter-delta RMS = RMS(p_after − p_before), momentum from 0, 1st step, element-weighted within group. Definition consistent & correct — NOT relative/momentum/mixed.
- Moonlight invariant (scaled_pre_lr_rms ≈ 0.2) HOLDS on every shape where NS converges (q/out 0.22, ffn 0.19) → the math is CORRECT: the √max(d_in,d_out) factor cancels the 1/√max(dim) RMS of the semi-orthogonal update → scaled RMS ≈ constant 0.2 (shape-independent).
- The old attention 3.9e-3 / adaln 1.46e-3 non-uniformity is NOT "Moonlight intrinsic" and NOT a CMuon bug — it is QUINTIC-NS DIVERGENCE at specific shapes. native torch.optim.Muon diverges identically (custom==native per shape).
- Per-shape (ns5): q 3.36e-5 ✓, k 1.09e-2 (NS diverge), v 1.14e-2 (NS diverge), content_gate 3.31e-5 ✓, attn_out 3.31e-5 ✓, ffn 2.7-2.8e-5 ✓, adaln 1.42e-3 (NS diverge). attention group 3.9e-3 = element-weighted mix of normal q/out (3.3e-5) + divergent k/v (1.1e-2).
- Root cause (case A, not a bug): NS convergence window is shape-dependent (∝ largest singular value after Frobenius-normalize ≈ 1/√c+1/√r); small short-edge shapes (k/v 640, adaln 1024) have larger max-σ and leave the basin at iteration 5.
- **ns4 is the safe choice** (delta RMS ≈0.2·lr on ALL shapes: q 2.78e-5 / k 4.87e-5 / adaln 3.42e-5 / ffn 3.17e-5). ns5 diverges for k/v+adaln; ns6 diverges for all.

**Verdict / Recommendation (08-28 audit, CORRECTED)**
- **RECOMMEND: CMuon BF16 momentum + ns4 (config A hybrid_cmuon_core), with `cmuon_ns_steps` changed 5→4**
- Faster (~3× step, ~10% total update), similar state / less peak memory
- ns5 is NOT safe (k/v [640,2560] + adaln [2560,1024] diverge at the 5th NS iteration, delta 46×–1000×); ns6 diverges for ALL shapes; **ns4 converges for ALL shapes (delta ≈0.2·lr, uniform across groups)**
- Do NOT use FP32 momentum (v1): +2× memory for only +4.5% speed
- CAVEAT: requires (1) ns4 re-measurement of update RMS (expect ≈0.2·lr on all groups) and (2) long-run quality validation (loss/FID) before adoption; this audit validates speed/memory/numerics/scale only

**Open questions**
- BF16 momentum drift over long runs (>50k updates)? (none seen in short smoke)
- After switching to ns4, do all groups give update RMS ≈0.2·lr? (expected; re-measure to confirm)
- Is there a more robust NS (spectral-norm pre-normalization, or a more stable polynomial) making ns5/ns6 safe on all shapes? (v1: use ns4 to sidestep; not pursued)
- Effect of CMuon (ns4, uniform ≈0.2·lr) on generation quality? (needs long-run FID)
- Production wiring (config branch + 2 tomls) + AdamW→Hybrid transition loader: DONE (config schema + _build_optimizer branch + preflight + 2 tomls + transition loader all implemented; 17/17 tests pass on salt1 HCU)
- REMAINING GATES before adoption: (1) switch cmuon_ns_steps 5→4 + re-measure update RMS, (2) long-run quality validation (loss/FID) — this task STOPS here; does NOT switch the live trainer / does NOT deploy / does NOT change code or toml
=== END COPY TO CHATGPT ===
