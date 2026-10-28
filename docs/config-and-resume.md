# SakuraMoon 配置驱动训练：接口与恢复契约

本仓库的训练执行完全由 `config/*.toml` 驱动。运行时代码不再读取任何历史
阶段策略（stage graph）、伪配置锁（locked constants）或部署期硬编码；
所有执行参数只来自配置，所有冲突在配置读取期 fail-closed。

## 1. 当前配置接口

只有以下顶层表参与执行：

```toml
[run]                  # run_id, label（仅展示）, ...
[paths]
[security]
[assets.qwen] [assets.vae]
[data]                 # mainset/cache/buckets/quality pipeline
[caption]
[model.dit]            # depth + active_slot_ids（拓扑唯一来源）
[model.vae] [model.qwen]
[train]                # resolution, local_batch, accumulation,
                       # max_updates, activation_checkpoint_mode, ...
[objective]            # flow matching 数值边界（t_eps, p_std, noise_scale...）
[optimizer]            # name, base_lr, reference_batch, lr_scaling, ...
[distributed]          # backend, world_size
[growth]               # enabled, ramp_updates（在线扩容切换）
[checkpoint]
[sampling]             # profile 驱动（用户自定 profiles 表）
[evaluation]
[irepa]                # 可选；缺表 = 特性不存在
[condition_diagnostics] # 可选；缺表 = 特性不存在（§1.1）
```

### 1.1 [condition_diagnostics]（可选，默认关）

条件侧独立诊断流（`train/condition_diagnostics_stream.py`，schema v1，
与 TrainingMetric schema 12 独立版本化；不修改 schema 12）：

```toml
[condition_diagnostics]
enabled = true            # 缺表或 false = 特性不存在：训练路径、优化器流程、
                          # schema-12 输出与未启用时逐字节一致
log_every_n = 500         # 每 N 个成功 update，每个 rank 追加一条记录
flush_every_records = 100 # 每 N 条记录 fsync 一次（默认 100；本流在训练线程
                          # 写盘，设 1 = 每条一次 fsync，热路径开销显著）
```

- 每个 rank 写自己的 `<artifact_dir>/diagnostics/rank-<n>/condition-diagnostics-v1.jsonl`
  （线程安全 `DurableJsonlSink`，O_APPEND/0o600）；记录字段：条件侧激活事实
  （token RMS、active/null 比例、global-path RMS、per-slot cosine）+
  pre/post-clip 条件侧 union 梯度范数与占比。
- 开销：仅 cadence 到期时重算一次 conditioner 前向（`no_grad`，取最后一个
  microbatch）；其余 update 只存引用，无张量运算。
- 重试的 update 会向同一 per-rank 文件追加，失败 update 可能重复出现
  （主 metrics 流同行为）。
- 实现注记：`tensor_rms` / `tensor_cosine` / `metric_float` /
  `TrainingSamplingError` 四个纯函数从 `train/condition_diagnostics.py`
  **逐字搬迁**（数值行为不变，`condition_diagnostics.py` 改为 re-export）
  到 `train/tensor_metrics.py`——per-update 诊断路径不应拖入 `eval.spec` /
  `conditioning.global_condition` 的采样期导入图。

关键执行字段：

| 字段 | 语义 |
| --- | --- |
| `train.resolution` | 训练分辨率（桶几何由此派生形状集合，无外部 `shape_count` 锁） |
| `train.local_batch` × `train.accumulation` × `distributed.world_size` | 有效全局批（`effective_global_batch()`，`[train]` 内部断言） |
| `train.max_updates` | **当前调用的绝对**成功更新终点；每个 resume 实时读取、双向权威（见 §3 第 6 条） |
| `train.activation_checkpoint_mode` | 激活检查点模式 |
| `model.dit.depth` | DiT 层数；必须等于 `active_slot_ids` 长度 |
| `model.dit.active_slot_ids` | 活跃槽 id 集合（拓扑唯一来源；缺省时按 depth 派生历史拓扑） |
| `distributed.world_size` | DDP 世界大小 |
| `optimizer.lr_scaling` | `linear_global_batch`（`base_lr * gb / reference_batch`）或 `none`（恒定 `base_lr`） |

派生量（配置读取期计算，运行时只读）：

- `effective_global_batch()` = `train.local_batch * train.accumulation * distributed.world_size`
- `scaled_learning_rate()` = `linear` 时 `base_lr * effective_global_batch() / reference_batch`，`none` 时 `base_lr`

### 独立优化器学习率

`optimizer.cmuon_lr_multiplier` 与 `optimizer.adamw_lr_multiplier` 均为正有限数，
默认 `1.0`。混合优化器在每个成功更新边界显式设置
`CMuon LR = scheduled_base * cmuon_lr_multiplier`、
`AdamW LR = scheduled_base * adamw_lr_multiplier`（所有 AdamW groups）。
CMuon 的 Moonlight、weight decay 和 LR 相关安全阈值使用自己的有效 LR；
非混合 AdamW 路径仅使用 AdamW multiplier。

指标 schema v12 的 `learning_rate` 对混合优化器仍表示共同调度基准，
`learning_rate_cmuon` / `learning_rate_adamw` 表示该次更新实际使用的有效 LR；
非混合路径前者为 `0.0`、后者等于原有 `learning_rate`。
修改 TOML 在受控的完整 checkpoint resume 后生效，不支持热重载。
恢复保留原 AdamW moments / quantization / step、CMuon momentum 和 SR RNG，
以当前配置和已恢复成功更新坐标重设两个 LR，不重启 warmup、不改变路由或状态格式。

可选 `[optimizer.condition]` 为同一显式 FQN 前缀列表提供独立裁剪和 AdamW LR 子组；
启用 B 时上文 AdamW LR 表示 rest，条件 LR 再乘 `condition.lr_multiplier`，另记
`learning_rate_condition`。不启用时保持原路径。阈值 0 禁裁但不禁非有限检查；生产路径
现在将 `[gradient].clip_norm` 传入更新器，而不是忽略该值固定裁剪到1。
完整开关、状态恢复限制（含 tagged→legacy fail-fast 与中性倍率1）见
[Condition 优化 A/B](condition-optimizer-ab.md)。

## 2. 拓扑与扩容

- 历史拓扑不连续：`depth=16` → 16 个非连续槽；`depth=20` = 16 槽 ∪ {2,8,14,20}；
  `depth=24` → 0..23 连续。派生规则在 `sakuramoon/model/slots.py`。
- `PackedDiT`/`DenseDiT` 的 `stable_slot_count` 必须等于 `max(active_slot_ids) + 1`
  （排他上界），由构造器校验，不再接受历史 24 锁。
- **在线扩容切换**（16→20 示例）：配置 `depth=20` + 从 16 槽 checkpoint resume +
  `growth.enabled = true`。运行时用 `new_slot_ids(target, source)` 计算差集作为
  `new_slot_ids`，新槽从 0 开始半余弦 ramp（`growth.ramp_updates`），锚定在恢复更新处。
  离线 `migrate_growth.py` 已删除；其 `growth_migration.json` 侧车仍被 checkpoint
  读取器接受（可选键）。

## 3. 恢复绑定（resume binding）

checkpoint 与配置之间的绑定分两类：

- **信息性**（不门控）：`growth.stage` / `growth.world_size` / `growth.resolution`
  元数据、run label、分辨率变化、checkpoint 的历史 terminal
  （`stage_budget.terminal_successful_update`，仅作兼容性包络）。
- **结构性**（严格校验，违规 fail-closed）：
  1. 槽拓扑：相等 = 普通 resume；严格子集 = 扩容切换（要求 `growth.enabled = true`、
     源 ramp 已完成、模块 `new_slot_ids` 与差集一致）。
  2. 身份完整性：checkpoint identity 与设备 ordinal 每次重绑到本 rank 设备。
  3. 节拍完整性：cadence 状态重绑到当前 `checkpoint.full_every_updates`。
  4. 确定性 alpha：ramp 锚点与 `ramp_updates` 一致。
  5. FQN 集合：默认 = 精确结构契约（不允许任何移除）；扩容切换可**新增**明确声明的
     新槽前缀 FQN（`_verify_fqn_sets`）；规范化 iREPA v4 → OFF 直接恢复可移除
     **恰好**已声明 locked-v1 iREPA 辅助的 2 个 FQN 及其优化器/路由状态
     （声明仅限 locked iREPA v1 元数据，任何其它移除仍 fail-closed）；
     优化器 group diff 同理（`_verify_group_diff`，顺序敏感）。
  6. 预算：`train.max_updates` 是**当前调用的唯一执行权威**（双向）：
     - `max_updates > 历史 terminal`：持久化 terminal 向上扩展到
       `max_updates`（后续保存的 update 必须仍在包络内可表示）；
     - `max_updates ≤ 历史 terminal`：历史 terminal 原样保留（不收缩、不改写
       源 checkpoint）；训练循环与强制终检点仍停在 `max_updates`；
     - 调低 `max_updates` 永不回滚模型/优化器/计数——它只表示“本次调用不超过
       该点继续训练”；
     - `max_updates ≤ 恢复 update`：零 update 干净完成（成功的 no-op，不是失败、
       不是回滚）：checkpoint 恢复/结构绑定后立即退出——不连接数据服务、
       不加载冻结编码器、不运行训练前检查（含 Qwen fast-path 探针）、
       无 forward/backward/optimizer.step；显式 `preflight-only` 调用不享受
       该短路，仍运行完整训练就绪预检。

     例（checkpoint @130k，历史 terminal 168k）：
     `max_updates=200k` → 训练到 200k；`150k` → 训练到 150k；`135k` → 训练到
     135k；`130k` / `120k` → 零 update 成功。

旧 v3/v4 checkpoint（无 `new_slot_ids`、growth 侧车、`growth_migration.json`）
均可加载：`new_slot_ids` 缺省为空元组，`stable_slot_count` 由活跃拓扑派生。

## 4. 遗留 `[stage]` 表

旧配置中的 `[stage]` 在配置读取边界被翻译一次（`normalize_legacy_config`，
代码库中唯一的遗留翻译）：

- `stage.resolution / local_batch / accumulation / max_updates` → `[train]`
- `stage.world_size` → `[distributed].world_size`
- `stage.depth` → `[model.dit].depth`
- `stage.name` / `run.stage` → `run.label`（仅展示）
- 与现有显式值冲突 → `ConfigurationError`；每次映射/废弃键产生一次性通知。

运行时代码永不看到 stage 词汇。

## 5. 采样与评估

- `sampling.profiles` 是用户自定的命名预设表（历史 preview/balanced/reference
  保留为模板预设）；`sampling.profile` 按名选择，NFE 由所选 solver/steps 计算。
- 生成元数据要求显式 `noise_scale` 与 `t_eps`（来自 `[objective]`，不再是代码锁）。
- 周期采样（`sampling.training`）的 locked 纵向队列保持 `image_count = 60` 不变量。
- 每次评估重新生成当前模型的特征；真实图像特征只在同一个 evaluator 实例内复用。
  不读取或写入跨运行的磁盘特征缓存，已有缓存文件不再使用，重启后重新提取真实特征。
- 本地 teacher / Inception 资产保留文件、大小和适用的结构检查，不计算或校验项目级 SHA。
  这些检查不保证同大小资产的内容一致；评估结果不再附带验证集 fingerprint。

## 6. 测试

- 单元测试位于 `tests/unit/`，Linux 权威门：`pytest tests/unit -q`
  （DTK 环境需先 `source /opt/dtk/env.sh`）。
- 配置模板全部经 `load_config` 加载验证（`base.toml` 含刻意的
  BENCHMARK 哨兵，加载时必须 fail-closed，属预期）。
- Windows 本机 `os.O_NOFOLLOW`/`os.O_DIRECTORY`/`fcntl`/符号链接缺失导致的
  少数失败为平台环境差异，非回归（已在 BASE 提交同环境对照确认）。

## Final output norm removal (isolated resume)

`model.dit.final_output_norm = "none"` is the new architecture default. It removes
only `dit.output_head.norm.weight` and the dense/packed final normalization;
final scale/shift and the original FP32 projection retain their names and values.
This deliberately changes the model function. `"rmsnorm"` is retained solely to
reconstruct/compare old artifacts. An old artifact without the field means
`"rmsnorm"`, not `"none"`.

The existing strict checkpoint loader accepts this **directional** architecture
change. It verifies and removes only that declared FQN, remaps optimizer slots
using saved `param_names`, and preserves retained tensors and metadata. Other
missing/unexpected names, wrong shapes/dtypes and changed locked group metadata
remain errors. New saves include the explicit architecture marker. No source
checkpoint is overwritten and no weight/gamma folding is performed.

The checked-in model has no model-EMA owner or serialized EMA. The strict
`checkpoint.final_norm.remove_final_norm_named_state` adapter supports an
externally verified FQN→tensor shadow map without rebuilding its history. If the
experiment deployment adds EMA, its actual serializer and counter/decay fields
must be verified and integrated before claiming EMA resume acceptance.

Use an isolated copy of the source checkpoint and an isolated TOML derived from
the experiment machine's **resolved config**, preserving optimizer/LR/clipping,
data cursor and backend settings. Set a new `run.run_id`, isolate **all** `paths`
outputs, and use a new `paths.checkpoint_dir` beneath `output_model/`. Set
`train.max_updates` to the saved successful update count plus the desired short
window; it is an absolute endpoint, not a number of additional steps. Start with
`model.dit.final_output_norm = "none"` and `kernels.equivalent_fusions = false`.
Do not run the production stack/restart scripts for this experiment.

These are existing executable entrypoints (paths are operator-supplied):

```bash
PYTHONPATH=src uv run --no-sync python -m sakuramoon.cli.train \
  --config "$ISOLATED_CONFIG" --config-root "$ISOLATED_CONFIG_ROOT" --root "$ISOLATED_ROOT" \
  --resume "$ISOLATED_CHECKPOINT_COPY" --preflight-only
PYTHONPATH=src uv run --no-sync python -m sakuramoon.cli.train \
  --config "$ISOLATED_CONFIG" --config-root "$ISOLATED_CONFIG_ROOT" --root "$ISOLATED_ROOT" \
  --resume "$ISOLATED_CHECKPOINT_COPY"
```

This is compatibility loading, not an offline conversion command. A subsequent
normal checkpoint save writes the norm-free model and preserved retained optimizer
state to the isolated output location. Validate a fresh process resume from that
new save before treating the experiment as accepted.

For asset-free architecture diagnostics (not full-model quality), run:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src uv run --no-sync python \
  -m sakuramoon.cli.validate_final_norm \
  --config train_g1_hybrid_cmuon_ns4_core.toml --dtype float32
```

The diagnostic preserves a shared nonzero projection and final modulation across
old/new heads, reports output RMS, loss and gradient scales, and checks finite
forward/backward. Synthetic old/new equality is deliberately not an assertion.
