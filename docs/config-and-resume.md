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
```

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

## 6. 测试

- 单元测试位于 `tests/unit/`，Linux 权威门：`pytest tests/unit -q`
  （DTK 环境需先 `source /opt/dtk/env.sh`）。
- 配置模板全部经 `load_config` 加载验证（`base.toml` 含刻意的
  BENCHMARK 哨兵，加载时必须 fail-closed，属预期）。
- Windows 本机 `os.O_NOFOLLOW`/`os.O_DIRECTORY`/`fcntl`/符号链接缺失导致的
  少数失败为平台环境差异，非回归（已在 BASE 提交同环境对照确认）。
