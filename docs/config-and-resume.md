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
| `train.max_updates` | **绝对**成功更新终点；每个 resume 实时读取，只允许延长、收缩 fail-closed |
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
  元数据、run label、分辨率变化。
- **结构性**（严格校验，违规 fail-closed）：
  1. 槽拓扑：相等 = 普通 resume；严格子集 = 扩容切换（要求 `growth.enabled = true`、
     源 ramp 已完成、模块 `new_slot_ids` 与差集一致）。
  2. 身份完整性：checkpoint identity 与设备 ordinal 每次重绑到本 rank 设备。
  3. 节拍完整性：cadence 状态重绑到当前 `checkpoint.full_every_updates`。
  4. 确定性 alpha：ramp 锚点与 `ramp_updates` 一致。
  5. FQN 集合：当前模型相对 checkpoint 的**新增** FQN 必须全部落在新槽前缀内
     （`_verify_fqn_sets`），**不允许任何移除**；优化器 group diff 同理
     （`_verify_group_diff`，顺序敏感）。
  6. 预算：`train.max_updates` 只允许延长 checkpoint 记录的 terminal；
     收缩抛 `ValueError`（fail-closed）。

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
