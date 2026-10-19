# Condition 优化 A/B（仅显式 opt-in）

本改动不冻结 DiT 或 TE，不改变 CMuon 路由，也不启用新的生产 recipe。测试/示例配置位于
`config/train_g1_condition_optim_ab.example.toml`，数值只是接口示例，**不是调参建议**。
所有训练阈值和倍率来自经过验证的 TOML；修改必须在完整 checkpoint resume 时生效，不热重载。

## 两个独立开关，共用一个成员列表

```toml
[gradient]
clip_norm = 1.0
clip_dtype = "float32"

[optimizer.condition]
parameter_prefixes = ["condition_tokens"]
independent_clipping = true
clip_norm = 0.37
rest_clip_norm = 1.4
independent_lr = true
lr_multiplier = 0.23
```

- `parameter_prefixes` 是 composite 的 canonical FQN 前缀（整组件匹配，不是子串或 glob）。
  默认没有开启的 condition policy；开启 A/B 时需显式给出列表。`condition_tokens` **不会**
  自动包含 `dit.conditioner.condition_global_projection`。可明确加入其它 AdamW 前缀，但每个
  前缀都必须匹配可训练参数，不允许空匹配、重复、嵌套重叠或 DDP/compile wrapper 名。
- **A**：`independent_clipping=true`。在 DDP 梯度同步、共同样本均值归一化之后，将
  选中参数和其余**所有可训练参数**作为互斥完备两组分别裁剪；不是先全局裁剪再裁条件。
  两组 norm 均有限才开始修改任一组。缺失梯度按该组零 norm 处理；非有限仍 fail/poison
  并清理，不是静默跳过。独立模式用 `clip_norm/rest_clip_norm` **替代**全局阈值。
- **全局模式**：A=false 时使用 `[gradient].clip_norm`，修复旧 runtime 虽接受配置却固定1
  的实现。阈值必须有限非负；0明确禁用缩放，但仍执行非有限检查。原配置1的算术路径保留。
- **B**：`independent_lr=true`。为选中条件参数建立真实 AdamW parameter subgroups。
  不改 betas、eps、decay 分类、参数 dtype、moment 格式，不把条件参数改路由为 CMuon。
  选择到任何 CMuon 参数会 fail-fast；还必须存在剩余 AdamW 参数用于定义 rest LR。
- A-only：B=false、倍率为中性1；B-only：A=false且不提供两组 clip 阈值。
  两个开关都false时不提供成员列表或非中性旋钮，以免配置被静默忽略。
- 省略整张 `[optimizer.condition]` 表完全保留旧参数组、序列化和恢复路径。
  B开启但倍率1是中性学习率，不代表序列化组拓扑仍是旧两组。

## 学习率相对谁？

令 `base(u)` 为现有共同 successful-update scheduler 在尝试更新u时给出的学习率：

- CMuon：`base(u) * optimizer.cmuon_lr_multiplier`；
- 其余 AdamW：`base(u) * optimizer.adamw_lr_multiplier`；
- 条件 AdamW：`base(u) * optimizer.adamw_lr_multiplier * optimizer.condition.lr_multiplier`。

没有独立 warmup 或新的 update counter。失败不会推进成功更新调度；恢复不重启 warmup。
`learning_rate_adamw` 保持 **rest AdamW** 的有效率，另有 `learning_rate_condition` 记录
本次实际使用的条件率，而非乘数/下一次更新率。非混合模式的旧 `learning_rate` 仍为 rest
AdamW有效率，`learning_rate_cmuon=0`。所有四种现有 optimizer 模式都有显式集成。

## 为什么不能简单把条件参数全部放到新组末尾？

TorchAO AdamW 按 `param_groups -> params` 逐参数更新，一条共享 stochastic-rounding
RNG 负责其中的随机舍入。普通“原组剩余参数 + 末尾条件组”的拆法会改变共享参数的随机数
分配，即使倍率为1也不再是相同更新。

因此实现按每个旧 decay/sensitive 组内部的**连续成员段**拆组。比如原顺序
`restA, conditionA, restB, conditionB` 仍保持这个顺序，只增加组边界。原 `group_name`
与 `ParameterSpec.group` 不变，以 `condition_lr_group: bool` 标记有效LR类型。可存在
多个同名子组，而不是保证只有一个condition物理组。Tensor LR为各段独立clone，避免
scheduler原地fill时意外联动。

这保留 flattened FQN 顺序、每参数状态及共享 SR traversal；不重排或重置 SR。CPU测试
验证顺序和状态；是否达到特定 vendor GPU 实现的中性逐bit更新仍需该环境的执行测试，
不能用CPU替代优化器测试冒充GPU数值等价。

## 完整 checkpoint 恢复

旧两组 AdamW checkpoint可恢复到 opt-in 子组；新子组checkpoint也可在新配置倍率下恢复。
恢复以 canonical FQN 重映射 state ID，而不是依赖组内位置。

在重映射前独立校验源schema/实际payload的组与FQN、重复/缺失/非法ID、decay归属、
相对顺序及非LR/WD锁定超参。禁止通过直接用current_groups覆盖源组掩盖坏betas/eps。
quantized moments（codes/qmap/scale）、每参数step、CMuon momentum、guard状态与SR RNG
都必须原样保留；仅LR/WD继续由当前配置治理。未知状态不能静默丢弃或重建。

B的恢复是显式升级路径；新子组直接加载为未开启B的旧拓扑不受支持，必须fail-fast，
不得假装完成回退。**中性回退请保持 `independent_lr=true`、成员列表不变，并设
`lr_multiplier=1.0`；不能简单删表或置 independent_lr=false 来恢复 tagged checkpoint。**
四种optimizer模式支持同family恢复，不等于支持 pure AdamW→hybrid 的跨family迁移组合：
启用B的这类跨family转换明确拒绝，不能静默丢失已拆组的身份。

## 指标解释

独立模式另记condition/rest各自raw norm、post norm、真实clip coefficient及是否clip。传统总体pre/post norm
仍可表示两组L2合成，但独立裁剪后总norm可能大于原全局阈值，这不是裁剪失效。
两组系数不同，aggregate coefficient仅为总体post/pre比（全零时为1），不是实际施加给所有
梯度的共同系数；`clip_fraction` 从两组真实coefficient计算any_group_clipped，不用汇总比的
舍入变化判断裁剪。
旧 `condition_encoder_grad_norm` 保持原诊断口径，不取代可配置成员组的raw norm。

关闭A/B时不输出新的condition优化指标，不把缺失值补为0；非有限不生成成功指标。

## 验证（CPU，不生成图片）

```bash
python -m pytest tests/unit/train/test_condition_clipping.py tests/unit/optim/test_condition_learning_rate.py tests/unit/optim/test_learning_rate_multipliers.py tests/unit/checkpoint/test_condition_groups.py -q
python -m ruff check src/sakuramoon/config/schema.py src/sakuramoon/optim/clip.py src/sakuramoon/optim/condition.py src/sakuramoon/train src/sakuramoon/telemetry src/sakuramoon/checkpoint/load.py tests/unit/train/test_condition_clipping.py tests/unit/optim/test_condition_learning_rate.py tests/unit/checkpoint/test_condition_groups.py
pyright --outputjson
```

Python版本与TorchAO/vendor依赖以项目支持环境为准；不为测试安装/替换生产包。
测试涵盖 arbitrary thresholds/multipliers、纯AdamW及三种hybrid、旧→新/新→新恢复、
中性/关闭、selector交错顺序和无效配置/非有限失败。真实量化checkpoint的CPU字节级
比较及vendor执行由独立验证负责。本地曾因未mask的既有测试触发CUDA：两项step缺Triton失败、
一个production-sized构造OOM；这些环境失败保留，不冒充功能验收。后续所有进程启动前均mask
CUDA/HIP/ROCR，只跑CPU。没有执行生产 forward/backward 或原生DTK GPU测试。
