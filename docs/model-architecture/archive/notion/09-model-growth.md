Here is the result of "view" for the Page with URL https://app.notion.com/p/3abae967ecf28103be8feea5e27f21e1 as of 2026-07-28T10:44:47.239Z:
<page url="https://app.notion.com/p/3abae967ecf28103be8feea5e27f21e1">
<ancestor-path>
<parent-data-source url="collection://69ca66ff-43e7-4128-bb7b-9f3751506705" name="组件决策记录"/>
<ancestor-2-database url="https://app.notion.com/p/250be554eacc40219065073dfcf66fd7" title="组件决策记录"/>
<ancestor-3-page url="https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff" title="新模型架构决策中心"/>
</ancestor-path>
<properties>
{"date:决定日期:is_datetime":0,"date:决定日期:start":"2026-07-28","url":"https://app.notion.com/p/3abae967ecf28103be8feea5e27f21e1","决策编号":"ARCH-10","序号":9,"影响":"高","标签":["架构","训练"],"状态":"待验证","组件决策":"09 模型规模与深度增长"}
</properties>
<content>
<callout icon="✅" color="green_bg">
	**架构决定已批准，待实现验证。** 16→20→24 均匀深度增长、半余弦 ramp、AdamW8bit state 迁移语义、阶段顺序和 transition 验收均已锁定。早期 FSDP2/bitsandbytes 实现说明已被组件 12 覆盖；当前训练系统为单卡原生→四卡 DDP、TorchAO AdamW8bit，具体参数组/LR/state 契约以组件 12-D/12-E 为准。
</callout>
# 组件边界
本组件决定固定宽度模型如何从 16 层增长到 20 层、再增长到 24 层，包括新增层位置、函数保持接入、优化器状态迁移和增长触发条件。分辨率与宽高比课程留给组件 10，基础优化器及分布式训练细节留给组件 12。
# 已锁定上游接口
- [06 Single-stream DiT 主干](https://app.notion.com/p/3abae967ecf281809823e98b5d91222e)：固定 `hidden_size=2560`、`head_dim=128`、20Q/5KV GQA、`intermediate_size=6912`，深度阶梯 `16 → 20 → 24`。
- 每个 block 约 76.03M 参数；block 主体在 16/20/24 层约为 1.216B/1.520B/1.825B，最终可训练参数约 1.85B–1.90B。
- 每个 block 已定义 `alpha_attn` 与 `alpha_mlp` 两个 growth switches；base blocks 为 1，新增 blocks 接入时为 0，因此增长瞬间可以严格保持原函数。
- [07 Timestep 与全局条件](https://app.notion.com/p/3abae967ecf2815b84fbda3c707728c2)：共享 condition projection 已在旧阶段训练；新增 block 只新增自己的 modulation bias，初值为 0。
- [08 x-pred 目标与采样系统](https://app.notion.com/p/3abae967ecf28167a869fb61c5ff0e96)：PMA 不跨深度增长边界；训练恢复默认使用 raw checkpoint 与配套训练状态。
# SANA 1.5 的实际做法
官方 [SANA 1.5 论文](https://arxiv.org/html/2501.18427v4) 将固定宽度 2240、FFN 5600 的模型从 20 层扩展到 60 层，比较了三种初始化：
- **Partial Preservation**：保留前 N 个旧层，新增层随机初始化。
- **Cyclic Replication**：循环复制旧层。
- **Block Replication**：把每个旧层连续复制多次。
论文选择 Partial Preservation；复制方案在其训练中出现不稳定甚至 NaN。新增层通过将 attention/MLP 输出投影置零实现恒等映射，并增加 QK Norm。根据 block importance，SANA 还删除了预训练模型最后两个 block。
该论文报告相对从头训练减少约 60% 训练步数，但其增长幅度为 `20→60`、主干是线性 DiT、训练使用 64 张 A100，不能把收益数字直接外推到本项目的 `16→20→24`、softmax GQA 单流和四张 RTX 5090。
# 适配原则
本项目采用 SANA 的核心原则，而不逐项照搬：
- 保留已有训练知识；新增层随机初始化，不复制旧层。
- 用已批准的独立 growth switches 实现严格恒等接入，不再把输出投影清零；这样新 block 内部保留非零函数，switch 才能获得有效梯度。
- 由于每次只增加 4 层，不删除任何旧层。照搬“删除最后两层”会在 `16→20` 时丢弃 12.5% 已训练主干，风险大于潜在收益。
- 保留已批准的 QK-RMSNorm，不因增长另加一套 normalization。
# 09-A：新增层位置与初始化
## A. 尾部追加 + Partial Preservation
最接近 SANA 的实现，checkpoint 映射简单，但两次各新增四层会把全部新容量集中到网络尾部，初期更偏向末端细化，不采用。
## B. 均匀插入随机新层，已批准
预先定义最终 24 个稳定 slot，外部文档使用 1-based `s01–s24`：
- 初始 16 层占用 `s01,s02,s04,s05,s07,s08,s10,s11,s13,s14,s16,s17,s19,s20,s22,s23`。
- `16→20` 时，在原始第 4、8、12、16 层后插入新层，即激活 `s06,s12,s18,s24`。
- `20→24` 时，在原始第 2、6、10、14 层后插入新层，即激活 `s03,s09,s15,s21`。
- 最终每组六层的来源顺序为“旧、旧、第二次新增、旧、旧、第一次新增”，新增容量均匀覆盖浅、中、深位置。
实现约束：
- state dict 使用稳定 slot ID，而不是当前 active list 下标；执行顺序由 checkpoint/config 的 `active_slot_ids` 明确记录。
- 旧层参数和 optimizer key 不因插层重编号。
- 新层使用基础模型同分布随机初始化，不复制相邻层；RMSNorm/QK-Norm weight=1、per-block modulation bias=0、`alpha_attn=alpha_mlp=0`。
- 不删除任何旧层，不使用 PMA 权重替代 raw checkpoint 执行增长。
## C. 均匀插入并复制相邻层
属于 replication 初始化；SANA 1.5 中复制方案更不稳定，本项目没有必要采用。
## D. 删除旧尾层再扩展
SANA 的依据来自 `20→60` block importance；这里每次只增加四层，删除两个旧层会损失过多已训练容量，不采用。
# 09-A 最终决定（已批准）
采用 B：两次增长均匀插入四个随机初始化的恒等 block，使用稳定的最终 slot ID 保持 checkpoint/optimizer 映射；完整保留旧层，不复制、不删除。
# 09-B：growth switch 渐开
## A. 可学习 alpha
把 `alpha_attn/alpha_mlp` 作为普通参数从 0 开始优化。它能自行选择打开速度，但不同层可能长期停留在不同幅度，alpha 也可能变负或瞬时过大；结果依赖 optimizer 超参数，不利于两次增长复现，不推荐。
## B. 直接从 0 切到 1
增长点函数保持，但第一个训练更新立即接入四个完整随机 residual，容易造成 loss/grad norm 瞬态，不采用。
## C. 固定半余弦 ramp，推荐
增长迁移完成后先在 `alpha=0` 下执行无参数更新的等价性检查。正式训练第 `k` 个成功 optimizer update 使用：
$$
p=\operatorname{clip}(k/R,0,1)
$$
$$
\alpha(k)=\frac{1-\cos(\pi p)}{2}
$$
具体协议：
- 仅新增四层的 `alpha_attn` 与 `alpha_mlp` 同步使用该值；所有旧层固定为 1。
- `R = clamp(ceil(0.02 × planned_new_depth_updates), 1000, 5000)`，即新深度阶段计划更新数的 2%，最少 1000、最多 5000 个成功 optimizer updates。
- ramp 在新深度的正常训练预算内部进行，不额外增加训练步数。
- alpha 是 checkpointed FP32 schedule state，不加入 optimizer、不做 weight decay；达到 1 后永久固定为 1。
- 只计算真正完成的 optimizer update；数据跳过、验证、仅 forward 和失败/回滚的 step 不推进 `k`。
- 两个 switch 分开保存在 checkpoint 中以保持结构和诊断能力，但 baseline 不错峰打开 attention 与 MLP。
- 第一次增长的四层完成 ramp 后保持 1；第二次增长只渐开新激活的 `s03,s09,s15,s21`。
- checkpoint/manifest 保存 growth generation、`R`、`k`、公式版本和 active slots，恢复后继续同一进度，不能按 wall clock 重算。
该方案在 `k=0` 严格保持旧函数，开始时变化平缓，并确保所有新增层最终完全参与计算。新层内部权重的梯度会被 alpha 缩放；第一次正式更新使用 `k=1` 的非零 alpha，因此不会永久冻结。
# 09-B 最终决定（已批准）
采用 C：新增层的 attention/MLP growth switches 同步执行固定半余弦 `0→1` ramp，时长为新深度阶段计划更新数的 2%，限制在 1000–5000 个成功 optimizer updates；alpha 不参与学习。
# 09-C：optimizer state 迁移
## 官方接口边界
PyTorch [Distributed Checkpoint state-dict API](https://docs.pytorch.org/docs/main/distributed.checkpoint.html) 会把 optimizer 内部 parameter ID 转为未并行模型的 canonical FQN，并能处理 `fully_shard`/FSDP2 的 DTensor 状态与不同 world size 的重新分片。因此增长迁移以 canonical FQN 和稳定 slot ID 为主键，不能直接序列化 Python parameter object 或当前 active list 下标。
该 API 仍标为 experimental，PyTorch 版本和 state-dict schema 必须写入 run manifest。`set_optimizer_state_dict` 只能在 backward 前或 optimizer step 后调用。
## A. 全部重置 optimizer
实现最简单，但会清空约 1.2B/1.5B 已训练参数的历史 moments，破坏增长点连续性并可能造成全模型瞬态，不采用。
## B. 旧状态按 FQN 保留，新状态置空，同一学习率，推荐
语义规则：
- 所有名称、shape、dtype 与 optimizer group 规则未改变的旧参数，完整保留 first/second moments、per-parameter step 和优化器后端所需量化元数据。
- 新增四层只创建空 optimizer state：moments=0、per-parameter step=0；8-bit quantization scale/metadata 由 optimizer backend 首次初始化，不从相邻层复制或手工伪造。
- 新旧 block 使用同一个 stage LR、betas、eps 和 scheduler，不给新层设置 LR multiplier，不采用 layer-wise LR decay。growth ramp 已经对新层有效更新做渐开，再叠加更高 LR 会增加不必要的瞬态。
- weight-decay 分组沿用模型规则：attention/MLP 大矩阵进入 decay group；RMSNorm/QK-Norm、modulation bias 和其他 bias 不衰减；alpha 是 schedule state，不进入 optimizer。
- 全局 scheduler 进入显式的 new-depth segment；具体 LR 数值由组件 12 决定，但旧层和新层必须共享，不允许 loader 因缺少新 state 而隐式重启整个 scheduler。
## C. 复制相邻层 optimizer moments
新层权重是随机初始化，复制 moments 会把不对应的梯度历史施加到新参数上；比仅复制权重更缺乏语义，不采用。
## D. 新层独立高学习率
可能加快追赶，但与 alpha ramp 共同作用后形成第二套增长调度，需要额外调参；不进入 baseline。
## 迁移事务
1. 在旧深度完成一个 optimizer step 后保存可恢复的 raw checkpoint，并校验模型、optimizer、scheduler、RNG、数据位置和 manifest 完整性。
2. 按目标 active slots 构建新拓扑；新增层由记录在 manifest 的 `growth_seed` 确定性初始化。
3. 以 canonical FQN 加载旧模型权重。missing keys 必须精确等于本次新增 slot 的 allowlist；任何旧参数 missing、shape 变化或 unexpected key 都中止迁移，不能用宽泛 `strict=False` 忽略。
4. 按组件 06 的 bottom-up 粒度执行 FSDP2 `fully_shard`，构造新 optimizer，再通过 named distributed state-dict 迁移旧参数状态；新增 FQN 保持未初始化/零状态。
5. 立即保存一份 `post_growth_pre_update` checkpoint；完成 alpha=0 等价性验证后才允许第一个训练 update。
6. 迁移报告保存 reused/new/dropped optimizer FQN 数量与列表、tensor shape、state checksum、optimizer/scheduler config hash。`dropped` 必须为 0。
## AdamW8bit 实现延期
- 当前训练 optimizer 选择保持 **AdamW8bit**，本组件不改为其他 optimizer。
- 09-C 只锁定目标语义：旧 state 保留、新 state 置零、相同 stage LR。此时不要求实现 adapter，也不因 PyTorch DCP 的实验状态阻塞后续架构讨论。
- AdamW8bit、FSDP2 与 checkpoint state 的具体保存/加载方式统一推迟到组件 12，在固定软件版本和目标机器上实现与测试。
- 在组件 12 验证前不宣称兼容，也不提前决定替代 optimizer；任何未来变更必须重新形成显式决定，不能由 loader 静默改变。
- bitsandbytes [官方 AdamW8bit 文档](https://huggingface.co/docs/bitsandbytes/en/reference/optim/adamw)作为后续实现依据保留。
# 09-C 最终决定（已批准）
旧参数 optimizer state 按 canonical FQN 完整保留，新增层 state 从零开始；新旧层使用相同 stage LR，不复制 moments、不设新层 LR multiplier。当前 optimizer 仍为 AdamW8bit，迁移实现与兼容性测试延期到组件 12。
# 09-D：增长触发与阶段解耦
## 原则：一次只改变一个主要轴
GPU world size、模型深度、训练分辨率、数据混合和 optimizer 类型不能在同一个 transition step 同时改变。否则出现 loss 或质量异常时无法定位来源，也无法选择干净的回滚点。
深度增长采用预先配置的预算触发加安全门槛，不使用“训练 loss 看起来平台了”作为自动触发器。扩散 loss 与生成质量并非严格单调，纯动态 plateau 判定容易过早或过晚增长。
## 推荐阶段顺序
1. **S0：单卡、16 层、256。** 按已批准方向执行正式前置训练，预计覆盖前几个 epoch；精确有效样本/FLOPs 预算由组件 10 在吞吐 benchmark 后锁定。
2. **S1：四卡、16 层、256。** 先只改变 world size。完成无更新一致性、shard restore 和 global-batch 检查，再进行 200–1000 个成功 optimizer updates；该窗口验证分布式训练和网络存储并发，不改变深度或分辨率。
3. **G1：四卡、20 层、256。** 从 S1 的 raw checkpoint 执行 `16→20`，完成 09-B 的 1000–5000 update ramp；alpha 达到 1 后至少再运行 `max(500, ceil(0.25R))` 个成功 updates，确认完整 20 层稳定。
4. **S2：四卡、20 层、512。** 只有 G1 通过后才提升分辨率；这是主要 512 预训练阶段，具体宽高比分桶和 FLOPs 配额由组件 10决定。
5. **G2：四卡、24 层、512。** 在分辨率和数据混合保持不变时执行 `20→24`，使用同样 ramp 与 post-ramp 稳定窗口。
6. **S3：四卡、24 层、512→768/1024 可选。** 完整 24 层稳定后，组件 10 才能决定是否进入更高分辨率；512 是最低最终要求，1024 不是必须阶段。
## 每次增长的资格门槛
到达预设 stage budget 后，只有同时满足以下条件才执行增长：
- 存在完整且已验证可恢复的 raw checkpoint；不得使用 PMA artifact 直接增长。
- 当前拓扑、分辨率、数据协议和 optimizer 配置 hash 与 stage manifest 一致。
- 最近 1000 个成功 updates 无 non-finite loss/gradient、无未解释的 loss spike，训练吞吐与显存没有持续退化。
- 固定验证集 loss、固定 prompt/seed 样本和 tag/NL 条件跟随没有相对最近已接受 checkpoint 的明确回退。
- 数据跳过率、网络读取错误和 batch/token 统计处于组件 11/12 设定阈值内。
若预算到达但门槛失败，保持当前深度并排查；不得自动增长。若指标仍持续改善但门槛正常，仍按预设预算增长，以保证最终 24 层获得足够训练预算，不无限等待 plateau。
## 配置与恢复
- 每个 stage 显式记录 `stage_id`、world size、active slots、resolution policy、target/min/max valid samples、target FLOPs、global batch、LR segment、growth generation 和允许的下一 transition。
- stage 触发按 consumed valid samples/FLOPs 与成功 optimizer updates计算，不按 wall clock；epoch 只用于可读日志。
- 每个 transition 前后各保留独立 raw checkpoint；失败时回滚到 transition 前，不在损坏状态上继续改变第二个轴。
# 09-D 最终决定（已批准）
采用 `S0→S1→G1→S2→G2→S3` 顺序：先单卡 16 层/256，再四卡同配置验证，然后增长到 20 层并完成 ramp，之后升到 512；第二次增长在 512 内完成，24 层稳定后才考虑 768/1024。增长由预设预算触发并受安全门槛约束，不由 loss plateau 自动触发。
# 09-E：增长验收、恢复与回滚
## A. alpha=0 函数等价性
每次增长完成拓扑构造、权重加载和 FSDP2/optimizer 初始化后，必须先保持新增 slot 的 `alpha_attn=alpha_mlp=0`，不执行参数更新。使用同一输入 batch、latent、text/style condition、timestep、噪声和随机状态，同时运行旧模型与新拓扑。
- reference/eager FP32 路径：新旧 `x_pred`、loss、有效 token reduction 和梯度输入必须逐元素一致；出现任何非有限值直接失败。
- BF16/FlashAttention 生产路径：使用项目已登记的 backend control tolerance 比较；增长引入的误差不得超过同一 backend 在“相同模型重复执行”控制测试的 `2×p99`，且不得超过配置中的绝对上限。不能因为 FA4 的非确定性而跳过等价性测试。
- 检查插入位置前后的旧 slot 输出，确认旧 block 的输入输出没有因为 slot 重排而改变；检查 active slot 顺序和 RoPE/token position 均未改变。
## B. ramp 稳定性
记录每个成功 optimizer update 的 `alpha`、loss、grad_norm、learning rate、nonfinite 标记、吞吐、显存和数据跳过率。
- ramp 期间不得出现 NaN/Inf、optimizer overflow、异常 checkpoint 或未解释的数据读取错误。
- `grad_norm` 和 loss 使用增长前最近 1000 个成功 update 的滑动统计作为参考；单点超过参考 p99 的 3 倍需标记，超过 10 倍或连续 3 个 update 超过 3 倍则停止并回滚。
- ramp 结束后至少执行 09-D 规定的 `max(500, ceil(0.25R))` 个稳定 update；其 loss、吞吐、显存和数据错误率不得相对增长前窗口出现持续性回退。
- 不要求随机训练 loss 每个点单调下降；以滑动中位数、p95 grad_norm 和固定验证集共同判断，避免把正常噪声当成失败。
## C. checkpoint/resume 一致性
在 `alpha=0`、ramp 中点和 `alpha=1` 各保存一次 transition checkpoint，并分别测试：
1. 同一 checkpoint 直接继续运行若干 update；
2. 重新启动进程、重新构造模型/optimizer 后恢复，再运行相同 update；
3. 比较模型、AdamW8bit state、scheduler、growth schedule state、RNG、data cursor 和日志计数。
在 reference 路径要求参数/状态一致；生产路径至少要求 loss、grad_norm、alpha、consumed samples 和下一 checkpoint 的统计在登记容差内一致。恢复后 `k/R`、active slots 和 `growth_seed` 不得变化。
## D. 失败与回滚
- 任一 A–C 验收失败，保留失败诊断包和 `post_growth_pre_update`，回滚到 transition 前 raw checkpoint；不使用 PMA artifact 作为回滚源。
- 回滚后允许继续旧深度训练或暂停排查，不自动重复增长。
- 任何旧参数 missing、optimizer state dropped、slot 顺序变化、数据 cursor 倒退或 scheduler 重置都属于硬失败。
- 验收脚本必须返回机器可读的 pass/fail manifest，记录代码/config/dependency hash；没有 manifest 不得把增长阶段标记为完成。
## E. 最低成本验证预算
每次增长只额外使用固定 transition 验证：一组相同输入的等价性检查、三个 checkpoint 点的 resume 检查，以及 ramp 后的最小稳定窗口。无需训练一条完整的 from-scratch 24 层对照，也不因为一次增长新增大规模消融。
# 09-E 最终决定（已批准）
采用上述 A–E 作为两次深度增长的统一验收协议。组件 09 的架构决定已闭合，待实现测试；当前不改变 AdamW8bit、slot 布局、ramp 公式或 `S0→S1→G1→S2→G2→S3` 阶段顺序。
# 09 最终摘要
- 模型始终固定宽度，仅按 `16→20→24` 增加深度。
- 两次各均匀插入 4 个随机初始化的新 slot；旧 slot 不重编号、不复制、不删除。
- 新 slot 通过固定半余弦 alpha ramp 接入；alpha 不由 optimizer 学习。
- 旧 AdamW8bit state 语义上保留，新 slot state 从零开始；具体 FSDP2/AdamW8bit adapter 留给组件 12。
- 阶段顺序为 `S0→S1→G1→S2→G2→S3`，一次只改变 world size、深度、分辨率、数据混合、optimizer 中的一个主要轴。
- 每次增长须经过 alpha=0 等价性、ramp 稳定性、checkpoint/resume 一致性与可回滚验证。
</content>
</page>
