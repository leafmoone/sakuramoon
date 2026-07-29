Here is the result of "view" for the Page with URL https://app.notion.com/p/3abae967ecf2816da90ccbee372b27c4 as of 2026-07-28T09:50:17.079Z:
<page url="https://app.notion.com/p/3abae967ecf2816da90ccbee372b27c4">
<ancestor-path>
<parent-data-source url="collection://69ca66ff-43e7-4128-bb7b-9f3751506705" name="组件决策记录"/>
<ancestor-2-database url="https://app.notion.com/p/250be554eacc40219065073dfcf66fd7" title="组件决策记录"/>
<ancestor-3-page url="https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff" title="新模型架构决策中心"/>
</ancestor-path>
<properties>
{"date:决定日期:is_datetime":0,"date:决定日期:start":"2026-07-28","url":"https://app.notion.com/p/3abae967ecf2816da90ccbee372b27c4","决策编号":"ARCH-11","序号":10,"影响":"高","标签":["架构","训练","数据"],"状态":"待验证","组件决策":"10 分辨率与宽高比课程"}
</properties>
<content>
<callout icon="✅" color="green_bg">
	**组件 10 架构决定已批准，进入待验证。** 首版固定为人工切换的 256→512 课程、17 个近等 token 宽高比桶和严格不放大的随机 crop；iREPA 关闭，768/1024 仅保留为 benchmark 后手动启用的独立可选阶段。
</callout>
# 组件边界
本组件决定训练分辨率、原生宽高比/裁剪策略、bucket 划分、阶段切换和按计算量分配预算。它不重新决定 Mage-VAE、latent 通道、文本协议、DiT 深度增长、数据 dropout 或 AdamW8bit。
# 已锁定上游接口
- [02 图像表示与 Mage-VAE](https://app.notion.com/p/3aaae967ecf2816096f0ea37634a2f7e)：官方 Mage-VAE、posterior mean、在线编码、无额外 patchify；latent 空间下采样 16×、128 channels。
- [05 多模态序列与位置编码](https://app.notion.com/p/3abae967ecf281de8dafc6dbecd04fe6)：图像 token 数随最终 bucket 的 latent 网格变化；裁剪后按完整目标画布重新生成面积归一化 2D RoPE，crop offset 只用于审计，不进入模型。
- [06 Single-stream DiT 主干](https://app.notion.com/p/3abae967ecf281809823e98b5d91222e)：16→20→24 深度，主干 attention 为 softmax GQA，不能把线性 attention 的分辨率成本假设套用进来。
- [09 模型规模与深度增长](https://app.notion.com/p/3abae967ecf28103be8feea5e27f21e1)：单卡 16 层/256 前置，四卡同配置验证，再增长和升分辨率；一次只改变一个主要轴。
- 目标域是二次元垂类，最终最低要求为 512；1024 不是强制验收阶段。
# 计算量基本关系
对 Mage-VAE 原生 16× 下采样且不额外 patchify：
- 256² → latent 16×16 → 256 image tokens。
- 512² → latent 32×32 → 1024 image tokens。
- 768² → latent 48×48 → 2304 image tokens。
- 1024² → latent 64×64 → 4096 image tokens。
单流 softmax attention 的主要二次项与总序列长度 L=text_tokens+style_tokens+image_tokens 的平方相关；MLP/投影项近似与 L 成正比。因此 512→1024 不是 4 倍总成本的简单关系，但 attention 部分约按图像 token 二次增长。实际预算必须从目标机器 benchmark 得到：记录 samples/s、image tokens/s、peak VRAM、DiT forward/backward time 和总 step time。
不能按“每个分辨率图片数相同”分配预算；否则 1024 图片会消耗远多于 256/512 的 FLOPs，也可能让训练分布被少数高分辨率样本主导。
# 10-A：训练阶段候选
## A. 直接从 512 开始
符合最终最低要求，阶段简单，但无法利用已批准的单卡 256 低成本排错窗口；早期每一步代价更高，不采用。
## B. 256→512，推荐 baseline
- S0 单卡 16 层/256：运行前置训练和长期稳定性检查。
- S1 四卡 16 层/256：只验证 world size 转换。
- G1 在 256、四卡、20 层完成 growth ramp。
- S2 四卡 20 层/512：主训练阶段。
- G2 在 512 阶段增长到 24 层。
- S3 24 层/512：最低质量收尾。
- 768/1024 作为通过 512 验收后的可选质量阶段，不阻塞首版。
优点是与 09 的已批准顺序一致，单卡阶段能提前发现隐藏问题，512 获得主要计算预算；缺点是分辨率变化会带来一次数据/attention 分布切换，需要独立验证。
## C. 256→512→768→1024
适合追求多尺度质量，但后两个阶段 attention 成本高，且项目不强制 1024；若启用必须由实测 FLOPs/质量收益批准，当前不作为 baseline。
## D. 多分辨率从一开始混合
可以减少硬切换，但会增加 sampler/bucket、全局 batch 和 loss 统计复杂度；在完成 256→512 baseline 前不采用。
# 10-A 最终决定
采用 B：256→512 为首版课程，768/1024 保留为后续可选阶段。256 不是只跑极短 smoke test，而是正式的低成本前置训练；512 是最低质量主阶段。阶段顺序继承组件 09：单卡 16 层/256 → 四卡 16 层/256 → 四卡 20 层/256 → 四卡 20 层/512 → 四卡 24 层/512；增长与升分辨率不在同一步发生。
# 10-B：原生宽高比与裁剪
## A. 固定正方形 resize/crop
实现最简单，但会丢失二次元数据集中常见的立绘、横构图和长幅构图，不采用。
## B. 等比 cover resize + bucket 精确裁剪，推荐
1. **一次解码确定尺寸。** JSON width/height 只用于解码前预分桶；图片只正常解码一次，应用 EXIF orientation 后以 tensor 的实际 `H×W` 为准。不得额外调用 header parser 或重复解码。
2. **先筛不放大的 eligible buckets。** 对当前阶段每个候选 bucket 计算 cover scale `s=max(W_b/W, H_b/H)`；只有 `s≤1` 才有资格。若没有任何 eligible bucket，则在当前阶段跳过并按原因计数；该样本仍可在较低分辨率阶段使用。
3. **再按宽高比选桶。** 在 eligible buckets 中，以 `|log((W/H)/(W_b/H_b))|` 最小者作为目标；不得因为宽高比最近的桶需要放大，就忽略另一个本可使用的桶。真实分布扫描后再在 10-C 增加裁剪保留率约束和极端比例处理。
4. **一次等比 resize。** 使用固定的高质量、带 antialias 的重采样实现，将图片等比缩小到同时覆盖 bucket 的最小尺寸；禁止拉伸变形、letterbox/padding 或先缩小再放大。
5. **训练随机裁剪。** 从 resize 后全部合法 offset 中均匀随机采出精确 `H_b×W_b` crop；随机数由 global seed、data pass 与样本 `id` 派生，恢复后可复现。固定验证集使用 manifest 中固化的 crop，不重新采样。
6. **不引入额外视觉增强。** 首版只做 EXIF、resize 和 crop，不加入主体检测/显著性模型、水平翻转、旋转、颜色扰动或随机插值核。主体保留主要通过宽高比匹配和 10-C 的裁剪保留率控制，而不是新增在线模型。
7. **坐标与元数据分离。** 保存源尺寸、实际 resize 尺寸、scale、crop box/offset 和目标 bucket 供审计；模型只接收最终 bucket `H_b/W_b`。2D RoPE 在 crop 后按完整目标画布重新生成，绝不编码源图尺寸、scale 或 offset。
8. **精确尺寸。** bucket 像素宽高均为 16 的倍数，crop 输出必须与 bucket 完全一致，直接形成整数 Mage-VAE latent 网格；不得在 latent 侧补齐或再次 resize。
该路径的 CPU 增量很小：随机 offset 和 crop 相对图片解码/resize 可忽略；不运行主体检测器也避免额外吞吐瓶颈。主要代价是严格不放大会降低高分辨率阶段的 eligible 样本数，因此 10-C 必须扫描并报告每阶段覆盖率。
# 10-B 最终决定
采用 B。关键实现契约为 `实际尺寸 → 筛选全部不放大的 eligible buckets → 选择宽高比最近者 → 等比 cover resize → 均匀随机 crop`；crop 参数只进入审计记录，不进入位置编码或全局尺寸条件。具体 bucket 集合、最低裁剪保留率和极端比例策略由 10-C 决定。
# 10-C：bucket 设计
## 设计约束
- 以下尺寸统一按 `W×H` 书写，像素宽高均为 16 的倍数。
- 同一阶段各 bucket 应尽量拥有相同 image-token 数；这比只要求像素面积落在宽松区间更重要，因为 single-stream softmax attention 对联合序列长度敏感。
- 256→512 的每个形状应严格放大 2 倍；可选 512→768 严格放大 1.5 倍，使课程切换只提高采样密度，不同时改变宽高比支持。
- 每个 bucket 记录 `pixel_h/pixel_w`、`latent_h/latent_w`、`image_tokens`、`aspect_ratio` 和相对方形 token 比例。
- bucket 只约束图像空间形状；文本长度桶是独立维度。
## A. 旧 5 桶草案
旧草案的宽高比位置合理，但矩形桶比方形多 9.4%–12.5% image tokens。按当前平均约 201 个 text+style tokens 估算，最极端桶相对方形会使联合 attention score 计算在 256 阶段增加约 14%，在 512 阶段增加约 22%；不采用这些具体尺寸。
## B. 以 512² 等效面积规则生成完整桶集合，推荐
不再手工挑选少数宽高比。先在 512 阶段定义统一生成规则：
- 目标面积 `A₀=512×512`；基准尺寸步长确定为 `q=32 px`。
- 短边 `S` 依次取 `256, 288, 320, …, 512`。
- 长边取 `L=q×round((A₀/S)/q)`，即把保持目标面积所需的长边舍入到最近的 32 px。
- 对每个 `S×L` 同时加入转置的 `L×S`，正方形只保留一次。
- 基准最大 bucket 宽高比确定为 `4:1`；不生成更极端的输出 shape，所有尺寸仍是 16 的倍数。
该规则在 512 阶段生成 17 个桶：
- `256×1024`、`1024×256`
- `288×896`、`896×288`
- `320×832`、`832×320`
- `352×736`、`736×352`
- `384×672`、`672×384`
- `416×640`、`640×416`
- `448×576`、`576×448`
- `480×544`、`544×480`
- `512×512`
这些桶拥有 1008–1040 image tokens，相对 1024 的误差仅为 `-1.56%～+1.56%`。相邻宽高比中心之间的最坏保留率约为 88.2%，明显高于 7 桶方案，同时仍只有 17 个有限 shape。
## 跨阶段映射
bucket 列表只在 512 基准上生成一次：
- 256 阶段使用每条边乘 `0.5`，得到 17 个 `128×512` 到 `512×128` 的等效 256² 桶；基准步长自动变为 16 px。
- 512 阶段直接使用基准列表。
- 可选 768 阶段每条边乘 `1.5`，基准步长为 48 px。
- 若启用 1024 阶段，每条边乘 `2`，基准步长为 64 px。
所有阶段因此拥有完全相同的宽高比中心，只改变采样密度和 image-token 预算。
## C. 步长比较与决定
- `q=64` 只产生 9 个桶，宽高比中点的最坏保留率约 80.6%，仍容易接近 20% 裁剪上限。
- 采用 `q=32`：产生 17 个桶，最坏保留率约 88.2%，计算量误差仅约 ±1.6%。
- `q=16` 会产生 33 个桶，裁剪更少但增加编译 shape、bucket 队列和 text/image shape 组合；收益相对 17 桶有限，不采用。
## 裁剪保留率
定义忽略整数取整后的保留率：
`retention = min((W/H)/(W_b/H_b), (W_b/H_b)/(W/H))`。
对宽高比位于 `0.25～4` 且最近 bucket eligible 的原图，17 桶的理论最坏保留率约为 88.2%，因此不会触及 80%。`retention≥0.80` 只作为安全阈值：一是处理原图比例超出 bucket 中心范围的长尾，二是最近 bucket 因严格不放大而不可用、只能考虑更远 eligible bucket 的情况。
以横图为例，最外侧 bucket 为 `4:1`：原图 `4:1` 的 retention 为 100%，`5:1` 恰为 80%，`6:1` 只有 `4/6=66.7%`。因此原图可在不新增输出 bucket 的情况下覆盖到 `5:1`；超过该临界点，或任何 eligibility 回退后低于 80% 的样本，不强行裁剪、不补边，按 `aspect_retention_reject` 跳过。整数 resize 取整只会造成很小误差，不能把正常范围内约 88.2% 的最坏值降到 80% 以下。
## 分布与扫描规则
- 不对 bucket 做均匀重采样或稀有比例过采样；每个 data pass 保持数据集自然宽高比分布，每个有效样本最多消费一次。
- 正式训练前用 metadata 做全量分配扫描，并在解码 dry run 中用实际 EXIF 后尺寸复核。分别报告每阶段 `no_upscale_reject`、`aspect_retention_reject`、每桶样本数和 retention 分位数。
- 扫描可以删除样本数为零的外侧 bucket，但不得根据频率合并中间 bucket；这样不会重新增加相邻比例间的裁剪损失。
- 最大输出 bucket 已固定为 `1:4～4:1`，扫描不再自动扩展 shape。超过 `1:5～5:1` 或 eligibility 回退后低于 80% 的长尾只统计并跳过；若拒绝占比异常，再单独回到架构决策，而不是由 loader 自动加桶。
- 17 个 shape 仍需在目标机记录首次编译时间、稳态吞吐和 bucket queue 等待；若 shape 调度成为实测瓶颈，再回退到 9 桶，而不是预先牺牲构图保留率。
# 10-C 最终决定
采用 `A₀=512²` 等效面积、`q=32`、512 基准短边下限 256、最大 bucket 比例 `4:1` 和转置闭包，共生成 17 个近等 image-token buckets。`retention≥0.80` 仅作为极端原图和 eligibility 回退的拒绝阈值；超过阈值不补边、不拉伸、不自动增加更极端 bucket。数据保持自然宽高比分布，不按桶重采样。
# 10-D：预算与阶段切换
## A. 等 epoch 或等样本预算
不同深度和分辨率的单样本计算量差异很大。给各阶段相同 epoch 会让后期 512 阶段消耗远多于早期，也无法表示 growth ramp 的真实成本；不推荐作为主预算单位。
## B. 只按 optimizer updates
updates 会随 global batch、microbatch 和梯度累积变化；不同 bucket/分辨率下同样 update 数代表的数据暴露量也不同。它适合定义 ramp 和稳定窗口，但不适合单独控制完整课程。
## C. 数据暴露 + DiT FLOPs 双门槛，推荐
每个 stage 同时配置并记录：
- `valid_samples` 与 `effective_data_passes = valid_samples / 当前阶段 eligible 样本数`，表示数据暴露。
- `consumed_image_tokens` 与按实际联合序列长度、active depth 估计的 `dit_train_flops`，表示主干计算。
- `successful_optimizer_updates`，用于 LR schedule、growth ramp、稳定窗口和 checkpoint 频率。
- `wall_time` 只用于成本预测，不触发 transition。
阶段必须同时达到配置的最小数据暴露和目标 FLOPs/updates，并通过组件 09 的稳定性门槛，才允许切换；loss plateau 只用于人工诊断，不自动提前或无限推迟阶段。
## 当前相对计算量估算
按平均约 201 个有效 text+style tokens、方形等效 image tokens 和组件 06 的固定宽度粗估，仅比较 DiT 主干：
- `16层/256`：1.00×。
- `20层/256`：约 1.25×/sample。
- `20层/512`：约 3.54×/sample。
- `24层/512`：约 4.25×/sample。
17 个图像 buckets 的 token 数只波动约 ±1.6%，不会显著改变上述结论。该估算不把在线 Qwen、VAE、通信和数据等待算入，最终预算必须用 1/4 卡 RTX 5090 benchmark 校正。
## 推荐阶段预算原则
- **S0 单卡 16层/256：** 保留用户已批准的“前几个有效 data passes”方向，承担从零训练的低成本排错和初始表示学习；精确 pass 数在单卡吞吐实测后写入配置，不能仅按日历时间结束。
- **S1 四卡 16层/256：** 继承组件 09，只运行 200–1000 个成功 updates，验证 world size、global batch、shard 恢复和吞吐；不重复一轮完整 S0。
- **G1 四卡 20层/256：** growth ramp `R` 及 post-ramp 窗口完全计入该阶段正常预算，不另加 epoch。
- **S2 四卡 20层/512：** 分配首版最大份额的训练 FLOPs，形成主要 512 表示。
- **G2 四卡 24层/512：** ramp 与 post-ramp 窗口计入 512 总预算，不因增长重复计算数据 pass。
- **S3 四卡 24层/512：** 在完整深度下完成最低成品质量收尾；768/1024 不计入首版承诺。
## 人工阶段切换与 data pass
所有阶段切换都由用户显式执行，不实现自动 transition controller。训练程序达到当前 stage 的预算时只执行以下动作：
1. 设置 `stage_ready=true` 并输出固定集、稳定性、吞吐和预算报告；不得自动停止或切换，除非当前 stage 配置显式设置了 `stop_at_budget=true`。
2. 用户决定切换后，当前作业在 optimizer-step 边界进入 drain：停止向 shuffle buffer 发布新 shard，完成各 worker 已经活跃的 shard。
3. 保存带 `COMPLETE` 标记的最终 raw checkpoint、`trainer_state.json`、`data_state.json` 和 `stage_report.json`，再正常退出。
4. 用户使用目标 stage 配置和明确的源 checkpoint 路径启动新作业；新 stage 以新的 `stage_id/pass_index/seed` 从完整 manifest 开始确定性 shuffle。
旧 pass 尚未遍历的 shard 记录为计划内 `stage_cut`，不伪装为已消费；跨 stage 再次看到先前样本属于新阶段的预期数据暴露，不计入异常恢复重放。如果预算恰在 pass 尾部达到，可以自然完成整个 pass 后退出。
## 预备配置文件布局
实现时预先提供一份共享配置与六份 stage overlay：
- `configs/train/base.yaml`：模型公共接口、数据 schema、caption/dropout 引用、optimizer 类型、checkpoint 格式和日志字段。
- `configs/train/stages/s0_16l_256_1gpu.yaml`
- `configs/train/stages/s1_16l_256_4gpu.yaml`
- `configs/train/stages/g1_20l_256_4gpu.yaml`
- `configs/train/stages/s2_20l_512_4gpu.yaml`
- `configs/train/stages/g2_24l_512_4gpu.yaml`
- `configs/train/stages/s3_24l_512_4gpu.yaml`
每份 overlay 至少显式包含：
- `stage_id`、`world_size`、`active_slot_ids/depth`、目标分辨率、bucket 生成参数与 `retention_min`。
- checkpoint 输入语义（from-scratch/resume/transition）、允许的唯一前序 stage 和是否执行 growth migration。
- global/local microbatch、gradient accumulation、activation checkpoint 和 varlen/dense 路径。
- LR segment、目标/最小 valid samples、effective passes、FLOPs、updates、checkpoint/validation 间隔。
- stage seed、ModelScope manifest revision、eligible 样本统计引用和 `stop_at_budget`。
目标机 benchmark 前无法确定的数值使用 schema 级 `REQUIRED_AFTER_BENCHMARK`/等价显式占位；启动器遇到占位必须失败，不得回落到代码默认值。配置经解析、继承和环境覆盖后的完整 resolved config、schema version 与 hash 保存到每个 checkpoint。
## 切换校验
手工不等于绕过约束。启动新 stage 前必须由 preflight 校验：
- 源 checkpoint `stage_id` 必须是目标配置允许的前序；禁止跳过 S1/G1/G2 或同时改变两个主要轴。
- config hash、model slot、Qwen/VAE、tokenizer、bucket 规则、optimizer state 和 manifest revision 的变更必须符合目标 transition allowlist。
- G1/G2 必须执行组件 09 的 growth migration、alpha=0 等价性和 post-growth checkpoint；普通 resume 不得误触发 growth。
- 检查未通过时拒绝启动，不做自动修复或隐式状态重置。
# 10-D 最终决定
采用数据暴露和实际 DiT FLOPs 双门槛，updates 负责局部调度；所有 `S0→S1→G1→S2→G2→S3` 切换均由用户手动执行。实现预先提供 base + 六份 stage overlays、drain/finalize 命令和 transition preflight；训练程序永不自动改变 world size、模型深度或分辨率。具体 S0/S2/S3 的 pass/FLOPs 数值在目标机 benchmark 后填写。
# 10-E：iREPA 与高分辨率扩展
## iREPA
首版不启用 iREPA，也不为它加入视觉教师、额外特征缓存、辅助 projection head 或额外 loss。原因是当前课程已经包含从 256 到 512、两次深度增长和在线 VAE/Qwen；再加入视觉表征对齐会增加模型之外的依赖、显存和 loss 权重调试，偏离“有限算力、不做大规模消融”的约束。
配置中必须显式保存 `irepa.enabled=false`；训练启动时不得因为依赖可用而自动开启。未来若重新评估，必须作为独立架构变更，明确教师模型、作用阶段、loss 权重、额外 FLOPs/显存、checkpoint 兼容性和固定集收益，不能直接继承本组件的批准。
## 可选 H1：24层/768
- 不属于首版最低验收，不计入 S0～S3 的硬性预算。
- 提前提供 `configs/train/stages/h1_24l_768_4gpu.yaml` 模板，但默认 `enabled=false`，预算、microbatch、累积步数和 activation checkpoint 保留强制 benchmark 占位。
- 只允许从已接受的 S3 raw checkpoint 手动启动；模型深度、optimizer 类型、数据协议和 bucket 比例中心不变，只将 512 基准桶各边乘 1.5。
## 可选 H2：24层/1024
- 提前提供 `configs/train/stages/h2_24l_1024_4gpu.yaml` 模板，默认 `enabled=false`。
- 1024 不是项目承诺；优先从已接受的 H1 checkpoint 进入。若跳过 H1 并从 S3 直接升到 1024，必须单独生成 transition benchmark/批准记录，不能只修改数值后启动。
- bucket 各边相对 512 基准乘 2；必须重新确定 microbatch、累积步数、checkpointing、吞吐和 eligible 子集。
## 手动启用门槛
H1/H2 只有在以下条件全部满足后才可由用户手动启用：
1. 前序阶段存在完整、可恢复且固定集质量已接受的 raw checkpoint。
2. 目标分辨率的 eligible 样本数、`no_upscale_reject`、`aspect_retention_reject` 和 17 桶占用已扫描；固定验证集使用无需放大的 eligible 子集，并单独报告大小。
3. 四卡 RTX 5090 实测不存在 OOM、host swap 或持续数据等待，并已经填写 global/local batch、gradient accumulation、activation checkpoint、samples/s、image tokens/s 与峰值显存。
4. 质量收益需同时体现在目标分辨率固定集和生成样本上；不得只因为训练 loss 下降就进入下一阶段。
5. 切换仍执行 10-D 的人工 drain/finalize/preflight，不自动连跑 H1/H2。
# 10-E 最终决定
首版固定 `irepa.enabled=false`。768 和 1024 只提供默认禁用的 H1/H2 配置模板，在 512 成品稳定、数据覆盖与目标机 benchmark 通过后由用户手动启动；未启用时不影响模型结构、checkpoint 或首版验收。
# 10 最终摘要
- 首版课程为 `S0→S1→G1→S2→G2→S3`，从 16层/256 发展到 24层/512；所有 transition 人工执行。
- 图片仅做 EXIF、一次等比缩小和可复现均匀随机 crop；严格不放大、不拉伸、不补边，crop offset 不进入模型。
- 512 基准按 `A₀=512²、q=32、最大4:1` 生成 17 个近等 image-token buckets；其他分辨率按比例缩放同一形状集合。
- 裁剪保留率最低 80%；极端长图或 eligibility 回退低于阈值时跳过并计数。
- 预算采用数据暴露与实际 DiT FLOPs 双门槛，updates 负责局部调度；具体数值在 benchmark 后写入显式配置。
- 首版关闭 iREPA；768/1024 是默认禁用、人工审批的可选阶段。
# 实施前验证项
1. 对完整 metadata 执行 256/512 bucket 分配扫描，并用解码 dry run 核对 EXIF 后实际尺寸；输出每桶数量、eligible 数、两类 reject 和 retention 分位数。
2. 在单卡和四卡 RTX 5090 上完成每阶段 microbatch、gradient accumulation、activation checkpoint、varlen/dense、吞吐与显存 benchmark，填写所有 `REQUIRED_AFTER_BENCHMARK` 字段。
3. 实现 base、S0～S3 overlays、默认禁用的 H1/H2 模板、resolved-config hash、stage finalize 和 transition preflight。
4. 执行 crop 可复现性、stage-cut/data-state、checkpoint resume、bucket queue 等待和 17-shape 编译缓存测试。
5. 只有实际考虑 H1/H2 时才评估 768/1024 质量收益；该评估不阻塞首版 512 实现。
<empty-block/>
</content>
</page>
