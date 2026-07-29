Here is the result of "view" for the Page with URL https://app.notion.com/p/3aaae967ecf281db800cfb1d6545f880 as of 2026-07-29T06:39:49.999Z:
<page url="https://app.notion.com/p/3aaae967ecf281db800cfb1d6545f880">
<ancestor-path>
<parent-data-source url="collection://69ca66ff-43e7-4128-bb7b-9f3751506705" name="组件决策记录"/>
<ancestor-2-database url="https://app.notion.com/p/250be554eacc40219065073dfcf66fd7" title="组件决策记录"/>
<ancestor-3-page url="https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff" title="新模型架构决策中心"/>
</ancestor-path>
<properties>
{"date:决定日期:is_datetime":0,"date:决定日期:start":"2026-07-27","url":"https://app.notion.com/p/3aaae967ecf281db800cfb1d6545f880","决策编号":"ARCH-3","序号":11,"影响":"高","标签":["数据","训练","系统"],"状态":"已接受","组件决策":"11 数据与缓存管线"}
</properties>
<content>
<callout icon="✅" color="green_bg">
	**已接受。** 组件 11 的数据语义、远端缓存、采样恢复、验证隔离、动态 caption、token 预算和在线生产队列均已闭合；实机相关容量与并发值按本文基准写入最终训练配置。
</callout>
# 背景
训练数据约 11M 张二次元图像，已通过 `/root/shared-nvme/ssh_help/repack_full_direct.py` 打包为 WebDataset。每个样本包含原图和同 key JSON。该决定提前讨论原顺序中的组件 11，不改变其他组件编号。
相关决定：<mention-page url="https://app.notion.com/p/3aaae967ecf281ba8f73fac2f9e4c4f3"/>、<mention-page url="https://app.notion.com/p/3aaae967ecf2816096f0ea37634a2f7e"/>、<mention-page url="https://app.notion.com/p/3aaae967ecf281fba3cfe0f5dc53fece"/>。
# 数据范围
每个 JSON 使用 `schema_version: 1`，训练所需字段为：
- 四类 tags：`general`、`artist`、`character`、`copyright`。
- `nsfw`，在最终 caption 中按 tag 处理。
- 五个等权 NL 候选：`long_names`、`long_no_names`、`short + vibes`、`nl2`、`nl3`。
- `dropout.candidate_tags` 及其 `candidate_source` 元数据。
- 图像 `id`、格式、宽高、release、年份和 join 状态继续保留。
数据库中未打包的 `aesthetics`、`fav_count`、`rating`、`meta`、`special`、其他 caption 和统计字段不补传，不建立 sidecar。
# Caption 采样决定
1. 首先执行整体条件 dropout，概率固定为 `0.10`。命中时丢弃全部 tags 与 NL，生成 CFG 无条件样本。
2. 未命中整体 dropout 时，tag 条件池由四类 tags 与 `nsfw` 组成。各类别拥有独立、配置化的 dropout 概率。
3. `dropout.candidate_tags` 不是第五类条件，而是四类 tags 上的删除掩码。`candidate_source` 使用单独的 candidate dropout 概率；命中时从四类 tag 列表中删除所有 canonical form 出现在 `candidate_tags` 的词，不追加 candidate 副本，也不扫描或改写 NL。
4. NL 以五个候选分支处理，其中 `short + vibes` 是一个分支。只在当前样本非空的分支中等概率抽取，并且每个样本最多出现一种 NL。
5. 五个 NL 分支在 dropout 字典中逐项列出，但配置值必须相同。
6. 在不触发任何 component dropout 时，最终条件由全部四类 tags、`nsfw` 和随机选中的一种 NL 组成。
7. 原始条件为空，或 component dropout 后全部条件为空时，不跳过样本，直接作为无条件样本训练。因此最终无条件比例允许高于显式整体 dropout 的 `0.10`。
8. WebDataset 中保留原始下划线 tag；仅在最终训练 caption 渲染时把 `_` 替换为空格。
9. caption 完成采样、dropout、归一化后形成 `caption_body`：所有非空 tags 用 `, ` 连接，NL 若存在则以双换行 `\n\n` 作为段落边界追加在 tags 之后，不加入字段或类别标签。仅 tags、仅 NL 与空 body 不添加多余空行。最后应用组件 03 已接受的 Krea 2 式手工 chat framing；数据层不自行追加 EOS/EOT 或角色 token。
# 配置契约
训练配置必须使用一个显式 dropout 字典列出所有条件源，不允许依靠代码默认值或合并掉字段：
- 整体：`all_condition`，固定为 `0.10`。
- Tags：`general`、`artist`、`character`、`copyright`、`nsfw`。
- Candidate：`candidate_source`，实际控制 JSON 中 `candidate_tags` 删除掩码；命中时跨 `general`、`artist`、`character`、`copyright` 删除所有匹配词，`candidate_source` 与 `candidate_tags` 本身均不作为新文本追加。
- NL：`long_names`、`long_no_names`、`short_vibes`、`nl2`、`nl3`。
五个 NL key 必须逐项存在且取相同值。除 `all_condition` 外的具体数值延后到训练配置定稿时填写。任一 key 缺失、出现未知 key、概率越界或 NL 值不一致时必须在启动阶段失败。
# Candidate 掩码与 Tag Shuffle 决定
- 当前打包逻辑先用 `character` tags 查询 character map 并合并对应 `popular_tags`，再与四类实际 tags 的规范化并集取交集：`candidate_tags = popular_tags(character records) ∩ (general ∪ artist ∪ character ∪ copyright)`。
- 因此 candidate 候选的生成入口来自 character records，但结果在数学上是四类 tags 并集的子集，并不保证只属于 `character` 类。
- `candidate_tags` 是为训练期 group dropout 留存的元数据，不是 caption 的第五类 tag。在 candidate dropout 未命中时，各词保持在原始类别；命中时从四类原始类别中一次性删除全部匹配词。
- candidate dropout 与四类 tag dropout 独立抽样。一个 candidate 词只保留原始那一份，不重复追加，也不对同一词建立 candidate/character 双份表示。
- 取消 tag 类别块之间的 shuffle。普通 caption 使用固定类别顺序 `nsfw → artist → character → copyright → general → NL`；若组件 04 后续批准把 artist 移入独立 style condition，则主 caption 改为 `nsfw → character → copyright → general → NL`，artist 不再参与普通 tag 序列。
- 只在每个类别块内部 shuffle tags；`nsfw` 通常为单值，NL 不参与任何 shuffle，始终完整追加在全部 tag 块之后。`candidate_tags` 仍只是跨四类原始 tags 的删除掩码，不构成可排序的独立类别块。
- 类别内 shuffle seed 由 global seed、data pass 和 `id` 派生，跨 pass 变化且可复现；验证集使用固定 seed，保证类别内顺序恒定。固定类别骨架用于降低冻结因果 Qwen 对同一样本产生的跨类别表示噪声。
- candidate 匹配使用与打包脚本一致的 canonical form：trim、lowercase，并把下划线与空格视为等价；原始 JSON 不变，最终 caption 渲染时才移除下划线。
- 数据 schema 保证 `general`、`artist`、`character`、`copyright` 四类实际 tags 互斥，不存在同一个 canonical tag 跨类别重复；loader 不增加跨类别归属优先级或二次 tag 去重。`candidate_tags` 与四类原始 tags 的重叠是预期的删除掩码关系，不属于类别重复。
# 过滤与去重决定
- 不在 loader 或训练前索引阶段再次去重；信任当前数据制作流程已经完成去重。
- 图片损坏、无法解码、格式不支持、尺寸无效，或因组件 02 的“不放大”约束而无法覆盖当前 bucket 时跳过，并按原因计数。
- 所有条件为空时不跳过，作为无条件训练样本。
- `nsfw` 不用于过滤，作为普通 tag 进入 caption，并在 dropout 字典中显式列出。
# 采样与 Shuffle 决定
- 三个 release 合并为一个样本池，默认不设置来源权重。
- 一个 data pass 中每个 shard 只读取一次，不使用有放回 shard 抽样。
- 每个 pass 使用由全局 seed 与 pass index 派生的 seed，对完整 shard 清单进行确定性 shuffle。
- shuffle 后按 manifest 的样本数均衡分配到 rank 和 DataLoader worker；同一 pass 内各 rank/worker 的 shard 不重叠。
- shard 内使用在线 sample shuffle buffer；具体容量在内存和吞吐测试后写入配置。
- 训练阶段按 optimizer steps 定义，不要求与 data pass 边界对齐；pass 用尽后以新 seed 开始下一轮。
- 生成轻量 manifest，至少记录 `repo_id`、不可变 `revision`、`path`、`release`、`bytes`、`sha256`、`samples`，用于均衡分配、进度、完整性检查与恢复。
- 不采用 uniform-with-replacement shard sampling，因为 2 GiB shard 的图片数不同，会产生隐式样本和 release 权重。
# ModelScope 读取与本地缓存决定
- 唯一远端数据源为 ModelScope dataset repo `leafmoone/webdataset_danbooru`；无需另设 NFS 或完整本地副本。
- 每次训练必须在 manifest 中固定不可变 tag 或 commit revision；训练中不得跟随可变化的 `master`。
- 强制使用官方 [`modelscope-hub`](https://github.com/modelscope/modelscope_hub) 下载接口，利用其 HTTP Range 续传、重试、SHA256 校验、原子合并和多进程文件锁；测试通过的精确版本写入环境锁文件。
- 训练进程按 manifest 顺序异步下载完整 shard 到本地缓存，完成大小与 SHA256 校验后才向 WebDataset reader 发布；不把 tar reader 直接连接到长时间 HTTP 响应。
- 本地 SSD/NVMe 缓存额度配置为 `300–500 GiB`。按当前约 2 GiB/shard 计算，可滚动容纳约 150–250 个 shard，无需保存整个数据集。
- 缓存使用 LRU 淘汰；当前正在读取、已经分配及下载中的 shard 均受保护。设置高低水位，避免每次写入都触发抖动，具体水位写入训练配置。
- 每个 rank 至少预取下一个已分配 shard；全机下载并发数与单文件 Range worker 数通过目标机器冷缓存测试确定，避免四个 rank 各自无界并发。
- 单张图片解码失败按既定过滤策略跳过并计数；网络或完整 shard 下载失败执行有界指数退避重试，多次失败后停止训练，不允许静默跳过整个 shard。
- ModelScope token 仅通过环境或凭据文件注入，不写入 manifest、日志、checkpoint 或实验配置。
- 只有冷缓存吞吐不达标或预取队列反复耗尽时，才扩大本地缓存或增加独立数据镜像；300–500 GiB 是当前接受容量。
# Shard 级恢复与 Checkpoint 分层决定
- 采用 shard 级 `at-least-once` 恢复：不追求严格逐样本恢复，不允许静默遗漏，允许异常恢复后少量样本重复。
- checkpoint 只在 optimizer step 边界提交。恢复时跳过该 checkpoint 已确认完整消费的 shard；当时尚未完成的 shard 从头重放。
- 每个 worker 同时只读取一个活跃 shard；下一个 shard 可以预下载到本地，但不得提前进入 sample shuffle buffer。这样重放上限约为每个 worker 一个 shard。
- `data_state.json` 至少保存 schema version、checkpoint ID、global step、stage、data pass index、shard 排列 seed、rank/worker 分配、完整消费 shard 集合、活跃 shard、处理/跳过/重放计数及数据拓扑。
- 只有样本已经越过训练消费边界、且该 shard 的全部样本都已消费或按规则跳过后，才能把 shard 标记为完整；下载完成或进入预取队列不等于完整消费。
- 当前 data pass 的恢复要求 `world_size` 和每 rank worker 数与 checkpoint 一致。拓扑变化时不得伪装为原位恢复，必须明确结束旧 pass 并以新 seed 开启新 pass，同时记录不连续事件。
- 不序列化 DataLoader 预取队列与 sample shuffle buffer；异常恢复时丢弃这些易失状态，并通过重放未完成 shard 重建数据流。
- 单独记录 `replayed_shards` 与 `replayed_samples`，用于监控重复量；严格逐样本恢复不作为第一版目标。
## 文件分层
- 模型产物目录只包含模型配置、训练得到的模型权重和模型相关组件；采用可独立加载的安全张量格式，能够脱离续训状态用于推理或发布。
- `trainer_state.json` 保存 global step、阶段、累计样本/FLOPs、配置与代码版本引用等普通训练元数据。
- `data_state.json` 单独保存上述 shard 恢复状态，不写入模型权重文件。
- optimizer、混合精度 scaler 和其他张量状态使用独立二进制或分片张量文件；RNG 状态使用独立可无损恢复的状态文件。它们不能为追求统一而强行编码进 JSON。
- 所有文件放在同一个带 checkpoint ID 的目录中并共同提交：先写临时目录、生成完整性清单，最后写 `COMPLETE` 标记并原子发布。恢复器只接受最新的完整 checkpoint，忽略半写入目录。
- 模型产物与续训 sidecar 独立加载但必须共享 checkpoint ID，禁止把不同 step 的模型、optimizer 与 data state 混用；具体分片格式由组件 12 的并行策略确定。
# 固定验证集与训练隔离决定
- 当前 WebDataset 的 JSON `id` 已保证在三个 release 间全局唯一，验证隔离直接使用整数 `id`；tar member 的物理 key 虽包含 release，但不作为逻辑隔离主键。
- 从全部数据制作和既有去重完成后的最终样本池中，以固定 seed 确定性抽取 2,000 张；不再次进行精确、感知或近重复去重，信任上游去重结果。
- 抽样按 `release × 宽高比分桶 × caption 可用情况` 分层，使验证集覆盖数据来源、构图比例及条件覆盖差异；具体配额由最终 manifest 的分布按比例计算。
- 生成不可变的 `validation_manifest.jsonl` 与 `validation_ids`，至少保存 `id`、原始 shard、release、图像格式与尺寸、512 bucket、固定 NL 分支及预处理 seed，并记录生成脚本版本和内容哈希。
- 所有训练阶段在样本进入 shuffle buffer 前按 `validation_ids` 排除；原训练 shard 无需重打包，任何验证 `id` 都不得形成训练 batch。
- 将相同的 2,000 个样本物理复制为少量独立 validation shard，避免验证时为零散样本下载大量 2 GiB 训练 shard。物理副本不改变隔离语义，训练 loader 不读取 validation 路径。
- 验证 caption 关闭整体与各组件 dropout；NL 分支、tag 渲染、resize/crop、bucket 和其他随机预处理全部由 manifest 固定，重复验证必须逐样本一致。
- 2,000 张均须满足不放大的 512 主验证要求。更高分辨率阶段只使用其中无需放大的固定 eligible 子集，并单独报告子集大小，不临时替换样本。
- 后续追加数据时只检查新样本 `id` 是否命中 `validation_ids` 并继续排除；不因此启动第二轮去重。
# 尺寸来源与单次解码决定
- JSON 中的 `image.width` 与 `image.height` 仅用于解码前的快速预分桶和队列调度，不作为最终图像几何真值。
- 不安装或调用 `imagesize`；训练本来就必须解码图像，额外的独立文件头扫描会重复解析而没有实际收益。
- 每个样本只执行一次正常图像解码。应用 EXIF 方向后，直接读取解码 tensor 的 `H × W` 作为最终实际宽高。
- JSON 与实际宽高一致时沿用预分桶结果；不一致时以实际宽高重新计算 bucket 并路由，同时增加 `dimension_mismatch` 计数。
- 不放大资格、resize/crop、裁剪偏移及位置编码所需坐标全部使用应用 EXIF 后的实际宽高。
- JSON 尺寸缺失但图片可正常解码时允许使用实际宽高；实际宽高无效、格式不支持或解码失败时按既定过滤策略跳过。
- 不修改 WebDataset 中的原始 JSON，日志分别报告缺失、宽高互换、普通不一致和解码失败。
# Token 预算预扫描与长文本决定
- 正式训练前，使用组件 03 锁定的 Qwen tokenizer、`, ` tag 分隔符、双换行 `\n\n` NL 段落边界和 Krea 2 式手工 chat 模板，对全部约 11M 条条件记录或其等价 metadata-only stream 做一次 token 预算扫描；不需要解码图片。
- 扫描至少分别报告 tags-only、五个 NL 分支、每个样本最长 NL 分支及完整无 dropout 条件的 token 分布，包括 P50、P90、P95、P99、P99.5、P99.9、最大值和候选预算下的溢出率。
- 同时用正式 dropout 配置生成代表实际训练分布的确定性视图，但 token 上限选择不能只依据 dropout 后的短文本；无 dropout 视图用于评估上界。
- token 上限不在组件 11 预先写死，由组件 03 根据扫描覆盖率及 4×32 GB 机器上的显存/吞吐基准确定；目标是在资源允许范围内尽可能降低截断率。
- 最终 caption body 渲染完成后才用真实 tokenizer 计算预算，不按字符数、词数或经验比例估算；预算必须为固定 assistant suffix 保留位置，并单独报告 system/user prefix 带来的 Qwen 计算长度。
- 溢出时优先保留全部 tags，再缩短或移除末尾 NL。若 tags 本身仍超长，则按当前确定性 shuffle 后的顺序逐个装入完整 tag，不从某个 tag 的 token 中间截断。
- 因类别内 tag 顺序跨 data pass 改变，极端超长样本在不同 pass 可保留不同 tag 子集；类别骨架固定，验证集的类别内顺序也固定，因此验证截断结果固定。
- 单个 tag 自身超过全部可用预算时删除该 tag 并记录 `oversized_tag`；不因此跳过图像。
- 训练持续记录 `caption_overflow`、`nl_truncated`、`tags_truncated`、`oversized_tag` 和最终 token 长度直方图，以检查真实截断率是否与预扫描一致。
# 在线生产者/消费者队列决定
- 每台四卡机器只运行一个 ModelScope 下载与本地 cache 协调器，统一维护 ready shard、文件锁、LRU 和下载并发，四个训练 rank 不各自建立无界下载器。
- 初始 DataLoader 配置为每 GPU 2 个持久 worker，共 8 个 worker；在 14 vCPU 上为四个训练主进程、下载器和系统线程保留余量。
- CPU worker 负责 JSON 解析、验证集排除、dropout、candidate 掩码、tag/NL 拼装、tokenizer、单次图像解码、EXIF、resize/crop 和 bucket 路由。
- tokenizer 与图像库在 worker 内禁止再次无界创建线程，避免 8 个进程叠加线程池导致 CPU 过度订阅。
- 每 rank 的 CPU ready queue 初始上限为 2 个完整 batch；等待传输的图像保持 `uint8`，token IDs 使用必要的最小整数表示，到 GPU 后再转换目标 dtype。所有队列必须有界并提供 backpressure。
- 每个 GPU rank 各自加载冻结 Qwen 与官方 Mage-VAE，使用 `eval()`、`inference_mode()` 和 BF16；不得为冻结模块构建梯度或保留反向激活。
- 每个 batch 即时执行 Qwen 文本编码与 Mage-VAE posterior mean，输出随即交给 DiT；不建立跨 batch 的 text embedding、latent 或 encoder activation 队列。
- Qwen、Mage-VAE 与 DiT 默认在同一 GPU 上串行执行，避免多个 CUDA compute stream 抢占算力并抬高峰值显存；仅允许 CPU 准备、下一批 shard 下载和 non-blocking H2D 与当前 GPU 计算重叠。
- Qwen 完成后释放 tokenizer/GPU 临时输入，VAE 完成后立即释放原图 tensor；只保留当前 microbatch 训练所需的 text states 与 posterior mean latent。
- 分别采集 ModelScope/cache wait、tar read、decode、tokenize、bucket wait、H2D、Qwen、VAE、DiT、ready queue depth、host RAM、pinned RAM 和 GPU peak memory 指标。
- `2 workers/rank + 2 ready batches/rank` 是接受的初始值，不是未经测量的永久常量；正式训练前比较 1/2/3 workers 与多个有界 queue depth，选择满足吞吐且内存稳定的最小配置。
# 随机性与在线处理
- 随机状态由全局 seed、epoch 和 sample key 派生，保证恢复训练后可复现。
- tag dropout、candidate dropout、NL 分支选择和 NL dropout 均在线执行。
- 因为每个 epoch 可形成不同 caption 视图，当前不缓存文本 embedding。
- 图像仍按组件 02 决定在线 resize/crop 和 Mage-VAE 编码。
# 强制系统依赖
`causal_conv1d` 与 `fla` 改为强制安装项。训练启动时必须验证模块可导入且 Qwen DeltaNet 使用快速 kernel；不允许静默回退到更慢、更占显存的 PyTorch 路径。
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
# 理由
- 现有 WebDataset 已包含目标训练真正需要的 tag 与 NL 多视图，重传额外标注没有直接收益。
- 在线采样允许同一图像跨 epoch 获得不同条件组合，符合 tags 优先、NL 辅助的目标。
- 最终渲染时去下划线既保留原始标注可追溯性，也使 Qwen 接收更自然的文本形式。
- 将概率全部配置化，能在不改数据和代码的情况下调整条件强度。
# 后果
- 不使用 aesthetic 字段做过滤、加权或额外条件。
- 全局 NL 来源比例会受到字段实际覆盖率影响；“等概率”定义为在单个样本当前可用分支之间等概率。
- 动态 caption 视图意味着文本编码继续在线执行。
- 具体 dropout 数值未写入本决定，训练配置在启动前必须补齐。
# 验证标准
- 在至少 100,000 个样本的 dry run 中，显式 `all_condition` 分支命中率应为 `10% ± 0.5` 个百分点；同时单独报告最终无条件样本总比例。
- 任一非空条件样本最多包含一种 NL 分支。
- 各 tag 类、candidate 与 NL dropout 的实测频率必须与配置值一致，允许统计误差但不得出现字段串线。
- 最终送入 tokenizer 的 tag 不含下划线，原始 JSON 保持不变；tags 只能以 `, ` 连接，NL 只能以双换行 `\n\n` 追加，且不得出现 `Tags:`、`Description:` 或类别标签。
- 固定 seed、epoch 与 sample key 时 caption 结果完全可复现。
- 缺少 `causal_conv1d`、`fla` 或未启用快速 kernel 时启动检查必须失败。
- 目标机器从空缓存启动时，端到端数据管线必须持续达到组件 01 的 `≥12 samples/s` 门槛，并报告 GPU 等待数据的时间比例。
- 对下载中断、损坏缓存和 token 失效做故障注入：前两者必须续传或重下且不发布坏 shard，认证失败必须明确终止。
- 缓存占用不得突破配置 quota；活跃 shard 保护不得造成无限增长。
- 做一次强制中断恢复测试：已完成 shard 不得重读，未完成 shard 必须从头重放，恢复后的 `replayed_samples` 必须与审计日志一致。
- 删除所有续训 sidecar 后，模型目录仍必须能够独立加载；删除或损坏任一必需 sidecar 后，续训必须明确失败而不是部分恢复。
- 在 checkpoint 写入中途强制终止时，恢复器必须忽略该半成品并回退到上一份带 `COMPLETE` 标记的 checkpoint。
- `validation_manifest.jsonl` 必须恰含 2,000 个唯一 `id`，且独立 validation shard 的内容与 manifest 完全一致。
- 对完整训练 manifest 做一次排除 dry run，验证 2,000 个 `id` 的训练消费计数全部为零；不要求再次执行图像去重。
- 固定 checkpoint 和验证配置重复运行两次时，验证样本、caption、bucket 与 crop 参数必须逐项一致。
- 在前 100,000 张 dry run 中报告 `dimension_mismatch`；不一致率超过 `0.1%` 时暂停正式训练并检查数据制作流程。
- 尺寸校验开启与关闭各运行同一基准；若单次解码校验导致端到端吞吐下降超过 `2%`，必须先定位重复解码或队列重路由问题。
- candidate dropout 未命中时，最终 tags 必须与原四类 tags 的并集一致且不增加副本；命中时，四类列表中所有 candidate 匹配词的残留计数必须为零，NL 内容保持原样。
- 固定 global seed、data pass 和 `id` 时 tag 排列完全一致；改变 data pass 时只有类别内顺序形成新的确定性排列，类别顺序固定，且 NL 始终位于末尾。
- token预算决定固定复用当前结果：`text_condition_max=512`，桶为 `[64,128,192,256,320,384,448,512]`。Artist从主caption移入style辅助segment不触发重新扫描；当前结果直接作为resolved config依据。tokenizer revision、协议版本、扫描代码版本、样本数和结果hash仍须保存。
- 文本预算已经定稿；目标机继续报告Qwen/DiT峰值显存和端到端吞吐，但只用于选择dense/varlen与batch配置，不重新选择512上限或8个桶。
- 对发生 tags 截断的样本跨两个 data pass 检查保留集合变化；任何输出都不得包含被截断一半的 tag。
- 在目标四卡机器上完成 worker/queue sweep；最终配置必须持续达到 `≥12 samples/s`，GPU 因 ready queue 为空造成的等待占比不得超过 `2%`，且不得触发 host swap、OOM 或持续队列增长。
- 冻结 Qwen 与 Mage-VAE 的参数和前向中间量必须通过自动检查确认无梯度；连续运行基准时 GPU peak memory 必须稳定而非逐步增长。
# 实施前待测与跨组件接口
- 分辨率阶段切换时继续当前 data pass 还是以新 seed 重开：与组件 10 联合决定。
- sample shuffle buffer 的具体容量：通过内存与吞吐测试写入配置。
- ModelScope 下载并发数、单文件 Range worker 数、缓存高低水位及 `300–500 GiB` 区间内的最终 quota：通过目标机器冷缓存测试写入配置。
# 延后项
- 各 tag 类、candidate 与 NL 的具体 dropout 概率：在训练配置中确定。
- tokenizer文本预算已由组件03锁定为condition 512和8个桶；Caption body分隔符、Krea 2式手工framing、无thinking、无有效`<|endoftext|>`及同模板空body的CFG空条件均已接受，不再延后。
- 若未来发现上游去重失效或需要重复样本权重，再另行更新本决定并保留变更记录。
<empty-block/>
</content>
</page>
