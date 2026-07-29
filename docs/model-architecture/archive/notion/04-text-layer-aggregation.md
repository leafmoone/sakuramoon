Here is the result of "view" for the Page with URL https://app.notion.com/p/3abae967ecf28104ad74d1b843728be4 as of 2026-07-29T06:39:47.422Z:
<page url="https://app.notion.com/p/3abae967ecf28104ad74d1b843728be4">
<ancestor-path>
<parent-data-source url="collection://69ca66ff-43e7-4128-bb7b-9f3751506705" name="组件决策记录"/>
<ancestor-2-database url="https://app.notion.com/p/250be554eacc40219065073dfcf66fd7" title="组件决策记录"/>
<ancestor-3-page url="https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff" title="新模型架构决策中心"/>
</ancestor-path>
<properties>
{"date:决定日期:is_datetime":0,"date:决定日期:start":"2026-07-28","url":"https://app.notion.com/p/3abae967ecf28104ad74d1b843728be4","决策编号":"ARCH-5","序号":4,"影响":"高","标签":["架构","训练","系统"],"状态":"已接受","组件决策":"04 文本多层聚合与双向适配器"}
</properties>
<content>
<callout icon="✅" color="green_bg">
	**已接受。** 主文本采用 7 层 token-dependent grouped gated mixing，并用 1 个双向 Attention-only block 修正因果表示；artist style 分支继续使用 4-query attention pooling + Style MLP。
</callout>
# 范围
本组件承接组件 03 输出的冻结 Qwen hidden states，决定取哪些层、如何沿层轴融合、是否增加双向文本适配器，以及如何投影到生成主干宽度。tokenizer、framing、caption body、最大文本长度由组件 03 决定；图文排列和联合位置编码留给组件 05。
# 已确认的外部实现
## Krea 2 实际结构
Krea 2 的官方 `encoder.py` 从 36 层 Qwen3-VL-4B 中默认选择 12 层：`(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)`，输出形状为 `[B, L, N, D]`。
官方 `mmdit.py` 的 `TextFusionTransformer` 不是直接平均或拼接：
1. 对每个文本 token 独立地，把 N 个层状态当作短序列，执行 2 个 `TextFusionBlock`；
2. 用 `Linear(N, 1)` 将层轴压成一个向量；
3. 在文本 token 轴执行 2 个不带 causal mask 的 `TextFusionBlock`，只应用 padding mask；
4. 再由 MLP 投影到 DiT 主干宽度。
因此这里所谓“双向适配”来自第 3 步：Qwen 本体保持冻结和因果，融合后的文本 token 在进入图文主干前获得左右文上下文。
- [Krea 2 官方 encoder.py](https://github.com/krea-ai/krea-2/blob/main/encoder.py)
- [Krea 2 官方 mmdit.py](https://github.com/krea-ai/krea-2/blob/main/mmdit.py)
# 本项目输入约束
- 指定 Qwen3.5-2B 共 24 层、hidden size 2048。
- 层拓扑按 `3 × linear_attention + 1 × full_attention` 重复 6 次；full-attention block 的零基索引为 `3, 7, 11, 15, 19, 23`。
- `output_hidden_states=True` 只做一次冻结前向；任何选层必须来自这一次输出，不允许为不同层重复运行 Qwen。
- padding、prefix 裁切和 suffix 保留严格沿用组件 03 的 mask。
# 已批准：artist style 分支
## 结构
对 artist span，从同一次冻结 Qwen 前向中取 7 个候选层的 hidden states，形成 `[B, A, 7, 2048]`，其中 A 为 artist span 的 token 数。处理顺序：
1. 对各层状态执行共享 `RMSNorm`，加入 learned layer embedding；
2. 将 `artist token × layer` 展成短源序列；
3. 使用 4 个 learned style queries 做一层 cross-attention pooling，输出 `4 × 1024` style slots；
4. 对每个 slot 应用共享的 residual Style MLP：`RMSNorm → SwiGLU(1024 → 2048 → 1024)`；
5. 用 `Linear(1024, d_dit)` 投影成 4 个 style tokens，送入生成主干。
## 已批准约束
- attention pooling 和 Style MLP 只属于 style 分支，不替代主文本多层聚合。
- 复用主 caption 的同一次 Qwen hidden states；禁止为了 artist 进行第二次在线 Qwen 前向。
- artist dropout 或无 artist 时，使用 4 个可学习 null style tokens，保持 shape 固定。
- style 分支不增加 tokenizer 特殊词，不改变 Qwen tokenizer。
- 4-query pooling 允许四个 style slot 学到不同视觉因素；不得把单个 pooled vector 线性展开成 4 个伪 token。
- 后续会话已锁定：Artist 从主文本 caption 移出，只进入 style 分支。在线 serializer 在同一 Qwen 输入中构造 Artist 辅助 segment并记录 token indices；主文本聚合不消费Artist tokens，首版不做第二次Qwen或离线style cache。
## 成本判断
style 分支只处理少量 artist tokens、7 个层状态和 4 个 query，增量计算远小于 Qwen 与生成主干。主要风险不是 attention pooling，而是若错误地增加第二次 Qwen 前向；该实现已明确禁止。
# 规模约束
若在 2048 宽度原样复制 Krea 2 的 4 个文本融合 block，注意力加 SwiGLU 粗略约为 2 亿参数，尚未计入输入/输出投影。对预期约 1B 的生成主干和四卡 RTX 5090 预算而言占比过高，不能把 Krea 2 的 13B 配比直接照搬。
# 已取代候选
- 2048 宽度原样复制 Krea 2 的 `2 + 2` TextFusionBlocks：规模过大，否决。
- 缩宽至 1024 的 `1 + 1` 完整 Transformer：仍包含不必要的 layer-axis pairwise modeling 与两个 SwiGLU FFN，已被最终轻量方案取代。
- 全局静态 7 层标量权重：无法按 token 和通道子空间改变选层，表达力不足，不采用。
# 已批准：主文本多层聚合与双向修正
## 输入层选择
从 Qwen 24 个 block 中取 7 个语义阶段输出：`after block 2, 4, 8, 12, 16, 20, 24`。block 编号按 1 起算，运行时必须显式映射 `hidden_states` tuple，不得直接把 block 编号当作 tuple 下标。
`4, 8, 12, 16, 20, 24` 对应六个“3 个 linear-attention + 1 个 full-attention”周期的结束；额外加入 `after block 2`，保留较早的词法和局部 tag 表示。
## 层轴：token-dependent grouped gated mixing
1. 各选中层先经过各自的 `RMSNorm`，再经过共享 `Linear(2048, 1024)`，得到 `[B, L, 7, 1024]`。
2. 将 1024 个通道固定分成 8 组，每组 128 维。
3. 使用共享的小型 gate scorer 为每个 token、每个层、每个通道组产生分数；仅沿 7 个层做 softmax。
4. 各通道组按自己的 7 层权重加权求和，重新拼成 `[B, L, 1024]`。
5. 保留最深层残差锚点：`x = z_deep + mix_gate × (z_mixed - z_deep)`。`mix_gate` 可学习，确保融合退化时仍可回到最终层表示。
约束：层轴不得使用 Transformer、self-attention 或 FFN；不得退化成对所有 token 共用的一组静态层权重。
## Token 轴：一个双向 Attention-only block
执行一次：`x = x + LayerScale × MHA(QKNorm(RMSNorm(x)), padding_mask)`。
- 使用 MHA，不使用 GQA；具体 head 数作为实现配置。
- attention 为非因果，只屏蔽 padding；有效 token 两两可见。
- 只包含 RMSNorm、Q/K Norm、QKV/O projections、attention、LayerScale 和 residual。
- 不包含 FFN、SwiGLU 或第二个 refinement block。
- 不增加文本 RoPE、绝对位置编码或相对位置 bias。Qwen hidden states 已携带位置与因果顺序信息；保留的 suffix states 已汇总整个 caption，一次双向 attention 即可将全局信息传播给前置 token。
- block 后再次将 padding 位置清零，防止 residual 保留无效状态。
## 输出
使用 `RMSNorm + Linear(1024, d_dit)` 投影为主文本条件序列。这里不做整句 pooling，保留原有有效 token 数和组件 03 的 mask。
## 成本判断
在 `d_adapter=1024` 下，原 `1 + 1` 完整 Transformer 的两个 block 约为 2500 万参数；最终 gated mixer 加一个 MHA Attention-only block 的融合核心约为 500 万参数，输入与输出投影另计。融合核心约减少 80%，但整体 step time 仍由冻结 Qwen 前向和生成主干主导，必须以实测吞吐为准。
## 与 style 分支的边界
以上 gated mixing 和 Attention-only block只处理主文本序列。artist style 分支保持此前批准的 `4-query cross-attention pooling + residual SwiGLU Style MLP + 4 style tokens`，不得因主文本轻量化而删除 Style MLP。
# 实现配置与外部依赖
以下项目不改变本组件已接受的架构：
1. MHA 的 head 数、`mix_gate` 与 LayerScale 初值在模型配置文件中明确；实现前用 shape/mask 单元测试确认。
2. `d_dit` 随最终生成主干宽度确定，输入适配宽度固定为 1024。
3. Artist只进入在线style辅助segment；实现必须复用同一次Qwen前向并使用结构化segment metadata，不得增加第二次在线Qwen或可缓存的artist-only特征路径。
# 最低成本验证
- 固定一小批 tags-only、NL-only、tags+NL、长文本和空条件，验证输出 shape、mask、padding 零泄漏及双向依赖。
- 在单卡 RTX 5090 上分别测量 Qwen、文本适配器和生成主干的前向时间及峰值显存。
- 只比较最终候选与一个轻量回退，不进行大规模层号消融。
</content>
</page>
