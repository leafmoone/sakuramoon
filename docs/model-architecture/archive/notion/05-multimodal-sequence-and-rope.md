Here is the result of "view" for the Page with URL https://app.notion.com/p/3abae967ecf281de8dafc6dbecd04fe6 as of 2026-07-29T06:42:44.624Z:
<page url="https://app.notion.com/p/3abae967ecf281de8dafc6dbecd04fe6">
<ancestor-path>
<parent-data-source url="collection://69ca66ff-43e7-4128-bb7b-9f3751506705" name="组件决策记录"/>
<ancestor-2-database url="https://app.notion.com/p/250be554eacc40219065073dfcf66fd7" title="组件决策记录"/>
<ancestor-3-page url="https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff" title="新模型架构决策中心"/>
</ancestor-path>
<properties>
{"date:决定日期:is_datetime":0,"date:决定日期:start":"2026-07-28","url":"https://app.notion.com/p/3abae967ecf281de8dafc6dbecd04fe6","决策编号":"ARCH-6","序号":5,"影响":"高","标签":["架构","训练"],"状态":"待验证","组件决策":"05 多模态序列与位置编码"}
</properties>
<content>
<callout icon="🚧" color="yellow_bg">
	**架构与文本数值均已批准，进入待验证。** 联合序列、2D RoPE、crop offset、全局尺寸元数据、`text_condition_max=512`、8个文本桶与DiT varlen packing均已确定。只保留四卡RTX 5090目标机上的varlen/dense吞吐、峰值显存和数值一致性验证。
</callout>
# 范围
本组件接收组件 02 的 Mage-VAE latent tokens、组件 04 的主文本 tokens 与 4 个 style tokens，定义它们进入 Single-stream DiT 前的排列、modality embedding、position coordinates 和 attention mask。DiT block 宽度、head 数和 GQA 比例由组件 06 决定；时间/尺寸调制接口与输出 head 由组件 07 决定。
# 继承约束
- Mage-VAE 原生 `H/16 × W/16` 网格、128 latent channels，不额外 patchify；512 方图为 1024 image tokens，1024 方图为 4096 image tokens。
- 主文本序列保留有效 token 粒度，不做句级 pooling；Qwen hidden states 已携带文本顺序，组件 04 的双向 Attention-only block 使用 NoPE。
- style 分支固定输出 4 个 style tokens；artist dropout 或无 artist 时，4 个可学习 null style tokens 仍是有效条件 token。
- 主干采用 single-stream full attention；不得重新引入独立 cross-attention 双流。
- 宽高比归一化 2D axial RoPE 方向已批准，且与 GQA 兼容。
# 外部实现核对
## Krea 2
Krea 2 以 `[text | image]` 拼接后执行全序列 attention。文本 position 三轴全为 0，图像 position 为 `(0, row, column)`；因此文本不增加 1D RoPE，空间轴只作用于图像，而 text-image attention 仍共享同一 RoPE attention。联合 mask 由有效 text mask 与全有效 image mask 拼接。
- [Krea 2 sampling.py](https://github.com/krea-ai/krea-2/blob/main/sampling.py)
- [Krea 2 mmdit.py](https://github.com/krea-ai/krea-2/blob/main/mmdit.py)
## HDM
HDM 技术报告和代码都保持 x/y 网格的各向同性，但归一化尺度并不相同：
- 报告公式：`r_h = sqrt(H/W)`、`r_w = sqrt(W/H)`，保持坐标面积 `r_h × r_w = 1`。
- 仓库 `bounding_box()`：最长轴固定到 `[-1, 1]`，短轴按宽高比收缩。
二者都让 x/y 相邻 token 的坐标步长相等；在恒定像素面积的宽高比桶中，只有报告公式还能让不同宽高比拥有相同坐标步长。该差异必须显式决定，不能笼统写成“照搬 HDM”。
- [HDM axial_rope.py](https://github.com/KohakuBlueleaf/HDM/blob/main/src/xut/modules/axial_rope.py)
- [HDM TechReport](https://github.com/KohakuBlueleaf/HDM/blob/main/TechReport.md)
# 已批准 05-A：联合序列布局
## A. `[text | style | image]`，推荐
- 三类输入分别加入 learned modality embedding：`text`、`style`、`image`。
- text 与 style 的 2D position 均为 `(0, 0)`，即在主干中不追加 1D 文本 RoPE。
- image 使用归一化的 `(y, x)` pixel-center coordinates。
- 所有有效 token 之间使用全双向 attention；只屏蔽 text padding。
- null style tokens 始终有效，不得当成 padding 屏蔽。
- image tokens 连续位于序列尾部，输出 head 只切出 image span。
优点：最接近 Krea 2/HDM 的单流接口，文本顺序由 Qwen 表示承载；为 style 保留独立类型，同时不增加第三个 RoPE 轴。
## B. 文本 1D + 图像 2D 三轴 RoPE
为文本分配单独 1D axis，图像使用 y/x，style 再定义特殊坐标。优点是主干显式感知文本顺序；缺点是重复编码 Qwen 已有顺序，压缩每个 head 可用于空间轴的维度，并增加 style 和未来编辑 token 的坐标规则。当前不推荐。
## C. 分块或单向 mask
限制 text/style/image 的可见关系。会偏离 single-stream 的联合建模目标，也会让后续编辑 token 接口复杂化，不采用。
# 05-A 最终决定
采用 `[text | 4 style | image]`，三种 learned modality embeddings，text/style 坐标固定为 `(0, 0)`，image 使用 2D 坐标，全双向 valid-token attention。padding 同时屏蔽 query 与 key，并在每个 block 后将 padding states 清零。null style tokens 始终有效；不增加分隔 token或文本 1D RoPE。
# 已批准 05-B：2D 坐标归一化
令最终 latent 网格为 `H × W`，坐标取 token cell center，不使用 align-corners。
## A. 面积归一化，推荐
定义 `r_y = sqrt(H/W)`、`r_x = sqrt(W/H)`：
- `y_i = (2(i + 0.5)/H - 1) × r_y`
- `x_j = (2(j + 0.5)/W - 1) × r_x`
于是 `r_y × r_x = 1`，且两个轴的相邻 token 步长严格相同：`Δy = Δx = 2/sqrt(HW)`。对于保持 `H × W` 近似恒定的宽高比桶，不同宽高比拥有相同局部 RoPE 相位步长；不同分辨率阶段则保持同一归一化画布，只是采样更密。
代价是长轴坐标范围可超过 `[-1, 1]`。因此推理宽高比应限制在训练 bucket 覆盖范围；极端比例属于 RoPE 坐标外推。
## B. 最长边归一化
定义 `r_y = H/max(H,W)`、`r_x = W/max(H,W)`，即 HDM 当前仓库 `bounding_box()` 的行为。长轴始终限制在 `[-1, 1]`，坐标范围更保守；但 `Δ = 2/max(H,W)`，同像素面积的不同宽高比会获得不同局部相位尺度，使 aspect-ratio bucket 之间的几何标尺不统一。
## C. 原始整数 row/column
最接近 Krea 2/FLUX 一类实现，但分辨率增长会直接扩大 position range；不符合已经批准的宽高比归一化方向，不采用。
# 05-B 最终决定
采用 A，即 HDM 技术报告的面积归一化公式，而不是仓库当前的最长边归一化实现。使用 token cell centers 且 `align_corners=False`。坐标表达归一化画布几何；绝对 target H/W 和像素规模另作为全局尺寸元数据。推理宽高比原则上限制在训练 buckets 的覆盖范围内。
# 已批准 05-C：RoPE 维度与频率
## A. 全 head 维度、y/x 各一半、可学习 per-head 频率
最接近 HDM 当前代码。纯图像 self-attention 的空间表达最强，但所有 text-image 相似度通道都会受到图像位置相位影响；per-head 频率也不适合后续 GQA 的共享 K heads。渐进分辨率训练中，可学习频率还可能过早适配低分辨率阶段。当前不推荐。
## B. 75% 空间 RoPE + 25% NoPE，固定共享频率，推荐
设 attention head dimension 为 `d_h`：
- `d_nope = d_h/4`
- `d_y = 3d_h/8`
- `d_x = 3d_h/8`
要求 `d_h` 可被 16 整除，使 y/x rotary pairs 都合法。若 `d_h=128`，分配为 `32/48/48`；若 `d_h=64`，分配为 `16/24/24`。
1/4 NoPE 通道为 text-image 提供与空间位置无关的语义相似度；其余 3/4 等分给 y/x，保留足够空间容量。这对应 Krea 2 在 `head_dim=128` 下的有效分配：第一个零 position axis 占 32 维，y/x 各占 48 维。
频率采用固定、所有 Q/K heads 与两个空间轴共享的几何频谱；不得使用 per-head 频率。这样 MHA、GQA 都可直接复用同一 position frequencies，KV head 被多个 query heads 共享时不会发生 RoPE 基不一致。
## 坐标单位和 theta
05-B 的面积归一化坐标在进入 RoPE 前乘固定 `position_scale=16`。这不改变宽高比归一化关系，只定义相位单位：
- 512 方图 latent 为 `32×32`，相邻坐标步长约为 1；
- 同为 1024 tokens 的 `16×64`，两个轴步长仍约为 1；
- 256 方图步长约为 2，1024 方图步长约为 0.5，表示同一画布上的不同采样密度。
固定 RoPE `theta=1000`，采用标准几何频率；Q/K 先执行 QK-RMSNorm，再应用 RoPE。`position_scale` 与 `theta` 都写入 checkpoint config，不设可学习参数。
## C. 50% 空间 RoPE + 50% NoPE
全局语义通道更多，但对于最高 4096 image tokens 的二维空间容量偏保守，也缺少比 B 更直接的强模型实现依据。不推荐。
# 05-C 最终决定
采用 B：每头 25% NoPE、37.5% y-RoPE、37.5% x-RoPE；面积归一化坐标乘固定 `position_scale=16`；固定 `theta=1000`；频率跨 heads、Q/K 和 x/y 共享，QK-RMSNorm 后应用 RoPE。要求 `head_dim` 可被 16 整除。`position_scale`、`theta` 与维度分配均保存到 checkpoint config，不设可学习参数。
# 已继承 05-D：crop offset
组件 02 已批准：resize/crop 后按最终 bucket 画布重新生成完整的面积归一化 position map，**不编码原图尺寸、缩放比例或 crop offset**。每个 crop 被视为一张完整目标图，而不是隐藏大画布中的窗口；不采用 HDM shifted-square crop offset map。
理由：当前训练目标是从文本生成完整画面，而不是恢复原图中的绝对窗口位置。编码 offset 会把随机数据增强变成可见条件，引入“被裁出来”的构图先验；推理时又不存在对应原图 offset，形成训练/推理接口不一致。crop 参数仍保存在数据审计与验证记录中，但不进入模型。
# 已批准 05-E：全局尺寸元数据
面积归一化 RoPE 主要表达画布几何和相对空间尺度；绝对输出分辨率需要显式送入全局条件。只使用最终 bucket/推理请求的目标像素尺寸，不使用源图尺寸。
## A. 直接提供 target H/W 的正交对数参数，推荐
定义：
- `size_scale = 0.5 × log2((H_px × W_px) / 512²) = log2(sqrt(H_px × W_px) / 512)`
- `aspect = log2(W_px / H_px)`
两者分别表示线性分辨率尺度和宽高比，且可无损恢复 `log2(H/512)` 与 `log2(W/512)`：
- `log2(H/512) = size_scale - aspect/2`
- `log2(W/512) = size_scale + aspect/2`
组件 07 对两个标量分别做固定 Fourier/sinusoidal embedding，再经一个共享小 MLP 投影到全局 conditioning vector，与 timestep embedding 相加或拼接后投影。原始两个标量及 embedding 配置保存到 checkpoint config。
## B. 四个冗余标量 `H, W, area, aspect`
信息重复、数值尺度不一致，增加配置和调试负担，没有额外信息，不采用。
## C. 不提供尺寸元数据
模型可从 token 数和坐标范围间接推断，但面积归一化有意统一了不同分辨率的画布尺度。仅依赖序列长度会让局部 block 难以稳定区分 512 与 1024 阶段，不推荐。
# 05-E 最终决定
采用 A，仅传入 `size_scale = 0.5 × log2((H_px × W_px)/512²)` 与 `aspect = log2(W_px/H_px)`。尺寸条件始终有效，不参加 caption/CFG dropout；conditional 与 unconditional 分支使用完全相同的两个值。推理时由请求的最终 H/W 重新计算。组件 05 固定数值定义，具体 Fourier embedding 维度和注入方式由组件 07 决定。
# 已批准 05-F：文本长度与联合序列 packing
## 批准前扫描截面（早期记录）
后台扫描正常运行；批准前阶段性结果约为 `7.17M / 11.27M`：动态 NL 等概率选择后的 `full_equal_choice` 为 p50=185、p90=322、p95=372、p99=482、p99.9=655；超过 512/640/768/1024 的比例约为 0.671%/0.119%/0.037%/0.012%。这些不是最终结果，但已说明固定 512 会产生可见截断，而把所有样本固定 pad 到 1024 又会浪费大量计算。
## A. 所有样本固定长度 dense padding
实现最简单，但绝大多数 caption 会为空耗 Qwen 和 DiT 计算；不采用。
## B. 离散文本桶 + DiT dense padding
按文本长度进入有限 buckets，并在联合序列内保留 text padding。可作为 correctness fallback，但 image tokens 会在序列中跨过一段 padding，且 DiT 仍计算无效 token，不作为生产默认路径。
## C. 离散文本桶 + DiT varlen packing，推荐
1. Qwen输入使用有限文本buckets。批准前暂定边界已被05-F最终数值覆盖；bucket长度始终指裁掉prefix后的全部condition tokens，Qwen实际前向还需包含prefix。
2. Qwen 与组件 04 文本适配器按 bucket dense 执行，并严格使用 text padding mask。
3. 进入 DiT 前移除每个样本的所有 text padding，紧凑构造 `[valid text | 4 style | image]`。
4. 将 batch 中样本打平成 varlen packed tensor，使用 `cu_seqlens`/等价 document boundaries 阻止跨样本 attention；单个样本内部仍为全双向 attention。
5. 为每个样本保存 `text_len`、style span、image span 和 `(H_latent,W_latent)`，输出后只 gather 连续 image span 并恢复网格。
6. varlen 生产路径不为了编译而给每个样本补到 128/256 的倍数；只允许 kernel 在内部做不可见的 tile 对齐。通过有限的 image buckets 与 text `max_seqlen` buckets 控制编译 shape。
7. dense fallback 只用于正确性测试和 kernel 不可用时的诊断；必须同时屏蔽 padding query/key 并在每个 block 后清零。正式训练是否允许 fallback 由组件 06 的目标机器 kernel benchmark 决定。
## 批准前截断候选（已取代）
早期在768/1024间选择的候选已被最终condition 512决定取代。现行规则见05-F：超过512时保留协议边界和高优先级tags，先裁NL尾部；tags仍过长时只删除完整低优先级tag，禁止截断半个token/tag。
# 05-F 最终决定
采用**离散文本长度桶 + DiT varlen packing**，并保留 dense bucket fallback。
1. Qwen 与组件 04 在有限长度桶内执行 dense batch，并使用正确 attention mask。
2. 进入 DiT 前移除文本 padding，按样本形成 `[valid text | 4 style | image]`。
3. 同一 GPU microbatch 内通过 document boundaries / `cu_seqlens` 打包；不同样本绝不互相 attention。
4. 保存各样本 span，只取回各自 image-token 输出计算 loss。
5. 不为 128/256 等逻辑长度重新补 padding；kernel 内部 tile 对齐不改变逻辑序列长度。
6. 保留 dense 实现，用于正确性对照、诊断以及目标机速度回退。
7. 截断只在完整字段或完整tag边界执行；先为协议边界和Artist辅助segment保留预算，再从NL尾部及低优先级主文本tags删减。
8. 固定 `text_condition_max=512` 与文本桶 `[64,128,192,256,320,384,448,512]`；对应完整Qwen dense lengths为 `[98,162,226,290,354,418,482,546]`。
9. bucket长度是去掉固定34-token prefix后的全部condition tokens，包含5-token suffix与Artist辅助segment；prefix/suffix计数必须由锁定tokenizer实测断言。
10. Artist只进入style辅助segment但复用当前扫描，不重新扫描。超过512约0.516%的样本按第7条截断；目标机benchmark无权改写上限或桶集合。
# 速度影响：批准前估算（历史）
以下只用于确定工程方向，不替代目标机 benchmark。局部扫描已覆盖约 901 万 / 1127 万样本：条件正文均值约 197 token，`p95=367`、`p99=473`；超过 512 的约 0.59%，超过 768 的约 0.037%。当前扫描尚未应用 10% 全条件 dropout 和字段 dropout，因此真实训练平均长度还会更短。
使用暂定桶 `192/256/320/384/512/640/768/1024` 时，平均执行长度约 246 token。正式桶集合加入 64/128 后，dropout 样本的平均执行长度会继续下降。
## Qwen / 组件 04
- 相对所有样本固定补到 1024，文本 token 工作量理论下降约 **76%**。
- 考虑 GPU 利用率、kernel 启动和固定开销，Qwen 文本分支本身预计约 **2.5–3.5 倍**吞吐，而不是理论上的 4.2 倍。
- 若未优化时文本分支占整步 10%–30%，按 Amdahl 定律，训练总吞吐预计提高约 **8%–25%**；高分辨率阶段 DiT 占比更高，收益更接近区间下端。
## DiT
按均值 197 个有效文本 token、4 个 style token 估算，联合序列平均长度约为：
<table>
<tr>
<td>训练分辨率</td>
<td>image tokens</td>
<td>packed 平均总长度</td>
<td>attention 计算量相对 dense length-bucket 的下降</td>
<td>相对固定 text=512</td>
<td>相对固定 text=1024</td>
</tr>
<tr>
<td>256</td>
<td>256</td>
<td>457</td>
<td>约 18%</td>
<td>约 65%</td>
<td>约 87%</td>
</tr>
<tr>
<td>512</td>
<td>1024</td>
<td>1225</td>
<td>约 7%</td>
<td>约 37%</td>
<td>约 64%</td>
</tr>
<tr>
<td>768</td>
<td>2304</td>
<td>2505</td>
<td>约 4%</td>
<td>约 21%</td>
<td>约 44%</td>
</tr>
<tr>
<td>1024</td>
<td>4096</td>
<td>4297</td>
<td>约 2%</td>
<td>约 13%</td>
<td>约 30%</td>
</tr>
</table>
这里的百分比是 attention score FLOPs 的估算，不是端到端速度。QKV、MLP、归一化、通信和 VAE 不会按同样比例下降；FlashAttention 下峰值显存也不能直接按平方比例推算。
相对一个已经合理实现的 dense length-bucket 基线，**varlen 单独带来的端到端训练吞吐增益**预计为：
- 256 阶段：约 **5%–15%**
- 512 阶段：约 **3%–10%**
- 768 阶段：约 **1%–5%**
- 1024 阶段：约 **0%–3%**
如果 varlen kernel 在 RTX 5090、GQA 和当前 head dimension 下效率不佳，1024 阶段可能与 dense bucket 持平甚至略慢，所以 dense fallback 是正式设计的一部分。
## 推理
- batch=1 且直接使用真实 prompt 长度时，本来就没有跨样本 padding；varlen packing 的收益基本为 **0%**，还可能有极小的调度开销。
- 常规 20–50 步采样中，Qwen 只执行一次，文本长度优化对端到端延迟通常约 **\<1%–3%**。
- 4–8 步少步数采样时，Qwen 占比提高，预计约 **2%–8%**。
- 多 prompt、长度差异较大的批量推理或 CFG cond/uncond 合批时，packing 才明显：512 分辨率预计约 **5%–15%**，1024 分辨率约 **1%–5%**。
# 验收门槛
在目标机分别对256、512、1024，至少测试dense bucket与varlen两条路径，记录samples/s、tokens/s、峰值显存以及输出数值误差。生产训练只在varlen带来至少约3%端到端吞吐提升，或显存收益能够换取更大microbatch时启用；否则该阶段使用dense bucket fallback。该验证只选择执行后端，`text_condition_max=512`与8个桶保持不变。
</content>
</page>
