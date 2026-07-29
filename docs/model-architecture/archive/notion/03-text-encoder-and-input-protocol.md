Here is the result of "view" for the Page with URL https://app.notion.com/p/3aaae967ecf281fba3cfe0f5dc53fece as of 2026-07-29T06:42:43.003Z:
<page url="https://app.notion.com/p/3aaae967ecf281fba3cfe0f5dc53fece" icon="🧾">
<ancestor-path>
<parent-data-source url="collection://69ca66ff-43e7-4128-bb7b-9f3751506705" name="组件决策记录"/>
<ancestor-2-database url="https://app.notion.com/p/250be554eacc40219065073dfcf66fd7" title="组件决策记录"/>
<ancestor-3-page url="https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff" title="新模型架构决策中心"/>
</ancestor-path>
<properties>
{"date:决定日期:is_datetime":0,"date:决定日期:start":"2026-07-27","url":"https://app.notion.com/p/3aaae967ecf281fba3cfe0f5dc53fece","决策编号":"ARCH-4","序号":3,"影响":"高","标签":["架构","训练","数据","系统"],"状态":"讨论中","组件决策":"03 文本编码器与输入协议"}
</properties>
<content>
<callout icon="✅" color="green_bg">
	**输入协议与文本预算已闭合，进入待验证。** checkpoint、手工 framing、caption body、Artist 辅助 segment、`text_condition_max=512` 与 8 个长度桶均已锁定。目标机只继续验证 dense/varlen 的执行效率，不得改写已批准的上限和桶集合。
</callout>
# 范围
本组件决定文本编码器的具体权重、加载与前向边界、tokenizer 输入协议、CFG 空条件和最大 token 长度。caption 字段选择、dropout、shuffle、下划线清理及截断优先级由 <mention-page url="https://app.notion.com/p/3aaae967ecf281db800cfb1d6545f880"/> 提供；多层 hidden states 的层选择与聚合方式留给组件 04。
# 03-A 已接受决定
## 指定 checkpoint
- 唯一指定模型：[spawner/Qwen3_5_2b_claude_heretic_spawner](https://www.modelscope.cn/models/spawner/Qwen3_5_2b_claude_heretic_spawner)。
- **不使用官方 Qwen3.5-2B 或 Qwen3.5-2B-Base 替换。** 官方版本不是自动回退项；若未来需要更换，必须新开决策并重新验证。
- 该仓库声明其权重来源于 Qwen3.5-2B-Base 的第三方 reasoning-distilled checkpoint，再经过额外处理。因此本文把它视为指定的社区后训练权重，不把它误记为官方 Base。
## 可复现锁定
ModelScope 当前仅公开可变的 `master` 修订，没有不可变 tag。下载时固定仓库和 `master`，并强制校验以下 SHA-256：
<table fit-page-width="true" header-row="true">
<tr>
<td>文件</td>
<td>SHA-256</td>
</tr>
<tr>
<td>`model.safetensors`</td>
<td>`a71a234d04b9a026f56232372c0f5c143caaf9ca3e2cea814ca1cc08ce56301f`</td>
</tr>
<tr>
<td>`config.json`</td>
<td>`ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4`</td>
</tr>
<tr>
<td>`tokenizer.json`</td>
<td>`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`</td>
</tr>
<tr>
<td>`tokenizer_config.json`</td>
<td>`49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c`</td>
</tr>
</table>
任一 hash 不匹配即停止加载，不允许继续训练。若仓库以后提供不可变 commit/tag，再把它补入配置，但仍保留 hash 校验。
# 模型与前向边界
- 配置对应 `Qwen3_5ForConditionalGeneration`：语言 hidden size 2048、24 层，线性注意力与 full attention 混合排列。
- 文本编码器全程冻结：`eval()`、全部参数 `requires_grad=False`，在 `no_grad/inference_mode` 下以 BF16 执行。
- 只执行语言路径；视觉编码器不得加载到执行图或被调用。
- 每个 caption 只做一次 `forward(..., output_hidden_states=True, use_cache=False)`。
- 不调用 `generate()`，不建立生成 KV cache，不生成任何回答或推理文本。
- hidden states 的取层与聚合不在本组件决定，转交组件 04。
- `causal_conv1d` 与 `fla/flash-linear-attention` 是明确安装要求；不得因缺少依赖静默切换到行为不同的实现。具体版本随训练环境 lockfile 固定。
# No-thinking 强约束
- tokenizer 虽包含 `<think>` 与 `</think>`，训练和推理输入都不得出现这两个 token。
- 不调用该仓库自带的 `apply_chat_template(..., add_generation_prompt=True)`；即使设置 `enable_thinking=False`，该路径仍可能插入空的 thinking 包装。
- 采用下面明确写死且版本化的 Krea 2 式手工 framing，只做一次文本 `forward`，不生成 assistant 内容。
- 训练、验证和推理必须调用同一个序列化函数。
# 03-B 已接受：Krea 2 式输入协议
参考 [Krea 2 官方 ](https://github.com/krea-ai/krea-2/blob/main/encoder.py)[`encoder.py`](https://github.com/krea-ai/krea-2/blob/main/encoder.py)，条件文本使用以下固定结构：
```plain text
<|im_start|>system
Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>
<|im_start|>user
{caption_body}<|im_end|>
<|im_start|>assistant
```
## 序列与 mask
- 前缀固定到 `user\n` 之后，`caption_body` 由组件 11 生成；后缀固定为 `<|im_end|>\n<|im_start|>assistant\n`。
- 不追加有效的 `<|endoftext|>`。该 token 只可作为 padding id，并且所有 padding 位置的 attention mask 必须为 `0`。
- 前缀、非空 caption body 和后缀 token 的 attention mask 均为 `1`。
- 与 Krea 2 一致，固定长度序列按 `prefix + caption + masked padding + suffix` 构造，使 suffix 保留在输出条件的末端。
- Qwen 前向完成后移除 prefix 对应的 hidden states；保留 caption 区、masked padding 区和 suffix hidden states。prefix 虽不交给生成主干，仍通过 Qwen 的因果上下文影响后续 hidden states。
- 当前 Qwen3.5 tokenizer 与 Krea 2 的 Qwen3-VL tokenizer 不同，因此不得复制 Krea 2 的硬编码 `prefix_idx=34` 和 `suffix_start_idx=5`。启动时用锁定 tokenizer 计算并断言 prefix/suffix token 数，协议文本或 tokenizer hash 变化即使旧 token 扫描失效。
- 本项目最终锁定 `text_condition_max=512`，但这是基于本项目扫描结果的独立决定，不是从 Krea 2 自动继承。四卡吞吐/显存基准只决定 dense/varlen执行路径。
## CFG 与推理
- CFG 空条件不是单独一个 `<|endoftext|>`，而是 `caption_body=""` 后经过同一完整模板；system、user 和 assistant 边界仍存在。
- 显式负面提示词也放入同一个 `caption_body` 位置，不创建第二套模板。
- 训练整体 dropout 产生空字符串 body；训练和推理内部自动包装，用户始终只输入普通 prompt。
- 所有输入都必须断言不含 `<think>` 与 `</think>`；不执行生成，不产生 chain-of-thought。
# 03-C 已接受：Caption body 序列化
- 对完成 dropout、candidate 删除、类别块 shuffle、类别内 shuffle 和下划线转空格后的所有非空 tags，使用精确分隔符 `, ` 依次连接。
- 不在文本中加入 `Tags:`、`Description:`、类别名或其他字段标签；`nsfw` 与四类 tags 均作为普通 tag 进入同一个逗号分隔序列。
- NL 若存在，始终位于全部 tags 之后；tags 与 NL 同时存在时插入两个连续换行符 `\n\n`，形成清晰的段落边界。
- 只有 tags 时，body 就是逗号分隔的 tags，不追加换行；只有 NL 时，body 直接等于 NL，不添加前导换行；两者都为空时 body 是空字符串。
- 不主动改写 NL 的首尾标点，不为 NL 自动添加句号；仅执行既定的首尾空白清理。
- body 组装必须先在字符串层完成，再整体放入 03-B 的 `{caption_body}` 位置并由锁定 tokenizer 编码。
```plain text
tags + NL: tag one, tag two, tag three\n\nA natural-language description.
tags only: tag one, tag two, tag three
NL only: A natural-language description.
empty: ""
```
# 已取代方案
- 03-C 最初批准的单换行 tags/NL 边界已被双换行 `\n\n` 段落边界取代。
- “原始 caption + `<|endoftext|>`”和“CFG 空条件仅一个有效 `<|endoftext|>`”已由本决定取代。
- Anima 式无 start/end 的原始 Qwen 输入不采用。
# 已接受风险
- 该权重经历过第三方 reasoning 蒸馏与额外处理，其 hidden-state 几何是否优于官方权重没有针对本项目的证据。此风险已知并接受。
- 风险处理方式是先做低成本表征与条件跟随验证；验证失败时记录失败原因并重新讨论模型来源，不静默替换为官方版本。
# 必须通过的实现检查
- [ ] 启动时打印并保存仓库标识、修订名、四项文件 hash、Transformers 版本及线性注意力依赖版本。
- [ ] 单元测试断言所有文本编码器参数无梯度、`use_cache=False`、视觉路径调用次数为 0。
- [ ] 对最终输入批次扫描，断言 `<think>` 与 `</think>` token id 出现次数为 0。
- [ ] 相同输入在 eval 模式重复前向，输出在约定数值容差内一致。
- [ ] 在单张 RTX 5090 上测量文本前向显存、吞吐和与数据加载重叠后的占比，再决定是否需要缓存；当前保持在线编码。
# 03-D 已接受：文本上限、长度桶与 Artist segment
- 固定 `text_condition_max=512`；该长度是去掉 34-token prefix 后的全部 condition tokens，包含 5 个 suffix tokens和在线 Artist 辅助 segment。完整 Qwen 最大长度固定为 546。
- 固定 `text_buckets=[64,128,192,256,320,384,448,512]`，对应完整 Qwen dense lengths `[98,162,226,290,354,418,482,546]`。
- 扫描截面采用当前结果：p50=169、p90=307，超过512约0.516%。超限时先保留协议边界和Artist辅助segment，再裁NL尾部与低优先级主文本tags；禁止半个tag/token。
- 加入448桶后，平均condition bucket=218.12、平均完整Qwen长度=252.12、平均padding=31.35 tokens、长度平方代理=71,442；相对无448桶，padding下降约4.1%、平方代理下降约1.9%，因此保留448。
- Artist不进入主文本caption，只进入同一Qwen序列末尾的辅助segment；serializer在线返回segment/token indices。Artist路径变化不触发重新扫描，复用本结果。
- 34/5 token计数、Qwen/tokenizer hash和协议版本必须在启动时断言并写入resolved config。目标机只比较dense/varlen吞吐、显存和数值一致性，不重新选择上限或桶。
# 变更记录
- 2026-07-27：用户确认继续使用指定的 ModelScope Qwen3.5-2B checkpoint，不采用官方版本；其余 03-A 边界全部批准。
- 2026-07-27：用户确认采用 Krea 2 式手工 chat framing；取消有效 `<|endoftext|>` 终止符，CFG 空条件改为同模板的空 caption body。
- 2026-07-27：用户批准 caption body 使用逗号加空格连接 tags，并用单个换行分隔末尾 NL；不添加字段标签。
- 2026-07-27：用户将 tags/NL 边界修订为双换行 `\n\n`，形成段落分隔；仅 tags、仅 NL 与空 body 不添加多余空行。
- 2026-07-29：用户锁定 condition 512 和 `[64,128,192,256,320,384,448,512]` 八桶；Artist只走style辅助segment且不触发重新扫描。
</content>
</page>
