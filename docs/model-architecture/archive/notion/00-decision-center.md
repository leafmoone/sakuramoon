Here is the result of "view" for the Page with URL https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff as of 2026-07-29T06:42:43.823Z:
<page url="https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff" icon="📐">
<ancestor-path></ancestor-path>
<properties>
{"title":"新模型架构决策中心"}
</properties>
<content>
<callout icon="📐" color="green_bg">
	**当前入口：会话重整 v2。** 已明确批准的现行架构以两份 v2 页面为准；原组件与本页后续内容保留论证和历史，不再直接生成训练配置。
</callout>
# 统一入口
- 现行确定方案：<mention-page url="https://app.notion.com/p/3acae967ecf281d69eb2c080768c6cd1">当前确定方案 v2（会话重整）</mention-page>
- 开放决定与执行清单：<mention-page url="https://app.notion.com/p/3acae967ecf28174bbaafbd053308860">待做与待确认清单 v2（会话重整）</mention-page>
- Artist 路径已固定为 **只进入 4-token style 分支**；serializer 在线构造并记录 token segment，主文本分支不消费 Artist tokens，不做第二次 Qwen 或离线 style cache。
- 文本预算已固定为 `text_condition_max=512` 与 8 个长度桶 `[64,128,192,256,320,384,448,512]`，Artist 去留不触发重新扫描。
- 当前只剩除 `all_condition=0.10` 外的 dropout 数值尚待用户确定；其余“待验证”项需要实现或实机证据，不要求重复确认架构。
# 历史评审（以下不作为现行配置）
这份方案的优点是主线统一：高压缩 latent、冻结文本编码器、联合图文建模、干净 latent 预测、渐进扩展，都围绕“把有限算力优先留给生成主干”展开。模块大多有公开先例，工程上也没有明显互斥。
主要问题不是单个模块错误，而是**证据外推过快**：
- Mage-VAE 很新，公开结果来自与 Mage-Flow 的协同设计；目标域上的重建质量和 latent 统计尚未验证。
- JLT 的公开验证集中在 FLUX.2 latent、130M、ImageNet 256；不能直接等价到 Mage-VAE、文本条件、约 1B 和 1024 分辨率。
- Krea 2 支持 GQA、单流、RMSNorm、轻量时间条件和多层文本特征的大方向，但草案中的具体实现并不等同于 Krea 2 的实现。
- 1024² 时 4096 个图像 token，再叠加约 1B DiT 和在线 2B Qwen 前向，与“算力有限”存在直接张力。
因此应把本文视为 **Architecture Candidate v0**，而不是“最终架构”。
# 决策状态
<table fit-page-width="true" header-row="true">
<tr>
<td>状态</td>
<td>含义</td>
<td>当前示例</td>
</tr>
<tr>
<td>方向接受</td>
<td>可以作为后续讨论基线</td>
<td>官方 Mage-VAE 原生 H/16、posterior mean、在线编码、指定社区 Qwen3.5-2B checkpoint（冻结、纯文本前向、no-thinking、Krea 2 式 framing、逗号 tags + 双换行 NL body）、WebDataset 动态 caption/dropout、单流基线、渐进分辨率</td>
</tr>
<tr>
<td>暂定</td>
<td>合理默认值，但要在组件讨论中确认</td>
<td>文本 token 上限与 padding 策略、GQA、RMSNorm、SwiGLU、x-pred</td>
</tr>
<tr>
<td>必须验证</td>
<td>没有测量结果前不得锁定</td>
<td>4096 token、在线 Qwen、层选择、双向适配器、深度增长、训练比例</td>
</tr>
<tr>
<td>推迟</td>
<td>不阻塞第一版</td>
<td>iREPA、编辑视觉分支、复杂优化器、额外结构技巧</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
</tr>
</table>
# 组件讨论顺序
<table fit-page-width="true" header-row="true">
<tr>
<td>序号</td>
<td>组件文档</td>
<td>必须关闭的问题</td>
</tr>
<tr>
<td>01</td>
<td>约束、预算与验收标准</td>
<td>GPU、总 GPU-hours、数据规模、目标质量、最大训练周期、失败止损线</td>
</tr>
<tr>
<td>02</td>
<td>图像表示与 Mage-VAE</td>
<td>重建质量、latent 统计、预处理、是否 patchify、是否预编码</td>
</tr>
<tr>
<td>03</td>
<td><mention-page url="https://app.notion.com/p/3aaae967ecf281fba3cfe0f5dc53fece"/></td>
<td>指定 checkpoint、Krea 2 式 framing、token 长度、NL/tags body 格式、CFG 空条件</td>
</tr>
<tr>
<td>04</td>
<td><mention-page url="https://app.notion.com/p/3abae967ecf28104ad74d1b843728be4"/></td>
<td>层选择、聚合方式、是否需要双向层、RoPE、gate 初始化</td>
</tr>
<tr>
<td>05</td>
<td><mention-page url="https://app.notion.com/p/3abae967ecf281de8dafc6dbecd04fe6"/></td>
<td>图文 token 排列、mask、1D/2D RoPE 维度分配、分辨率与比例元数据</td>
</tr>
<tr>
<td>06</td>
<td>Single-stream DiT Block</td>
<td>宽度/深度、GQA 比例、MLP 宽度、Norm、门控与 kernel 可用性</td>
</tr>
<tr>
<td>07</td>
<td>时间条件与输出 Head</td>
<td>时间嵌入、每层调制接口、初始化、输出预条件与 latent 通道</td>
</tr>
<tr>
<td>08</td>
<td>x-pred 目标与采样系统</td>
<td>噪声路径、timestep 分布、loss weighting、sampler、CFG、EMA/PMA</td>
</tr>
<tr>
<td>09</td>
<td><mention-page url="https://app.notion.com/p/3abae967ecf28103be8feea5e27f21e1"/></td>
<td>起始规模、插层位置、函数保持初始化、优化器状态迁移、增长触发条件</td>
</tr>
<tr>
<td>10</td>
<td><mention-page url="https://app.notion.com/p/3abae967ecf2816da90ccbee372b27c4"/></td>
<td>按 FLOPs 而非图片数分配预算、阶段切换条件、多宽高比分桶、iREPA 去留</td>
</tr>
<tr>
<td>11</td>
<td>数据与缓存管线</td>
<td>WebDataset schema、过滤去重、caption/tag 采样、latent/文本多视图缓存</td>
</tr>
<tr>
<td>12</td>
<td><mention-page url="https://app.notion.com/p/3abae967ecf281ebadadd176e1b492db"/></td>
<td>AdamW8bit 可行性、精度、checkpointing、并行策略、吞吐与恢复测试</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
</tr>
</table>
# 工作规则
1. 每次只讨论一个组件，先列接口约束、候选方案、风险和最低成本验证。
2. 结论分为“已接受”或“待验证”；“待验证”必须带明确测试、通过阈值和失败回退。
3. 每个组件闭合后，在组件决策库中新建独立文档，包含背景、决定、理由、备选、后果、实现接口和验证标准。
4. 上游决定改变时，更新受影响文档并标记取代关系，不静默覆盖历史。
# 当前最优先问题
**组件 12 已闭合，待目标机验证：** 训练系统采用 S0 单卡原生→S1 起四卡 DDP、TorchAO AdamW8bit mixed BF16/FP32 与同步 stochastic-round RNG；FA4 varlen、完整 block checkpoint、可回退 regional compile、WSD LR、canonical-FQN checkpoint/增长迁移和生产故障矩阵均已批准。实现阶段强制执行 AI/模型开发者 + Infra/性能开发者双角色独立审查，安装并验证高性能 kernel，对全路径分段计时和采样 profile，优先消除串行小算子、host sync 与 GPU idle。数据供给门槛为 12 samples/s，四卡完整 512 训练门槛为 6 samples/s，每卡峰值不超过 27.2 GB。下一步进入实现与 4×RTX 5090 canary，不再继续架构选项讨论。
# 公开依据
- [Mage-Flow / Mage-VAE 技术报告](https://arxiv.org/abs/2607.19064)
- [Krea 2 Technical Report](https://www.krea.ai/blog/krea-2-technical-report)
- [JLT: Clean-Latent Prediction in Latent Diffusion Transformers](https://arxiv.org/abs/2605.27102)
- [指定 Qwen3.5-2B checkpoint（ModelScope）](https://www.modelscope.cn/models/spawner/Qwen3_5_2b_claude_heretic_spawner)
<database url="https://app.notion.com/p/250be554eacc40219065073dfcf66fd7" inline="false" data-source-url="collection://69ca66ff-43e7-4128-bb7b-9f3751506705">组件决策记录</database>
<empty-block/>
<page url="https://app.notion.com/p/3acae967ecf281d69eb2c080768c6cd1">当前确定方案 v2（会话重整）</page>
<page url="https://app.notion.com/p/3acae967ecf28174bbaafbd053308860">待做与待确认清单 v2（会话重整）</page>
</content>
</page>
