Here is the result of "view" for the Page with URL https://app.notion.com/p/3aaae967ecf281ba8f73fac2f9e4c4f3 as of 2026-07-27T08:05:45.566Z:
<page url="https://app.notion.com/p/3aaae967ecf281ba8f73fac2f9e4c4f3" icon="✅">
<ancestor-path>
<parent-data-source url="collection://69ca66ff-43e7-4128-bb7b-9f3751506705" name="组件决策记录"/>
<ancestor-2-database url="https://app.notion.com/p/250be554eacc40219065073dfcf66fd7" title="组件决策记录"/>
<ancestor-3-page url="https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff" title="新模型架构决策中心"/>
</ancestor-path>
<properties>
{"date:决定日期:is_datetime":0,"date:决定日期:start":"2026-07-27","url":"https://app.notion.com/p/3aaae967ecf281ba8f73fac2f9e4c4f3","决策编号":"ARCH-1","序号":1,"影响":"高","标签":["架构","训练","数据","系统"],"状态":"已接受","组件决策":"01 约束、预算与验收标准"}
</properties>
<content>
<callout icon="✅" color="green_bg">
	**决定：**项目按单机 4×RTX 5090、90–180 天、11M 二次元数据、512 最低成品分辨率，从零训练一个以 tag 控制为第一优先级的基础模型。采用固定宽度、渐进增加少量 block 的增长原则；具体宽度与层数留待组件 06/09。
</callout>
# 背景
项目目标是在有限但可持续的消费级四卡算力上，从零训练二次元垂类文生图模型。数据已整理为约 11M 样本的 WebDataset，经网络存储读取。项目不要求追平现有模型，但将 Anima 作为理想效果参考。
# 已接受约束
<table fit-page-width="true" header-row="true">
<tr>
<td>项目</td>
<td>决定</td>
</tr>
<tr>
<td>训练方式</td>
<td>生成主干从零训练，不继承现有 DiT 权重</td>
</tr>
<tr>
<td>硬件</td>
<td>单机 4×RTX 5090，每卡 32GB</td>
</tr>
<tr>
<td>训练周期</td>
<td>90–180 天</td>
</tr>
<tr>
<td>名义算力预算</td>
<td>8,640–17,280 GPU-hours</td>
</tr>
<tr>
<td>规划有效预算</td>
<td>按 80% 可用率估算约 6,900–13,800 GPU-hours</td>
</tr>
<tr>
<td>数据</td>
<td>约 11M 二次元垂类 WebDataset</td>
</tr>
<tr>
<td>数据位置</td>
<td>网络存储；必须通过开训前吞吐验证</td>
</tr>
<tr>
<td>最低成品分辨率</td>
<td>512 等效面积</td>
</tr>
<tr>
<td>高分辨率</td>
<td>1024 非强制；768/1024 仅作预算充足时的可选收尾</td>
</tr>
<tr>
<td>参考效果</td>
<td>Anima，仅作方向性校准，不设必须追平的硬门槛</td>
</tr>
</table>
# 能力优先级
采用顺序门槛，而不是把所有能力混成一个加权平均分：
1. **Tag 控制是硬门槛。** 人数、角色、服装、颜色、姿态、镜头、画风等标签能力不能为了审美提升而明显退化。
2. **审美是第二目标。** 在线条、配色、细节、默认构图和严重伪影率上持续改善。
3. **自然语言是补充能力。** 支持简短描述与 tags + NL 混合输入，不以长篇复杂提示词为第一版重点。
4. **其他能力不阻塞基础模型。** 写实、长文本渲染和复杂空间推理不作为第一版发布门槛。
# 训练规模规划
在 11M 样本规模下，512 等效面积的四卡聚合吞吐用于判断架构是否符合周期：
<table fit-page-width="true" header-row="true">
<tr>
<td>聚合吞吐</td>
<td>一轮数据时间</td>
<td>90 天有效轮数</td>
<td>180 天有效轮数</td>
</tr>
<tr>
<td>4 samples/s</td>
<td>31.8 天</td>
<td>2.3</td>
<td>4.5</td>
</tr>
<tr>
<td>6 samples/s</td>
<td>21.2 天</td>
<td>3.4</td>
<td>6.8</td>
</tr>
<tr>
<td>8 samples/s</td>
<td>15.9 天</td>
<td>4.5</td>
<td>9.0</td>
</tr>
</table>
**暂定工程门槛：**
- 完整训练路径在 512 等效面积下应达到四卡聚合至少 6 samples/s。
- 若低于 4 samples/s，必须先用分段计时定位 VAE、文本编码、网络读取或 DiT 瓶颈。VAE 预编码仅在端到端基准证明有净收益后采用；同时比较文本缓存、网络存储流水线、本地 staging，仍不足时再缩减主干。
- 吞吐统计必须包含网络读取、解码、文本编码、VAE、DiT 前反向和梯度同步。
# 网络存储门槛
WebDataset 的顺序 shard 读取适合网络存储，但不能假定网络一定够快。开训前应完成：
- 冷缓存条件下，数据管线独立持续提供至少目标训练吞吐的 2 倍，即至少 12 samples/s。
- 连续运行至少 2 小时，无长时间 shard stall、worker 超时或样本重试风暴。
- 训练顺序可复现，能够定位导致 loss spike 的具体 shard 和 sample。
- 若网络不达标，必须增加本地 shard staging/cache；不能让 GPU 长期等待网络。
# 模型增长原则
接受 SANA 1.5 的**方法**，不照搬其深度：
- 固定 hidden width，仅增加 block 数。
- 保留已训练 block；新增 block 随机初始化。
- 新 block 的 attention 输出投影和 MLP 输出投影零初始化，使新增 block 初始近似恒等。
- 不采用循环复制或整块复制初始化。
- 增深后先在原分辨率稳定训练，再单独提升分辨率。
- 不在同一个切换点同时增深、升分辨率和改变主要数据分布。
SANA 1.5 验证的是从 20 层扩展到 60 层的 Linear DiT，并报告达到相同指标时节省约 60% 训练步数；本项目仅增加少量层，**不得预设能获得相同幅度的节省**。
# 宽浅方向的架构约束
Krea 2 公开说明其采用稍宽、较浅的模型，以减少 FSDP2 通信次数并提高大矩阵效率，但没有公开最终 hidden size 和层数。其主要基础设施收益依赖 FSDP2、NVLink 和大规模集群，不能直接等价到当前无 NVLink 的四卡 DDP。
尽管如此，宽浅方向对长图像序列仍有合理优势：在相近参数量下可减少 attention 的逐层开销和激活深度。公开模型配置表明 20–30 层属于常见区间；结合从零训练风险，当前接受 **d=2048、16→20→24 blocks** 作为主候选，d=2560、12→14→16 仅保留为更激进的宽浅备选。精确数值仍由组件 06/09 最终确认：
<table fit-page-width="true" header-row="true">
<tr>
<td>候选</td>
<td>单 block 参数</td>
<td>增长示例</td>
<td>最终主干参数</td>
</tr>
<tr>
<td>d=2048, m=5504</td>
<td>约 44.3M</td>
<td>16→20→24 blocks</td>
<td>约 0.71B→0.89B→1.06B</td>
</tr>
<tr>
<td>d=2560, m=6912</td>
<td>约 69.5M</td>
<td>12→14→16 blocks</td>
<td>约 0.83B→0.97B→1.11B</td>
</tr>
</table>
这说明“只增加少量层”仍会显著改变参数量：d=2560 时每增加 1 层约增加 69.5M，每增加 2 层约增加 139M。参数增幅可控，但不能视为不变。
# VAE 与主干输入接口约束
- 直接使用 [Microsoft 官方 Mage-Flow](https://huggingface.co/microsoft/Mage-Flow) 发布的 Mage-VAE 权重与官方实现，并在训练期间冻结。Mage-VAE 是通用图像 VAE，不是二次元特化 VAE；二次元线稿、眼睛、手部、纯色区域和高频纹理的重建适配性必须在组件 02 单独验收。
- 采用官方原生 latent：`128 channels @ H/16 × W/16`，不执行额外的 latent packing、unpacking 或数值分布变换。主干输入对应 `in_channels=128, patch_size=1`。
- 官方实现已经在固定 `t=0` 下对 adaLN modulation 做常量折叠，无需为获得该优化转换 checkpoint。项目若需要统一调用接口，只允许增加不改变权重、latent 形状和数值的薄封装。
- 取消 DiT 侧额外的 `2×2 patchify`，直接把 H/16 网格展平为序列，再做 `128→hidden_size` 线性输入投影。取消 patchify 不等于取消展平和输入投影。
- 512 输入对应 `32×32=1024` 个图像 token；1024 输入对应 `64×64=4096` 个图像 token。
- 现有 WebDataset 不含 latent。是否预编码不在本阶段强制决定，组件 11/12 应实测在线 VAE 编码与离线 latent 两条完整数据路径。
- 以 512 为例，单份 BF16 原生 latent 约 256 KiB；11M 样本约 2.88 TB（未计元数据、分桶和冗余）。网络存储下预编码可能把计算瓶颈转换成 I/O 瓶颈，并限制随机裁剪，因此只能依据实测净吞吐、存储占用和增强策略决定。
# 评估与发布标准
- 固定建立二次元评估集，覆盖纯 tags、tags + NL、纯 NL、多角色与构图压力测试。
- Tag 结果作为第一排序键；只有 tag 能力不退化时，才用审美与 NL 指标选择 checkpoint。
- Anima-Base 用于 tag、风格覆盖和可控性的方向性对比；Anima-Aesthetic 用于默认审美方向性对比。
- 不规定必须战胜 Anima；记录同 prompt 盲评结果和主要差距。
- 256 阶段只用于学习语义与基础构图，不满足最终发布条件；最终模型必须在 512 等效面积上完成充分训练和评估。
# 失败止损条件
- 512 完整训练吞吐经优化后仍低于 4 samples/s。
- 四卡训练无法稳定恢复 checkpoint，或反复出现不可定位的 NaN/Inf。
- 网络存储无法满足 2 倍数据供给门槛，且没有本地 staging 方案。
- 模型增深后在预定稳定窗口内不能恢复增深前的验证质量；具体窗口由组件 09 定义。
# 备选方案
- **直接训练最终深度：**工程最简单，但前期低分辨率阶段浪费更多计算，暂不采用。
- **照搬 SANA 20→60 层：**与 VAE、softmax 单流 DiT 和四卡条件不匹配，不采用。
- **同时扩宽和扩深：**难以进行函数保持迁移，不采用。
- **1024 作为硬目标：**成本过高且不是产品要求，不采用。
# 后果
**正面：**
- 训练预算集中在 tag 学习与 512 主质量。
- 允许通过早期浅模型减少训练成本。
- 不被 1024 和通用能力拖累。
**负面：**
- 从零训练意味着无法依赖 Anima/Cosmos 的通用先验。
- 宽度一旦确定，后续不能低风险扩宽，因此组件 06/09 的宽度决策必须谨慎。
- 网络存储成为训练稳定性的外部依赖。
# 延后到后续组件
- VAE 的二次元重建验收与缩放/归一化：组件 02；原生 `128ch @ H/16`、DiT `patch_size=1` 已在本组件固定。
- 文本编码器与 tag/NL 协议：组件 03/04。
- DiT 精确宽度和 block：组件 06。
- x-pred 与 sampler：组件 08。
- 深度增长阶段与恢复窗口：组件 09。
- 分辨率预算：组件 10。
- 缓存和网络数据实现：组件 11/12。
# 依据
- [Anima 模型卡](https://huggingface.co/circlestone-labs/Anima)
- [SANA 1.5 论文](https://arxiv.org/pdf/2501.18427)
- [Krea 2 Technical Report](https://www.krea.ai/blog/krea-2-technical-report)
- [RTX 5090 官方规格](https://www.nvidia.com/en-gb/geforce/graphics-cards/50-series/rtx-5090/)
- [Mage-Flow 官方模型与 VAE 配置](https://huggingface.co/microsoft/Mage-Flow)
- [Mage-Flow 官方 Mage-VAE 实现](https://github.com/microsoft/Mage/blob/main/mage_flow/models/modules/mage_vae.py)
<empty-block/>
</content>
</page>
