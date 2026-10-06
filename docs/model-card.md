# 模型说明与发布信息

## 用途与范围

SakuraMoon 用于动漫风格文本到图像研究，包含训练、恢复、采样和离线评估代码。它不是现成的通用 Diffusers pipeline；其 trainable composite、文本 framing、编码器资产和 checkpoint 格式构成一个整体。

当前代码中的生产结构是 20 层、hidden 2560 的 PackedDiT，输入/输出为 128 通道 latent，使用 x-prediction 流匹配。其他深度由稳定 slot 机制支持，具体结构应读取模型产物中的架构描述。详见[架构文档](architecture.md)。

## 必须随发布说明的资产

| 资产 | 当前可确认的信息 | 正式发布还需提供 |
|---|---|---|
| Trainable composite | safetensors 分片；含 DiT、文本适配、条件编码，可能含 iREPA 辅助投影器 | 发布 ID、训练阶段、分辨率、采样建议、来源 checkpoint |
| 文本编码器 | Qwen3.5 架构，24 层、hidden 2048；固定 tokenizer framing | **实际派生权重的来源、版本及授权范围** |
| Mage-VAE | 本地实现按 Microsoft Mage 的特定 checkpoint 结构加载；128 通道、16 倍下采样 | 训练使用的确切权重来源和获取方式 |
| PE-Spatial 教师 | iREPA 训练辅助，推理不需要 | 启用 iREPA 时的教师版本与许可 |
| 数据集与 caption | 配置中引用分片清单和 ModelScope 数据源；训练有 tags、自然语言与角色/风格条件 | 数据来源、授权、过滤方法、可再分发范围和已知偏差 |

**文本编码器来源尚不能仅凭文件夹名确定。** 本轮核对的部署资产 README 描述了第三方蒸馏/修改来源，而非未经修改的官方 Qwen 权重。README 的描述仍需作者确认；不能指示复现者随意下载官方 `Qwen/Qwen3.5-2B` 替代后宣称结果等价。

代码加载器只从用户准备的本地目录读取资产，不自动下载“看起来相近”的替代权重。

## 训练与推理约定

- 主目标是 JLT 时间采样下的速度空间 MSE，模型输出干净 latent。
- 可选 iREPA 使用冻结视觉教师和训练中的图像 hidden 对齐，仅影响训练。
- CMuon / AdamW8bit 的精度、状态和数值保护属于训练配置；切换硬件不代表可以忽略优化器恢复契约。
- 默认示例 profile 是 Euler 28、Heun 25 和 Heun 50；新 checkpoint 应以自身 `resolved_config.toml` 为准。
- 当前推理程序输出 PNG 和完整采样参数记录。种子一致不保证不同设备/内核逐像素一致。

## 已知限制

NVIDIA 上的完整模型推理已经过实际权重验证，但生产多卡训练仍需要单独验证。长宽比、超出训练分辨率的生成质量、文字渲染、复杂组合提示词和风格/角色泛化能力没有因“能出图”就得到质量保证。

代码提供 FID/IS/KID/CMMD 和概念条件评估能力；这里不沿用历史 `reports/` 中的数值作为当前模型的新评估结论。对外发布指标应注明模型版本、真实图像集合、提示词集合、采样参数、样本数与随机种子。

## 许可证与第三方来源

根目录目前没有为 SakuraMoon 自有代码指定许可证。作者应在正式发布前确定代码与模型权重各自的许可证，并核对训练数据及派生文本编码器的授权。此整理不代替作者作许可选择，也不把第三方组件的许可证自动套用到整个项目。

已保留的第三方来源入口：

- [Microsoft Mage](https://github.com/microsoft/Mage)：Mage-VAE 结构；本地实现的来源 commit 见 `src/sakuramoon/encoders/mage_vae.py`。
- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)：架构/原始模型参考；不作为部署派生权重已确认来源。
- `src/sakuramoon/pe_spatial/NOTICE` 与 `LICENSE.PE`：仓库内 PE 视觉编码器的现有声明。

对外发布前还应把服务器上未同步到 Git 的训练策略变更与发布代码核对一致；仅保存配置文件而遗漏对应的 caption 或数据服务代码，会使复现行为不同。
