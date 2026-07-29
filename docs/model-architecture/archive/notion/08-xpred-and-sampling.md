Here is the result of "view" for the Page with URL https://app.notion.com/p/3abae967ecf28167a869fb61c5ff0e96 as of 2026-07-29T06:39:51.577Z:
<page url="https://app.notion.com/p/3abae967ecf28167a869fb61c5ff0e96">
<ancestor-path>
<parent-data-source url="collection://69ca66ff-43e7-4128-bb7b-9f3751506705" name="组件决策记录"/>
<ancestor-2-database url="https://app.notion.com/p/250be554eacc40219065073dfcf66fd7" title="组件决策记录"/>
<ancestor-3-page url="https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff" title="新模型架构决策中心"/>
</ancestor-path>
<properties>
{"date:决定日期:is_datetime":0,"date:决定日期:start":"2026-07-28","url":"https://app.notion.com/p/3abae967ecf28167a869fb61c5ff0e96","决策编号":"ARCH-9","序号":8,"影响":"高","标签":["架构","训练"],"状态":"待验证","组件决策":"08 x-pred 目标与采样系统"}
</properties>
<content>
<callout icon="✅" color="green_bg">
	**架构决定已批准，待实现验证。** 08-A clean-latent loss、08-B Heun-50 sampler、08-C CFG 与 08-D PMA 权重平滑均已批准。
</callout>
# 上游接口
- [02 图像表示与 Mage-VAE](https://app.notion.com/p/3aaae967ecf2816096f0ea37634a2f7e)：官方 Mage-VAE、posterior mean、原生 `128ch @ H/16`、在线编码、无额外 patchify。
- [07 Timestep 与全局条件](https://app.notion.com/p/3abae967ecf2815b84fbda3c707728c2)：网络最终输出 image span 的 `x_pred`，`prediction_type=x`、`out_channels=128`。
- 用户已指定 x-pred 与采样实现以固定参考项目 [JLT](https://github.com/akatsuki-neo/JLT) 为准。
# 08-A：clean-latent 噪声路径与 loss
## JLT 实际定义
令 Mage-VAE posterior mean 为 clean latent `x`，采样独立高斯噪声：
$$
\epsilon \sim \mathcal{N}(0,I)
$$
每个样本采一个 scalar timestep：
$$
u\sim\mathcal{N}(-0.8,0.8^2),\qquad t=\operatorname{sigmoid}(u)
$$
线性插值路径固定为：
$$
z_t=t x+(1-t)\epsilon
$$
因此 `t=0` 表示纯噪声端，`t=1` 表示 clean latent 端，不能与常见的反向记号混用。
网络直接输出：
$$
\hat{x}=f_\theta(z_t,t,c)
$$
但 JLT 并不直接计算无权重的 `MSE(x_pred,x)`。它将 target 与预测都转换到 velocity 空间：
$$
d(t)=\max(1-t,0.05)
$$
$$
v_{target}=\frac{x-z_t}{d(t)},\qquad v_{pred}=\frac{\hat{x}-z_t}{d(t)}
$$
$$
\mathcal{L}=\operatorname{MSE}(v_{pred},v_{target})
$$
这等价于：
$$
\mathcal{L}=\frac{\lVert \hat{x}-x\rVert^2}{\max(1-t,0.05)^2}
$$
即 x-pred output 加上 capped endpoint weighting；最大权重为 `1/0.05²=400`。实现成普通 x-MSE 会改变算法，不允许静默替换。
## 08-A 最终配置（已批准）
- `prediction_type: x`。
- `flow_matching: false`；不启用 JLT 的 direct velocity ablation 分支。
- `P_mean: -0.8`、`P_std: 0.8`。
- `noise_scale: 1.0`，训练噪声与采样初始噪声都使用单位高斯。
- `t_eps: 0.05`，target、prediction 和推理 x-to-v 转换必须调用同一个 helper，禁止三处复制出不同公式。
- baseline 每张图只采一个 scalar `t`，该样本全部 image tokens 共用；`async_timesteps: false`。tokenwise timestep 属于额外训练技巧，不进入首版。
- `z_t` 构造、x-to-v 转换、平方误差和 reduction 使用 FP32；DiT forward 保持 BF16。
- 每个样本先对其所有有效 latent token 和 128 channels 求平均，再对 local batch 求平均；不能对 packed 全部元素直接全局平均，否则大图或较长 image span 会获得更高权重。
- DDP/FSDP 梯度按正常 global batch 平均；最后一个不足 batch 需保持等权语义。
- 不增加 min-SNR、P2、perceptual、VAE reconstruction、velocity auxiliary 或 feature-alignment loss。
## Mage latent 尺度
Microsoft 官方 Mage-Flow 的 VAE config 和调用路径没有 `scaling_factor`、`shift_factor`、`latent_mean` 或 `latent_std`；`vae.encode()` 的输出直接送入生成主干，生成输出也直接送入 `vae.decode()`。因此：
- 保持官方 Mage latent 原生尺度，不加入 SD/FLUX 风格的外部缩放或逐通道标准化。
- `noise_scale=1.0` 按 JLT 固定；不得为了让 latent 看似 unit variance 临时改变 decode 接口。
- 正式训练前对固定 50k–100k 个训练 crop 记录全局及逐通道 mean/std、P1/P50/P99、绝对值最大值和 BF16 overflow；该扫描只做诊断，不根据结果自动归一化。
- 若发现非有限值或极端尺度导致 200–1000 step smoke 持续不稳定，应重新打开组件 02/08 的 latent convention，而不是在 dataloader 中静默修正。
# 候选与否决
## A. JLT x-output + velocity-space MSE，推荐
完全保持参考实现的 target geometry、endpoint weighting 和采样转换，模型输出接口也与 07 一致。
## B. 直接无权重 x-MSE
虽然同样预测 `x`，但删除 `1/max(1-t,t_eps)^2` weighting，不是当前指定的 JLT 训练目标，不采用。
## C. direct v-pred flow matching
是 JLT 代码中的对照分支，不是已选 clean-latent prediction；会改变 output head 语义和 checkpoint protocol，不采用。
## D. tokenwise asynchronous timesteps
需要每个 image token 独立 timestep 条件，并改变当前 07 的 per-sample global condition broadcast；作为额外技巧推迟，不进入 baseline。
# 08-A 最终决定
批准原生 Mage latent、JLT `P_mean=-0.8/P_std=0.8`、`noise_scale=1`、`t_eps=0.05`、x-output 转 velocity-space FP32 MSE、每样本等权 reduction，并关闭 direct-v 与 tokenwise timestep。
# 08-B：ODE 时间网格与采样器
## JLT 固定参考行为
采样从单位高斯开始：
```plain text
z_0 ~ N(0, I)
timesteps = linspace(0, 1, num_steps + 1)
```
每次模型输出 `x_pred` 后，用与训练完全相同的 helper 转为向量场：
$$
v(z,t,c)=\frac{x_{pred}(z,t,c)-z}{\max(1-t,0.05)}
$$
Euler：
$$
z_{next}=z+(t_{next}-t)v(z,t,c)
$$
Heun：
$$
v_1=v(z,t,c)
$$
$$
z_E=z+(t_{next}-t)v_1
$$
$$
v_2=v(z_E,t_{next},c)
$$
$$
z_{next}=z+(t_{next}-t)\frac{v_1+v_2}{2}
$$
JLT 对前 `num_steps-1` 个区间使用 Heun，最后一个区间固定使用 Euler。这样最后一次模型求值发生在 `t=1-1/num_steps`，不会在 clean endpoint `t=1` 再评估网络。
## B. JLT Heun-50，推荐
正式评估和默认高质量推理固定：
- `sampler: heun`。
- `num_steps: 50`。
- `time_schedule: linear`，包含端点 `0` 和 `1`。
- 前 49 个区间 Heun，最后 1 个区间 Euler。
- `t_eps: 0.05` 与训练共享配置和代码，不允许 sampler 单独覆盖。
- 初始噪声 `noise_scale=1.0`；不增加 churn、ancestral noise 或每步随机噪声。
- 不做 Karras、log-SNR shift、resolution-dependent shift、dynamic thresholding、x-pred clipping 或 latent clipping。
- 最终 `z_1` 直接按官方 Mage latent convention 送入 decoder，不再缩放。
数值实现：
- ODE state、`t`、`dt`、x-to-v 转换和 Euler/Heun 累加使用 FP32。
- 每次 DiT forward 前将 `z` 转为 BF16，`x_pred` 返回后立即转回 FP32；这不改变 solver 公式。
- 使用 per-sample seed 生成初始噪声，使同一 seed 的输出不依赖 batch size、GPU rank 或 CFG 是否合批。
- conditional/unconditional 路径必须复用相同的 `z`、`t`、size、aspect 和初始噪声。
计算量：
- Heun-50 的向量场评估次数为 `2×49+1=99 NFE`。
- 不使用 CFG 时为 99 次 DiT forward。
- JLT 参考代码将 conditional/unconditional 分开调用；启用 CFG 时为 198 次 DiT forward。
- 若显存允许，可把 conditional/unconditional 沿 batch 维合并，一次 forward 产出两支；数学结果不变，仍为 99 NFE，但每次 forward 的 token batch 约翻倍。
- 预览可提供 `Euler-20` 或 `Heun-20` 作为非验收快速模式，但任何质量对比、checkpoint 选择和正式报告必须使用 Heun-50。
## 其他候选
### A. Euler-50
NFE 仅 50，速度更快，但相对 JLT 最终默认会引入 solver quality 变化；只保留为 correctness/debug baseline，不作为正式默认。
### C. 最后一段也使用 Heun
需要在 `t=1` 评估 x-to-v，依赖 denominator clamp 解释 endpoint 向量场，也偏离 JLT；不采用。
### D. DPM/Karras/自适应 ODE solver
可能以更少 NFE 达到相似质量，但需要独立调参与评估，违反当前“采样依据 JLT”和减少消融的约束；推迟到模型收敛后，不进入首版。
# 08-B 最终决定（已批准）
采用 JLT 的 linear-time Heun-50 + final Euler，FP32 solver state，正式评估固定 99 NFE；快速预览模式不得混入模型质量验收。
# 08-C：Classifier-Free Guidance
## 训练侧 unconditional 定义
训练时保留已批准的全条件 dropout：
- 每个样本以 `0.1` 概率触发 global unconditional dropout。
- 触发后 `caption_body=""`，再经过组件 03 的同一完整 Krea 2 式 framing；不是单独 EOT，也不是零长度 text sequence。
- 对应的 4 个 artist style slots 必须使用组件 04 已批准的 4 个 learned null style tokens；不得保留原样本 artist style 旁路。
- timestep、size、aspect、image latent 和噪声不改变。
- 各字段独立 dropout、自然缺失以及 candidate 删除仍可能额外产生空 body；因此实际 unconditional 比例允许高于 10%，但必须单独记录“global dropout 命中率”和“最终空条件率”。
- unconditional 样本使用同一个 x-pred loss，不增加单独 CFG loss。
## 推理侧条件分支
默认负分支：
```plain text
caption_body = ""
-> 完整 system/user/assistant framing
-> frozen Qwen
-> 主文本聚合器
-> 4 learned null style tokens
```
正分支使用同一结构化序列化协议：主文本与Artist辅助segment分离，Artist只生成style tokens。若用户提供negative prompt，则negative主文本替代空body，并从其结构化Artist字段/segment构造负分支style tokens；没有Artist时仍使用null style tokens。
### 来源边界
- 完整 system/user/assistant prefix/suffix 直接来自 [Krea 2 官方 encoder.py](https://github.com/krea-ai/krea-2/blob/main/encoder.py)，不是本项目自行设计。
- Krea 2 官方 [sampling.py](https://github.com/krea-ai/krea-2/blob/main/sampling.py) 在没有 negative prompt 时显式设置 `negative_prompts=[""]`，再调用同一个 encoder。因此“空 body 仍经过完整模板”也是 Krea 2 官方推理行为。
- 本项目只做两项适配：用锁定的 Qwen3.5 tokenizer 动态计算 prefix/suffix token 数；给自定义 artist style 旁路增加 4 个 learned null style tokens。后者不是 Krea 2 公共模型的结构。
- `no-thinking` 不是先生成 `<think>`/`</think>` 再删除；实现完全不调用会自动插入 thinking wrapper 的 generation chat-template 路径，而由统一 serializer 直接构造无 thinking token 的 assistant prefix。训练与推理输入中这两个 token 的出现次数必须始终为 0。
- Krea 2 CLI 的 `guidance=4.5` 使用 `cond + g*(cond-uncond)`；本组件采用 JLT 的 `uncond + s*(cond-uncond)`。两者参数定义相差 1，不能把 Krea 的 4.5 直接当作本项目的 `cfg_scale`。
cond/negative 的 Qwen hidden states、文本聚合结果和 style tokens 在 ODE loop 外各计算一次，并在所有 99 NFE 中复用；不得每个采样步重复执行 Qwen。
## JLT guidance 公式
先对两支 `x_pred` 分别执行与 08-A/08-B 相同的 x-to-v：
$$
v_{pos}=\frac{x_{pos}-z}{\max(1-t,0.05)},\qquad
v_{neg}=\frac{x_{neg}-z}{\max(1-t,0.05)}
$$
再计算：
$$
v_{cfg}=v_{neg}+s(v_{pos}-v_{neg})
$$
- `s=1` 精确退化为 conditional velocity，不需要运行 negative 分支；实现应走单分支快速路径。
- 默认初始 `cfg_scale=2.9`，沿用 JLT 正式评估配置；它是 sampler/runtime 参数，不写死在模型权重。
- 初始 `cfg_interval=[0,1]`，即所有实际网络求值点启用 guidance；区间外若未来启用，则令 `s=1`，不是切换成纯 unconditional。
- 不加入 CFG rescale、dynamic thresholding、channel-wise guidance、style-only guidance 或不同 timestep 的手工 scale schedule。
- 由于 x-to-v 是关于 `x_pred` 的仿射映射且两支共享 `z,t`，先在 x-space 线性组合再转 v 在数学上等价；实现仍统一在 velocity helper 后 guidance，以逐行对齐 JLT 和减少接口分叉。
## 合批与 packed varlen
- 默认提供 batch-concatenated CFG：把 positive/negative 视为边界独立的 `2B` 个 packed samples，复制同一份 `z,t,size,aspect`，但使用各自 text/style spans。
- FlashAttention varlen 的 `cu_seqlens` 必须保证两支及不同样本完全隔离；positive 与 negative 的文本长度可以不同。
- 若 `2B` 超过显存，回退到 sequential CFG；两条路径必须在相同输入上通过 FP32 solver tolerance 比较。
- 合批只减少 kernel launch 和 FSDP 通信调用次数，不减少理论 DiT FLOPs；实际加速由目标机 benchmark 决定。
- `cfg_scale=1` 时禁止构造 `2B` batch，避免无意义的双倍计算。
## 默认值与低成本校准
- baseline 和 checkpoint 对比固定 `Heun-50, cfg=2.9, interval=[0,1]`，同一 prompt、seed、尺寸。
- 文本条件模型与 JLT 的 ImageNet class condition 不同，因此 `2.9` 是可靠起点，不宣称是最终最优值。
- 模型形成基本条件跟随后，只做一次小型固定集校准：先用快速模式比较 `cfg={2.0,2.9,4.0}`，再用 Heun-50 复核候选；这属于推理参数校准，不训练额外模型。
- 正式报告必须记录 sampler、steps、CFG scale、interval、negative body hash 和 seed；不得只写“CFG on”。
# 08-C 最终决定（已批准）
采用 JLT velocity-space CFG：训练 global unconditional dropout=0.1；推理空 body + null style tokens；默认 `cfg=2.9`、全区间 guidance；支持数学等价的合批和顺序执行，不增加 CFG rescale。
# 08-D：模型权重平滑，EMA 与 PMA
## 名称与参考实现
- JLT 在线维护两份 FP32 EMA：`decay=0.9999` 和 `0.9996`，正式采样默认使用第一份。
- 本节的 PMA 指 **Pre-trained Model Averaging**，不是 Polyak Moving Average。它将同一预训练轨迹稳定阶段的多个 checkpoint 离线合并。
- [Krea 2 技术报告](https://www.krea.ai/blog/krea-2-technical-report) 在预训练中采用 PMA，并报告其表现与 EMA 接近，同时避免 EMA 的显著内存开销；其依据为 [Model Merging in Pre-training of Large Language Models](https://arxiv.org/abs/2505.12082)。
## A. JLT 双 FP32 EMA
最终约 1.9B 可训练参数时：
- 一份完整 FP32 EMA 约 `1.9B×4 bytes=7.6 GB`；两份约 15.2 GB。
- 若与 FSDP2 一样按 4 卡分片，双 EMA 仍约占每卡 3.8 GB 常驻显存；单 EMA 约 1.9 GB/卡。
- EMA 必须在每个 optimizer step 后以 FP32 更新；不能用 BF16 保存 `0.9999` EMA，否则 `1-decay=1e-4` 的增量会丢失精度。
- 双 EMA 的算术开销不大，但显存会直接挤压高分辨率 activation 和 CFG/验证 batch。
- JLT 的普通参数列表 EMA 不能直接复制到 FSDP2；必须实现 DTensor shard-local EMA、分片 checkpoint 和一致的增长迁移。
该方案最贴近 JLT，但 JLT 的双 EMA 不是 x-pred 数学定义的一部分，在当前显存约束下不推荐。
## B. 单个 sharded FP32 EMA
只保留 `decay=0.9999`，每卡约增加 1.9 GB；比双 EMA 稳妥，但仍增加常驻显存和 FSDP2 状态复杂度。可作为 PMA 工具链失败时的回退，不作为首选。
## C. PMA 离线 checkpoint 合并，推荐
生产训练不维护在线 EMA；正常保存模型权重快照，在同一稳定训练段结束时离线执行简单算术平均：
$$
\theta_{PMA}=\frac{1}{N}\sum_{i=1}^{N}\theta_i
$$
默认协议：
- `merge_method=simple_mean`，初始 `N=10`；不先引入可调权重、Fisher merge 或 task vector。
- 只合并**完全相同模型拓扑与参数 schema**的 checkpoint。
- 不跨越 `16→20→24` 深度增长边界，不跨越新层插入、参数重置或不兼容的配置变更。
- 优先选取同一 fixed-depth stable-LR 窗口中等间隔的 10 份模型快照；精确间隔由组件 09/12 根据阶段 optimizer steps、累计 FLOPs 和 checkpoint I/O 决定。
- 分辨率或数据混合发生变化时默认开启新的 PMA window；若要跨阶段合并，必须先证明模型拓扑、loss 和数据分布连续，不能自动执行。
- 合并所有可训练模型参数，包括 DiT、文本聚合器、style 分支、condition encoder、output head、RMSNorm 和 growth switches；排除冻结 Qwen 与 Mage-VAE。
- 逐 shard/逐 tensor 流式读取，使用 FP32 accumulator，最后输出 BF16 模型权重；不一次将 10 个 checkpoint 全部载入内存。
- PMA 产物是独立 model artifact，带 source checkpoint IDs、step/FLOPs、权重、代码/config hash 和 merge manifest。
- 续训恢复默认仍使用最新 raw checkpoint 对应的 optimizer/RNG/data state；不得把 PMA 权重与某个旧 optimizer state 拼接。
- 若 PMA 用作下一阶段初始化，作为显式 stage transition，并由组件 09 决定 optimizer state 重置/迁移和新增层初始化。
## 成本
- 训练每 step 的 GPU 显存与计算增量近似为零。
- 1.9B BF16 模型每份约 3.8 GB；10 份模型-only 快照约 38 GB，不含文件系统开销，符合当前存储量级，但必须与 300–500 GiB 数据缓存分别设 quota。
- 一次 10-checkpoint merge 约读取 38 GB 并写出约 3.8 GB，属于阶段边界离线 I/O，不占正式训练 step。
- 模型-only snapshot 与 optimizer/FSDP resume state 继续遵守组件 11 的文件分层；PMA 不要求复制十份 optimizer state。
## 验证与回退
1. 对 raw latest 与 PMA-10 使用同一固定 prompt、seed、Heun-50、CFG 和分辨率评估 loss、条件跟随、审美与伪影率。
2. PMA 结果出现明显退化时，先检查是否跨越增长/阶段边界或混入异常 checkpoint；不直接增加复杂 merge 算法。
3. 工具链必须用两个相同 checkpoint 合并得到 bitwise/容差等价结果，并验证分片 merge 与 consolidated merge 一致。
4. PMA 失败或阶段快照不足时发布 raw latest；需要在线平滑回退时只启用单个 sharded FP32 EMA `0.9999`，不启用 JLT 双 EMA。
# 单卡 256 前置训练与 PMA 边界
- 计划先以单张 RTX 5090 在 256 分辨率执行前几个 epoch，用于在进入四卡高分辨率阶段前暴露数据、loss、条件、采样和长期稳定性问题。该阶段产生有效优化更新，属于正式训练阶段，不再归类为 200–1000 step smoke test。
- PMA 的数学定义与 GPU 数量无关；只要 checkpoint 来自同一参数轨迹、模型拓扑和稳定训练窗口，单卡阶段同样可以使用 PMA。
- 单卡 256 阶段若形成足够长的 stable-LR 窗口，可在该窗口内部生成 PMA artifact，用于固定集评估和判断是否进入下一阶段；不得为了凑足 10 份而混入 warmup、异常 loss 区间或过密的近重复 checkpoint。快照不足时直接评估 raw latest。
- 默认仍以 raw latest 及其配套 optimizer、scheduler、RNG 和 dataloader state 恢复训练。PMA 不与任意一个旧 optimizer state 拼接；若明确采用 PMA 初始化下一阶段，必须作为显式 stage transition，定义 optimizer state 的重置或迁移。
- 不把单卡 256 checkpoint 与四卡/更高分辨率 checkpoint 放入同一个 PMA window；深度增长、新层插入或参数 schema 变化后必须新开窗口。
- 单卡训练无法覆盖 FSDP2 collective、四卡 shard restore、global batch 语义和网络存储并发。因此转四卡后仍必须先执行短分布式验证，再进入昂贵训练段。
- 从单卡转四卡时，学习率与 scheduler 必须按 global batch 和累计有效样本数定义，不能按 local step 直接续接。训练日志除 epoch 外必须同时记录 consumed valid samples、optimizer steps、image tokens 和累计 FLOPs；11M 数据上的一个 epoch 等于约 1100 万个有效样本，阶段退出不能只以“前几个 epoch”表述。
# 08-D 最终决定（已批准）
采用 PMA-10 simple mean 作为唯一默认权重平滑方案，不维护在线 EMA。PMA 可用于单卡 256 正式前置训练，但仅在同拓扑、同稳定训练窗口内合并；raw checkpoint 负责恢复，PMA artifact 负责评估/发布。跨 GPU 数量本身不禁止 PMA，跨拓扑、增长边界或不连续训练阶段禁止自动合并。
</content>
</page>
