Here is the result of "view" for the Page with URL https://app.notion.com/p/3abae967ecf281809823e98b5d91222e as of 2026-07-28T10:33:03.330Z:
<page url="https://app.notion.com/p/3abae967ecf281809823e98b5d91222e">
<ancestor-path>
<parent-data-source url="collection://69ca66ff-43e7-4128-bb7b-9f3751506705" name="组件决策记录"/>
<ancestor-2-database url="https://app.notion.com/p/250be554eacc40219065073dfcf66fd7" title="组件决策记录"/>
<ancestor-3-page url="https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff" title="新模型架构决策中心"/>
</ancestor-path>
<properties>
{"date:决定日期:is_datetime":0,"date:决定日期:start":"2026-07-28","url":"https://app.notion.com/p/3abae967ecf281809823e98b5d91222e","决策编号":"ARCH-7","序号":6,"影响":"高","标签":["架构","训练"],"状态":"待验证","组件决策":"06 Single-stream DiT 主干"}
</properties>
<content>
<callout icon="🧪" color="yellow_bg">
	**06-A 至 06-E 的主干结构继续有效。** 06-F 的早期 FSDP2 与“单卡只做短验证”方案已被组件 10/12 正式覆盖：当前为 S0 单卡正式 16层/256、S1 起四卡 DDP、TorchAO AdamW8bit。FA4 varlen、完整 block checkpoint 与 compile 边界迁移到组件 12-C，以后以组件 12 为训练系统唯一来源。
</callout>
# 范围
本组件接收组件 05 的 packed `[valid text | 4 style | image]` 联合序列。以下接口已经固定，不在本组件重新讨论：
- Single-stream、样本内全双向 attention。
- Mage-VAE 原生 image tokens，不额外 patchify。
- head dimension 必须可被 16 整除；2D RoPE 为 25% NoPE、37.5% y、37.5% x。
- QK-RMSNorm 后应用 RoPE。
- text/style/image modality embeddings 和全局尺寸元数据由组件 05/07 定义。
- 训练目标 x-pred 与采样由 JLT 参考实现约束。
# 外部基线
## Krea 2
官方推理代码的主干配置为 `hidden=6144`、`28 blocks`、`48 Q heads / 12 KV heads`、`head_dim=128`。它确实偏宽，但绝对深度仍为 28；按其 GQA、attention output gate 与 SwiGLU block 粗算，仅主干就是约 12B 量级，不能把 6144 宽度直接迁移到四卡 5090 项目。
- [Krea 2 inference.py](https://github.com/krea-ai/krea-2/blob/main/inference.py)
- [Krea 2 mmdit.py](https://github.com/krea-ai/krea-2/blob/main/mmdit.py)
## SANA 1.5
官方配置保持 `hidden=2240`：1.6B 使用 20 层，3.2B/4.8B 通过深度增长扩到 40/60 层。它支持“最终宽度先固定、再增层”，但 40/60 层不适合当前 4096 image-token 的标准 softmax attention 主干。
- [SANA 1.5 1.6B/20 层定义](https://github.com/NVlabs/Sana/blob/main/diffusion/model/nets/sana_multi_scale.py)
- [SANA 1.5 配置目录](https://github.com/NVlabs/Sana/tree/main/configs/sana1-5_config/1024ms)
## Anima
官方 base 权重结构可读出 28 个 blocks、`hidden=2048`、4× MLP；其 block 同时包含 self-attention 与 cross-attention，约 2B 总规模。它证明 28 层可接受，但不能和本项目的单流 GQA block 按层数直接等价。
- [Anima base 权重](https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/diffusion_models/anima-base-v1.0.safetensors)
# 已批准 06-A：宽度与深度
以下参数量只估算 block 主体，假定 4:1 GQA、Krea 2 式 attention output gate、SwiGLU intermediate 约 `8d/3`；不含输入/输出投影、组件 04、全局条件和 checkpoint metadata。单 block 主导参数近似为 `11.5d²`。
## A. 2048 宽，20 → 24 → 28 层
- Q/KV heads：16/4，head_dim=128。
- block 主体约：0.96B → 1.16B → 1.35B。
- 优点：激活宽度较小，28 层与 Anima 深度接近。
- 缺点：高分辨率下顺序 kernel 更多；相同参数预算时，attention 的 `N²dL` 项高于更宽更浅方案。
## B. 2560 宽，16 → 20 → 24 层，当前推荐
- Q/KV heads：20/5，head_dim=128，保持严格 4:1 GQA。
- block 主体约：1.21B → 1.51B → 1.81B。
- 16 层相对 2048×24：block 参数只增加约 4%，但 attention score 计算约减少 17%，顺序 block 数减少三分之一。
- 增到 20 层时参数和线性计算相对 16 层增加 25%；20 增到 24 层再增加 20%。
- 2560 宽矩阵更适合 5090 Tensor Core；固定宽度也让组件 04/05/07 的投影接口始终不变。
- 最终 24 层连同外围组件预计接近 1.9B；与 Anima 的容量目标接近，但单流 GQA 的图文通信路径更短。
## C. 3072 宽，12 → 16 → 20 层
- Q/KV heads：24/6，head_dim=128。
- block 主体约：1.30B → 1.74B → 2.17B。
- attention 层数更少，但每 token 激活、MLP 矩阵和输入/输出投影明显增大；12 层的迭代重整深度偏低，20 层又超过当前稳妥预算，不推荐。
# 06-A 最终决定
采用 B：
1. 从一开始固定 `hidden_size=2560`、`head_dim=128`、`Q heads=20`、`KV heads=5`。
2. 深度阶梯为 `16 → 20 → 24`；不改变宽度，不重建输入/输出投影。
3. 16 层用于 256 阶段建立表示，20 层作为主要 512 预训练容量，24 层用于后续 512 收敛与 768/1024 收尾。
4. 每次只增加 4 层，使用近似恒等接入；具体插层位置、优化器状态继承和增长时点由训练策略组件决定。
5. 24 层是同一模型的最终结构。若目标机 benchmark 证明高分辨率阶段成本过高，优先减少该阶段训练步数，不在中途改变宽度。
# 已批准 06-B：attention gate 与残差结构
这里有三种名称相近但作用不同的 gate，必须分开定义：
1. **内容 gate**：由当前 token hidden state 产生，控制 attention 输出的不同通道。
2. **timestep residual gate**：由 diffusion timestep/global condition 产生，控制整个 attention/MLP 残差分支在当前噪声级的强度。
3. **growth switch**：只用于深度增长，让新增 block 在接入瞬间严格等于恒等映射。
## A. 只用 timestep gate，不用 attention 内容 gate
结构最省参数：
`x = x + g_attn(c) × W_o(Attn(Q,K,V))`
优点是每层少一个 `d×d` 投影；最终 24 层约少 1.57 亿参数，DiT 单步预计快约 5%–9%。缺点是 token/channel 选择完全交给 attention value、输出投影和外层 timestep gate；对于 4:1 GQA 和 text/style/image 单流混合，动态通道路由能力更弱。
## B. Krea 2 式内容 gate + timestep gate + growth switch，推荐
attention 分支定义为：
```plain text
h = mod_attn(RMSNorm(x), condition)
q, k, v, z = fused_qkvg(h)
a = Attention(q, k, v)
a = W_o(a * sigmoid(z))
x = x + alpha_attn * g_attn(condition) * a
```
MLP 分支定义为：
```plain text
h = mod_mlp(RMSNorm(x), condition)
m = W_down(SiLU(W_gate(h)) * W_up(h))
x = x + alpha_mlp * g_mlp(condition) * m
```
具体约束：
- 内容 gate `z` 来自与 Q/K/V 相同的 modulated pre-norm hidden state，维度为 `d`。
- `sigmoid(z)` 在 attention 输出、`W_o` 之前逐 token、逐通道相乘，与 Krea 2 一致。
- Q/K/V/z 使用一个 fused projection 或一次 grouped GEMM，减少 kernel launch；逻辑 shape 仍分别保存，便于 checkpoint 转换。
- 内容 gate projection 不使用 bias；初始化时 `z` 近 0，因此 `sigmoid(z)` 约为 0.5，不会关闭分支。
- `g_attn` 和 `g_mlp` 是由 timestep/global condition 生成的逐通道 gate，shape 为 `[B,1,d]`；具体共享 MLP 由组件 07 决定。
- `alpha_attn`、`alpha_mlp` 是每个 block 各一个标量 growth switch。初始 16 个 base blocks 设为 1；新增 blocks 接入时设为 0，以保证共享 timestep gate 已经训练为非零后，新层仍是严格恒等映射。
- 新增 block 的 `alpha` 由训练学习或按训练策略短暂 warm up；不把它用于 CFG，也不随 timestep 改变。
- 不额外加入常驻的 per-channel LayerScale/ReZero；其功能已由 timestep gate 覆盖。
参数和速度影响：
- 在 `d=2560`、24 层时，内容 gate 增加 `24×2560² = 157,286,400` 参数。
- 相对无内容 gate 的 block 主体，参数与线性投影 FLOPs 增加约 9.5%；相对含 gate 的完整 block 约占 8.7%。
- 由于高分辨率 attention score、VAE、Qwen 和通信不受影响，预计端到端训练/推理变慢约 5%–9%，1024 阶段更接近下端。
- growth switches 总共仅 48 个标量，对参数和速度没有可测影响。
优势：
- 内容 gate 为 GQA 补充逐 token、逐通道的动态选择，尤其适合单流中语义 token、style token 与 image token 的异质混合。
- timestep gate 保留 diffusion 各噪声级对 attention/MLP 分支强度的控制。
- 独立 growth switch 解决深度增长时的函数保持问题；仅将新层的 modulation 偏置清零并不足以保证恒等，因为共享 timestep 投影在增长时已经非零。
## C. 不使用内容 gate，但给所有层增加 LayerScale/ReZero
可以稳定深层残差，但与逐通道 timestep gate 功能重叠；全程压小所有残差还可能拖慢 16 层起步阶段的学习。它也没有解决 GQA 内容选择性问题，不推荐。
# 06-B 最终决定
采用 B：
1. 保留 Krea 2 式 `sigmoid(W_g h)` attention 内容 gate。
2. 保留 attention/MLP 两个逐通道 timestep residual gates。
3. 每个 block 增加两个独立标量 growth switches；base blocks 为 1，新增 blocks 为 0。
4. 不增加额外 LayerScale/ReZero。
5. 接受最终约 1.57 亿参数和约 5%–9% 端到端计算成本，换取 GQA 单流下更强的动态通道路由和可靠的深度增长接口。
# 已批准 06-C：SwiGLU、对齐与 bias/dropout
已批准的 hidden size 为 `d=2560`。SwiGLU 使用三个矩阵：
```plain text
gate, up = split(W_gate_up(x))
hidden = SiLU(gate) * up
out = W_down(hidden)
```
其中 `W_gate` 与 `W_up` 在实现中融合成一次 `d → 2m` GEMM，checkpoint schema 仍显式记录两个逻辑 projection 或记录可无损拆分的顺序。
## A. intermediate=6784，向下取最近 128 倍数
- 相对精确 `8d/3=6826.67` 缩小约 0.63%。
- 参数和 FLOPs 最低，但主动低于等参数的标准 SwiGLU 容量；对当前质量优先目标没有明显必要。
## B. intermediate=6912，向上对齐 128/256，推荐
- `6912 = 54×128 = 27×256`。
- 相对精确 `8d/3` 增加约 1.25%；相对 6784 的 MLP 矩阵只增加约 1.89%。
- 每层 SwiGLU 参数为 `3×2560×6912 = 53,084,160`。
- 连同已批准的 4:1 GQA、attention 内容 gate 与输出投影，每个 block 主体约 76.03M 参数；16/20/24 层分别约 1.216B/1.520B/1.825B。
- 加上输入输出投影、文本适配器和全局条件模块后，最终可训练部分仍约为 1.85B–1.90B，符合 06-A 预算。
## C. intermediate=7168，对齐 512
- `7168 = 14×512`，但比精确 `8d/3` 增加 5%。
- 512 对齐不会在 RTX 5090 上稳定换来足以覆盖额外计算的收益；容量增长也偏离已批准参数预算，不采用。
# 06-C 最终决定
## SwiGLU
- 固定 `intermediate_size=6912`，配置文件直接保存整数，不在运行时重新计算。
- `gate/up` 使用 fused projection，逻辑顺序固定为 `[gate, up]`；激活使用精确 `SiLU(gate) * up`。
- `down` projection 独立；不在 SwiGLU 内再加 normalization 或卷积。
- 优先使用 fused SwiGLU kernel；没有合适 kernel 时必须有数值等价的 PyTorch fallback。
## Bias
block 内以下线性层全部 `bias=False`：
- Q、K、V、attention content gate、attention output projection。
- SwiGLU gate、up、down。
- 采用 RMSNorm 和显式 timestep scale/shift 后，block linear bias 信息重复；关闭 bias 也更利于 projection fusion。
输入投影、最终 x-pred head 是否保留 bias 由它们各自的小节决定，不受这里约束。
## Dropout
以下全部设为 0：
- attention probability dropout。
- attention output dropout。
- SwiGLU activation/output dropout。
- residual dropout。
- DropPath/stochastic depth。
原因是 11M 已去重数据、在线字段/CFG dropout 和多分辨率训练已经提供足够随机性；block dropout 会降低 fused kernel 效率，并干扰后续函数保持式增层。条件 dropout 属于组件 03/11 的数据协议，继续保留，不算 block dropout。
# 已批准 06-D：RMSNorm、QK-Norm 与残差初始化
## A. RMSNorm 数值协议
- 所有 RMSNorm，包括主干 pre-RMSNorm、Q/K RMSNorm 和最终主干 norm，统一 `eps=1e-6`。
- 输入先转 FP32 计算平方均值、倒数平方根和缩放，再转回输入 dtype；BF16 主干输出保持 BF16。
- norm weight 以 FP32 保存和更新；如果 FSDP/mixed-precision wrapper 自动 cast，forward 必须显式恢复 FP32 累计并在返回前 cast 到 BF16。
- 不使用 LayerNorm，不加入 mean subtraction，不使用 post-norm。
- norm weight 不参与 weight decay，具体 optimizer 分组在训练组件记录。
## B. QK-Norm 参数
- Q 与 K 使用两个独立的 RMSNorm：`q_norm` 和 `k_norm`。
- 每个 norm 的参数 shape 为 `[head_dim]`，在所有 heads 间共享；不使用 `[num_heads, head_dim]` 的 per-head 参数。
- 不对 V 做 RMSNorm。
- 顺序固定为：Q/K projection → Q/K RMSNorm → 2D RoPE → attention kernel。
- Q/K 的 norm weight 不共享；它们的统计尺度可能不同，共享会减少自由度但没有明确收益。
- 输出进入 FlashAttention/SDPA 前强制为 BF16，避免 FP32 norm weight 或 RoPE buffer 将整个 attention kernel 上推到 FP32。
## C. 残差初始化
基础 16 层从头训练时：
1. 主干 pre-RMSNorm weight 初始化为 1。
2. timestep modulation 的最终 scale/shift/gate projection 采用 zero initialization；因此初始 block 近似恒等。
3. 已批准的 base-block `alpha_attn=alpha_mlp=1` 保持不变；初始残差由 zero-init timestep gates 关闭，而不是再叠加一套 LayerScale。
4. attention output projection、SwiGLU down projection 使用常规 BF16-compatible Xavier/Kaiming 初始化，不再做全程 `1/sqrt(L)` 缩放。
增层时：
1. 新 block 的 `alpha_attn=alpha_mlp=0`，其余权重可从相邻已训练 block 复制或按训练策略初始化。
2. 由于 growth switch 独立于 timestep，已有共享 timestep modulation 即使非零，新 block 仍严格等于恒等映射。
3. alpha 在新阶段开始后学习打开；不修改旧 block 的 alpha，不重置已有 optimizer state。
## D. 精度与速度影响
- FP32 RMSNorm 累计和独立 QK-Norm 的参数/计算量相对 2560 宽主干可忽略，通常低于 1% 端到端开销。
- 主要实现要求是 norm/rope 后显式 cast BF16，否则会破坏 FlashAttention/SDPA 的 kernel 选择。
- `eps=1e-6` 与当前 JLT/HDM 参考实现一致；统一配置便于 checkpoint、复现和故障排查。
# 06-D 最终决定
采用上述 RMSNorm/QK-Norm 和残差初始化协议：
- 全部 RMSNorm `eps=1e-6`，FP32 累计，BF16 输出。
- Q/K 独立 head_dim RMSNorm，跨 heads 共享参数；V 不归一化。
- timestep modulation 最终投影 zero-init；新增层只用独立 alpha 做恒等接入。
- 不使用 post-norm、LayerNorm、per-head QK-Norm 或额外深度缩放。
# 已批准 06-E：attention kernel、GQA layout 与 varlen
## 当前上游能力
FlashAttention-4 的 CuTeDSL 接口面向 Hopper/Blackwell，公开提供 `flash_attn_varlen_func`；它接受 flattened Q/K/V、`cu_seqlens_q/k`、非因果模式和少于 Q heads 的 KV heads。当前接口也包含 SM 12.x 路径，适合 RTX 5090 的目标 benchmark。
- [FlashAttention-4 README](https://github.com/Dao-AILab/flash-attention#flashattention-4-cutedsl)
- [FlashAttention-4 varlen interface](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py)
- [FlashAttention-4 varlen tests](https://github.com/Dao-AILab/flash-attention/blob/main/tests/cute/test_flash_attn_varlen.py)
PyTorch SDPA 支持 `enable_gqa=True`，但官方仍将 GQA 标为 experimental，并明确不支持 GQA Nested Tensor。因此 SDPA 用于 dense bucket fallback，而不是 packed varlen 生产路径。
- [PyTorch scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
## A. 生产后端：FlashAttention-4 varlen，推荐
每个 GPU 在本地 microbatch 内构造：
```plain text
q: [total_tokens, 20, 128]
k: [total_tokens,  5, 128]
v: [total_tokens,  5, 128]
cu_seqlens: [local_batch + 1], int32, contiguous, CUDA
```
调用协议：
```plain text
flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_local_seqlen,
    max_seqlen_k=max_local_seqlen,
    softmax_scale=1/sqrt(128),
    causal=False,
    window_size=(None, None),
    softcap=0.0,
    deterministic=False,
    return_lse=False,
)
```
具体约束：
- 不把 5 个 KV heads `repeat_interleave` 成 20 heads；由 kernel 原生执行 4:1 GQA。
- Q/K/V 必须为 contiguous BF16；`cu_seqlens` 必须为 contiguous CUDA int32。
- Q/K 在 reshape 成上述 layout 前完成独立 RMSNorm 和 RoPE；位置频率跨 heads 共享并 broadcast，不物化 20 份。
- attention 输出 shape 为 `[total_tokens,20,128]`；已批准的内容 gate reshape 为相同 shape，在 flatten 回 `[total_tokens,2560]` 和 `W_o` 之前相乘。
- 不传 block mask、padding mask、causal mask、ALiBi、softcap 或 attention dropout。样本隔离只由 `cu_seqlens` 表示。
- `softmax_scale` 显式保存为 `1/sqrt(head_dim)`，不设可学习 temperature。
- 生产训练使用 nondeterministic 高速 backward；不承诺 bitwise reproducibility，数据顺序、seed 和 checkpoint resume 仍需可复现。
- FlashAttention 调用封装为独立 backend module；projection、norm、RoPE 与 MLP 可以 compile，不能为了 fullgraph 强行退回 padding dense attention。
## B. dense bucket fallback：PyTorch SDPA
fallback layout 为 `[B,H,L,D]`，使用有限长度 bucket 和正确的二维 bool mask：
- 首选 `F.scaled_dot_product_attention(..., enable_gqa=True, is_causal=False, dropout_p=0)`。
- 如果当前 PyTorch/SDPA backend 的 GQA kernel不兼容，只在 correctness/debug 路径中显式扩展 K/V；禁止作为生产默认实现。
- SDPA fallback 不接收 packed `cu_seqlens`，必须先恢复 dense bucket；其输出必须重新移除 padding 后再进入统一后续路径。
- 不引入第三套 xFormers/FlexAttention 生产后端，减少编译与数值分叉。
## C. FlashAttention-2
当前官方 CUDA 支持列表主要写明 Ampere/Ada/Hopper，而目标机为 Blackwell。项目不把 FlashAttention-2 设为 RTX 5090 的正式依赖；只有目标机实测证明其版本稳定且快于 FA4 时，才可作为临时 backend，不改变 checkpoint。
## D. 安装与配置
- 目标训练环境明确要求安装 `flash-attn-4`；若使用 CUDA 13，则按上游建议使用对应 `cu13` extra。
- 依赖版本、CUDA、PyTorch、GPU compute capability 和最终 `pack_gqa` 选择写入 run manifest，不写进模型权重 tensor。
- `pack_gqa=None` 使用上游自动策略作为初始值；在目标机对 `None/True/False` benchmark 后固定最快选项。
- backend 名称保存在 checkpoint config；加载时若生产 backend 不可用，必须显式提示正在使用 dense fallback，不能静默改变。
## E. 正确性验收
1. 对 MHA-reference/SDPA dense 与 FA4 varlen 比较 forward、dQ、dK、dV；BF16 初始容差采用上游测试同级的 `atol=rtol=3e-2`。
2. 构造两个不同长度样本，修改样本 A 的 token 后，样本 B 输出与梯度不得发生跨样本变化。
3. 覆盖空文本条件、4 style tokens、非方形 image buckets、最长 `L_max + 4 + image_tokens` 和 microbatch=1。
4. 检查 content gate 前后的 reshape 顺序，确保 head/channel 没有错位。
5. 在 256/512/768/1024 及真实文本长度分布上记录 samples/s、tokens/s、峰值显存；沿用组件 05 的 3% 启用门槛。
# 06-E 最终决定
- 生产路径采用 FlashAttention-4 `flash_attn_varlen_func`，原生 BF16 20Q/5KV GQA。
- dense PyTorch SDPA 是唯一 correctness/runtime fallback。
- 不扩展 KV heads，不构造跨样本 attention mask，不要求生产 backward 完全确定性。
- 目标环境必须安装 FA4；最终版本和 `pack_gqa` 仅通过四卡 5090 benchmark 锁定。
# 06-F：activation checkpoint、FSDP2 与四卡并行
## A. 并行策略
采用**四卡 1D FSDP2 full-shard 数据并行**：
- world size 固定为 4，使用单一 1D CUDA `DeviceMesh`。
- 不做 tensor parallel：`20 Q / 5 KV` 的 KV heads 无法在 4 卡上均匀切分；为并行重排 heads 会破坏已批准的 GQA 结构或引入 KV 复制。
- 不做 context/sequence parallel：组件 05 的 document-boundary varlen 已在每个 GPU 的 local microbatch 内完成，跨卡切分同一样本会额外引入通信和边界逻辑。
- 不做 CPU parameter offload：120GB 内存不抵消 PCIe 往返延迟；只允许在 OOM 诊断时临时启用，不作为生产配置。
- 每卡处理自己的 WebDataset 样本和 packed varlen 序列；样本不跨 GPU attention。
- Qwen3.5-2B 与 Mage-VAE 冻结、`eval()`、`inference_mode()`，在每卡复制，不纳入 FSDP；Qwen hidden states 和 VAE latents 在进入 DiT 前释放不再需要的中间激活。
## B. FSDP2 wrap 粒度
使用 PyTorch FSDP2 `fully_shard`，从内到外执行：
1. 对每个 DiT block 单独 `fully_shard(block)`，每个 block 形成一个通信 group。
2. 最后对 root model `fully_shard(model)`，只收集尚未分组的输入/输出 projection、condition head 和其他外围参数。
3. block 的 `reshard_after_forward=True`，释放 forward 的 unsharded parameters，backward 时重新 all-gather；root group 使用 `False`，避免 forward 结束后立刻为 backward 再聚合 root 参数。
4. 注册相邻 block 的 forward prefetch；让下一个 block 的 all-gather 与当前 block 计算重叠。
5. 不只对 root 调用 `fully_shard`；根组会把全部 block 合成一次大通信，峰值显存和通信重叠都更差。
每个最终 block 主体约 76MB BF16 参数；按 block 分组能让 4 卡 all-gather/reduce-scatter 和计算流水化。FSDP2 的 sharded state dict 用于训练恢复；独立的 consolidated model checkpoint 仍遵守组件 11 的“模型权重与训练状态分离”规则。
- [PyTorch FSDP2 fully_shard](https://github.com/pytorch/pytorch/blob/main/docs/source/distributed.fsdp.fully_shard.md)
- [PyTorch FSDP2 implementation](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fully_shard.py)
## C. Activation checkpoint
checkpoint 边界为**完整 DiT block**，不是 attention/MLP 内部：
- 使用 non-reentrant checkpoint：`use_reentrant=False`。
- 因为已批准所有 dropout/DropPath=0，生产路径使用 `preserve_rng_state=False`。
- 256 阶段默认关闭 checkpoint；如果 microbatch 目标无法满足再开启。
- 512 阶段默认每 2 个 block checkpoint 一次。
- 768/1024 阶段默认每个 block checkpoint。
- 发现 OOM 时先按 `1024 → 768 → 512` 顺序降低 microbatch，再改变 checkpoint 频率；不通过逻辑 padding 或额外 patchify 解决。
- 预计高分辨率 checkpoint 增加约 20%–35% block compute，但释放的 activation memory 用于保持 microbatch=1 或增加梯度累积，通常比直接 OOM 更有价值。
- 检查点 wrapper 在 FSDP2 bottom-up `fully_shard` 之前挂载；必须通过单 block forward/backward 和 FSDP 多卡测试后再锁定实际 wrapper API。
- [PyTorch checkpoint source](https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py)
## D. Mixed precision 与 optimizer 边界
- DiT linear/attention/MLP compute dtype：BF16。
- FSDP `param_dtype=BF16`、`output_dtype=BF16`；reduce dtype 初始使用 BF16，减少四卡 PCIe/NCCL 通信。
- norm weight 的 FP32 storage/update 由 06-D 保留；norm 子模块需有 FP32-accumulation 的 mixed-precision override，不能被普通 BF16 forward cast 静默覆盖。
- Qwen/VAE 的 frozen forward 不参与 gradient scaling。
- AdamW8bit 与 FSDP2 DTensor parameter 的兼容性在优化器组件单独验证；06-F 不把 optimizer state 强行塞进模型 checkpoint。
- 训练状态使用 FSDP2 sharded Distributed Checkpoint；模型发布权重另行导出 consolidated state。
## E. torch.compile 边界
`torch.compile` 不作为首次生产训练的硬依赖：
- FA4 varlen 是独立 backend/custom kernel，保留 graph boundary；不得为了 `fullgraph=True` 把它替换成 padded dense attention。
- 首次训练使用 eager PyTorch + FA4 + fused SwiGLU，先验证 loss、resume 和多卡稳定性。
- 后续只对 RMSNorm、condition modulation、fused QKVG projection、content gate 和 SwiGLU 等纯张量部分做局部 compile benchmark。
- 允许 `dynamic=True, fullgraph=False`，但按有限 resolution/text bucket 管理 compile cache；packed total token 数不强制逻辑 padding。
- FSDP2 root/child hooks、checkpoint wrapper 和 FA4 backend 三者组合若触发 graph break，不视为错误；局部 compile 低于 3% 端到端收益时保持 eager。
- compile 配置写入 run manifest，不写入模型权重；resume 时必须能关闭 compile 继续加载。
## F. 运行配置与验收
首个四卡配置：
- `torchrun --nproc_per_node=4`，单机 1D FSDP2 mesh。
- 各阶段动态设置 microbatch；1024 阶段以每卡 microbatch=1 作为可行性基线。
- 用 gradient accumulation 达到目标 global batch，不跨 GPU 拼接 varlen 序列。
- 记录每阶段 samples/s、有效 tokens/s、峰值显存、all-gather/reduce-scatter 时间、checkpoint 重算时间和 compile cache 命中情况。
- 对比 DDP、FSDP root-only、FSDP per-block 三条诊断路径；生产只保留 per-block FSDP。
- 先完成单卡 dense correctness，再完成四卡 dense FSDP，再切换四卡 FA4 varlen；任何顺序都不能跳过。
## G. 单卡先行与切换多卡
单卡先行是**实现验证策略**，不是正式训练策略。推荐采用以下顺序：
1. **单卡 dense correctness**：world size=1 时绕过 FSDP，使用 PyTorch SDPA dense 路径，在 256 和 512 阶段运行约 200–1000 steps。验证 x-pred loss、无 NaN、text/style/image span、空条件、CFG 两分支、checkpoint 保存与恢复。
2. **单卡 FA4 varlen**：在 RTX 5090 上用同一批固定输入切换 FA4，运行约 100–500 steps；验证 packed 与 dense 的输出/梯度容差、无跨样本泄漏和 `20Q/5KV` reshape。
3. **四卡 dense FSDP2**：使用 `torchrun --nproc_per_node=4`，先关闭 FA4 varlen，只验证 per-block FSDP、gradient reduction、activation checkpoint、mixed precision、optimizer step 和 shard-level resume。运行约 100–500 steps。
4. **四卡 FA4 varlen**：最后打开生产 backend，在 256/512/1024 至少各做短 benchmark，再开始正式多分辨率训练。
实现要求：
- 单卡和多卡共用同一个 model forward、loss、checkpoint schema、data protocol 和 backend selector；仅由 `world_size` 决定是否启用 FSDP。
- 不先用单卡训练完整 epoch 或积累正式模型权重；单卡只承担 bring-up。正式训练从四卡 FSDP2 通过 smoke test 后开始。
- 单卡 smoke 不要求覆盖 1024 长时间训练；1024 的可行性由四卡 FSDP2 + checkpoint + FA4 microbatch=1 验证。
- 四个阶段任何一个失败，都先修实现并从对应阶段重跑，不把单卡结果直接当作多卡结果。
# 06-F 历史决定（已被覆盖）
本节原 FSDP2 路线保留用于追溯，但不再是实现规范。组件 10 与组件 12 后续批准的决定具有更高优先级：
1. S0 是单卡 16层/256 的正式低成本训练阶段，不只是短 smoke。
2. S1 起使用同机四卡 DDP，不使用 FSDP2 baseline；只有 DDP 失败时才重新讨论 FSDP2。
3. optimizer 为 TorchAO AdamW8bit，BF16 大矩阵写回启用 stochastic rounding。
4. checkpoint/FA4/compile 的当前规则以组件 12-C 为准；06-E 的 FA4 varlen 接口继续有效。
5. frozen Qwen/VAE 每卡复制和 `20Q/5KV` GQA 结构不变。
# 06-F 当前状态
06-A 至 06-E 保持待验证；06-F 的训练系统内容已迁移到组件 12，不得从本节旧 FSDP2 描述生成配置。
</content>
</page>
