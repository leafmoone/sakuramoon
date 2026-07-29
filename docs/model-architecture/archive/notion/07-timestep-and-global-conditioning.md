Here is the result of "view" for the Page with URL https://app.notion.com/p/3abae967ecf2815b84fbda3c707728c2 as of 2026-07-28T07:16:43.924Z:
<page url="https://app.notion.com/p/3abae967ecf2815b84fbda3c707728c2">
<ancestor-path>
<parent-data-source url="collection://69ca66ff-43e7-4128-bb7b-9f3751506705" name="组件决策记录"/>
<ancestor-2-database url="https://app.notion.com/p/250be554eacc40219065073dfcf66fd7" title="组件决策记录"/>
<ancestor-3-page url="https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff" title="新模型架构决策中心"/>
</ancestor-path>
<properties>
{"date:决定日期:is_datetime":0,"date:决定日期:start":"2026-07-28","url":"https://app.notion.com/p/3abae967ecf2815b84fbda3c707728c2","决策编号":"ARCH-8","序号":7,"影响":"高","标签":["架构","训练"],"状态":"待验证","组件决策":"07 Timestep 与全局条件"}
</properties>
<content>
<callout icon="🚧" color="yellow_bg">
	**结构路线已完成，进入实现验证。** 07-A 与 07-B 已批准：采用共享轻量 AdaLN 条件调制，以及 image-span-only 的 128 通道 x-pred output head。
</callout>
# 范围
本组件接收：
- JLT 规定的连续 `t∈[0,1]`。
- 组件 05 已确定的 `size_scale` 和 `aspect` 两个标量；两者在 conditional/unconditional CFG 分支中始终相同，不参加 dropout。
- 组件 06 的每个 block 接口：`shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp`，以及独立的 `alpha_attn/alpha_mlp` growth switches。
- 最终 image span 和 128-channel Mage latent 输出接口。
本组件不改变 x-pred 的 JLT noise schedule、loss weighting 或采样器；这些继续以 [JLT](https://github.com/akatsuki-neo/JLT) 为准。
# 外部参考
## JLT timestep embedding
JLT 使用连续 `t∈[0,1]` 的固定 sinusoidal embedding，再经过两层 SiLU MLP；其 embedding dim 为 256。这里保留该输入协议，不把 timestep 离散化，也不乘 HDM 风格的额外 `time_factor=1000`。
- [JLT model_jit.py](https://github.com/akatsuki-neo/JLT/blob/main/model_jit.py)
## Krea 2 lightweight modulation
Krea 2 使用共享 timestep MLP 和共享到 `6d` 的投影；每个 block 只保存一个很小的 learned modulation offset，再切分为 attention/MLP 的 scale、shift、gate。这样不会为每个 block 各自保存完整 condition MLP。
- [Krea 2 mmdit.py](https://github.com/krea-ai/krea-2/blob/main/mmdit.py)
# 07-A：共享全局 condition 与 per-block modulation
## A. 每个 block 独立 AdaLN-Zero projection
每层从 condition 直接生成 6 组 `d` 向量：
`condition → [shift_a, scale_a, gate_a, shift_m, scale_m, gate_m]`
对于 `d=2560`、24 层，condition hidden=1024 时会产生约 0.38B projection 参数；若 condition hidden=d，则接近 0.94B。两者都不符合轻量调制目标，不采用。
## B. 共享 condition MLP + 共享 6d projection + per-block bias，推荐
输入编码：
```plain text
t                    -> fixed sinusoidal dim 256
size_scale           -> fixed sinusoidal dim 64
aspect               -> fixed sinusoidal dim 64
concat dim 384
                  -> Linear(384, 1024) -> SiLU -> Linear(1024, 1024)
                  -> Linear(1024, 6 * 2560)
```
共享输出 `u∈R^(6d)` 加入每个 block 的 learned `bias_l∈R^(6d)`，然后切分：
```plain text
u + bias_l
  -> shift_attn, scale_attn, gate_attn,
     shift_mlp,  scale_mlp,  gate_mlp
```
具体使用：
```plain text
h_a = (1 + scale_attn) * RMSNorm(x) + shift_attn
a   = AttentionContentGate(h_a)
x   = x + alpha_attn * gate_attn * a

h_m = (1 + scale_mlp) * RMSNorm(x) + shift_mlp
m   = SwiGLU(h_m)
x   = x + alpha_mlp * gate_mlp * m
```
初始化和参数：
- 最后 `1024 → 6d` projection 的 weight/bias zero-init；每层 `bias_l` zero-init；因此基础 16 层初始近似恒等，和 06-D 一致。
- `alpha_attn/alpha_mlp` 由 06-B 负责：base blocks=1，新增 blocks=0。
- `condition_dim=1024`、timestep Fourier dim=256、size/aspect 各 64；这些整数写入 config。
- 共享 condition MLP 与 `6d` projection 约 17M 参数；24 层 per-block modulation bias 仅约 0.37M，远低于 per-block 独立 projection。
- `size_scale`、`aspect` 直接加入同一 global condition；不注入 attention token，不改变 05 的位置编码。
- condition MLP 使用 SiLU，线性层 bias 保留；06-C 的 `bias=False` 只约束 DiT block 内 projections，不约束 condition encoder。
- condition 输出为 `[B,6d]`，进入 block 时 broadcast 成 `[B,1,d]`；packed varlen 只在 block 内 broadcast，不复制成每 token 的独立 condition tensor，除非 fused kernel 需要。
- 生产 CFG 的 conditional/unconditional 只改变 text/style tokens；timestep、size 和 aspect 完全一致。
## C. 每个 block 独立小 MLP
比 B 更有层表达力，但每层仍需 `1024→6d` 或低秩适配器；参数、通信和 checkpoint 复杂度增加，没有足够收益，不采用。
# 07-A 最终决定（已批准）
采用 B：
1. 固定 JLT 风格 timestep sinusoidal dim=256，输入 `t∈[0,1]`。
2. `size_scale` 和 `aspect` 各使用 64 维固定 Fourier/sinusoidal embedding。
3. 拼接后使用 `384→1024→1024` 两层 SiLU condition MLP。
4. 使用一个共享 `1024→6d` projection，再加每层 zero-init `6d` bias。
5. 切分为 6 组 per-block modulation，接入 06 的 attention/MLP pre-RMSNorm。
6. final modulation projection zero-init；不引入每层独立完整 AdaLN MLP。
# 07-A 批准说明
批准 `dim=256/64/64`、`condition_dim=1024`、共享 `6d` projection 和 per-block zero-init bias。
该路线属于常规、已有实证的轻量 AdaLN 实现：原始 DiT/JLT 常用每个 block 独立 modulation MLP；HDM 等模型使用全层 shared AdaLN；Krea 2 则明确采用 light modulation with per-block bias，并报告其替代每层 MLP 后没有牺牲模型表现。这里按 Krea 2 路线实现，不视为新的实验性结构。
下一项：07-B 最终 image-span RMSNorm、x-pred output head 初始化与输出通道。
# 07-B：最终 image-span 与 x-pred output head
## 约束
- 训练目标与采样全部沿用 JLT 的 x-pred 路径：网络直接输出 clean latent 估计 `x_pred`，采样时按 `v_pred=(x_pred-z_t)/clamp(1-t,t_eps)` 转为 ODE velocity。
- Mage-VAE 原生 latent 为每个空间位置 128 通道；组件 01 已决定取消额外 patchify。
- Single-stream 主干输出仍含 text、4 个 style token 和 image token，但只有 image span 需要预测。
- 不学习额外 variance/sigma；输出通道固定为 128，不扩成 256。
## A. 无条件 final RMSNorm + Linear
直接使用 `Linear(RMSNorm(image_hidden),128)`。参数最少，但删除了 JLT 与 Krea 2 都保留的 final timestep modulation，使输出读出无法在最后一步按噪声级与尺寸条件调整，不采用。
## B. JLT 式条件化 final RMSNorm + Linear，推荐
执行顺序：
```plain text
joint hidden
  -> gather image spans only                         [total_image_tokens, 2560]
  -> RMSNorm(eps=1e-6, FP32 accumulation)
  -> (1 + final_scale) * norm + final_shift
  -> Linear(2560, 128, bias=True)
  -> packed x_pred                                  [total_image_tokens, 128]
```
final modulation 来自 07-A 的共享 global condition hidden：
```plain text
condition_hidden [B,1024]
  -> SiLU
  -> Linear(1024, 2 * 2560)
  -> final_scale, final_shift
```
具体决定：
- final modulation 是独立的共享 `1024→2d` projection，不复用 block 的 `6d` tensor，也不增加 per-block 参数。
- `final_scale/final_shift` 按样本 broadcast 到该样本的 image tokens；不复制到 text/style tokens。
- final RMSNorm 完全遵守 06-D：`eps=1e-6`、FP32 累计、BF16 输出、weight 不做 weight decay。
- final modulation projection 的 weight/bias zero-init；因此初始化时 final norm 不被条件偏移。
- `Linear(2560,128,bias=True)` 的 weight/bias 也 zero-init，与 JLT 一致；06-C 的 `bias=False` 不约束最终输出层。
- 不执行 JLT 的 `unpatchify`：输出保持 Mage 原生每位置 128 通道，仅依据每个样本的 latent `H×W` 元数据恢复空间布局。
- 训练损失可直接在 packed image-token layout 上计算，避免为不同宽高构造额外 padding；进入 VAE decoder 前再恢复 `[B,128,H_lat,W_lat]`。
- 输出不包含 learned variance、logvar、epsilon 或 velocity 的额外通道；checkpoint 明确记录 `prediction_type=x` 和 `out_channels=128`。
参数与速度：
- final modulation 约 `1024×5120 = 5.24M` 参数；output projection 约 `2560×128 = 0.33M`，合计约 5.57M。
- 相对约 1.9B 模型不到 0.3%；只处理 image span，预计训练和推理端到端开销低于 1%。
- zero-init output head 会让第一个 optimizer step 主要更新输出层，随后梯度进入主干；这是 DiT/JLT 的常规稳定初始化，不需要为此加入 warmup bypass。
## C. 同时预测 x 与 variance/velocity
会增加输出通道、损失权重和采样接口，违反“x-pred 与采样依据 JLT”的已定约束，也没有当前训练目标需要，不采用。
# 07-B 最终决定（已批准）
采用 B：image span only、条件化 final RMSNorm、`d→128` zero-init head、无额外 patchify、无 learned variance。
# 07 实现验证
1. zero-init 后任意输入的 `x_pred` 必须为全零，且第一个 optimizer step 的输出层梯度有限、无 NaN。
2. 第二个及后续 optimizer step 必须能向 final modulation、condition encoder 和主干传播非零梯度。
3. dense 与 packed 路径的 image-span 输出、loss 和梯度在 BF16 容差内一致；text/style token 不进入 output projection。
4. 覆盖非方形 latent、不同 image-token 数、CFG 空条件与 16→20→24 层增长后的 checkpoint 加载。
5. checkpoint config 固定记录 timestep/size/aspect embedding 维度、`condition_dim=1024`、`prediction_type=x`、`out_channels=128` 与无额外 patchify。
# 07 状态
组件 07 已封板为待验证。后续只允许根据实现 smoke test 修正数值或 tensor layout，不重新打开全局条件与 output head 的结构选择。下一步进入组件 08：x-pred 目标与采样系统。
<empty-block/>
</content>
</page>
