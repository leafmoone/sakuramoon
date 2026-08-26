# AnimeSR-Mage-UFlow：纯图片 4× 二次元超分模型完整执行计划书

**版本：** v2.0  
**冻结日期：** 2026年8月26日  
**项目类型：** 独立的纯图片单图超分项目  
**主要任务：** 二次元图片盲超分，典型输入 `256×256`，输出 `1024×1024`，固定 4×  
**唯一复用的预训练组件：** Mage-VAE  
**不复用内容：** 不复用 SakuraMoon 的模型、条件系统、训练目标、代码、数据服务状态或 checkpoint  
**硬件原则：** 正式模型结构不随硬件变化；只自动调整 micro-batch、梯度累积、检查点和 attention backend

---

# 一、最终方案摘要

最终模型采用：

> **冻结 Mage-VAE + 原生 LQ 像素条件编码器 + LQ-centered residual flow + 多尺度 U-Flow Transformer + 可选 decoder grounding**

核心链路：

```text
HR 高质量原图
    │
    ├─ 冻结 Mage Encoder ──────────────────────> z_hr
    │
    └─ 随机 anime 退化 ──> LQ
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
       Bicubic 放大到 HR 尺寸       Pixel Condition Encoder
                │                           │
                ▼                           ├─ p64
       冻结 Mage Encoder                    ├─ p32
                │                           ├─ p16
                ▼                           └─ 可选 decoder features
              z_lr
                │
                ├─ δ = z_hr - z_lr
                │
                └─ residual flow state r_t
                              │
                              ▼
                   U-Flow Transformer
                              │
                              ▼
                    预测 residual velocity
                              │
                              ▼
                 ẑ_hr = z_lr + δ̂
                              │
                              ▼
                   冻结 Mage Decoder
                              │
                  可选 decoder grounding
                              │
                              ▼
                          SR 图像
```

正式主干目标为：

```text
约 121M～128M 可训练参数
```

若 Mage-VAE ceiling 测试证明 decoder 对细线、小字或规则纹样的损失不可接受，再加入：

```text
约 4M～6M decoder adapters
```

最终上限约：

```text
125M～134M 可训练参数
```

---

# 二、为什么采用这条路线

## 2.1 纯视觉模型，不使用文本或多模态条件

超分是强空间对齐的图像恢复问题，LQ 图像本身已经提供了主体、构图、颜色和大部分结构。VOSR 已经证明，生成式超分可以在不依赖文本到图像模型的情况下训练，并在结构保真和 hallucination 控制上取得竞争力。

因此本项目不包含：

```text
文本编码器
prompt
CFG
condition token
artist token
DINO 语义条件
T2I 预训练主干
多模态联合 attention
```

---

## 2.2 使用 Mage-VAE，但不假设它没有信息损失

Mage-VAE 提供原生：

```text
128 channels
16× spatial downsampling
单次前向 encode/decode
```

并以高保真、低编解码成本为设计目标。官方实现直接输出 `[B,128,H/16,W/16]`，不需要再次 patch packing。

但 latent SR 仍存在一个风险：LQ 中已经存在的细线、边界和局部颜色可能在 VAE 压缩后被弱化。PGSR 的主要发现就是，单纯依赖 latent condition 容易使最终细节与输入像素证据脱节；在 VAE 前保留像素特征，并在轨迹和 decoder 两端利用它们，可以改善保真度。

因此本项目保留一个轻量 Pixel Condition Encoder，但不会默认同时堆叠多个重复条件模块。

---

## 2.3 使用 residual flow，而不是从纯噪声重新生成整张图

RFMSR 将 source distribution 放在 LQ latent 附近，而不是从纯高斯噪声出发，从而缩短 transport distance，并通过两阶段训练同时获得一步输出和多步 refinement。

本项目进一步把状态平移到显式 residual 坐标：

```text
网络只预测 z_hr 相对 z_lr 缺少的部分
```

这样可以：

- 让低频结构通过显式 skip 保留；
- 减少模型重复生成输入中已经存在的内容；
- 让零初始化输出自然退化为 LQ anchor；
- 降低轻度退化时过度重画的倾向。

---

## 2.4 使用 U 形多尺度主干，而不是平铺 4096-token DiT

高分辨率图像恢复不适合所有层都在 `64×64` latent 网格上运行宽大 MLP。Uformer 和 Restormer 的共同思路是，在高分辨率层使用高效局部建模，在低分辨率 bottleneck 建立长距离联系，并通过 U 形 skip 保留细节。

因此正式主干使用：

```text
64×64 → 32×32 → 16×16 → 32×32 → 64×64
```

而不是：

```text
64×64 × 24 个等宽全程 block
```

---

## 2.5 Anime 专用退化和损失是项目成败的核心

APISR 指出，anime 超分的主要问题包括：

- 手绘线条扭曲；
- 淡线消失；
- 预测型压缩伪影；
- 不自然色块和颜色伪影。

其工作使用了面向动画生产和视频压缩的退化方式，以及 anime-domain 与通用视觉特征结合的感知监督。

因此本项目不会只使用：

```text
Gaussian blur + JPEG + Gaussian noise
```

而是建立独立的 anime degradation pipeline。

---

# 三、v1.0 范围与非目标

## 3.1 v1.0 必须支持

```text
单张图片
固定 4×
典型 256 → 1024
任意长宽比 bucket
1-step 默认推理
4-step quality 推理
真实网络压缩与复合退化
插画、动画截图、漫画和黑白线稿
整图推理与 tiled inference
```

## 3.2 v1.0 不支持

```text
视频时序一致性
任意倍率
×2 / ×3 / ×8
文本控制
参考图控制
角色重绘
图像编辑
2048 输出专项训练
4K 单次整图训练
通用自然照片
```

## 3.3 明确删除的冗余设计

正式链路不包含：

- 平铺 24×768 DiT；
- 多次重复注入同一个 `p64`；
- 独立 degradation encoder；
- size、aspect、profile、文本等复杂条件 token；
- cross-attention；
- ControlNet；
- x-pred 和 v-pred 双头；
-独立 latent endpoint loss；
- DWT head；
- pixel loss 之外重复的独立 color loss；
- paired synthetic 数据上的 re-degradation loss；
- 基础模型 GAN；
- decoder adapters 与 post-refiner 同时存在。

---

# 四、数据与张量合同

## 4.1 训练尺寸

正式课程分辨率：

| LQ | HR | HR latent |
|---:|---:|---:|
| 128×128 | 512×512 | 32×32 |
| 192×192 | 768×768 | 48×48 |
| 256×256 | 1024×1024 | 64×64 |

非正方形 bucket 保持同等面积，并要求：

```text
HR 高宽均为 64 的整数倍
LQ 高宽均为 16 的整数倍
```

这样可以保证：

```text
Mage latent 高宽可被 4 整除
U-Flow 能稳定执行两次 2× downsample
```

---

## 4.2 Mage-VAE 使用合同

```text
Mage Encoder:
  输入 RGB，范围统一
  输出 [B,128,H/16,W/16]

Mage Decoder:
  输入 [B,128,H/16,W/16]
  输出 RGB
```

训练目标 latent 使用：

```text
deterministic posterior mean
```

不对 target latent 采样，避免给 SR GT 引入随机性。

Encoder 参数：

```text
requires_grad = false
eval mode
encode 使用 no_grad
```

Decoder 参数：

```text
requires_grad = false
eval mode
Stage II decode 不能放在 no_grad 中
```

原因是 Stage II 需要将像素 loss 的梯度穿过冻结 decoder 传回 U-Flow。

---

## 4.3 LQ latent anchor

定义：

\[
z_{\mathrm{lr}}
=
E_{\mathrm{Mage}}
\left(
\operatorname{Bicubic}_{4\times}(x_{\mathrm{lr}})
\right)
\]

其中：

```text
LQ 256
→ bicubic 到 1024
→ Mage Encoder
→ z_lr [128,64,64]
```

HR target：

\[
z_{\mathrm{hr}}
=
E_{\mathrm{Mage}}(x_{\mathrm{hr}})
\]

不得直接使用：

```text
E(LQ 256) = [128,16,16]
```

和：

```text
E(HR 1024) = [128,64,64]
```

进行插值。

---

# 五、Residual Flow 数学定义

## 5.1 Residual target

\[
\delta=z_{\mathrm{hr}}-z_{\mathrm{lr}}
\]

模型不直接生成完整 `z_hr`，而是生成缺失 residual。

---

## 5.2 Source state

\[
r_0=\sigma\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
\]

当：

```text
σ = 0
```

时，source residual 为零，模型成为完全确定性的 LQ-centered flow。

---

## 5.3 Flow path

\[
r_t=(1-t)r_0+t\delta,
\qquad t\in[0,1]
\]

目标速度：

\[
v^\star=\delta-r_0
\]

模型：

\[
\hat v=v_\theta(r_t,t,c)
\]

主损失：

\[
L_{\mathrm{FM}}
=
\mathbb E
\left[
\left\|
\hat v-v^\star
\right\|_2^2
\right]
\]

---

## 5.4 一步推理

从 `t=0` 做一次 Euler：

\[
\hat\delta
=
r_0+v_\theta(r_0,0,c)
\]

\[
\hat z_{\mathrm{hr}}
=
z_{\mathrm{lr}}+\hat\delta
\]

默认 Faithful 模式：

```text
σ = 0
r0 = 0
```

因此模型完全确定。

---

## 5.5 四步推理

```text
solver = Heun
time points = [0.00, 0.25, 0.50, 0.75, 1.00]
```

最后一步允许退化成 Euler，以减少一次网络评估。

只有在完整验证集中，四步相对一步确实取得稳定质量提升时，才发布四步模式。

---

## 5.6 Source noise 配置

初始训练配置：

```text
75% 样本：σ = 0
25% 样本：σ ~ Uniform(0.02, 0.15)
```

`σ` 作为一个标量条件输入。

推理模式：

| 模式 | σ |
|---|---:|
| Faithful | 0 |
| Balanced | 0.05～0.10 |
| Experimental | 最大0.15 |

在 M3 smoke 阶段同时测试：

```text
全部 σ=0
75% zero + 25% small noise
50% zero + 50% small noise
```

选择规则：

- 非零噪声必须稳定改善 LPIPS/DISTS；
- edge displacement 不得恶化；
- 眼睛、文字和规则线条错误率不得增加；
- 若没有明确收益，正式模型固定 `σ=0`，同时删除 sigma conditioner。

---

# 六、Pixel Condition Encoder

## 6.1 输入

```text
原生 LQ RGB
典型尺寸 256×256
```

不先放大到 1024 后再运行重型 pixel encoder。

因为固定 4× 时：

```text
LQ 256 下采样4× → 64
```

正好对应：

```text
HR 1024 经 Mage 下采样16× → 64
```

---

## 6.2 结构

```text
RGB 256×256
  │
  ├─ Stem 3×3 Conv, 48ch
  │
  ├─ 2 × PixelConditionBlock, 48ch
  │
  ├─ Downsample
  │
  ├─ 128×128, 96ch, 2 blocks
  │
  ├─ Downsample
  │
  ├─ 64×64, 128ch, 3 blocks
  │
  ├─ Downsample
  │
  ├─ 32×32, 192ch, 3 blocks
  │
  ├─ Downsample
  │
  └─ 16×16, 256ch, 4 blocks
```

每个 PixelConditionBlock：

```text
LayerNorm2d
→ depthwise 3×3
→ pointwise expansion 2×
→ SiLU / SimpleGate
→ pointwise projection
→ residual
```

预计参数：

```text
9M～12M
```

---

## 6.3 输出用途

| 输出 | 用途 |
|---|---|
| `p64` | U-Flow 64×64 输入条件 |
| `p32` | U-Flow encoder 32×32 stage 条件 |
| `p16` | bottleneck 条件和全局退化摘要 |
| `p128` | 可选 decoder grounding |
| `p256` | 可选 decoder grounding |
| `GAP(p16)` | 全局 degradation/content summary |

不再增加第二个 degradation encoder。

---

# 七、U-Flow Transformer 主干

## 7.1 总体结构

```text
64×64，dim384，4 local blocks
      │
      ▼ Downsample
32×32，dim512，6 local blocks
      │
      ▼ Downsample
16×16，dim768，8 global blocks
      │
      ▼ Upsample + encoder skip
32×32，dim512，6 local blocks
      │
      ▼ Upsample + encoder skip
64×64，dim384，4 local blocks
      │
      ▼
128-channel velocity
```

总 block 数：

```text
28
```

---

## 7.2 各 stage 规格

| Stage | Grid | Dim | Depth | Q Heads | KV Heads | FFN | Attention |
|---|---:|---:|---:|---:|---:|---:|---|
| Encoder S0 | 64×64 | 384 | 4 | 6 | 2 | 1152 | 8×8 window |
| Encoder S1 | 32×32 | 512 | 6 | 8 | 2 | 1536 | 8×8 window |
| Bottleneck | 16×16 | 768 | 8 | 12 | 4 | 2304 | global |
| Decoder S1 | 32×32 | 512 | 6 | 8 | 2 | 1536 | 8×8 window |
| Decoder S0 | 64×64 | 384 | 4 | 6 | 2 | 1152 | 8×8 window |

统一：

```text
head_dim = 64
qk_norm = true
norm = RMSNorm
activation = SwiGLU
dropout = 0
position = continuous 2D RoPE
```

---

## 7.3 Restoration block

```text
x
├─ RMSNorm
├─ stage FiLM
├─ GQA attention
├─ output projection
├─ LayerScale
└─ residual

x
├─ RMSNorm
├─ stage FiLM
├─ SwiGLU
├─ depthwise 3×3 on expanded branch
├─ output projection
├─ LayerScale
└─ residual
```

`LayerScale` 初值：

```text
1e-3
```

高分辨率 stage：

```text
普通 window
shifted window
普通 window
shifted window
```

交替排列。

---

## 7.4 Downsample 与 Upsample

Downsample：

```text
PixelUnshuffle(2)
→ 1×1 projection
```

Upsample：

```text
1×1 expansion
→ PixelShuffle(2)
```

这样比直接 stride convolution 更完整地保留局部信息。

Skip fusion：

```text
concat(decoder_feature, encoder_skip)
→ 1×1 projection
```

不使用直接相加，避免强制两个 feature space 完全一致。

---

## 7.5 输入条件融合

初始 64×64 feature：

\[
h_{64}
=
P_r(r_t)
+
P_z(z_{\mathrm{lr}})
+
g_{64}P_p(p_{64})
\]

其中：

- `P_r`：128→384；
- `P_z`：128→384；
- `P_p`：128→384；
- `g64`：可学习 sigmoid gate；
- pixel projection 使用小值初始化；
- `z_lr` 路径不做零初始化。

在 32×32：

\[
h_{32}
\leftarrow
h_{32}
+
g_{32}P_{32}(p_{32})
\]

在 16×16：

\[
h_{16}
\leftarrow
h_{16}
+
g_{16}P_{16}(p_{16})
\]

pixel condition 只在 encoder 对应 stage 注入一次。Decoder 通过 U 形 skip 继承这些信息。

---

## 7.6 全局条件

只保留：

```text
timestep embedding
sigma embedding
GAP(p16)
```

形式：

```text
g =
MLP(
  sinusoidal(t)
  || sinusoidal(σ)
  || projection(GAP(p16))
)
```

生成五组 stage-level FiLM：

```text
Encoder64
Encoder32
Bottleneck16
Decoder32
Decoder64
```

每个 stage 共享一次 projection，不为每个 block 单独构建大型 conditioner。

允许增加小型 per-block bias，但不允许每层独立 `g→6D` 大 MLP。

---

## 7.7 输出头

```text
RMSNorm
→ 3×3 Conv, 384→384
→ SiLU
→ 3×3 Conv, 384→128
```

最后一个卷积：

```text
weight zero-init
bias zero-init
```

初始化时：

```text
v̂ = 0
ẑ_hr = z_lr + r0
```

当 Faithful 模式 `σ=0` 时，初始化模型等价于直接输出 aligned LQ latent。

---

# 八、参数预算

| 模块 | 参数预算 |
|---|---:|
| 28 个 U-Flow blocks | 约105M |
| Pixel Condition Encoder | 9–12M |
| Down/Up/Skip projections | 3–5M |
| 输入和输出层 | 2–3M |
| Stage conditioner | 3–5M |
| **核心可训练总量** | **121–128M** |
| 可选 decoder adapters | 4–6M |
| **开启 grounding 后** | **125–134M** |

预计 BF16 纯 SR 权重：

```text
约 242～268MB
```

Mage-VAE 权重大小不在代码中写死，由启动时实际统计并写入 manifest。

完整训练 checkpoint 预计：

```text
约 1～2GB
```

精确大小由优化器状态格式和 EMA 配置决定。

---

# 九、Decoder Grounding 决策门

## 9.1 默认 v1 核心

```text
ẑ_hr
→ 原始冻结 Mage Decoder
→ SR
```

默认不修改 decoder。

---

## 9.2 M0 ceiling 触发条件

以下任意条件持续失败时，开启 decoder grounding：

```text
细线 edge F1 显著下降
线宽误差中位数 > 0.35 HR pixel
小字 OCR 准确率下降 > 2个百分点
规则网格或漫画网点发生明显结构破坏
平涂区域 ΔE00 中位数 > 1.5
```

阈值先在 2000 张验证图和合成线条图上校准。

---

## 9.3 Grounding 方案

优先在 Mage Decoder 中间 stage 注入：

```text
decoder 128 stage ← p128
decoder 256 stage ← p256
decoder 512 stage ← upsample(p256)
```

Adapter：

```text
Norm
→ 1×1 projection
→ depthwise 3×3
→ SiLU
→ zero-init 1×1
→ residual
```

不在最终 1024 RGB 层直接注入，避免复制 LQ aliasing 和压缩 block。

如果 Mage Decoder 内部结构无法稳定暴露，则二选一使用：

```text
Mage Decoder
→ 5M～8M zero-init residual refiner
```

不得同时使用内部 adapter 和 post-refiner。

---

# 十、独立数据系统

## 10.1 项目独立性

项目重新实现自己的：

```text
manifest
eligibility index
cache
queue
socket
worker lease
validation selection
checkpoint state
```

不共享任何其他项目的：

```text
代码
运行状态
socket
queue 文件
checkpoint schema
resolved config
```

现有 Danbooru WebDataset 数据源可以继续作为原始数据来源，但必须建立本项目独立的数据合同。

---

## 10.2 数据服务职责

```text
读取 manifest
下载和校验 shard
本地 shard cache
确定性 per-cycle shuffle
worker lease / ack
validation shard 排除
断点恢复
错误显式上报
```

数据服务只负责供应原始样本，不负责：

```text
Mage 编码
GPU 退化
perceptual feature
模型条件
```

---

## 10.3 SR 数据索引

生成：

```text
data/index/sr-eligibility-v1.parquet
data/index/shard-summary-v1.parquet
data/index/filter-report-v1.json
data/index/sr-validation-v1.json
```

每个样本至少记录：

```text
sample_id
shard_path
width
height
format
quality
anime_classification
anime_completeness
year
eligible_512
eligible_768
eligible_1024
clean_score
sampling_group
```

---

## 10.4 HR GT 过滤

硬排除：

```text
损坏图
无法解码
明显 AI corruption
not_painting
尺寸不足
严重预放大痕迹
严重 JPEG block
严重 ringing
大范围彩色噪点
过度锐化 halo
crop retention < 0.80
```

优先池：

```text
polished
illustration
bangumi
comic
quality ∈ {masterpiece, best, great, good}
```

辅助池：

```text
monochrome
rough
3d
normal quality
```

辅助池合计默认不超过：

```text
20%
```

---

## 10.5 图像级 clean score

对候选 HR 计算：

```text
blockiness score
ringing score
blur score
flat-region noise
upscale suspicion
edge overshoot
```

采用两阶段筛选：

```text
第一阶段：metadata 和尺寸过滤
第二阶段：首次读取时计算 clean score 并缓存
```

不得要求一次性解码全部数据后才能开始项目。

---

# 十一、Anime 退化管线

## 11.1 Profile 分布

| Profile | 比例 |
|---|---:|
| P0 Clean Downsample | 10% |
| P1 Mild Web | 25% |
| P2 Normal Web | 35% |
| P3 Anime Codec | 20% |
| P4 Severe | 10% |

P0 仍执行 4× 下采样，只是不加额外污染。

---

## 11.2 在线 GPU 退化

在线执行：

```text
isotropic Gaussian blur
anisotropic Gaussian blur
motion blur
sinc / ringing filter
area / bicubic / bilinear / nearest resize
anti-alias on/off
Gaussian noise
Poisson noise
chroma noise
gamma shift
banding
posterize
dither
unsharp mask
JPEG approximation
chroma subsampling approximation
```

初始范围：

```text
Blur sigma:
  mild    0.1～0.6
  normal  0.2～1.2
  severe  0.5～2.0

Gaussian noise:
  mild    0～2 / 255
  normal  0～5 / 255
  severe  2～10 / 255

JPEG quality:
  mild    80～98
  normal  55～90
  severe  30～70
```

最终输出尺寸必须严格为 HR 的四分之一。

---

## 11.3 退化顺序随机化

允许：

```text
blur → resize → noise → codec
codec → resize → noise
resize → codec → resize
blur → codec → sharpen → resize
```

不得固定为单一顺序。

---

## 11.4 真实 Codec Bank

真实 codec 不在每个 worker 中反复启动 ffmpeg。

离线构建：

```text
WebP
AVIF
H.264
H.265
AV1
MPEG-4
4:2:0 / 4:2:2
limited/full range mismatch
二次转码
```

规模：

```text
50k～100k HR crop
每个 crop 1～2 个 codec 版本
预计 20～50GiB
```

Codec Bank batch 占比：

```text
10%～20%
```

---

## 11.5 退化确定性

\[
\text{degradation seed}
=
H(
\text{global seed},
\text{sample id},
\text{data cycle},
\text{exposure index}
)
\]

要求：

```text
相同 checkpoint
相同 data cycle
相同 exposure
恢复后逐像素生成相同 LQ
```

---

# 十二、Loss 设计

## 12.1 Phase I

只使用：

\[
L_{\mathrm{FM}}
=
\|\hat v-v^\star\|_2^2
\]

Phase I 不运行：

```text
Mage Decoder backward
perceptual network
GAN
decoder adapters
pixel loss
```

---

## 12.2 Phase II

\[
L=
L_{\mathrm{FM}}
+
\lambda_{\mathrm{pix}}L_{\mathrm{Charb}}
+
\lambda_{\mathrm{edge}}L_{\mathrm{edge}}
+
\lambda_{\mathrm{flat}}L_{\mathrm{flat}}
+
\lambda_{\mathrm{perc}}L_{\mathrm{perc}}
\]

初始权重：

```toml
[loss]
flow = 1.00
pixel_charbonnier = 1.00
edge = 0.10
flat = 0.05
perceptual = 0.05
```

---

## 12.3 Pixel loss

```text
RGB Charbonnier
epsilon = 1e-3
```

控制：

- 颜色；
- 结构位置；
- 低频；
- 整体亮度。

---

## 12.4 Edge loss

组合：

```text
Sobel X/Y
Laplacian
soft edge map
```

控制：

- 线条位置；
- 线宽；
- 连续性；
- halo；
- 边缘过冲。

---

## 12.5 Flat-region loss

从 HR 生成：

\[
M_{\mathrm{flat}}
=
\mathbf 1
[
\|\nabla x_{\mathrm{hr}}\|<\tau
]
\]

在 mask 内约束：

```text
RGB residual variance
高频能量
chroma variance
```

主要抑制：

- 皮肤颗粒；
- 头发色块噪点；
- 天空假纹理；
- 平涂区域色带过度锐化。

---

## 12.6 Perceptual loss

使用双特征：

```text
70% anime-domain backbone
30% 通用视觉 backbone
```

anime-domain backbone：

```text
优先使用许可清晰的 Danbooru/anime 分类预训练权重
若许可证不满足项目发布要求，则独立训练一个 ResNet-50
```

该网络：

```text
冻结
只用于 loss
不随推理模型交付
```

通用特征可使用 LPIPS/VGG 类 backbone。

感知 loss 默认在随机：

```text
512×512 HR crop
```

上计算，以降低显存。

---

## 12.7 删除的 loss

v1 不使用：

```text
DWT loss
独立 color loss
latent endpoint loss
paired re-degradation loss
GAN loss
OCR loss 作为训练目标
```

OCR 只用于验证，避免模型为了文字指标破坏普通线条。

---

## 12.8 梯度贡献校准

固定权重只是初始值。训练前 2000 update 统计各 loss 对 U-Flow 参数的梯度范数。

目标相对贡献：

| Loss | 相对 FM 梯度 |
|---|---:|
| Pixel | 0.25～0.50 |
| Edge | 0.05～0.15 |
| Flat | 0.03～0.10 |
| Perceptual | 0.05～0.15 |

超出范围时调整 loss 权重，而不是仅比较 loss 数值。

---

# 十三、训练阶段

## M0：Mage-VAE Ceiling

样本：

```text
2000 张真实 anime 图
+ 500 张合成线条/网格/文字图
```

类别：

```text
插画
动画帧
漫画
黑白线稿
小字
规则网格
密集排线
大面积平涂
极端宽高比
```

输出：

```text
vae-ceiling-report.json
latent-statistics.json
per-sample.parquet
comparison-grids/
```

指标：

```text
PSNR
SSIM
LPIPS
edge F1
edge displacement
line width error
flat-region ΔE00
OCR accuracy
```

决策：

```text
通过 → 不使用 decoder grounding
失败 → 启用内部 adapter 或 post-refiner
```

---

## M1：数据索引与退化系统

工作：

1. 扫描 metadata；
2. 生成 eligibility；
3. 建立 SR validation；
4. 实现 independent data service；
5. 实现 GPU degradation；
6. 构建 codec bank；
7. 验证 resume 后确定性；
8. 连续运行 10k batch 压力测试。

通过条件：

```text
train/validation 零重叠
退化逐像素可复现
data wait < 5% step time
没有静默跳样本
worker 错误能够终止训练
```

---

## M2：确定性 Pixel Baseline

训练一个：

```text
5M～10M pixel U-Net / NAFNet-like
```

目的：

- 验证数据和退化；
- 建立轻量保真下限；
- 判断 128M flow 模型是否真正有价值。

曝光：

```text
0.5M～1.0M 样本
```

不需要追求最终质量。

---

## M3：Flow Smoke Model

结构：

```text
dim = [192, 256, 384]
depth = [2, 2, 4, 2, 2]
约 15M～20M
```

曝光：

```text
100k～200k 样本
```

必须验证：

```text
flow 方向正确
一步输出朝 HR 靠近
4-step 不比1-step差
非正方形 bucket 正确
window mask 正确
resume 连续
DDP 与单卡等价
decoder gradient 能传回主干
无 window seam
```

M3 未通过，不启动正式模型。

---

## M4：Phase I Residual Flow

最小曝光：

```text
6M
```

目标曝光：

```text
10M
```

最大曝光：

```text
16M
```

课程：

### I-A：20%

```text
100% 128→512
```

### I-B：30%

```text
50% 128→512
30% 192→768
20% 256→1024
```

### I-C：30%

```text
20% 128→512
30% 192→768
50% 256→1024
```

### I-D：20%

```text
10% 128→512
20% 192→768
70% 256→1024
```

最后：

```text
额外 0.5M 样本可使用 100% 256→1024
```

---

## M5：Phase II One-Step Faithful

目标曝光：

```text
2M
```

最小：

```text
1M
```

最大：

```text
4M
```

尺寸：

```text
20% 192→768
80% 256→1024
```

micro-batch 比例：

```text
50% random-t flow batch
50% t=0 one-step decode batch
```

random-t batch：

```text
只计算 L_FM
```

one-step batch：

```text
L_FM
+ pixel
+ edge
+ flat
+ perceptual
```

必须保留 random-t velocity loss，否则四步能力会退化。

---

## M6：可选 Perceptual Adapter

不属于 v1 必须交付。

进入条件：

```text
Faithful Base 已通过保真门槛
但真实 LQ 人评显示明显偏软
```

优先尝试：

```text
U-Flow 后8个 block 的 rank-8/rank-16 LoRA
+ decoder adapter 低学习率
```

先不加入 GAN。

只有 LoRA + perceptual/edge 仍不能改善视觉锐度时，才单独评估轻量 conditional PatchGAN。

---

# 十四、优化器与调度

## 14.1 Phase I

```toml
[optimizer]
name = "adamw"
lr = 0.00015
betas = [0.9, 0.95]
eps = 1e-8
weight_decay = 0.05

[scheduler]
warmup_fraction = 0.03
type = "cosine"
min_lr_ratio = 0.10

[gradient]
clip_norm = 1.0
```

不对以下参数做 weight decay：

```text
norm
bias
LayerScale
position parameters
gates
```

---

## 14.2 Phase II

| 参数组 | LR |
|---|---:|
| U-Flow | `2e-5` |
| Pixel Encoder | `5e-5` |
| Decoder adapters | `1e-4` |
| Mage-VAE base | `0` |

---

## 14.3 EMA

按样本数计算 EMA：

```text
half_life_samples = 500k
```

EMA 更新不绑定固定 update 数，避免硬件和 batch 变化后语义改变。

---

# 十五、硬件自适应方案

## 15.1 正式模型不随硬件改变

固定：

```text
U-Flow-128M
同一参数量
同一 loss
同一曝光预算
```

硬件只改变：

```text
micro-batch
gradient accumulation
activation checkpointing
attention backend
compile
worker 数
墙钟时间
```

---

## 15.2 启动 benchmark

```bash
python -m anime_sr.cli.benchmark \
  --config config/benchmark.toml \
  --resolutions 512 768 1024 \
  --microbatches 1 2 4 \
  --checkpoint-modes none selective full \
  --attention-backends sdpa flash
```

每组：

```text
20 warmup
100 timed iterations
```

记录：

```text
samples/s
latent tokens/s
peak allocated memory
peak reserved memory
VAE encode time
U-Flow forward/backward
decoder backward
data wait
```

选择：

```text
peak_reserved <= 总显存88%
无 OOM
无非有限值
吞吐最高
```

---

## 15.3 推荐硬件档

| 单卡显存 | 用途 |
|---:|---|
| 16GB | M0、数据测试、smoke、小尺寸 Phase I |
| 24GB | 正式 Phase I 最低档；Phase II需全 checkpointing |
| 32–48GB | 正式推荐 |
| 80GB以上 | 更大 micro-batch、减少重算 |

默认：

```text
BF16
DDP
SDPA correctness backend
Flash/window backend after parity test
activation checkpointing
```

默认不使用 FSDP。

只有以下情况启用 FSDP：

```text
24GB micro-batch1 仍无法训练
或未来参数量超过约400M
```

---

## 15.4 Token-based 累积

Phase I：

```text
target_latent_tokens_per_update = 131072
```

Phase II：

```text
target_latent_tokens_per_update = 65536
```

梯度累积：

\[
A=
\left\lceil
\frac{T_{\mathrm{target}}}
{W\cdot B_{\mathrm{micro}}\cdot H_{\mathrm{latent}}\cdot W_{\mathrm{latent}}}
\right\rceil
\]

最大 accumulation：

```text
64
```

超过 64 时：

```text
token target 减半
学习率乘 0.7
曝光总量不变
```

---

# 十六、评测体系

## 16.1 固定评测集

### VAE Ceiling

```text
2500 张
```

### Synthetic Paired

```text
5000 张
每个 degradation profile 1000 张
```

### Stress Set

```text
500 张
```

包含：

```text
眼睛
睫毛
发丝
文字
手指
网格
漫画网点
淡线
平涂
摩尔纹
```

### Real-LQ

```text
最低 200 张
目标 500 张
```

---

## 16.2 对比模型

```text
Bicubic
Pixel Baseline
APISR
Real-CUGAN
Real-ESRGAN Anime
Faithful 1-step
Faithful 4-step
Mage-VAE ceiling
```

大型通用生成式 SR 只作为 hallucination 和视觉上限参考，不作为主要目标。

---

## 16.3 指标

通用：

```text
PSNR-RGB
PSNR-Y
SSIM
LPIPS
DISTS
ΔE00
```

Anime 专项：

```text
edge precision / recall / F1
edge displacement
line width error
line continuity
flat-region variance
flat-region high-frequency energy
chroma noise
ringing score
OCR accuracy
```

人工错误：

```text
线条新增
线条消失
眼睛形状变化
文字笔画变化
手指变化
脸型变化
颜色漂移
平涂纹理化
halo
window seam
tile seam
```

---

## 16.4 Faithful 硬门槛

轻度退化：

```text
PSNR-Y 相对 Pixel Baseline 下降不得超过0.20dB
edge displacement 不得恶化
flat artifact rate 不得上升
```

中度退化：

```text
LPIPS 或 DISTS 至少改善5%
人评相对 APISR/Pixel Baseline 偏好率 ≥ 60%
```

重度退化：

```text
人评偏好率 ≥ 60%
严重结构错误率不得高于通用生成式 SR
文字和眼睛错误单独统计
```

一步与四步：

```text
若4-step相对1-step改善 < 2%
则不把4-step作为正式卖点
```

---

# 十七、推理与部署

## 17.1 推理模式

### Faithful

```text
σ = 0
steps = 1
deterministic
```

### Faithful Quality

```text
σ = 0
steps = 4
```

### Balanced

```text
σ = 0.05～0.10
steps = 1或4
固定seed
```

默认始终为 Faithful。

---

## 17.2 大图推理

单张 LQ 大于训练 crop 时：

```text
优先整图运行
显存不足再 tile
```

默认：

```text
LQ tile = 256
LQ overlap = 64
HR overlap = 256
blend = cosine feather
padding = reflect
```

窗口和 tile 坐标必须使用全图绝对坐标生成 2D RoPE，避免每个 tile 都从零开始导致位置不一致。

---

## 17.3 发布文件

```text
anime-sr-mage-uflow-faithful.safetensors
mage-vae/
inference-config.toml
model-manifest.json
metrics.json
README.md
```

Manifest：

```text
model version
git commit
Mage-VAE SHA256
dataset id
data index version
degradation version
training exposures
optimizer
EMA
metric results
supported scale
supported input contract
```

---

# 十八、Checkpoint 合同

Raw checkpoint 包含：

```text
model
EMA
optimizer
scheduler
global exposure count
optimizer update count
resolution curriculum state
RNG state
data cycle
degradation version
resolved config
hardware profile
Mage-VAE hash
git commit
manifest.json
COMPLETE
```

写入流程：

```text
temporary directory
→ write files
→ fsync
→ manifest
→ COMPLETE
→ atomic rename
```

保留：

```text
最近3个 raw checkpoint
最近10个 model-only checkpoint
最佳 faithful checkpoint
最佳 perceptual checkpoint
最终 release checkpoint
```

---

# 十九、项目目录

```text
anime-sr/
├── config/
│   ├── base.toml
│   ├── data.toml
│   ├── benchmark.toml
│   ├── smoke.toml
│   ├── stage1_flow.toml
│   └── stage2_faithful.toml
│
├── docs/
│   ├── design.md
│   ├── data-contract.md
│   ├── degradation-v1.md
│   └── evaluation-protocol.md
│
├── src/anime_sr/
│   ├── vae/
│   │   ├── mage.py
│   │   └── decoder_grounding.py
│   ├── data/
│   │   ├── manifest.py
│   │   ├── service.py
│   │   ├── client.py
│   │   ├── index.py
│   │   ├── buckets.py
│   │   ├── pipeline.py
│   │   ├── degradation.py
│   │   └── codec_bank.py
│   ├── model/
│   │   ├── pixel_encoder.py
│   │   ├── window_attention.py
│   │   ├── restoration_block.py
│   │   ├── conditioning.py
│   │   ├── uflow.py
│   │   └── output_head.py
│   ├── flow/
│   │   ├── path.py
│   │   ├── solver.py
│   │   └── sampling.py
│   ├── train/
│   │   ├── stage1.py
│   │   ├── stage2.py
│   │   ├── losses.py
│   │   ├── optimizer.py
│   │   ├── ema.py
│   │   └── checkpoint.py
│   ├── eval/
│   │   ├── vae_ceiling.py
│   │   ├── paired.py
│   │   ├── real_lq.py
│   │   ├── line_metrics.py
│   │   └── report.py
│   └── cli/
│       ├── index_dataset.py
│       ├── build_codec_bank.py
│       ├── benchmark.py
│       ├── train.py
│       ├── evaluate.py
│       └── infer.py
│
└── tests/
    ├── test_flow_direction.py
    ├── test_one_step_endpoint.py
    ├── test_degradation_determinism.py
    ├── test_bucket_alignment.py
    ├── test_window_attention.py
    ├── test_tile_coordinates.py
    ├── test_decoder_gradient.py
    ├── test_ddp_equivalence.py
    └── test_resume.py
```

---

# 二十、执行顺序

```text
1. 建立独立项目与配置 schema
2. 接入并冻结 Mage-VAE
3. M0：VAE ceiling
4. 建立数据 manifest、eligibility 和 validation
5. 实现 anime degradation
6. 构建 codec bank
7. 训练 Pixel Baseline
8. 实现 Pixel Condition Encoder
9. 实现 U-Flow smoke 模型
10. 验证 flow、window、resume 和 decoder gradient
11. 运行硬件 benchmark
12. 启动 Base-128M Phase I
13. 启动 Phase II Faithful
14. 完整 paired 与 real-LQ 评测
15. 实现 tiled inference
16. 导出 release checkpoint
17. 质量不足时再评估 perceptual LoRA
```

在第 1～10 项完成前，不实现：

```text
GAN
DWT
DINO
语义 encoder
2048 训练
视频模块
任意倍率
```

---

# 二十一、最终冻结配置

```text
项目:
  pure image anime SR
  fixed ×4
  independent codebase
  no text
  no CFG
  no T2I backbone

预训练:
  Mage-VAE only
  frozen encoder
  frozen decoder

输入:
  LQ RGB
  typical 256×256

目标:
  HR RGB
  typical 1024×1024

Anchor:
  z_lr = MageEncode(Bicubic4x(LQ))

Residual:
  delta = z_hr - z_lr

Flow:
  r0 = sigma * epsilon
  rt = (1-t) * r0 + t * delta
  v-pred only

Pixel Encoder:
  256 / 128 / 64 / 32 / 16
  9M～12M

U-Flow:
  64² d384 depth4 local
  32² d512 depth6 local
  16² d768 depth8 global
  32² d512 depth6 local
  64² d384 depth4 local

参数:
  core 121M～128M
  optional grounding 125M～134M

训练:
  Phase I flow only
  target 10M exposures
  Phase II flow + pixel losses
  target 2M exposures

推理:
  Faithful 1-step default
  4-step optional quality mode
  tiled inference for large images
```

---

# 二十二、v1.0 完成检查表

- [ ] 项目代码与其他项目完全独立；
- [ ] 仅加载 Mage-VAE 预训练权重；
- [ ] Mage-VAE ceiling 完成；
- [ ] decoder grounding 已按 ceiling 结果决定；
- [ ] SR eligibility index 完成；
- [ ] validation 与 train 零重叠；
- [ ] degradation 和 resume 逐像素确定；
- [ ] Codec Bank 完成；
- [ ] Pixel Baseline 完成；
- [ ] Flow smoke 全部测试通过；
- [ ] Base-128M Phase I 完成；
- [ ] Phase II Faithful 完成；
- [ ] 一步输出通过保真硬门槛；
- [ ] 四步输出经过独立验收；
- [ ] 至少200张真实 LQ 完成人工盲评；
- [ ] tiled inference 无可见接缝；
- [ ] release manifest 包含模型、VAE、数据和代码指纹；
- [ ] 未通过验收的可选模块不进入发布模型。

**最终交付物是一套约 128M 参数、以恢复为优先、仅接受图片输入的 4× AnimeSR 模型。它不再是删除文本后的生成 DiT，而是围绕图像恢复重新设计的 residual-coordinate U-Flow。**