# AnimeSR-Mage-UFlow 设计文档（v2.0，2026-08-26 冻结）

> 权威规格：`docs/plan-v2.0.md`（2320 行完整计划书）。本文是工作性浓缩：
> 架构、数学、里程碑与决策日志。旧 RFMSR 方案见 `docs/superseded/rfmsr-design-v1.md`。

## 1. 定位

纯图片 4× 二次元超分（blind SR：输入真实退化 LQ，无参考）。不引入
文本/CFG/T2I/cross-attn/ControlNet/GAN/DWT/DINO/OCR-loss（v1 明确不做）。

唯一复用的预训练组件：**Mage-VAE**（microsoft/Mage-Flow，MIT，128 通道、
16× 下采样、对称一步扩散编解码）。已 vendor 进
`src/anime_sr/vae/mage_vae_impl.py`（仅换 loguru→stdlib logging，附来源头）。

M0 前置 bake-off（2026-08-25，RTX 4060 bf16）已 PASS：1px 线
linewidth_ratio=1.0、线稿 PSNR 44–52.6dB、flat ΔE76=1.24、encode+decode
0.135s@1024²、峰值显存 ~1GB（见 `docs/evidence/m0-bakeoff-2026-08-25.md`）。

## 2. 流水线

```
HR 像素 ──冻结 Mage-VAE encode (no_grad, 后验均值)──> z_hr [B,128,H/16,W/16]
LQ 像素 ──4× bicubic 放大──> 冻结 Mage-VAE encode──> z_lr（LQ-centered 源分布）
δ = z_hr − z_lr；r0 = σε；rt = (1−t) r0 + t δ
模型 v_θ(rt, t, c) 预测速度（v-pred only，σ 标量条件 + t + stage FiLM）
一步：δ̂ = r0 + v_θ(r0, 0, c)；ẑ = z_lr + δ̂
四步：Heun，时间点 [0, .25, .5, .75, 1]，末步可退 Euler（少一次网络评估）
解码：冻结 Mage Decoder t=0 单步（Stage II 起梯度可穿过，参数冻结）
```

推理模式（§17.1）：Faithful（σ=0，1 步，确定性，默认）/ Quality（σ=0，4 步）/
Balanced（σ=0.05–0.10，固定 seed，实验性）。

## 3. 模型结构

### 3.1 Pixel Condition Encoder（§6，9–12M，原生 LQ 像素条件，不先放大）

```
RGB 256²  Stem 3×3 48ch → 2 blocks
→ 128² 96ch ×2 → 64² 128ch ×3 → 32² 192ch ×3 → 16² 256ch ×4
Block = LayerNorm2d → depthwise 3×3 → pointwise ×2 → SiLU/SimpleGate → pointwise → residual
输出：p64/p32/p16（条件）+ p128/p256（可选 decoder grounding）+ GAP(p16) 全局摘要
```

### 3.2 U-Flow 主干（§7，核心 121–128M）

| Stage | Grid | Dim | Depth | Q/KV heads | FFN | Attention |
|---|---:|---:|---:|---|---:|---|
| enc S0 | 64² | 384 | 4 | 6/2 | 1152 | 8×8 window |
| enc S1 | 32² | 512 | 6 | 8/2 | 1536 | 8×8 window |
| bottleneck | 16² | 768 | 8 | 12/4 | 2304 | global |
| dec S1 | 32² | 512 | 6 | 8/2 | 1536 | 8×8 window |
| dec S0 | 64² | 384 | 4 | 6/2 | 1152 | 8×8 window |

统一：head_dim=64、GQA、qk_norm、RMSNorm、SwiGLU、dropout=0、continuous 2D
RoPE（tile 用全图绝对坐标）、LayerScale 1e-3、pixelunshuffle2/1×1 下上采样、
window 用 normal/shifted 双模式消缝。block = RMSNorm→FiLM→GQA attention→
LayerScale→residual + RMSNorm→FiLM→SwiGLU(depthwise 3×3 在扩展支)→LayerScale
→residual。输出头：RMSNorm→3×3(384→384)→SiLU→3×3(384→128) 零初始化 → 128ch velocity。

### 3.3 Decoder Grounding（§9，默认关）

M0 ceiling 触发条件（§9.2，任一持续失败即开启 4–6M adapter）：
细线 edge F1 显著下降 / 线宽误差中位数 > 0.35 HR px / 小字 OCR 降 > 2pt /
规则网格或网点结构破坏 / 平涂 ΔE00 中位数 > 1.5（阈值先在 2000 验证图+合成线图上校准）。
方案：decoder 128/256/512 stage 注入（Norm→1×1→dw3×3→SiLU→zero-init 1×1→residual），
或独立 5–8M zero-init post-refiner，二者不可并用。

## 4. 训练（§13/§14）

- **Phase I（M4）flow-only**：10M 曝光（6M–16M），课程 I-A 20% 全 128→512 →
  I-B/I-C/I-D 渐增 256→1024，末尾 +0.5M 全 256。loss 仅 L_FM。
- **Phase II（M5）one-step faithful**：2M 曝光（1M–4M），20% 192 + 80% 256；
  50% random-t flow batch（只算 L_FM）+ 50% t=0 decode batch
  （L_FM+pixel+edge+flat+perceptual）；2000-update 梯度校准窗
  （pixel 0.25–0.50 / edge 0.05–0.15 / flat 0.03–0.10 / perc 0.05–0.15）。
- 优化器：AdamW β=[0.9,0.95] wd=0.05（norm/bias/layerscale/position/gates 不衰减），
  warmup 3% + cosine 至 10% 底；Phase II 分组 LR：U-Flow 2e-5 / pixel-enc 5e-5 /
  adapters 1e-4 / VAE 0。clip 1.0。EMA 按样本 half-life 500k。
- 硬件（§15）：模型结构不随硬件变，只变 micro-batch/累积/ckpt/attention backend；
  token-based 累积（Phase I 目标 131072 latent tokens/update，Phase II 65536），
  累积超 64 → token 目标减半、LR×0.7。

## 5. 里程碑与门（§13/§16）

| M | 内容 | 门 |
|---|---|---|
| M0 | VAE ceiling：2000 真实 + 500 合成（线条/网格/小字/平涂/极端宽高比），PSNR/SSIM/LPIPS/edge F1/线宽/ΔE00/OCR | 触发条件不过 → 开 grounding（§9.2） |
| M1 | 数据索引+退化：manifest/eligibility、确定性 GPU 退化、codec bank 50–100k | 零重叠、逐像素可复现、data wait <5% step、worker 错误可终止训练 |
| M2 | 5–10M pixel baseline（U-Net/NAFNet 类），0.5–1M 曝光 | 保真下限 + 数据/退化验证 |
| M3 | 15–20M smoke U-Flow（dim 192/256/384，depth 2/2/4/2/2），100–200k 曝光 | 9 项正确性检查（§13 M3），不过不启动正式训练 |
| M4 | Phase I 10M | — |
| M5 | Phase II 2M | 梯度校准窗通过 |
| M6 | （可选）LoRA+perceptual adapter，仍不加 GAN | 人评偏软才进入 |

评测（§16）：VAE ceiling 2500 张 / synthetic paired 5000（每 profile 1000）/
stress 500（眼睛/睫毛/发丝/文字/手指/网格/网点/淡线/平涂/摩尔纹）/
real-LQ 最低 200 目标 500。对比：Bicubic、Pixel Baseline、APISR、Real-CUGAN、
Real-ESRGAN Anime、1-step、4-step、VAE ceiling。Faithful 硬门槛见
`docs/evaluation-protocol.md`。

## 6. 决策日志

- **2026-08-26 v2.0 冻结**：取代 RFMSR 0.3B+GAN+DWT 旧设计（旧设计
  0.3–0.5B 主干+0.17B VAE、两阶段 GAN、400–800 A100·h 预算）。新方案
  纯 128M 主干 + 冻结 VAE，bf16 推理 3–6GB 显存档，10M+2M 曝光。
- **M0 bake-off PASS**（08-25，4060）：见 §1 引用。弱点仅 ≤4px 致密周期
  结构（4px 网格 SSIM 0.69、8px 棋盘 0.26），anime 实际线稿不在风险区。
- **训练机（待定）**：sakrua2 双卡 Hygon DCU（DTK 26.04）MFU 基准
  DiT-0.3B 代理 best 28.8%@50T 假设峰值（≈14.4 TF/s/卡，b32 no-ckpt；
  b64+ckpt 11.2 TF/s），略低于 30% 门。A100 直开 or DTK 直开待用户拍板
  （DTK 线 wall-clock 估 35–73 天 vs 400–800 A100·h）。
  证据：`docs/evidence/mfu_bench_dtk.py` + `/root/mfu_gpu{0,1}_*.log` on sakrua2。
- **仓库位置**：sakuramoon 仓库 `SR` 分支（dev@9f39922 起），与 T2I 主线
  完全隔离（§8 独立代码库原则）；VAE 为 vendor 拷贝而非运行时依赖。
