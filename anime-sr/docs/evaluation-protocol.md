# 评测协议（plan §16 + §9.2 + §17.1 摘录；细节以 plan-v2.0.md 为准）

## 1. 评测集（§16.1，与 `docs/data-contract.md` §5 一致）

| 集合 | 规模 | 用途 |
|---|---:|---|
| VAE ceiling | 2000 真实 + 500 合成 | M0 门（decoder 上限），含插画/动画帧/漫画/黑白线稿/小字/规则网格/密集排线/大面积平涂/极端宽高比 |
| Synthetic paired | 5000（每 degradation profile 1000） | 配对指标、profile 级归因 |
| Stress set | 500 | 眼睛/睫毛/发丝/文字/手指/网格/漫画网点/淡线/平涂/摩尔纹 |
| Real-LQ | 最低 200 / 目标 500 | 真实退化泛化（吸收旧「20+ crops」决策） |

所有集合与训练零重叠（`sr-validation-v1.json` 记录证明）。

## 2. 对比模型（§16.2）

Bicubic、Pixel Baseline（M2）、APISR、Real-CUGAN、Real-ESRGAN Anime、
Faithful 1-step、Faithful 4-step、Mage-VAE ceiling。
大型通用生成式 SR 仅作 hallucination 与视觉上限参考，不作主要目标。

## 3. 指标（§16.3）

- 通用：PSNR-RGB / PSNR-Y / SSIM / LPIPS / DISTS / ΔE00。
- Anime 专项：edge precision/recall/F1、edge displacement、line width error、
  line continuity、flat-region variance、flat-region high-frequency energy、
  chroma noise、ringing score、OCR accuracy。
- 人工错误清单（逐项统计）：线条新增/消失、眼睛形状变化、文字笔画变化、
  手指变化、脸型变化、颜色漂移、平涂纹理化、halo、window seam、tile seam。

## 4. Faithful 硬门槛（§16.4，Faithful Base 通过才可谈质量卖点）

**轻度退化**：PSNR-Y 相对 Pixel Baseline 下降 ≤ 0.20dB；edge displacement
不得恶化；flat artifact rate 不得上升。
**中度退化**：LPIPS 或 DISTS 至少改善 5%；人评相对 APISR/Pixel Baseline
偏好率 ≥ 60%。
**重度退化**：人评偏好率 ≥ 60%；严重结构错误率不得高于通用生成式 SR；
文字与眼睛错误单独统计。
**一步 vs 四步**：4-step 相对 1-step 改善 < 2% → 不把 4-step 作为正式卖点。

## 5. M0 VAE ceiling 门（§9.2，决定 decoder grounding 开关）

任一条件持续失败 → 开启 §9.3 grounding（4–6M adapter 或 5–8M zero-init
post-refiner，二者不并用）：

- 细线 edge F1 显著下降
- 线宽误差中位数 > 0.35 HR pixel
- 小字 OCR 准确率下降 > 2 个百分点
- 规则网格或漫画网点发生明显结构破坏
- 平涂区域 ΔE00 中位数 > 1.5

阈值先在 2000 张验证图 + 合成线条图上校准。
M0 实测（2026-08-25，`docs/evidence/m0-bakeoff-2026-08-25.md`）：
linewidth_ratio=1.0、线稿 PSNR 44–52.6dB、flat ΔE76=1.24 → **PASS，默认
`grounding_enabled = false`**（弱点仅 ≤4px 致密周期结构，非 anime 线稿风险区）。

## 6. 推理模式（§17.1）

| 模式 | σ | steps | seed |
|---|---|---:|---|
| Faithful（默认） | 0 | 1 | 确定性 |
| Faithful Quality | 0 | 4（Heun [0,.25,.5,.75,1]，末步可退 Euler） | 确定性 |
| Balanced | 0.05–0.10 | 1 或 4 | 固定 seed |

大图（§17.2）：优先整图，显存不足再 tile；LQ tile 256 / overlap 64（LQ）/
256（HR），cosine feather 融合，reflect padding；窗口与 tile 的 2D RoPE 必须
用**全图绝对坐标**（否则 tile 间位置不一致）。

## 7. 发布物（§17.3）

release manifest 记录：git commit、VAE SHA256（`FrozenMageVAE.weights_sha256()`）、
数据版本（manifest/degradation 版本）、评测集哈希。
（注：此条为用户冻结计划书要求，对 anime-sr 子树优先于仓库
「不维护哈希」通则。）
