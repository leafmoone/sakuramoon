# 退化管线 v1（plan §11 摘录；细节以 plan-v2.0.md 为准）

在线 GPU 退化：每条训练样本按 profile 权重采样一条退化链，输出尺寸
**精确为 HR/4**（固定 4×，§10.1）。所有参数在数据加载时由确定性 seed
派生：`H(global_seed, sample_id, data_cycle, exposure_index)`（§11.5）。

## 1. Profile（§11.1，权重见 `config/data.toml [degradation]`）

| Profile | 权重 | 含义 |
|---|---:|---|
| P0_clean | 10% | 只做 4× 下采样（area/bicubic），不额外加污损 |
| P1_mild_web | 25% | 轻度 web：小模糊 + 轻噪 + 高质 JPEG |
| P2_normal_web | 35% | 常规 web：中等模糊/噪 + 中质 JPEG（主力退化） |
| P3_anime_codec | 20% | 动漫编码（codec bank 喂入，见 §4） |
| P4_severe | 10% | 重度：大模糊 + 重噪 + 低质 JPEG + 可能 banding/posterize |

## 2. 在线 GPU 算子（§11.2）

- **模糊**：isotropic / anisotropic（方向采样）、motion、sinc（带通）；
  σ 范围（/px）：mild 0.1–0.6，normal 0.2–1.2，severe 0.5–2.0。
  P1 ⑤（producer 提速）：iso/sinc 走 separable 两个 1D 卷积（outer-product
  2D 卷积的精确等价，O(L²)→2·O(L)）；所有卷积边界由零填充改为
  **reflect 填充**（模糊/锐化不再在 crop 边缘产生暗晕）——LQ 数值与
  零填充版有 ULP/边界级差异，属 P1 计划内变更（canary 旧树不受影响）。
- **下采样滤波器**：area / bicubic / bilinear / nearest 随机选一；
  抗混叠（anti-alias）开关独立采样。
- **噪声**：gaussian / poisson / chroma；σ（/255 基准）：mild 0–2，
  normal 0–5，severe 2–10。
- **JPEG 近似**（在线快速近似，非完整 DCT）：质量 mild 80–98 /
  normal 55–90 / severe 30–70。
- **色彩/位深**：gamma 抖动、banding、posterize、dither、chroma 子采样近似。
- **锐化**：unsharp（少量样本）。

## 3. Codec Bank（§11.3，离线预生成）

- 50k–100k 个 HR crop，每 crop 1–2 个版本（不同编码器/参数），
  总 20–50GiB；占在线 batch 的 10–20%（`codec_bank_batch_fraction`）。
- 用于 P3_anime_codec 与部分 P4（真实编码块效应，在线近似无法完全替代）。

## 4. 正确性要求（§11.4）

- 确定性：同 seed 复现逐像素一致（resume 安全）。
- 数值安全：全程 fp32 计算退化，clamp 回 [-1,1]；不产生 NaN/Inf。
- 速度：退化耗时 < 5% step time（§10 数据服务纪律）；算子全部 GPU 原生
  或预计算 LUT。
- 审计：每个 batch 记录 profile 直方图 + 关键参数均值，进 telemetry。

## 5. 与旧方案差异

RFMSR v1 的退化配方是「4× 固定下采样 + 单 JPEG」单链；v2.0 是
5 profile × 多算子采样链 + 离线 codec bank，覆盖度对齐 §16.1 的
5000 张 synthetic paired（每 profile 1000 张验证）。
