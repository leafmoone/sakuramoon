# 数据契约（plan §10, §11 摘录；细节以 plan-v2.0.md 为准）

## 1. 张量契约（§4）

| 张量 | 形状 | dtype | 范围 |
|---|---|---|---|
| HR 像素 | [B, 3, H, W] | fp32 | [-1, 1]，H/W 为 64 倍数（latent 可被 4 整除） |
| LQ 像素 | [B, 3, H/4, W/4] | fp32 | [-1, 1]，LQ 为 16 倍数 |
| z_hr / z_lr / ẑ | [B, 128, H/16, W/16] | fp32 存储（bf16 训练） | Mage-VAE latent，无 patch packing、无 BN 归一 |
| 条件 c | pixel encoder 多尺度输出 p64/p32/p16(+GAP) + σ 标量 + t | bf16 | — |

固定 4×：LQ 256 → HR 1024 → latent 64。非方形 bucket：面积约束、HR 边为
64 倍数、LQ 边为 16 倍数。

## 2. 数据源与 bucket（§10.1）

- 原始数据沿用既有 **Danbooru WebDataset** 数据源（独立数据服务，不依赖
  SakuraMoon 运行时）。
- 训练 bucket：LQ 128 / 192 / 256 → HR 512 / 768 / 1024（latent 32 / 48 / 64）。
- 分辨率课程见 `config/stage1_flow.toml`（I-A…I-D）。

## 3. Manifest 与 eligibility（§10.2–10.3）

输出文件（`config/data.toml [data].outputs`）：

```
sr-eligibility-v1.parquet      # 每张图的 eligibility 判定（含原因码）
shard-summary-v1.parquet       # 分片级统计（张数、尺寸分布、字节数）
filter-report-v1.json          # 过滤漏斗（各排除规则命中数 + 人审抽样结果）
sr-validation-v1.json         # 验证集清单（zero-overlap 证明）
```

数据服务纪律（仓库规则 + plan §10）：本地 manifest、验证分片、按字节数
检查；不维护额外溯源/兼容/审计层。train/validation 零重叠，resume 后
退化必须逐像素可复现（seed = H(global_seed, sample_id, data_cycle,
exposure_index)），无静默跳样本，worker 错误必须能终止训练，data wait
< 5% step time，10k batch 连续压测通过。

## 4. 过滤规则（§10.4，`config/data.toml [filter]`）

- 硬排除：nsfw / gore / blood / logo / watermark / signature / text-heavy-ui；
  其中 35% 抽样人审后再丢弃（quality-gated）。
- 优先级池：quality ∈ {masterpiece, best, great, good}。
- 辅助池（monochrome / rough / 3d）上限 20%。
- crop 保留率 ≥ 80% 主体才保留；clean-score 惰性计算 + 缓存。

## 5. 验证集（§16.1）

| 集合 | 规模 | 说明 |
|---|---:|---|
| VAE ceiling | 2000 真实 + 500 合成 | 插画/动画帧/漫画/黑白线稿/小字/规则网格/密集排线/大面积平涂/极端宽高比 |
| Synthetic paired | 5000（每 degradation profile 1000） | 已知退化的配对数据 |
| Stress set | 500 | 眼睛/睫毛/发丝/文字/手指/网格/漫画网点/淡线/平涂/摩尔纹 |
| Real-LQ | 最低 200 / 目标 500 | 真实世界 LQ（旧「20+ crops」决策被此集吸收） |
