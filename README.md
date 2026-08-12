# SakuraMoon

本地文生图训练项目，仅保留模型、数据服务、训练、模型保存、采样和
FID/IS/KID/CMMD 评估。

## 目录

```text
config/                   TOML 运行配置
src/sakuramoon/model/     DiT
src/sakuramoon/encoders/  Qwen 与 VAE
src/sakuramoon/data/      WebDataset 与数据服务
src/sakuramoon/train/     训练循环
src/sakuramoon/eval/      FID/IS/KID/CMMD 与 VAE 重建评估
output_model/             新模型与评估结果（运行后生成）
```

## 运行环境（DTK 双卡）

- 代码仓库：`/root/private_data/sakuramoon`（50G 网络盘，只放代码与配置）
- 运行根目录：`/sakuramoon-runtime`（本地磁盘；`data/`、`cache/`、`model/`、`output_model/` 都落在其中，
  避免写满 `/root/private_data`）
- 数据队列状态：`config/data-service-mainset.json`（随项目放网络盘，迁移时随项目 tar 自动携带；
  **不要**放回 `/sakuramoon-runtime/cache/`，否则容器重建/迁移会丢失队列位置并从 cycle 0 重来）
- 两张 DCU：`ROCR_VISIBLE_DEVICES=0,1` / `CUDA_VISIBLE_DEVICES=0,1`
- 日志目录：`/root/sakuramoon-logs/`

## 1. 数据服务（终端 1）

负责下载/预取训练分片并提供给训练进程，先于训练启动。
队列位置（`cycle` + 各分片状态）保存在 `config/data-service-mainset.json`，属必要迁移文件：

```bash
cd /root/private_data/sakuramoon
.venv/bin/python -u -m sakuramoon.cli.data_service \
  --config train_s0.toml \
  --config-root /root/private_data/sakuramoon/config \
  --root /sakuramoon-runtime
```

查看日志：`tail -F /root/sakuramoon-logs/data-service-accelerate.log`

## 2. 训练（终端 2，等数据服务就绪）

双卡 accelerate 训练：

```bash
cd /root/private_data/sakuramoon
.venv/bin/python .venv/bin/accelerate launch \
  --multi_gpu --num_processes 2 --num_machines 1 \
  --mixed_precision no --dynamo_backend no --main_process_port 29500 \
  -m sakuramoon.cli.train \
  --config train_s0.toml \
  --config-root /root/private_data/sakuramoon/config \
  --root /sakuramoon-runtime
```

断点续训：追加 `--resume /sakuramoon-runtime/output_model/s0/ckpt_<n>_raw-<n>-update-cadence`，
使用 `/sakuramoon-runtime/output_model/s0/` 下最新的完整 checkpoint。全新训练不要加 `--resume`。

完整 checkpoint 每 1000 个 update 写入
`/sakuramoon-runtime/output_model/s0/ckpt_<n>_raw-<n>-update-cadence/`
（目录内含 `COMPLETE` 与 `manifest.json`）。

查看日志：`tail -F /root/sakuramoon-logs/train-accelerate.log`

## 3. 上传 checkpoint 到 ModelScope（终端 3）

发布器每 `INTERVAL_SECONDS`（默认 600 秒）把 `SOURCE_ROOT` 下带 `COMPLETE` 的完整
checkpoint 镜像上传到私有仓库 `leafmoone/sm_train_state/s0`，上传后做远端校验。
训练输出在 `/sakuramoon-runtime` 时直接把它作为 `SOURCE_ROOT`，无需先同步到仓库：

```bash
cd /root/private_data/sakuramoon
SOURCE_ROOT=/sakuramoon-runtime/output_model/s0 \
STATE_ROOT=/root/.sm-train-state-publisher \
  bash scripts/publish_train_state.sh loop
```

注意：`STATE_ROOT` 也要放在本地盘（`/root/...`），因为发布器用硬链接暂存，跨文件系统
（NFS 的 `/root/private_data`）会失败。

可选环境变量：`REPO_ID=leafmoone/sm_train_state`、`REPO_PATH=s0`、
`INTERVAL_SECONDS=600`、`UPLOAD_WORKERS=4`。

发布日志：`tail -F /root/private_data/.sm-train-state-publisher/publisher.log`

## 评估

`config/train_s0.toml` 的 `[evaluation].every_updates` 控制 FID/IS/KID/CMMD
周期，默认每 1000 个成功 update 运行一次。四项生成指标复用同一批生成图；
真实验证集的 Inception feature、CLIP feature 和 Inception logits 按数据指纹、
样本数与分辨率一次性缓存到
`output_model/evaluation/s0/cache/real-features-*.pt`。结果写入
`output_model/evaluation/s0/step-<update>.toml` 和 `latest.toml`。缓存契约或
验证集发生变化时不会静默复用不匹配的缓存。

VAE 重建评估必须只暴露一张 GPU，并从验证集抽取两个互不重叠的确定性子集：
第一组计算 reconstruction FID、LPIPS、PSNR 与 MS-SSIM，第二组计算
real-real FID。示例：

```bash
cd /root/private_data/sakuramoon
.venv/bin/python -u -m sakuramoon.cli.vae_reconstruction \
  --config train_s0.toml \
  --config-root /root/private_data/sakuramoon/config \
  --root /sakuramoon-runtime \
  --sample-count 512 \
  --batch-size 16 \
  --comparison-count 16
```

结果、逐样本元数据、重建图片和对比网格默认写入
`output_model/evaluation/vae-reconstruction/`。缺失模型、样本不足、两个子集
重叠、非有限指标或输出冲突都会直接失败，不会缩小样本或静默跳过。
