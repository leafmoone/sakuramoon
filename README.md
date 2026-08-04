# SakuraMoon

本地文生图训练项目，仅保留模型、数据服务、训练、模型保存、采样和 FID/IS。

## 目录

```text
config/                   TOML 运行配置
src/sakuramoon/model/     DiT
src/sakuramoon/encoders/  Qwen 与 VAE
src/sakuramoon/data/      WebDataset 与数据服务
src/sakuramoon/train/     训练循环
src/sakuramoon/eval/      FID/IS
output_model/             新模型与评估结果（运行后生成）
```

## 全新训练

先在终端 1 启动数据服务：

```bash
cd /root/shared-nvme/sakuramoon
PYTHONPATH=src uv run --no-sync python -m sakuramoon.cli.data_service \
  --config train_s0.toml \
  --config-root /root/shared-nvme/sakuramoon/config \
  --root /root/shared-nvme/sakuramoon
```

数据服务就绪后，在终端 2 开始训练：

```bash
cd /root/shared-nvme/sakuramoon
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run --no-sync python -m sakuramoon.cli.train \
  --config train_s0.toml \
  --config-root /root/shared-nvme/sakuramoon/config \
  --root /root/shared-nvme/sakuramoon
```

不要添加 `--resume`；运行命令使用 `--no-sync`，只用现有环境，不读取或生成
项目锁。依赖声明只维护 `pyproject.toml`。模型写入 `output_model/s0/`。

`config/train_s0.toml` 的 `[evaluation].every_updates` 控制 FID/IS 周期，默认
为 1000 个成功 update。结果写入 `output_model/evaluation/s0/step-<update>.toml`
和 `latest.toml`。
