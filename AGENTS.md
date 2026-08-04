# SakuraMoon 项目规则

本仓库只维护文生图模型、数据服务、训练、模型保存、采样和 FID/IS。

- Python 依赖只维护 `pyproject.toml` 中的兼容版本范围；现有环境使用 `uv run --no-sync`。
- 训练参数从 `config/*.toml` 读取，不在代码里复制训练参数。
- 不为配置、依赖、数据、模型或模型保存文件计算、保存或校验项目级哈希。
- Qwen 与 VAE 只从 `model/qwen_3.5_2B/` 和 `model/vae/` 加载；缺失时直接报错。
- 不读取、打印或提交 `.env`、令牌、私钥、模型权重、数据集、缓存和运行产物。
- `reference/` 只供人工参考，生产代码不得导入其中内容。
- 数据服务使用本地 manifest、验证分片和按字节数检查；不维护额外溯源、兼容或审计层。
- 新模型写入 `output_model/`，不得创建名为 `checkpoints` 的输出目录。
- CLI 正常输出进度与异常 traceback，不把错误统一压缩成 JSON。
- 只保留能保护当前核心路径的测试，不恢复历史任务文档、审查材料、兼容测试或旧输出。
- 修改后至少运行 Ruff、Pyright 和相关单元测试；GPU 测试按当前硬件可用性执行。
- 保留与当前任务无关的用户改动，不使用 `git reset --hard` 或 `git checkout --` 覆盖工作树。
