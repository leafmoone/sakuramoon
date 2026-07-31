# 可观测性与评估补充决定

状态：用户新增要求，纳入实施路线图；具体 benchmark 后数值允许通过配置修订。

## 训练指标

- 默认使用 Weights & Biases；所有指标同时先写本地持久化 JSONL，再异步上传，网络异常不得改变训练和 checkpoint 语义。
- 每个 successful optimizer update 至少记录高噪区域、低噪区域和总 loss；high noise 固定为 `t<0.95`，low noise 固定为 `t>=0.95`。分桶只用于观测聚合，总 loss 始终由完整 batch 的严格 JLT loss 计算。另记录 pre-clip grad norm、post-clip grad norm、clip fraction、learning rate、timestep 分布、有效 batch、image/text tokens、samples/s、显存和 non-finite 计数。
- 按数据、Qwen、VAE、条件聚合、DiT forward、loss、backward、DDP、clip、optimizer、checkpoint 和 evaluation 分段记录 wall-time。GPU 区间使用 CUDA events，常驻计时不得逐段强制同步。
- 任务开发耗时与训练运行耗时分开记录。实现/审查任务写入 `docs/model-architecture/progress/time-log.jsonl`，训练性能写入运行 artifact 与 W&B。

## FID 与 IS

- FID/IS 必须由 TOML 配置控制，按 successful optimizer updates 和 stage end 触发，不能写死在训练循环。
- 示例初值为每 10,000 个 successful updates 运行一次 10,000 样本趋势评估；stage 正式验收使用 50,000 样本。该数值是路线图示例，目标机 benchmark 后可显式修改。
- 固定并记录生成 checkpoint、prompt/condition manifest、seed、尺寸、CFG、Heun-50/99 NFE、样本数、IS splits、特征提取器版本和 real-stat SHA-256。
- 趋势 FID/IS 与正式 FID/IS 使用不同 artifact kind，禁止混报。2,000 张 VAE 重建集不能作为正式 FID real reference。
- FID/IS 是趋势与回归指标，不单独构成发布门槛；现行优先级仍是 tag 控制、审美质量、NL 跟随、宽高比与分辨率。
- 评估由完整 checkpoint 驱动，默认作为显式 evaluator job 执行；其 GPU 占用和训练暂停成本必须计入报告，不得无记录地与训练争抢 GPU。

## 配置要求

运行配置至少包含 `[logging]`、`[wandb]`、`[timing]`、`[profiling]`、`[evaluation]`、`[evaluation.fid]` 和 `[evaluation.is]`。W&B key 与 ModelScope token只允许写环境变量名，不允许进入 resolved config、日志或 artifact。
