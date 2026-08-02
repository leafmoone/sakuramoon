# 可观测性与评估补充决定

状态：用户新增要求，纳入实施路线图；具体 benchmark 后数值允许通过配置修订。

## 训练指标

- 默认使用 Weights & Biases；project/entity/run 等运行设置从 TOML 读取，API key 只从 `WANDB_API_KEY` 环境变量读取。所有指标每个 successful update 先立即 append 到本地持久化 JSONL，再异步上传；S0 当前 `flush_every_updates=1`，即每条 append 后执行一次 `fsync`。网络异常不得改变训练和 checkpoint 语义。
- 每个 successful optimizer update 至少记录高噪区域、低噪区域和总 loss；high noise 固定为 `t<0.95`，low noise 固定为 `t>=0.95`。分桶只用于观测聚合，总 loss 始终由完整 batch 的严格 JLT loss 计算。另记录 pre-clip grad norm、post-clip grad norm、clip fraction、learning rate、timestep 分布、有效 batch、image/text tokens、samples/s、显存和 non-finite 计数。
- 按数据、Qwen、VAE、条件聚合、DiT forward、loss、backward、DDP、clip、optimizer、checkpoint 和 evaluation 分段记录 wall-time。GPU 区间使用 CUDA events，常驻计时不得逐段强制同步。
- 任务开发耗时与训练运行耗时分开记录。实现/审查任务写入 `docs/model-architecture/progress/time-log.jsonl`，训练性能写入运行 artifact 与 W&B。

## FID 与 IS

- FID/IS 必须由 TOML 配置控制，按 successful optimizer updates 和 stage end 触发，不能写死在训练循环。
- S0 当前 TOML cadence 为每 1,000 个 successful updates；趋势/验收样本数、IS splits、batch size、output reserve、extractor/preprocess/real-stat 路径和版本尚未决定，正式 `eval.toml` 必须逐项硬失败，不能从示例值推断。
- evaluator 固定并记录生成 checkpoint、validation prompt/condition 选择、seed 44、尺寸、CFG 2.9、Heun-50/99 NFE、样本数、IS splits、特征提取器/预处理版本和 real-stat provenance。用户不提供配置 SHA-256；程序可以为已读取的 artifact 自动生成内部内容 ID，用于同一次发布的一致性和 no-clobber，不把它当作用户待填参数。
- 趋势 FID/IS 与正式 FID/IS 使用不同 artifact kind，禁止混报。2,000 张 VAE 重建集不能作为正式 FID real reference。
- FID/IS 是趋势与回归指标，不单独构成发布门槛；现行优先级仍是 tag 控制、审美质量、NL 跟随、宽高比与分辨率。
- 评估由完整 checkpoint 驱动，默认作为显式 evaluator job 执行；正式 stage-end 必须比较同 lineage 的 raw、PMA-10 与 accepted，缺少任一角色或本地 evaluator identity 时硬失败且不下载替代资产。其 GPU 占用和训练暂停成本必须计入报告，不得无记录地与训练争抢 GPU。
- validation prompt 从固定两个 tar 的 2,099 个 image+JSON 对读取，并复用生产 typed caption parser；tags-only 记录不得被伪装为 NL caption。单 RAW、2 样本 manual-quality 的 bounded 模式只允许显式 `engineering_only`，永远标记为 `synthetic_bounded_engineering_only`，不能冒充 FID/IS 或发布证据。

## 配置要求

训练运行配置至少包含 `[logging]`、`[wandb]` 与 `[timing]`；profiling/evaluation 可显式关闭。正式 evaluator 配置必须包含 `[evaluation]`、extractor、FID、IS 与 manual-quality 分组。W&B key 与 ModelScope token 只允许通过 `WANDB_API_KEY` 和 `MODELSCOPE_API_TOKEN` 环境变量进入进程，不允许进入 resolved config、日志或 artifact。
