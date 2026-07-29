# SakuraMoon Agent Rules

本文件适用于整个仓库。子目录若增加更具体的 `AGENTS.md`，只能收窄本文件规则，不能放宽安全、配置或验证门槛。

## 规范与范围

1. 开工前完整阅读任务文件、模型架构入口、现行决定、开放事项、可观测性决定、路线图和对应审查证据。
2. 冲突优先级固定为 `current/confirmed-decisions.md` > `current/open-items.md` > 本地补充决定 > 历史组件页。
3. `docs/model-architecture/archive/` 只读。历史候选、推荐值或被取代决定不得回流到实现或配置。
4. 不调用或恢复 Notion MCP；本地 `docs/model-architecture/` 是唯一文档入口。
5. 一次只执行一个任务 ID，只修改任务文件列出的允许路径，不顺手完成后续任务。

## 配置与环境

- 使用 `uv` 管理 Python 环境和依赖；不以临时 `pip` 安装、系统包漂移或未锁 Git HEAD 替代工程锁定。
- 所有训练参数只从 `config/*.toml` 进入运行时。schema 必须严格拒绝缺失、未知、错误类型和越界值；训练语义字段禁止代码默认值。
- `all_condition=0.10` 已锁定。其他 dropout 数值必须保持显式未决，用户决定前不得编造或从历史文档推断。
- 禁止静默 fallback，禁止自动更改 batch、accumulation、world size、backend、LR、token 上限、功能开关或 checkpoint 频率；preflight 硬失败不得提供绕过开关。

## 凭据与资产

- 不读取、输出、记录、上传或提交 `.env` 的内容。只允许在代码和文档中记录环境变量名，例如 `MODELSCOPE_API_TOKEN`。
- `.env`、私钥、模型权重、DB、dataset/cache、参考嵌套仓库、checkpoint、W&B、profile、trace 和训练 artifact 永不进入 Git。
- 大模型和 DB 只以 manifest/schema 边界进入仓库：来源、不可变 revision、文件路径、bytes、SHA-256、许可证、访问限制和必要的非敏感摘要。
- `reference/` 保持根仓忽略状态，不作为 submodule 或 vendor 代码提交。引用外部实现前必须锁定 remote、commit、许可证并单独审查归属与兼容性。
- 资产规则和当前参考仓库记录以 `docs/model-architecture/progress/asset-policy.md` 为准。

## 验证与证据

- 仅 import、shape 检查或 mock 不能替代任务要求的真实数据、Qwen、VAE、kernel、forward/backward/update 或质量验证。
- 当前 1GPU 结果不能外推为 4GPU 结论。要求四卡的项目在真实 4xRTX 5090 可用前必须保持阻塞，不得缩减 world size 关闭门槛。
- 每个任务分别维护 task 状态、实现报告、AI/模型正确性审查、Infra/性能审查、测试报告、耗时、artifact 索引和适用的 before/after 性能证据。
- 实现代理先从 AI/模型正确性与 Infra/性能两方面自检；实现结束后由独立审查代理复核。审查问题交回原实现代理修复，再由原审查代理复审。
- kernel、optimizer、DDP、checkpoint、growth 和故障注入任务分别使用两个独立审查代理。
- 主代理在任务验收通过后运行最终验证并创建一个原子 commit。实现和审查代理不得创建 commit。

## 工作树安全

- 保留不属于当前任务的用户改动，不执行 `git reset --hard`、`git checkout --` 或递归删除工作区。
- 使用 `apply_patch` 进行人工文件编辑。测试和生成命令不得写入任务允许路径之外。
- Secret 扫描只针对 Git 索引或显式 tracked manifest，绝不以扫描为由读取被忽略的 `.env`。
