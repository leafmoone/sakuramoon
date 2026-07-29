# SakuraMoon

SakuraMoon 是一个从零训练二次元文生图基础模型的本地工程。实现按
[`IMPLEMENTATION_ROADMAP.md`](docs/model-architecture/progress/IMPLEMENTATION_ROADMAP.md)
中的任务依赖顺序推进；任务通过实现、独立审查和最终验证后，才可形成单独的原子提交。

## 规范来源

实现依据按以下优先级解释，低优先级内容不得覆盖高优先级决定：

1. [`current/confirmed-decisions.md`](docs/model-architecture/current/confirmed-decisions.md)
2. [`current/open-items.md`](docs/model-architecture/current/open-items.md)
3. [`current/observability-and-evaluation.md`](docs/model-architecture/current/observability-and-evaluation.md) 等本地补充决定
4. `docs/model-architecture/archive/` 中的历史组件页

`archive/` 只用于追溯，不是配置来源。项目的模型架构文档入口是
[`docs/model-architecture/README.md`](docs/model-architecture/README.md)；本工程不再调用或恢复 Notion MCP。

## 工程约束

- 所有训练参数必须显式来自 `config/*.toml`，运行代码不得提供隐式训练默认值，也不得从历史候选方案补值。
- 除 `all_condition=0.10` 外的 dropout 数值仍未决定；相关生产配置在用户明确决定前必须保持不可运行。
- 环境和 Python 依赖统一由 `uv` 管理。R002 完成前不得假定环境已安装或锁定，也不得用临时安装替代锁文件。
- 当前单卡 RTX 5090 证据只能关闭明确标为 CPU 或 1GPU 的门槛，不能替代任何 4GPU DDP、NCCL、吞吐或恢复验收。
- 禁止静默 fallback，禁止自动改变 batch、world size、backend、LR 或绕过 preflight。

## 资产与凭据

`.env`、模型、数据库、数据集、缓存、参考仓库、checkpoint、W&B、profile 和训练产物均位于 Git 边界之外。Git 只保存可审计的来源、revision、哈希、schema、许可证与 manifest；详细规则见
[`asset-policy.md`](docs/model-architecture/progress/asset-policy.md)。

ModelScope 与 W&B 凭据只能通过环境变量注入。不得读取、打印、写入配置、日志、artifact 或提交本地 `.env` 中的值。

协作和任务执行规则见 [`AGENTS.md`](AGENTS.md)。
