# SakuraMoon 资产与 Git 边界

状态：R001 建立的仓库基线。资产内容的完整锁定由后续 `A001` 与数据任务执行；本文件只定义允许进入 Git 的元数据边界，并记录 R001 要求的参考仓库来源证据。

## 基本原则

1. Git 保存代码、配置、schema、来源、不可变 revision、哈希、许可证和审查证据，不保存凭据、模型权重、数据库内容、数据样本、缓存或运行产物。
2. `.gitignore` 是事故防线，不是授权机制。任何大文件或外部资产即使未命中 ignore，也必须先经过任务范围、来源、许可证和体积审查。
3. 当前架构与资产选择以 `current/confirmed-decisions.md` 为最高优先级。历史 Notion 候选和参考工程的默认值均不得成为配置来源。
4. 本仓库当前不使用 Git LFS，不允许以 LFS 规避资产边界。确需提交的小型测试 fixture 必须由独立任务明确列出来源、许可证、字节数和 SHA-256。

## 路径边界

| 本地路径或类型 | Git 状态 | 允许提交的替代物 |
|---|---|---|
| `.env`、`.env.*`（`.env.example` 除外）、私钥 | 忽略 | 环境变量名、空值示例、脱敏规则 |
| `model/`、模型权重扩展名 | 忽略 | `assets/manifest.toml` 中的 repo、revision、路径、bytes、SHA-256、配置/tokenizer 摘要和许可证 |
| `db/`、本地数据库扩展名 | 忽略 | schema、来源/revision、非敏感统计、文件级 bytes 与 SHA-256；不得包含数据行或可还原内容 |
| `data/`、`cache/` | 忽略 | 不可变 dataset/shard manifest、校验和、样本计数和缓存策略 |
| `reference/` | 忽略 | 本文件与审查 artifact 中的 remote、HEAD、license 和引用说明 |
| `checkpoints/`、`artifacts/`、`outputs/`、`samples/`、`runs/` | 忽略 | 运行 manifest、checksum、artifact kind 和外部/本地受控路径索引 |
| `wandb/`、`profiles?/`、`traces/`、`diagnostics/` | 忽略 | 脱敏后的摘要、指标 schema、报告和 trace 索引 |

## 模型与数据库 manifest 边界

后续 `A001` 可以提交 manifest 与校验工具，但不得提交实际资产。每个模型条目至少包含：稳定 asset ID、用途、本地逻辑路径、上游 repo、不可变 revision、逐文件相对路径/bytes/SHA-256、许可证或访问条款、tokenizer/config 哈希及冻结策略。Qwen 必须锁定 ModelScope `spawner/Qwen3_5_2b_claude_heretic_spawner`，Mage-VAE 必须锁定 Microsoft 官方来源；R001 不宣称已经验证本地权重身份或文件哈希。

DB 条目至少包含：稳定 asset ID、来源与不可变 revision、schema 版本、文件级 bytes/SHA-256、访问限制和允许输出的聚合统计。manifest 不得包含原始记录、caption、用户数据、访问 URL 中的签名参数或任何凭据。

凭据只按环境变量名引用。`MODELSCOPE_API_TOKEN` 和 `WANDB_API_KEY` 的值不得进入命令行、resolved config、exception、日志、W&B 或 artifact。

## 参考仓库锁定记录

以下值于 `2026-07-29T08:19:08Z` 从各嵌套仓库本地 Git 元数据和对应 HEAD 的许可证 blob 只读提取。根仓继续忽略整个 `reference/`；这些目录不是 submodule，也不表示允许复制其代码。

| 参考仓库 | origin URL | HEAD | 根许可证 | 许可证文件证据 |
|---|---|---|---|---|
| HDM | `https://github.com/KohakuBlueleaf/HDM.git` | `5fef7c4b71fe8386b497176021fe458810fdb7c0` | CC BY-NC-SA 4.0；含非商业与相同方式共享限制 | `LICENSE`; Git blob `c0469af8f694e33b950ee20a8c133e2d53c62343`; SHA-256 `c02a039a41caa77050b5d48880810777ec9e9c7c8bb740dca882611478f7319b` |
| JLT | `https://github.com/akatsuki-neo/JLT` | `aca236efa97aab3b7d865fd3d99a270431cf6ae5` | MIT，Copyright (c) 2026 akatsuki-neo | `LICENSE`; Git blob `df6ff06c9eaf46916c4983fd74677b803ae82cea`; SHA-256 `0e691a1683455d3cfa0808c60d9cb467accbb842355d2e43074c8efc0aef6532` |
| krea-2 | `https://github.com/krea-ai/krea-2.git` | `db3984fbc6e13b34c0064990fc2d95ac64d00058` | Apache License 2.0（根仓代码） | `LICENSE.md`; Git blob `b09cd7856d58590578ee1a4f3ad45d1310a97f87`; SHA-256 `50e6751797c50dedd75ef1b8a0d9e42f5f8472e9fbce91f34718e9f97b0c780a` |

krea-2 的 `assets/hf_samples/LICENSE.pdf` 是单独的 **KREA COMMUNITY LICENSE AGREEMENT**，不能按根目录 Apache-2.0 处理。该文件在锁定 HEAD 的 Git blob 为 `767c9b0bbb7609079f0ca16d5eb4fe447ca46b5b`，SHA-256 为 `b82a2805162bde714a4eb27b9063c4fc3345d08a30be055134a6160e5430ba74`。R001 不导入这些示例资产。

检查时 HDM 与 krea-2 工作树无变更；JLT 除 HEAD 外存在未跟踪的 `.ipynb_checkpoints/`。该本地状态不改变锁定 commit，且被根仓 `reference/` 边界整体排除。

## 引用与例外流程

- 参考实现只用于理解或对照。使用任何代码前，任务必须记录精确文件/行来源、适用许可证、修改说明和兼容性结论；HDM 的非商业许可证尤其不得被误当作宽松代码许可证。
- 远端 branch、tag 或默认分支不是不可变标识。manifest 和证据只接受 commit/revision 与逐文件哈希。
- 任何边界例外必须先更新现行决策或任务文件，列出允许路径、最大体积、许可证和回滚点，再由独立审查与主代理提交；不得通过 `git add -f` 临时绕过。
