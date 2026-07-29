# SakuraMoon Agent Rules

本文件适用于整个仓库。子目录若增加更具体的 `AGENTS.md`，只能收窄本文件规则，不能放宽安全、配置或验证门槛。

## 规范与范围

1. 开工前完整阅读任务文件、模型架构入口、现行决定、开放事项、可观测性决定、路线图和对应审查证据。
2. 冲突优先级固定为 `current/confirmed-decisions.md` > `current/open-items.md` > 本地补充决定 > 历史组件页。
3. `docs/model-architecture/archive/` 只读。历史候选、推荐值或被取代决定不得回流到实现或配置。
4. 不调用或恢复 Notion MCP；本地 `docs/model-architecture/` 是唯一文档入口。
5. 任务包内可复用同一个实现代理，但必须按依赖顺序逐个关闭任务 ID；每个 ID 保持独立 task、diff、测试、状态和原子 commit，不得把多个 ID 合并提交。
6. `traceability.toml` 只允许首次 bootstrap；后续新增、移动或修改条款必须保留既有 ID，只为新条款分配从未使用的新 ID。不得按行号、顺序或当前文本重新编号。
7. 每个实现任务的允许路径必须包含 `docs/model-architecture/progress/traceability.toml`。同一原子 commit 内更新受影响 requirement 的逐条映射、状态与证据；提交自身使用 `task:<TASK_ID>`，后续提交可使用完整 40 位 commit SHA。
8. 修改 `current/` 时必须递增 source 与 registry revision、追加连续 SHA-256 changelog 并更新受影响 fingerprint；不得缩小 canonical source 或 heading scope。archive 仍保持只读。

## 配置与环境

- 使用 `uv` 管理 Python 环境和依赖；不以临时 `pip` 安装、系统包漂移或未锁 Git HEAD 替代工程锁定。
- 下载或安装依赖时优先选择与目标 Linux/Python/CUDA ABI 匹配的官方或上游开源 wheel，并在锁文件中固定 URL、版本与 SHA-256。只有在没有适用 wheel，或已有可复现证据证明候选 wheel 不兼容时，才允许从固定 Git commit 编译；源码构建必须记录工具链、ABI、构建变量和失败诊断，并仅在不争用当前任务资源时后台执行。
- 所有训练参数只从 `config/*.toml` 进入运行时。schema 必须严格拒绝缺失、未知、错误类型和越界值；训练语义字段禁止代码默认值。
- `all_condition=0.10` 已锁定。其他 dropout 数值必须保持显式未决，用户决定前不得编造或从历史文档推断。
- 禁止静默 fallback，禁止自动更改 batch、accumulation、world size、backend、LR、token 上限、功能开关或 checkpoint 频率；preflight 硬失败不得提供绕过开关。

## 凭据与资产

- 不读取、输出、记录、上传或提交 `.env` 的内容。只允许在代码和文档中记录环境变量名，例如 `MODELSCOPE_API_TOKEN`。
- `.env`、私钥、模型权重、DB、dataset/cache、参考嵌套仓库、checkpoint、W&B、profile、trace 和训练 artifact 永不进入 Git。
- 大模型和 DB 只以 manifest/schema 边界进入仓库：来源、不可变 revision、文件路径、bytes、SHA-256、许可证、访问限制和必要的非敏感摘要。
- `reference/` 保持根仓忽略状态，不作为 submodule 或 vendor 代码提交。引用外部实现前必须锁定 remote、commit、许可证并单独审查归属与兼容性。
- 资产规则和当前参考仓库记录以 `docs/model-architecture/progress/asset-policy.md` 为准。

## 执行协议

- 启动子代理时，第一步必须直接调用 `collaboration.spawn_agent`。禁止通过 `exec`、JavaScript、`shell_command`、打印字符串或其他间接方式尝试调用。
- `collaboration.spawn_agent` 直接调用失败时，只报告失败并立即停止该次启动；不得自动重试，不得输出 `ready`、`spawn`、`no-op` 等占位信息。
- 在依赖已满足、接口已冻结、写路径与 GPU/NVMe 资源不冲突时，尽量保持并发槽位满载；子代理完成后立即将释放槽位复用于可安全推进的后续工作。依赖未满足时只允许只读预审，不得提前实现或关闭任务。
- 低中风险任务按里程碑包执行：Foundation=`R002,D001,C001,A001`；Data=`D010-D015`；Encoders/Conditioning=`T020-T024`；Dense Model=`M030-M033`；Training Utilities=`T050-T053`。
- 低中风险任务包复用一个实现代理；包结束后由独立审查代理逐 ID 给出 AI/模型正确性与 Infra/性能结论。问题只交回受影响任务，不重跑无关任务。
- `K001`、`T040`、`T041`、`T042`、`T043`、`T054` 和所有正式 stage canary 保持单独实现、独立 AI reviewer、独立 Infra reviewer。
- 当前优先完成 CPU/单卡范围。所有多卡实现、DDP/NCCL 验证与正式多卡 stage 暂不执行，并保持显式 blocked/pending；单卡证据不得关闭四卡门槛。
- 允许并行执行依赖已满足、接口已冻结、不会写同一文件且不争用同一 GPU/NVMe 的任务。后台结果返回前，不得关闭依赖该结果的任务或里程碑。
- 每任务运行针对性 unit/contract/小型真实 GPU 测试；17x8 shape、100k 扫描、1,000-step canary 和完整恢复矩阵只在对应里程碑集中运行一次。
- 生产 FA4 和正式 canary 前允许显式配置 dense SDPA reference，执行真实 data/Qwen/VAE/DiT/loss/checkpoint 的 1-10 step engineering smoke；该证据不得作为 S0 放行。

## 验证与证据

- 仅 import、shape 检查或 mock 不能替代任务要求的真实数据、Qwen、VAE、kernel、forward/backward/update 或质量验证。
- 当前 1GPU 结果不能外推为 4GPU 结论。要求四卡的项目在真实 4xRTX 5090 可用前必须保持阻塞，不得缩减 world size 关闭门槛。
- 每个任务分别维护 task、实现摘要、针对性测试结果、耗时和 commit；共同环境、资产、benchmark 与里程碑证据由任务包共享引用，不重复复制。
- `perf_baseline.json`/`perf_after.json` 只为真正的性能任务生成，普通任务不得创建 N/A 占位文件。
- 实现代理先从 AI/模型正确性与 Infra/性能两方面自检；低中风险任务在包级审查，审查问题交回原实现代理修复并由原审查代理复审。
- 主代理在每个任务实现验收后运行最终验证并创建一个原子 commit。实现和审查代理不得创建 commit。

## 工作树安全

- 保留不属于当前任务的用户改动，不执行 `git reset --hard`、`git checkout --` 或递归删除工作区。
- 使用 `apply_patch` 进行人工文件编辑。测试和生成命令不得写入任务允许路径之外。
- Secret 扫描只针对 Git 索引或显式 tracked manifest，绝不以扫描为由读取被忽略的 `.env`。
