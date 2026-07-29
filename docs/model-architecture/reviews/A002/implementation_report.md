# A002 Implementation Report

## 结果

A002 已将用户确认的两条资产执行边界写入现行决定、仓库规则、路线图与资产策略，并通过合同测试建立机器保护。实现没有读取或执行 `reference/` 内容、没有读取 `.env`、没有下载/哈希/加载模型、没有使用 GPU，也没有创建性能占位。

首轮独立 AI 与 Infra 审查均判定 FAIL/changes required。现已完成 `review_remediation.md` 所列收敛修复；状态仍是等待原 reviewer 复审，而不是已通过。

## 实现摘要

- `model/qwen_3.5_2B` 与 `model/vae` 明确为已准备的唯一 Qwen TE/Mage-VAE 本地目录；缺失或漂移硬失败，禁止自动下载、补下载、联网替换、隐式 cache 命中或 fallback。
- `reference/` 明确只供人工理解/对照且可完全不使用；生产代码、测试、preflight、训练、评估和运行时禁止静态 import、动态加载、`sys.path` 注入、执行、调用或以子进程启动其中代码。
- 独立 scanner 扫描 `src/`、`tests/`、`tools/` 及其自身；读取前拒绝越界、symlink、非普通文件和身份漂移。AST provenance/taint 覆盖 loader alias/`getattr`/`partial`、reference 组合路径/read-then-exec、动态 loader、进程与 search-path API。
- `from_pretrained` 来源必须绑定 A001 `VerifiedAssetSelection.verified_root()` 的 Qwen/Mage-VAE asset ID，并显式 `local_files_only=True`、禁用 remote code 和 cache。任何模型下载 API都禁止；唯一 ModelScope dataset transport 还必须锁定路径、函数、repo、40-hex revision 与 repo type。
- 参考 Git metadata helper 只允许三个命令，并用命令行配置、隔离环境、关闭 stdin 的组合阻断 local/system/global fsmonitor、hook、pager、external diff、interactive filter 与 prompt。
- 追踪将两条要求拆为独立 profile：`local_model_execution_boundary` 映射模型配置与 asset/encoder 模块；`reference_execution_boundary` 明确 config N/A，并覆盖现在及未来全部 `src/sakuramoon/**` 生产模块。
- 现行 confirmed source revision 从 1 递增到 2，registry revision 从 5 递增到 6；新增从未使用的 `DOC-006` 和 `DOC-007`，保留全部既有 ID，并追加连续 SHA-256 changelog。两条 requirement 的实现路径列出 A002 的规范、追踪、合同、fixture、task、耗时与证据文件。

## AI / 模型正确性自检

- manifest 中两条 ready model 的逻辑路径与用户确认值精确一致，合同没有读取模型 payload。
- 本任务只证明资产路径/执行策略和静态调用边界，不把它表述为 Qwen load、Mage-VAE posterior mean、round-trip、forward 或质量证据；这些门槛仍属于 T020/T021。
- Dataset 的 ModelScope 下载职责仍隔离在 `src/sakuramoon/data/` 与 dataset manifest CLI；该例外不授权任何模型下载。

## Infra / 性能自检

- Scanner 只在测试/验证阶段解析仓库 Python 文本；`verified_root` 只在模型构造前重验已锁文件身份，不进入训练 step 热路径。
- 检查不访问网络、GPU、大模型、数据库、ignored reference 内容或 `.env`。
- 本地模型不合约时维持 fail-closed，不新增网络恢复、远端替换或慢路径 fallback。

## 独立审查重点

- AI reviewer：确认 `DOC-006/007` 的措辞没有把静态边界误写为真实模型执行证据，并检查 dataset 下载例外不能被模型加载器借用。
- Infra reviewer：检查 AST 规则对 alias、动态 loader、组合路径和 subprocess 的覆盖，以及只读参考仓库 Git 身份审计没有执行其中代码。
