# A002 Implementation Report

## 结果

A002 已将用户确认的两条资产执行边界写入现行决定、仓库规则、路线图与资产策略，并通过合同测试建立机器保护。实现没有读取或执行 `reference/` 内容、没有读取 `.env`、没有下载/哈希/加载模型、没有使用 GPU，也没有创建性能占位。

首轮与第二轮独立 AI/Infra 审查均判定 FAIL/changes required。现已完成 `review_remediation.md` 所列第二轮收敛修复；状态仍是等待原 reviewer 复审，而不是已通过。

## 实现摘要

- `model/qwen_3.5_2B` 与 `model/vae` 明确为已准备的唯一 Qwen TE/Mage-VAE 本地目录；缺失或漂移硬失败，禁止自动下载、补下载、联网替换、隐式 cache 命中或 fallback。
- `reference/` 明确只供人工理解/对照且可完全不使用；生产代码、测试、preflight、训练、评估和运行时禁止静态 import、动态加载、`sys.path` 注入、执行、调用或以子进程启动其中代码。
- 独立 scanner 扫描 `src/`、`tests/`、`tools/` 及其自身，并忽略版本控制已忽略的 notebook checkpoint；读取前拒绝越界、symlink、非普通文件和身份漂移。AST provenance/taint 在模块和函数作用域内按语句顺序传播，赋值会 kill 旧 provenance；调用摘要跨函数传播 reference 参数与返回值，但不混淆不同函数中的同名变量。
- `from_pretrained` 覆盖 Transformers、Diffusers 与 ModelScope，以及 alias、computed `getattr`、`partial` 和 `.__call__`。来源必须是 `require_runtime_assets_ready` 返回的 capability，或由 `require_verified_selection` 验证的窄 wrapper 参数，再取 Qwen/Mage-VAE 的 `verified_root()`；annotation/cast 不构成证明。调用必须显式 `local_files_only=True`，拒绝 `trust_remote_code`、`cache_dir` 和来源未知的 `**kwargs`。
- 任何模型下载 API 都禁止。唯一 dataset transport 是 `src/sakuramoon/data/modelscope.py::fetch_dataset_shard` 对 `modelscope.hub.snapshot_download.snapshot_download` 的调用，repo、锁定 config revision provenance 与 `repo_type="dataset"` 必须精确；Hugging Face 没有 dataset 例外。
- Reference taint 覆盖 shell 字符串、路径组合、文件内容、属性/容器、函数参数/返回值、动态 loader、进程与 search-path API（包括 `sys.path` slice）。测试 Git 例外仅接受精确测试文件/函数及固定命令 shape，拒绝 `-c`、alias、未知 argv 或附加命令。
- 参考 Git metadata helper 只允许三个命令，并用命令行配置、隔离环境、关闭 stdin 的组合阻断 local/system/global fsmonitor、hook、pager、external diff、interactive filter 与 prompt。
- 追踪将两条要求拆为独立 profile：`local_model_execution_boundary` 映射模型配置与 asset/encoder 模块；`reference_execution_boundary` 明确 config N/A，并覆盖现在及未来全部 `src/sakuramoon/**` 生产模块。
- A002 初始实现将 confirmed source revision 从 1 递增到 2、registry 从 5 递增到 6；首轮修复、后续 R002 和本轮修复依次将 registry 递增为 7、8、9。新增的稳定 ID 仍为 `DOC-006` 与 `DOC-007`，既有 ID、source fingerprint 与 SHA-256 changelog 均未重写。本轮不修改 `current/` 或 archive，只更新两条 requirement 的实现路径、task、测试和证据。

## AI / 模型正确性自检

- manifest 中两条 ready model 的逻辑路径与用户确认值精确一致，合同没有读取模型 payload。
- 本任务只证明资产路径/执行策略和静态调用边界，不把它表述为 Qwen load、Mage-VAE posterior mean、round-trip、forward 或质量证据；这些门槛仍属于 T020/T021。
- Dataset 的 ModelScope 下载职责仍隔离在 `src/sakuramoon/data/` 与 dataset manifest CLI；该例外不授权任何模型下载。

## Infra / 性能自检

- Scanner 只在测试/验证阶段解析仓库 Python 文本；`require_verified_selection`/`verified_root` 只在模型构造前重验已锁文件身份，不进入训练 step 热路径。
- 检查不访问网络、GPU、大模型、数据库、ignored reference 内容或 `.env`。
- 本地模型不合约时维持 fail-closed，不新增网络恢复、远端替换或慢路径 fallback。

## 独立审查重点

- AI reviewer：复验 root 重赋值、annotation/cast、未知 `**kwargs`、ModelScope/computed loader/HF 下载限定路径，以及精确 dataset revision provenance。
- Infra reviewer：复验 reference 属性/容器/函数/shell/search-path taint、跨作用域同名正例和测试 Git 精确例外；确认只读参考仓库 Git 身份审计没有执行其中代码。
