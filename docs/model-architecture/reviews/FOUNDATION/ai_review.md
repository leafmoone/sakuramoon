# Foundation AI/模型正确性独立审查

审查范围：当前 `HEAD=c8a7c76` 已提交的 `R002`、`D001`、`C001` 与已撤销的 `A001`；工作树中的 `T043` 实现不属于本次审查。审查以 `current/confirmed-decisions.md`、`current/open-items.md`、`current/observability-and-evaluation.md` 和现行路线图为准，未读取 `.env`、未访问或执行 `reference/`、未运行训练长跑或多卡路径。

## 结论

| 任务 | AI/模型正确性结论 | 说明 |
|---|---|---|
| `R002` | **PASS** | 锁定依赖没有引入未决 dropout、历史候选或训练默认值；import/CUDA 可见性没有被表述为 kernel 数值放行。uv 工具自身未被工程强制到 0.12.0 是 Infra 阻塞项，不改变本行的模型语义结论。 |
| `D001` | **FAIL** | 稳定 requirement ID 在缺少完整 Git 历史时没有可信的 bootstrap locator 绑定；同时仓库级 module glob 使“关键模块反向找到具体要求”的门槛可被通用边界条款吞掉。当前 live checker 通过不能覆盖这两个负合同缺口。 |
| `C001` | **FAIL** | 当前 schema/loader 本身通过针对性正确性检查，但任务和实现报告仍声称存在已撤销的模型 repo/revision、tokenizer SHA 与 `microsoft/Mage-Flow` schema 约束；这些证据与当前本地资产决定和实际 schema 不一致，不能据此关闭独立审查。 |
| `A001` | **FAIL（仅保留边界）** | 重型 manifest/hash/capability 范围保持撤销，不要求恢复；但保留的“只检查组件实际需要文件”边界被共享的全模型检查破坏，单独加载 VAE 会因 Qwen 缺失失败，单独加载 Qwen 也会因 VAE 缺失失败。 |

## 阻塞发现

### F-AI-1 `D001` bootstrap identity 在浅历史/无 Git 场景可被重绑

`tools/verify_traceability.py:472-489` 的 `_validate_bootstrap_bindings()` 只验证 baseline ID 是否仍存在，没有把 ID 绑定到报告中声明的 bootstrap locator digest。真实 locator 所有权只来自 `_validate_registry_history()` 收到的最早 snapshot，因此浅克隆、源码包或无 `.git` 的隔离校验会把当下可能已篡改的 registry 当成 bootstrap。

独立负探针交换当前 registry 前两个 requirement ID，并仅把交换后的 snapshot 传给 history validator，结果为 `errors=[]`。这与 `D001` 实现报告所称“Anchored all 219 bootstrap requirement ID-to-source bindings”不符，也不能执行 AGENTS.md 的稳定 ID 规则。

### F-AI-2 `D001` 全仓 glob 令 module 反向追踪失去区分度

`traceability.toml:738` 与对应 requirement 映射使用 `src/sakuramoon/**`。因此任意新模块（例如 `src/sakuramoon/completely_unmapped.py`）都会被 `reference_execution_boundary` 命中，即使它没有任何领域要求映射。现有负测试在断言 unmapped module 前主动把该 profile/requirement 的 glob 改成 `src/sakuramoon/__init__.py`（`tests/unit/docs/test_verify_traceability.py:373-379`），没有测试 live registry 的真实行为。

通用“不得执行 reference”要求确实作用于所有模块，但它不能替代模块对应的架构/数据/训练 requirement。checker 需要区分横切边界映射与可满足 reverse inventory 的领域映射，或明确禁止 module-root blanket glob 充当唯一反向依据。

### F-AI-3 `C001` 当前证据仍描述已撤销的资产身份 schema

`docs/model-architecture/reviews/C001/implementation_report.md:10,14` 与 `progress/tasks/C001.md:43-44` 声称模型 revision 为 40 位 commit、VAE repo 固定为 `microsoft/Mage-Flow`、`tokenizer_sha256` 保持在 resolved config。当前 `QwenAssetConfig`/`VaeAssetConfig` 只保留固定本地路径和加载语义；唯一 `Commit` 字段是远端 dataset revision。这一实现变化符合后来的本地资产简化决定，但 C001 的任务与审查证据没有随之修订。

修复应只更新 C001 的现行实现摘要/证据边界，不得把已撤销的本地 repo/revision/hash 字段加回 schema。

### F-AI-4 `A001` 组件加载被无关模型文件耦合

`src/sakuramoon/assets/__init__.py:11-32` 的 `_REQUIRED_FILES`/`require_local_models()` 一次要求 Qwen 与 VAE 全部文件。`load_local_mage_vae()` 在选择 VAE 前调用该函数，`load_local_qwen()` 同样先调用它。结果违反 `asset-policy.md:8` 的“组件实际需要的文件”边界，并使组件级故障诊断错误扩大。

应拆成 component-specific 路径/必需文件检查，并补两类负测试：只有 VAE 文件时 VAE 检查通过；只有 Qwen 文件时 Qwen 检查通过。无需引入 manifest、hash、capability 或 hostile-local-environment 层。

## 非阻塞观察

- `.gitignore:26` 仍写“Model weights and local databases must be represented by manifests”，与 A001/A002 撤销后的“不维护本地资产 manifest”表述冲突。该行不改变 ignore 行为，但应改成只说明权重/DB 不得 tracked，避免旧边界回流。
- 多卡、DDP、NCCL、正式 stage 与训练长跑均不在 Foundation CPU 证据范围内，本审查没有关闭任何此类门槛。

## 独立验证

```text
uv run --frozen pytest -q tests/unit/docs/test_verify_traceability.py tests/unit/config tests/contracts/config tests/unit/assets/test_local_models.py
95 passed in 11.33s

uv run --frozen ruff check tools/verify_traceability.py src/sakuramoon/config src/sakuramoon/assets tests/unit/docs/test_verify_traceability.py tests/unit/config tests/contracts/config tests/unit/assets/test_local_models.py
All checks passed

uv run --frozen pyright tools/verify_traceability.py src/sakuramoon/config src/sakuramoon/assets tests/unit/docs/test_verify_traceability.py tests/unit/config tests/contracts/config tests/unit/assets/test_local_models.py
0 errors, 0 warnings

uv run --frozen python tools/verify_traceability.py --format json
ok=true; 221 requirements; 221 source nodes; 67 production modules; 234 runtime config keys; 16 archive files
```

这些正向结果证明现有测试与 live registry 自洽；它们不抵消上述独立负探针暴露的合同缺口。
