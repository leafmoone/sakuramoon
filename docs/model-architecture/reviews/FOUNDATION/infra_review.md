# Foundation Infra/性能独立审查

审查范围：当前 `HEAD=c8a7c76` 已提交的 Foundation CPU 范围。未读取 `.env`，未访问/执行 `reference/`，未运行 GPU kernel、训练长跑、DDP、NCCL 或多卡测试。工作树内未提交的 `T043` 不作为 Foundation 实现结论。

## 结论

| 任务 | Infra/性能结论 | 说明 |
|---|---|---|
| `R002` | **FAIL** | package lock、空 cache/venv 冷重建、许可证与当前工具入口证据充分；但声称固定的 `uv 0.12.0` 只存在于报告，工程配置没有 `required-version` 或等价可执行约束。 |
| `D001` | **FAIL** | checker 为 CPU 启动/CI 工具，不进入训练热路径，当前耗时可接受；但浅历史 bootstrap 可重绑和 module-root blanket glob 使 fail-closed 追踪门槛不成立。 |
| `C001` | **PASS（实现）/BLOCKED（证据关闭）** | loader、strict schema、redaction 和 writer 为启动期 CPU 工作；95 项共享 targeted suite、Ruff 与 strict Pyright 通过。C001 的陈旧资产字段声明必须更正后才能把包级证据标为完成。 |
| `A001` | **FAIL（仅保留边界）** | 撤销重型审计是正确范围；共享全模型存在性检查造成组件级无关故障和重复检查，属于可用性/故障边界回归。 |

## 阻塞发现

### F-INFRA-1 `R002` 未在仓库中强制 uv 工具版本

`progress/tasks/R002.md:29` 要求“安装并锁定 uv”，实现报告又明确声称“固定 uv 0.12.0”。但 `pyproject.toml:63-71` 的 `[tool.uv]` 只有 cache 与 platform 约束，没有 `required-version = "==0.12.0"` 或等价的 tracked tool bootstrap/check。当前执行 `uv --version` 的确是 0.12.0，不能证明另一环境会拒绝不同 uv 版本；不同 resolver/build frontend 版本仍可处理 lock、Git build 与 cache 语义。

应把 0.12.0 变成工程可机检约束，并增加错误 uv 版本的负验证。现有 `uv lock --check` 和冷重建证据可保留，不需要重跑 GPU 或训练。

### F-INFRA-2 `D001` fail-closed 历史和反向库存门槛可被绕过

与 AI 审查的 F-AI-1/F-AI-2 相同，影响 Infra 的具体结果是：在 shallow checkout/源码包中 baseline ID 交换不会报错；新增生产模块会被 `src/sakuramoon/**` 的通用 reference boundary 自动覆盖。CI 显示 `ok=true` 不能等价于“stable ID 未重绑、所有生产模块有领域 requirement”。修复需增加真实无 Git/浅历史负测试和 live blanket-glob 负测试。

### F-INFRA-3 `A001` component preflight 具有无关依赖

`require_local_models()` 把两个组件的文件列表合并，VAE/Qwen loader 又分别重复自身检查。任何一侧缺失都会阻塞另一侧的独立加载与诊断。应拆开检查并保留固定路径、缺失硬失败、无下载/fallback 的简化语义。

## 已核实边界

- `uv lock --check`：通过，95 packages。
- R002 记录的 cache-warm fresh-venv 与 empty-cache source rebuild 明确区分；cold build 不宣称 bit-for-bit，可接受。
- R002 没有把 import/CUDA visibility 当作 kernel forward/backward 证据；K001/T021/T040 的真实 kernel 证据不由本审查重复关闭。
- C001 的 deterministic merge、unknown/missing/type/range 拒绝、未决 sentinel、secret value 不落 resolved config/error、symlink path 拒绝均有 targeted tests；配置解析不在训练热路径。
- A001 撤销后没有恢复本地模型 hash、manifest、capability、TOCTOU 或 reference scanner；这些非目标不得因本次 FAIL 被重新引入。
- 当前 traceability checker 在含未提交 T043 的共享工作树上通过 `221/221`，module/key 数为 `67/234`；这只是当前快照的正向结果。

## 独立验证

```text
uv lock --check
Resolved 95 packages

uv run --frozen pytest -q tests/unit/docs/test_verify_traceability.py tests/unit/config tests/contracts/config tests/unit/assets/test_local_models.py
95 passed in 11.33s

uv run --frozen ruff check <Foundation targeted paths>
All checks passed

uv run --frozen pyright <Foundation targeted paths>
0 errors, 0 warnings

uv run --frozen python tools/verify_traceability.py --format json
ok=true; requirements=221; source_nodes=221; archive_files=16; production_modules=67; runtime_config_keys=234
```

没有生成 `perf_baseline.json`/`perf_after.json`，因为这些 Foundation 任务不是训练性能任务；没有执行长跑或任何多卡项目。

## R002 post-remediation rereview (`4ade567`)

本节追加于历史审查之后；历史 findings 与结论保持不变。对 `4ade567` 的独立复审结论为 **PASS**：工程仍以 `[tool.uv] required-version = "==0.12.0"` 硬约束工具版本，环境锁已明确区分 cold-rebuild 捕获时点与 remediation 后输入，dependency sources/build variables 未变化，且已有 cold rebuild 确实使用 uv 0.12.0，因此无需重复冷构建。Foundation 逐任务 Infra/性能结论为 R001 remediation、R002、D001、C001、A001 全部 PASS。

复验：Foundation targeted `156 passed`；trace `235/235`、0 errors；`uv lock --check`、Ruff、Pyright（0 errors/0 warnings）和 `git diff --check` 均通过。未关闭 GPU kernel、DDP/NCCL、长跑或正式 stage 门槛。不可变复审记录见 `r002_infra_rereview.md`。
