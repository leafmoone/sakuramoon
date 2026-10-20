# Condition observability reliability repair

> **历史快照（已合并）**：本可靠性修复已随 PR #2 合并入 `dev`（`31d51cc`）。仅作为证据保留（含 fd 异常注入 receipt）；旧 `src/sakuramoon/tests/**` 路径仅为历史记录，非运行入口。当前正式合同：[配置与恢复](config-and-resume.md)、[条件优化器](condition-optimizer-ab.md)、[Guard遥测](guard-window-telemetry.md)、[Owner契约](guard-window-owner-contract.md)。

## 范围与基线

- 目标分支：`feat/condition-observability`。
- 核验的远端基线：`97d1af079a7c0565e36dad514911e11f7797db7c`；分支以 `dev` 的 `4a94dfe` 为祖先。
- 使用独立 Git worktree；保留原 checkout 的未提交内容，不修改 `main`/`dev`，不重写历史。
- 新机器仅在 `/tmp/condition-observability-reliability/` 隔离工作区验证；没有部署到生产 checkout，没有操作生产训练进程、配置或模型文件。

## 修改与异常行为

### DurableJsonlSink

- 在获取文件描述符前完成锁和普通属性初始化；获取后用异常保护覆盖 fstat、regular-file/0600 检查、目录打开、目录 fsync/close。
- 构造失败时单次尝试释放取得的 fd；目录清理失败不替换原始目录 fsync 异常。
- `close()` 即使 fsync 抛异常也尝试关闭 fd；进入关闭后标记 closed，避免失败后重试关闭已复用的 fd。
- 只有一个 close 错误且此前 fsync 成功时，仍将该 close 错误作为主异常传播；已有主异常时，清理失败仅附加 best-effort warning。with 块原始异常对象保持不变，包括清理自身被中断的情况。
- JSONL 排序/紧凑格式、换行、0600、符号链接/regular-file 检查、append flags、短写循环、fsync 节律和公开接口不变。
- 将既有的非空 spatial metric 映射不变量显式写为断言，使该文件进入 changed-file Pyright gate 后也能通过；不改变正常记录的数据或 schema。

### 可选条件诊断

- 只在 sink 写入处将普通 `OSError` 降级为告警，先停用 recorder，再清理；后续 observation 不再检查/写入该 recorder，不按训练步重复报错。
- recorder 关闭采用 Guard Window 已有的 best-effort stderr/traceback 模式；只在清理边界捕获普通 `Exception`，损坏的 stderr 也不能替换训练异常。
- production 构造遇普通 I/O 错误停用当前 rank 的 stream；配置/结构错误仍传播。最终关闭采用与 Guard Window 相同的异常保留守卫。
- Guard Window 构造纳入资源拥有者的 try/finally，构造失败也清理已创建的 condition recorder。
- 没有在模型 forward、loss、backward、optimizer 或训练主循环外新增宽泛捕获；record 构造和结构契约错误仍为真实失败。

### 离线审计 CLI

- 退出码：正常 `0`，读取/container 错误 `1`，参数/窗口错误 `2`，`--out` 写入失败 `3`。
- 读取出错仍先输出已读行统计和 `container_error` 详情，再返回非零。读取中路径消失、最终 stat 也失败时，保留 partial report，metadata 置 null 并附加错误详情。
- 参数错误和输出文件写入失败有明确 stderr；不覆盖输入文件。增加 Unicode 解码读取故障的容器错误处理。

## 故障注入与回归

- 构造 open 后 fstat/文件类型/权限失败、目录 fsync 和目录 close 失败；验证 close 调用及真实 fd 的 EBADF。
- 写入 fsync、最终 close fsync、双重 fsync/close、清理中断和损坏 logging handler；断言原始异常对象与单次清理。
- recorder 写入失败后 close 再抛普通异常、with 清理失败、broken stderr；断言停用、无重复尝试、原始训练异常及 ExceptionGroup 对象不替换。
- 三次真实 CPU SGD 更新：有故障诊断与无 recorder 的 loss、参数和 Torch RNG 状态逐值相同。
- 执行 production 源码中实际资源拥有者的 try/finally 片段，验证 Guard Window 构造失败后 condition recorder 仍关闭。
- 审计注入读失败、真实截断 gzip、路径消失、无效参数和输出写失败；既测 partial 统计，也以 subprocess 检查进程非零退出码。
- CI unit job 加入 sink 和 production-wiring 新测试文件。

## 已运行验证

| 环境 / 检查 | 结果 |
| --- | --- |
| Linux CPU / Python 3.12.14，完整 pyproject train+eval+dev 依赖，原基线 CI unit 集合 | 222 passed |
| 同环境，修复后的完整 CI unit 集合（含新故障注入测试及 CPU gloo world 1/2/4） | 277 passed，3 个依赖弃用警告 |
| 同环境，既有 G1 / t-bin / spatial telemetry 额外回归 | 16 passed |
| 同环境，Ruff 全仓库 | PASS |
| 同环境，Pyright：相对 dev 的 changed Python files，含新增文件，共 30 文件 | 0 errors / 0 warnings |
| shell：逐个 `bash -n scripts/**/*.sh` | PASS |
| `scripts/check_configs.py --config-root config` | PASS |
| 新机器原生 Python 3.11.9 / Torch 2.9，相关故障注入、诊断、审计及 Guard Window wiring CPU 测试 | 105 passed，4 个依赖警告；原生 Ruff PASS |

原生测试屏蔽 CUDA/HIP/ROCR 设备，并仅在测试 harness 内阻止可选 vendor FlashAttention 导入，使用已有 CPU fallback；未安装/替换生产依赖。harness 在测试前后断言 accelerator 未初始化。原生 vendor 环境不能替代 GitHub 的 Python 3.12 CPU gate：其基线完整 gate 曾遇测试 harness 的 spawn/stdlib 隔离问题，未标为通过。正式 gate 使用上述干净 CPU 环境。

本地 CI 等价检查不是 GitHub Actions 的通过声明；push 后由对应提交的远端 static/unit/contracts 检查给出最终结论。

## 剩余风险与未验证项

- **没有运行 DCU/GPU 测试、生产训练或真实模型恢复**；不宣称加速器等价、吞吐、长期稳定或样本质量得到验证。
- fsync 失败意味着不能保证该条记录持久化；短写失败可能留下部分尾行。继续训练时诊断数据可能缺失，这是可选遥测的明确降级，未增加修复/截断/自动重试逻辑。
- fd close 失败只保证进行单次尝试，不能承诺操作系统已释放 fd；不盲目重试以免关闭复用的描述符。
- 可选 recorder/production 关闭守卫按 Guard Window 保持 `BaseException` 的解释器传播规则；raw sink 仅在已有主异常的清理边界保留原异常。没有引入异常分组框架。
- 原生环境有既有 TorchAO 弃用和 torchvision image 扩展加载警告；本轮未修改这些依赖。
- 额外全目录 Gitleaks 扫描在基线和候选均报告相同两处旧代码表达式误报（key-set 比较和 tensor dtype 断言），不在本轮改动内；不添加全局忽略。分支历史扫描无泄漏，提交前另扫描完整 staged diff。
