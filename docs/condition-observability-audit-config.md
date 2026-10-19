# 命令 2：审计工具与配置校验精简

## 基线和范围

重新 fetch 并核验远端 `feat/condition-observability` 与本地干净 worktree 均为命令 1 提交 `5ce7f2b6ab0723caf2b55d892baff1fcdf6411ca`；对应 GitHub Actions run `39222522533` 的 unit/static/contracts 全通过。`dev` 保持 `4a94dfe564e8ef439eafee71be64dcc5092962c9`。本轮只在独立 worktree 和本地 WSL CPU 环境开发/验证，没有 SSH、上传文件到训练主机或操作生产进程。保留原 dev checkout 的无关未提交内容，不修改 main/dev，不重写历史。

## 1. 缺失区间：删除按数值跨度逐个扫描

`audit_training_metrics._missing_ranges()` 现在接收调用方已排序、去重且限制在窗口内的已有 update 序列：

- 游标从 start 开始，在相邻已有 update 之间生成闭区间缺口；末尾 sentinel `end + 1` 收尾，不构造巨幅 range。
- 缺失数量为 `end - start + 1 - len(selected)`；只与存在的 update 数量有关，不随区间列表截断。
- 最多保存 1000 个缺口；发现第 1001 个即返回 truncated。恰好 1000 个不标截断；无需先存全部缺口再切片。
- 输入排序原本就在 audit 路径中，本轮复用该结果；gap scan 为 O(k)，附加区间存储受 1000 上限约束。整体 audit 仍需读取行、保留 update 状态并排序，不是常数空间算法。
- 保留含首尾的窗口、空文件 `no_update_rows`、部分边界推断、presence coverage、相同 payload 去重和缺失记录不等于未训练的语义。

证据：10**12 宽窗口集成用例；缺口数量 999/1000/1002 的截断边界；300 个固定 seed 小窗口与逐点参考算法等价；独立审查额外执行 30,690 个有界穷举 case，通过。大窗口测试不设脆弱的耗时阈值。

## 2. 冲突 update：不再 last-read-wins

审计报告 schema 由 2 升为 **3**。忽略 `recorded_at_unix_ns` 后，同一 update 有两个不同 payload 即视为冲突；这是保守的整条 payload 规则，不只比较 loss。

- 冲突 update 的所有记录从 loss、梯度、clip、nonfinite 聚合中剔除；不挑 first/last/min/max 作为真实值。
- `statistics.rows` 是无冲突 update 数，`source_rows` 是这些 update 的来源行数；相同 payload 的重复行只提供一次指标值。
- `window.status=conflicting_updates`、`conflicting_updates` 明细、`conflicting_update_count` 和 `statistics.conflicting_updates_excluded` 明确暴露不确定性。
- `window.coverage`、`distinct_updates`、`selected_rows`、`missing_count` 仍表示文件中是否有该 update，不冒充可信统计覆盖。coverage 为 1 也可能有冲突，调用方必须同时看 status/excluded。
- `invalid_numeric_values` 仍是原来的文件级输入诊断计数，不是无冲突子集聚合；`nonfinite_total=0` 不能证明冲突行或实际训练都有限值。
- CLI 正常为 0，读容器失败 1，参数错误 2，输出写失败 3，窗口内冲突 **4**。先写 report，再报告冲突；container error 优先于冲突，输出失败优先于两者。

测试验证交换冲突行读取顺序时统计不变、梯度/clip/nonfinite 冲突值也被剔除、相同 payload 不冲突、窗口外冲突不影响当前窗口、全部冲突时统计为空，以及 subprocess 退出码/partial report。没有把一个新 `complete=true` 字段当作数据质量证明。

## 3. 配置路径闭包

`check_configs.py` 是离线 TOML 继承图校验，不是可信 runtime `load_config` 的替代品。

1. source 和 target 都使用 canonical resolved identity；每条 extends 先确认根目录边界，不能因 outside 文件存在就标 resolved。
2. 未声明的越界目标均 invalid，包括不存在的路径、`..`、文件符号链接、目录符号链接和 symlink/../ 组合。`sub/../base.toml` 最终仍在 root 内则允许。
3. `[external]` manifest 的 key 仍相对配置 root，而非引用它的 source。现在允许显式 `../outside/parent.toml`，只授权其精确 canonical TOML 文件，不授权目录、同名兄弟或 glob 展开；不支持 absolute/anchored key。
4. 已有仓库 root-relative missing parent 声明继续有效。声明对象不论是否存在都作为 opaque external leaf，不读取其 TOML，也不声称验证 external 内部的环。
5. manifest 自身必须 resolve 在 root 内，不能用外部 manifest symlink 替换 allowlist。source 别名规范化，防止漏报本地别名循环。
6. 独立审查发现旧代码将 manifest 排除出图，却允许它作为 resolved parent，能隐藏外部继承或环。本轮明确拒绝继承 manifest（除非显式 external）；同样拒绝已存在但没有已检查图节点的非 TOML 隐藏叶子，不再把未遍历对象标 resolved。

覆盖：存在/缺失越界；内部合法 `..`；三类 symlink escape；exact external、siblings/globs、不合法 manifest；文件/目录 symlink external；source canonical parent；普通环、alias 环、symlink resolution loop；manifest 外逃和 direct/alias 隐藏继承。测试用 read guard 证明未读取外部 TOML，而非只检查报告文字。

**边界限制：**检查针对稳定文件树快照，不是抵御并发恶意替换目录/符号链接的文件系统沙箱。运行时业务 schema、密钥、模型/数据资产、external 内部闭包和非常深图的递归资源限制不在本轮保证内。canonical 去重后 files 计数不重复算 alias，也不算 explicit external；普通仓库仍为 15 files / 13 references / 1 external。

## 4. 删除与保留决定（按契约，不按行数）

| 代码/测试 | 决定 | 证据及额外保护价值 |
| --- | --- | --- |
| audit 按 `range(start, end+1)` 扫描及累加 | 删除/替换 | 有序 selected 已存在；随机等价、截断边界和实际 microbenchmark 证明保持语义且不受巨大空洞拖累 |
| 冲突列表的再次 `sorted` 和 `.get(update,set())` 默认分配 | 删除 | selected 已排序，每个 selected 都在 ingest 时建立 payload_digests；直接筛选保序且未放宽契约 |
| checker 单调用 `_target` 薄封装 | 删除，表达式移入原 resolve try | 只有一处调用、无额外校验/异常转换；边界及 resolve 失败仍在同一调用点处理 |
| GuardWindow `_seal` 的 dumps → encode → loads → sink.write | 删除重复 JSON roundtrip | sink.write 已执行 `allow_nan=False`、排序、紧凑 JSON 序列化，且不改变传入 dict；最终格式仍由同一个 sink 决定 |
| `_seal` 手工 write/close | 换成已有 sink context manager | 减少重复生命周期，同时修复 write 失败时未关闭 fd；真实 fd EBADF、主异常 identity、无 formal/temp、后续不重试的测试证明 |
| guard_publish 旧 Known shared-sink gaps 文案 | 删除过期说明 | 命令 1 已修复 sink 构造和 fsync 的 fd 生命周期；改成实际 prepare/primitive 资源所有权，不以过期 leak 为 latch 理由 |
| `_make_temp` close marker | 保留但移到 syscall 前，改名 close_attempted | close 失败后不能安全重试可能复用的 fd；原 close 错误仍传播。既有测试从允许两次收紧为恰好一次 |
| GuardWindow disabled latch / continuity / monotonic snapshot 检查 | 全部保留 | fd 修复不意味着失败窗口可恢复可比性；mid-window rollback、duplicate/rewind/skip、snapshot/publication fault 测试保护不同失效面 |
| session `.claimed` O_EXCL、session path/optimizer variant/routing/owner 身份验证 | 全部保留 | 不让新进程接管旧窗口、不把错误 optimizer/owner 当真；构造与 merge 校验是不同信任边界，不可互相代替 |
| publish 的 prepare/commit/postcommit cleanup 分支与 warning | 全部保留 | `os.link` 是唯一 no-clobber commit；EEXIST 为正常竞态，其他 precommit 错误传播，postcommit unlink 失败只能警告；竞态及异常身份测试覆盖 |
| production `_construct_*` / `_close_*` 和嵌套 finally | 保持文件无改动 | 构造分别处理可选 I/O 与配置错误；finally 即便一个 cleanup 异常仍关闭另一个；普通 close 不覆盖训练异常/ExceptionGroup，BaseException 规则不改 |
| condition_diagnostics record 验证、sink OSError guard、close Exception guard、broken-stderr guard | 保持文件无改动 | 分别保护结构错误、可选写入、可选清理、告警失败四个边界；不能合并成吞掉 model/loss/optimizer 错误的宽泛捕获 |
| 各模块 best-effort warning 小函数 | 保留 | 消息前缀、traceback 与异常类型范围不同；统一会引入跨层依赖或通用异常框架，无证据收益 |
| audit CLI、check_configs、guard_window_merge | 保留为独立入口 | 调用方分别为运营 JSONL 审计、CI 配置图、rank/owner 窗口合并；输入/schema/失败语义不同，不是同名字段就能合并 |
| sink 故障注入 vs recorder vs production wiring 测试 | 保留 | 分别测试 fd 原语、停用状态与真实 CPU 更新不变性、真实 production try/finally 和训练异常身份；低层测试不能替代调用方接线 |
| byte-identical / same-payload / conflict 及 CLI subprocess 测试 | 保留 | payload 分类、聚合选择、进程退出码是不同契约；同一 fixture 已融合顺序独立与窗口范围断言，不为静态文案另堆测试 |
| 小窗口 brute-force 参考函数 | 仅保留在测试 | 故意使用简单旧算法作为独立 oracle，不能复用新 gap 算法自证正确；只在 bounded 小区间运行 |

没有为减少测试数量而删除身份、原子发布、训练异常和路径安全覆盖；未新增通用异常框架或依赖。

## 5. 代码变化

相对命令 1，按内容行（LF）统计，不把 Windows CRLF 当作修改：

| 文件 | 修改前 → 后（行） |
| --- | --- |
| scripts/audit_training_metrics.py | 496 → 533 |
| scripts/check_configs.py | 227 → 288 |
| src/sakuramoon/telemetry/guard_publish.py | 191 → 184 |
| src/sakuramoon/train/guard_window_stream.py | 587 → 584 |
| tests/unit/config/test_config_closure.py | 155 → 477 |
| tests/unit/telemetry/test_guard_publish.py | 340 → 340 |
| tests/unit/test_training_metrics_audit.py | 321 → 471 |
| tests/unit/train/test_guard_window_stream.py | 710 → 756 |

核心 stream/publication 两文件净减 10 行；审计与路径安全/冲突语义及对应测试有净增加。这不是把总行数变少当作验收目标。

## 6. 性能证据

同一 WSL Ubuntu / Linux 6.18.33.2 / Python 3.12.14。`perf_counter_ns`，先 warmup，再取 7 组样本中位数（配置 5 组）；gap 小 case 每组 10 次、1m case 每组 1 次，均逐次核对结果。基线用命令 1 detached worktree，候选用当前 worktree，均无 GPU；输入纯合成，不读训练指标。没有在旧 O(span) 算法上跑 10**12，也不外推其时间。

| case | 命令 1 中位数 ms | 本轮中位数 ms |
| --- | ---: | ---: |
| gap：span 10,000 / present 5 | 0.3043195 | 0.0004582 |
| gap：span 1,000,000 / present 5 | 30.921245 | 0.0007680 |
| gap：100,000 全部存在 | 3.2546698 | 1.5073982 |
| gap：span 5,000 / present 2,500 / 截断 | 0.3019207 | 0.0821530 |
| gap：span 10**12 / present 5 | 未运行 | 0.0004955 |
| 端到端 audit：span 1,000,000 / 5 行，含读取/解析/统计 | 31.282674 | 0.070792 |
| 完整仓库配置 check：15 files / 13 refs / 1 external | 175.663696 | 274.581775 |

所有可比较 case 的 missing/count/ranges/truncation/window/loss 摘要逐值相同。配置检查此轮样本较慢；新增 canonical/边界检查有成本，WSL 跨盘文件元数据和调度也有较大方差，**不宣称配置性能优化或稳定倍率**。sub-microsecond gap 数字受计时开销影响，只是实测样本，不是部署 SLA。

原始每组样本、benchmark 脚本和验证日志保留在本地 `tmp/condition-observability-evidence/command2-*`，不提交运行产物。可复现核心方法：导入两个版本的脚本，分别传入 `(1, 1_000_000, {1,101,400000,999999,1000000})`（候选用同内容 sorted list），先核对返回 tuple，再用 `perf_counter_ns` 测量。端到端生成相同五条 `successful_update`/`main_loss=0.5`/`schema_version=12` JSONL，并调用 `audit(..., start=1, end=1_000_000, loss_field='main_loss')`；配置调用 `check(repo / 'config')`。

## 7. 验证与限制

- 命令 1 baseline 本地 CI 等价集合：277 passed；Ruff、30 个 dev-diff Python 文件 Pyright、真实 config tree、逐文件 shell syntax 均通过。
- 本轮最终完整 CI 等价集合：**315 passed，3 个依赖弃用警告，无 skip**；全仓 Ruff PASS，30 文件 Pyright **0 errors / 0 warnings**，真实配置树和逐文件 shell syntax PASS。远端 CI 以交付说明中同一提交的 GitHub Actions 为准，不把本地 runner 等价检查冒充远端 CI。
- 额外整个 `tests/unit/config`：116 passed，无 skip；symlink 用例在 Linux 实际运行。
- 独立审查：audit/guard 103 相关用例 + 30,690 bounded exhaustive gap case；配置 blocker 修复后 closure 38 passed，无 skip。后续只对同一边界测试增加截断阈值参数及更强统计断言。
- 未运行 DCU/GPU、生产训练/恢复、吞吐或质量验证；未验证外部 schema-2 消费者。消费审计输出的调用方需识别 schema 3 和 exit 4。
- fd close 失败仍只能承诺单次尝试，fsync 失败不保证数据持久化，post-link directory-entry crash durability 仍不承诺。这些边界未被精简掩盖。
