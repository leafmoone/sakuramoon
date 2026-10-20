# Guard Window Telemetry（Phase 2B/2C/2D）

按成功更新窗口密封的逐 rank 优化器守卫计数遥测。默认关闭；启用后每个 rank
每 `window_updates` 个**成功**更新写一行 JSONL 记录，离线合并器把同一
`session_id` 的各 rank 窗口对齐成带完整性的覆盖报告。

本遥测与 `TRAINING_METRIC_SCHEMA_VERSION`（schema 12）完全独立，版本号
`SCHEMA_VERSION = 1` 由本模块自管；正常训练路径不新增任何集合通信。

## 配置（`[guard_window]`）

```toml
[guard_window]
enabled = true          # 缺省 false = 特性不存在（默认关闭）
session_id = "s20260215" # enabled 时必填；单一合法路径组件（无 / \ ..，≤120）
window_updates = 100     # 每窗口成功更新数，正整数
```

- 段缺失 / `enabled = false`：不构造流、不挂 observer、无额外快照/IO/collective，
  训练路径与关闭时逐字节等价。
- `session_id` 复用（同一存储上同 rank 已存在 `.claimed`）→ 该 rank 拒记，
  **不影响训练**；真实 load/rollback = 新进程 = 新 `session_id`。
- 生产 fresh 与 resume 都实际加载 checkpoint（fresh 走 bootstrap save+restore），
  因此身份中的 `launch_kind` 由生产调用点**显式**传入
  （`fresh_bootstrap` / `user_resume`），绝不用"是否存在 restored 对象"推断。

## 布局

```
<artifact_dir>/guard_window/session-<session_id>/rank-<n>/
  .claimed                     # O_EXCL 每 rank 会话声明（防 session_id 复用）
  guard-window-u<a>-<b>.jsonl  # 每窗口一行记录（原子 no-clobber 发布）
```

发布原语（`telemetry/guard_publish.py`）：同目录 `0600` 临时文件；
prepare 回调负责写入 + fsync + 关 fd（helper 本身不重复 fsync，只调用
prepare）；`os.link` 为提交点（no-clobber，`TARGET_EXISTS` = 冲突，
不覆盖）；成功后文件已 fsync 且原子可见，但不保证目录项 crash durability；
post-commit 临时清理的普通错误只留 warning；用户中断时 commit outcome
可能为 unknown。

## 记录 schema（`SCHEMA_VERSION = 1`，单行 JSON）

| 字段 | 说明 |
| --- | --- |
| `schema_version`, `run_id`, `session_id`, `rank`, `world_size` | 固定身份 |
| `start_update`, `end_update`, `window_id` | 窗口闭区间（成功更新号） |
| `complete` | 是否满窗；运行中止尾记录 `false` |
| `run_outcome` | 正式满窗记录为 `null`；尾记录 `clean` / `raised` / `unknown`。**仅观测循环范围**（训练 loop + telemetry block），不覆盖其后 production cleanup；CLI 不得据此声称整个 session 成功 |
| `routing` | 真实优化器 routing 全量：`fqn/shape/chunk_count/chunk_dim/roles/role`（语义标签与 canonical 计数域并存，如 adaln 6-chunk 共享一个 canonical role） |
| `owner_partition` | 用 `stable_owner(fqn, chunk, world_size)` 重算的 `{"<owner>": [fqn#chunk, ...]}` |
| `launch` | `resume_source_checkpoint`, `resume_successful_updates`, `launch_kind`, `optimizer`（真实类型绑定的 variant 名，见下）, `owner_mapping_version`, `last_published_checkpoint`（封窗时刻最近周期 checkpoint 名，仅元数据不读盘） |
| `baseline.source` | `fresh` / `restored_rank0`（生产两种 launch 都是真实 checkpoint 读取，诚实记 `restored_rank0`） |
| `baseline.counts` / `absolute` / `delta` | 计数器端点与差值（R 域 + O 域，map 按已知域归一 sparse 0） |
| `delta_status` | 每字段 `ok` / `unknown`（缺键）/ `rollback`（端点回退）；合并器拒绝 `unknown`/`rollback`，稀疏合法 0 与未知缺字段不混淆 |
| `delta_coverage` | `complete`（流只在该值下写正式记录） |
| `absolute_history` | 每字段 `rank_provenance` / `rank0_owner_only`（restore 后 O 域绝对值来自 rank0 checkpoint 复制，跨 rank 求和无意义）/ `local`（fresh/synthetic 非 restore） |

### variant 与字段域

`launch.optimizer` 由**真实类型** isinstance 绑定（rescue 子类先判），取 config
真实 name：

- `hybrid_cmuon_canonical_ns4_fp32_rescue`：必需 5 个 O 标量
  （`bf16_attempts`, `bf16_safety_failures`, `fp32_attempts`, `fp32_rescues`,
  `fp32_rescue_failures`）加 1 个 role map `rescue_by_role`。
- `hybrid_cmuon_guarded_canonical`：无 O 域；记录中**不出现** O 字段，
  绝不以 0 冒充。

合并器按 variant 强制 required 字段；variant 声明 rescue 但 O 字段缺失 =
硬失败，不得降级成 base/legacy。

## 流语义（`train/guard_window_stream.py`）

- 读面：只读优化器公开 int/map 属性，从不调用 `state_dict()`（那会复制
  动量/权重）。
- hook 位置：`run_single_gpu_training` 成功更新处，condition observer 之后、
  主 observer 之前；失败尝试 fail-closed 中止运行，不进窗。
- **update 号连续性**：hook 每成功更新触发一次且 `successful_updates` 恰好
  +1；首拍必须等于窗口起点，后续必须 `prev+1`。重复/倒退/跳号 → 显式
  不可比，整会话锁存 **DISABLED**（不是 defensive return 后继续封窗）。
- **逐快照单调性**：每个成功快照对上一有效快照（首拍对 baseline）比较
  所有标量与归一化 map 的每个 key；任何回退（含恢复前中窗回退）→ 锁存
  DISABLED。end-baseline 差值永不掩盖中间回退。
- **失败锁存**：快照异常 / 发布失败 / 目标已存在 / 构造期基线异常 → 整会话
  DISABLED（stderr + traceback，永不抛入训练；诊断输出本身 best-effort，
  broken stderr 也不中断训练）。选择"整会话停用而非逐窗恢复"的原因：共享
  `DurableJsonlSink` 在 ctor 写失败/close 异常路径有 ≤1 FD 的泄漏缺口，
  逐窗恢复会把该泄漏放大到无界；停用的显式缺口（尾缺失）本身可解释。
  只有新会话（新进程 + 新 `session_id`）重新记录。
- 配置级错误（非 guarded 优化器、非法 `session_id`/`window_updates`/
  `launch_kind`）→ `CMuonSafetyError`，启动即 fail-closed（不静默 disable）。
- 尾记录：`close(run_outcome)` 至多写一条 `complete=false` 记录（仅当存在
  已观测未封快照）；中止不写"complete 的 abort 尾"。

## 合并器（`cli/guard_window_merge.py`，独立进程）

命令（独立进程，可选外部 manifest 锚点）：

```bash
uv run --no-sync python -m sakuramoon.cli.guard_window_merge   --input <rank-0 目录/rank-1 目录/...>   [--legacy <旧格式文件...>]   --output <merged.json>   --start-update <u> --end-update <v>   [--routing-manifest <外部锚点 JSON>]
```

输出 no-clobber（目标已存在即拒）；`--routing-manifest` 缺省用各记录
embedded routing（embedded vs external 锚点区分见下）。`--legacy` 是显式
mode：只接受旧格式，输出 coverage 为 `unknown`/`partial`，不得用 legacy
记录填补缺 rank/身份或把现代坏数据降格通过。

输出合并 JSON：每 rank 窗口序列、O 域汇总与 `session_completion`
恒为 `not_asserted`（只给请求数据域的 coverage，不给"整个 session
成功"）。缺 rank / 窗口 gap / 坏尾 / schema 违例是**整批抛错、无 formal
输出**，不是"带缺口的 complete 报告"。**abs vs delta**：O 域本地 **delta** 经 owner 归属后
跨 rank 求和是合法的；恢复历史 **absolute**（`absolute_history=
rank0_owner_only` 字段，如 restore 后 rank≠0 的 `fp32_rescues` 绝对值）
来自 rank0 checkpoint 复制，**不可跨 rank 求和**。
owner 静态 partition 完整**不等于**每 step 有 grad/NS eligible（grad=None
合法），不能据此造 attempt 分母。
校验边界：routing 必须**非空**（base 也拒空，空域不得背书正 owner 事件）；
非 owner rank 在 owner 静默窗口内 O delta 必须全 0（baseline/absolute 可因
rank0 复制恢复为正，不误拒合法 resume）；同 chunk 全员缺失在仅 embedded
routing 时不可辨别（embedded_writer_manifest vs external_manifest 区分，
不给抗篡改声称）；外部 manifest 给定时同缺硬失败。

## 性能边界

微基准（Windows CPU，真实 `_snapshot` + 真实 publish stdlib-fsync 回调，
3-spec/9-chunk/world4 合成 fixture，5×1000 次）：snapshot 中位 ≈0.89µs，
单次 fsync+link 发布中位 ≈1.5ms，record ≈1.6KB。**合成小规模适配器
路径的非生产测量**，不推广到实际模型 routing、POSIX DurableSink 或
native 环境。最终 hosted Linux CI 只跑功能（不设速度断言）；本文件
不引用未随提交验证的测试总数。

采集是 **only-success** 观测：失败候选计数不进入正式成功窗，流无法
恢复全部失败候选；"failed_steps 完整记录"不是本特性交付。`run_outcome`
的 scope = observed loop（训练 loop + telemetry block），不代表整个
session 的所有 cleanup 成功。

## 测试归属

- `tests/unit/telemetry/test_guard_publish.py` — 发布原语（POSIX 用例 Linux
  CI 实跑；Windows skip 并注明）。
- `tests/unit/train/test_guard_window_stream.py` — 窗口/单调性/连续性/
  锁存/variant/尾记录（fake 计数器 + 非 POSIX sink 适配器，**不是** gloo/
  真实构造器证据）。
- `tests/unit/train/test_guard_window_production_wiring.py` — 生产 ctor/
  close 双故障（含 broken stderr）、bootstrap/resume 显式 launch_kind、
  原异常/ExceptionGroup/SystemExit 身份保持。
- `tests/unit/optim/test_guard_window_gloo.py` — 真实 base/rescue 构造器 +
  CPU gloo ws1/2/4 真实 step/计数/恢复（7 子进程；CPU 边界明示，非 native/
  完整 production 验收）。
- `tests/unit/config/test_guard_window_config.py` — 默认关闭/回读/校验。
- `tests/unit/cli/test_guard_window_merge.py` — 合并器正反例与变异红
  （空 routing、整 O 删除、无 variant、零 owner 正 delta、update 跨度
  伪造、计数守恒）。
- 最终 hosted Linux CI 实跑（真实 `DurableJsonlSink`，不设速度断言）与
  未验 native 边界以提交后的 CI 结果为准。

## 未验证边界（明确不覆盖）

以下项**未运行、未验证**，CI 也不能代验（hosted Linux x86_64 CPU only）：

- Hygon DCU / DTK / RCCL 多 rank 实机；
- GPU / NCCL 多 rank 实机；
- native TorchAO AdamW8bit / SR 路径（测试用 CPU 适配器边界）；
- 生产 checkpoint 文件级 restore 与真实模型吞吐（测试用真实
  optimizer outer state 与合成 fixture）；
- 发布后目录项（directory entry）crash durability（只保证文件已
  fsync + 原子可见）。

本特性在这些环境上启用前需按本文件的契约另做实机验收。
