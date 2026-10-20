# Phase 1 审查后定向修复报告（feat/condition-observability）

> **历史快照（已合并）**：本报告工作已随 PR #2 合并入 `dev`（`31d51cc`）。仅作为证据保留；旧 `src/sakuramoon/tests/**` 路径仅为历史记录，非运行入口。当前正式合同：[配置与恢复](config-and-resume.md)、[条件优化器](condition-optimizer-ab.md)、[Guard遥测](guard-window-telemetry.md)、[Owner契约](guard-window-owner-contract.md)。

对应 root 2026-10-17 执行单。工作区：隔离 worktree `D:/WORK/.sakuramoon-p1fix`
（detached HEAD @ 6a85df8）。基线：origin/dev + 本分支既有提交。
**修订记录**：2026-10-18 复审（85cbbf3）两条 P1 后重做 2a——
(1) 首版 2a 的 `torch.empty(rows)` 无 device 参数，CUDA 下逐行 DtoH、
无收益（§1.2/§2 已更正）；(2) 余弦数学从 stream 回收到共享模块的
批量原语 `tensor_cosine_rows`，标量 `tensor_cosine` 逐字未动。
§4.1 的 CI/本地测试口径一并更正（CI unit 门 46/0 vs 本地 7 文件集合
68/5）。同日复审补充后第二轮（本提交）：等价测试加 full-active
batch 与 available-GPU 分支、全部测试输入显式 float32（85cbbf3 的
临时 buffer 还依赖全局 default dtype，同属临时张量问题）；非有限
测试加"有限输入导致非有限结果"（float32 溢出 inf/inf→NaN→advisory
None）；新增 default-dtype 独立性回归；§2 并入复审 4060 合成
GPU wall-clock（不支持泛称 GPU 加速，收益主体仍是 2b 的 K→1）。

## 1. 已修复的问题与对应测试

### 1.1 非有限值处理（执行单第 1 条）

- **`_slot_cosine` 非有限输入/结果 → `None`**（advisory，不抛异常）：
  `src/sakuramoon/train/condition_diagnostics_stream.py:150`。
  输入非有限（`torch.isfinite(...).all()` 检查）或余弦结果非有限时返回
  `None`；此前 NaN/inf 会经 `float()` 透传进记录并落到 JSONL。
- **builder 有限性校验覆盖全部 fact key**：
  `src/sakuramoon/train/condition_diagnostics_stream.py:358`
  （`sorted(required - {"active_samples", "null_samples"})`，`required` 含
  `slot_cosine_*`；此前只遍历 `_BASE_FACT_KEYS`，slot 余弦不在校验范围）。
  迭代改为 sorted 保证确定性。
- **`json.dumps(..., allow_nan=False)` 可通过**：由 sink 写入路径
  （`DurableJsonlSink`）+ 测试显式断言覆盖。

测试（`tests/unit/train/test_condition_diagnostics_stream.py`）：
- `test_slot_cosine_nonfinite_inputs_are_advisory_none` (:216) —
  NaN 左输入 / inf 右输入 / NaN bias 均 → `None`，有限输入仍为 float。
- `test_record_builder_validates_slot_cosine_finiteness` (:234) —
  `slot_cosine_03` 注入 nan/inf/-inf 均被 builder 拒绝（`must be finite or
  null`）；合法记录与含 null slot 的记录 `json.dumps(allow_nan=False)` 通过。

### 1.2 性能优化（执行单第 2 条）

**2a `_slot_cosine` 去逐样本标量转换**（复审 P1 后重做）：

- 共享工具集中提供**真批量**原语 `tensor_cosine_rows`
  （`src/sakuramoon/train/tensor_metrics.py`）：与标量 `tensor_cosine`
  同一数学定义（float32、逐行 `x.norm() * y.norm()` 分母 + 1e-12
  阈值 + dot/分母商），`[N,D]` 进 `[N]` 出，**在输入设备上整张量
  内核完成**——无 Python 行循环、无逐行内核启动、无逐行
  device→host 拷贝；零范数行抛 `TrainingSamplingError`（与标量原语
  同一错误契约），stream 照旧 catch → `None`。
- 标量 `tensor_cosine` **逐字未动**（9826ffe 版本，采样时诊断的
  逐位等价保证不受影响）；stream 的 `_slot_cosine` 只剩诊断
  advisory 层（非有限输入/结果 → `None`、空 active → `None`、
  `double().mean()`），不再持有第二份余弦实现（85cbbf3 把
  norm/dot/quotient 复制进 stream 的问题，按复审意见已回收到
  共享模块）。
- **85cbbf3 首版 2a 的设备 bug（复审 P1-1）如实记录**：
  `torch.empty(rows)` 无 device 参数，CUDA 环境下是 CPU 张量，
  `buffer[row] = CUDA_scalar` 逐行 DtoH——本机全 CPU 验证从未暴露，
  首版报告"设备张量累积、约 5 次同步"的说法在 CUDA 下不成立。
  复审 RTX 4060 profile 证据：旧版 40 次 DtoH，85cbbf3 新版 42 次
  （两次 CPU buffer 写入 × 20 行 + 2 次 pinned），**无收益**。
  本版彻底删除累积 buffer：所有中间量由输入张量在输入设备上派生，
  整次调用 device→host 同步恰好 5 次（2×isfinite 输入 + 阈值 any()
  + isfinite(cosines) + 最终 mean 的 float），与行数无关。
- **数值等价而非逐位等价（如实）**：批量归约内核与逐行
  `torch.dot` 的 float32 加法顺序不同，逐位相同不可达（本机
  2000 组随机种子/形状/尺度共 5671 次对比实测最差 **6.6e-7
  绝对差**，均值近零时相对差放大至 ~6e-3）。诊断是 advisory，
  跟踪的是 ~4e-3 量级的权重更新漂移，1e-6 级噪声低三个数量级；
  等价测试按 `abs=1e-6` 断言——**实测余量约 1.52×（不是 10×）**，
  且这只是基于 torch 2.13.0+cpu 的一次定界，不是跨环境硬保证
  （见 §4.3-2）。

**2b 消除 microbatch 重复计算**（root 指定实现方式，选项 1）：

- loop 把 microbatch 索引传进 `measure()`：
  `src/sakuramoon/train/loop.py:70`（协议
  `Callable[[BatchT, int], torch.Tensor]`）、`loop.py:264`
  （`self.loss_fn(batch, microbatch_index)`）。
- `measure()` 新增同性质参数 `is_final_microbatch`
  （`src/sakuramoon/train/runtime.py:969`）；门控
  `if every is not None and is_final_microbatch:`（`runtime.py:1025`）
  ——**只在到期 update 的最后一个 microbatch 上算 facts**，到期
  update 的条件器前向次数 K→1；非到期/非最后 microbatch 零计算、
  零保留。
- 失败模式守卫（执行单点名）全部保持：
  - facts 仍在**同一 update、同一权重状态**的前向数据上计算
    （`measure()` 体内、`optimizer.step()` 之前；未挪进 observation
    hook，`maybe_record` 只读已存 facts）；
  - cadence 不到期零计算零保留（门控第一个条件仍是 `every is not None`，
    默认路径字节等价）；
  - `RuntimeMeasurement.condition_diagnostics` 仍只存标量 dict
    （无 `prepared.inputs` 引用，无显存常驻）。
- 生产接线：`runtime.py:1409`（`is_final_microbatch=microbatch_index ==
  config.train.accumulation - 1`，由 `measure_batch` 闭包传入）。
- 连带更新（loop 协议变更的既有调用方）：
  `src/sakuramoon/perf/harness.py:448`（`measure_batch` 加索引参数）、
  `tests/unit/train/test_g1_alpha_runtime.py`、
  `tests/unit/train/test_irepa_lambda_binding.py`（loss_fn 签名 +
  strict 模式注解补齐，见 1.4）。

测试：
- 数值等价：`test_slot_cosine_vectorized_matches_rowwise_reference`
  —— 批量 `_slot_cosine` 与旧逐行标量路径（逐行 `tensor_cosine`，
  参考实现保留在测试里）按 `abs=1e-6` 等价（torch 2.13.0+cpu
  实测边界 6.6e-7，**实测余量约 1.52×**；逐位不等价的依据写在
  测试 docstring）。**边界不是跨环境通用上界**：复审在 torch 2.11
  CPU、full-active B=20/M=15360 上实测到 1.47e-6，已**超过** 1e-6
  ——不同 torch 版本/形状会漂移，所以测试红了是预期内的信号：
  **重测定界，不静默放宽**（写进测试 docstring 与 §4.3-2）。复审
  补充后：覆盖 **full-active batch**（20/23 行全 active 同时归约，
  不再只有 ≤4 行窗口；full-active 边界 torch 2.13.0+cpu 实测
  1.8e-7）+ **available-GPU 分支**（CUDA 可用时同比较在设备上跑；
  **DCU/GPU 上的实际边界仍未测量**，1e-5 是保守占位、非已验证
  上界，last42 复测定界）；全部输入显式 `dtype=torch.float32`。
- default-dtype 独立性：`test_slot_cosine_is_independent_of_global_default_dtype`
  —— 85cbbf3 的 `torch.empty(rows)` 无 dtype，`set_default_dtype(float64)`
  下新旧结果分歧（0.9599999785423279 vs 0.96）；当前路径不创建任何
  未显式类型的临时张量（生产代码 grep 零 buffer 创建），测试把这条
  钉死（wrapper 与共享原语都断言）。
- 共享原语契约：`test_tensor_cosine_rows_matches_scalar_and_error_contract`
  —— 逐行值在等价界内、零范数行抛 `TrainingSamplingError`、
  非法形状/空拒绝、`[N]` 返回。
- 非有限：`test_slot_cosine_nonfinite_inputs_are_advisory_none` 增加
  **有限输入导致非有限结果**分支（1e20 量级 → 范数积溢出 inf/inf
  → NaN → advisory None，旧实现会落 NaN 进记录）。
- 非诊断路径不受影响 + 接线钉住：结构守卫扩展
  （:817 起）——
  - (e) `measure` 方法源必须含
    `if every is not None and is_final_microbatch:`（最后 microbatch
    门控 + 默认路径短路的精确形态）；
  - `is_final_microbatch=microbatch_index` 与
    `config.train.accumulation - 1` 必须在 runtime 源里（接线存在）；
  - (f) loop 源必须含 `self.loss_fn(batch, microbatch_index)`
    （索引信号从 loop 来，不是自算计数器）。
  既有的 (a) 计算点在 `measure` 体内 / (b) 不在 `maybe_record` /
  (c) 构造函数无 `composite` / (d) `update_number=next_growth_update`
  断言不变。

### 1.3 `sampling.py` 重复导入块（执行单第 3 条）——实测后结论：**保持现状**

- 尝试把 3 个 `from sakuramoon.train.condition_diagnostics import (...)`
  合并为 1 个后，项目自身的 ruff isort 规则
  （`combine-as-imports` 默认 false：含别名的导入必须独立成句）
  要求把 `_metric_float`/`_tensor_rms` 两个别名导入拆回独立语句，
  即三块结构是 CI `ruff check .` 全仓门的规范形态，合并会让 CI 红。
  非别名的 5 个名字已经在同一个块里，无可合并项。
  已恢复 `sampling.py` 到 HEAD（本提交对它的 diff 为零）；
  共享张量工具、必要校验、有效测试均未删。

### 1.4 连带 strict 注解（loop 协议变更使 3 个既有文件进入 CI 改动集）

CI static 对 `origin/dev...HEAD` 改动文件跑 pyright（strict）。
loop 协议变更必须更新 3 个既有调用方文件，使其进入改动集后暴露出
**既有的** strict 违规（这些文件在 origin/dev 上从未被 CI 检查过）：

- `tests/unit/train/test_g1_alpha_runtime.py`：`tmp_path: Path`、
  `observation: SuccessfulLoopObservation`、lambda→typed def、
  `torch.manual_seed` 的 torch-stub 缺口按项目既定 ignore 模式标注。
- `tests/unit/train/test_irepa_lambda_binding.py`：同上 +
  `_bare_runtime` 的 `__new__` 属性赋值 ignore + `loss_fn` 联合类型
  收窄（`isinstance` 替代 `hasattr`，对这两个测试类行为等价）。
- `src/sakuramoon/perf/harness.py:235`：accelerate 延迟导入的 ignore
  补 `reportMissingTypeStubs`（与 `production.py:15` 的既有模式一致）。

未新增任何测试删除；未改断言语义。

## 2. 诊断开销优化前后的测量

### 2a（复审 P1 重做后）

**GPU 侧（复审 RTX 4060 / torch 2.11.0+cu128 profile，B=20/M=64）**：

| 版本 | DtoH 拷贝 | 总同步 |
|---|---|---|
| 6a85df8 旧版（逐行 float()） | 40（pinned） | 40 |
| 85cbbf3 首版（CPU buffer 逐行写入） | 40（pageable）+2（pinned） | 42 |
| 本版（共享批量原语） | **分析值 5 次**（未在本机复测，无 GPU） | 5 |

本版的 5 = 2×isfinite 输入 + 阈值 `any()` + isfinite(cosines) + 最终
`float(mean)`，全部是整张量谓词/标量归约，**与行数无关**；无累积
buffer、无逐行内核启动、无 Python 行循环（中间量全部由输入张量在
输入设备上派生，设备归属由构造保证，不存在 85cbbf3 那种 CPU buffer）。
**本机无 GPU，4060 上的实际 DtoH 计数请复审用同一 profile 方法复测
后确认。**

**合成基准（CPU wall-clock，非生产，n=300 均值）**：

| 配置 | 旧（逐行 float 转换） | 本版（共享批量原语） | 说明 |
|---|---|---|---|
| B=20, M=64 | 551.9 µs/次 | 147.5 µs/次 | 0.27× |
| B=20, M=256 | 652.8 µs/次 | 202.7 µs/次 | 0.31× |

CPU 上 3.5–3.7× 来自消除 Python 行循环（2B 次逐行调度和 2B 次
float() 调用）；GPU 上的收益机制则是逐行 DtoH 同步消除（40→5）。
DCU/last42 真实数字需训练机复验，不冒称生产性能。

**合成 GPU wall-clock（复审 RTX 4060 / torch 2.11.0+cu128，B=20，
非生产、非 DCU、非端到端；"新"列针对 85cbbf3，非本提交）**：
M=64 旧 2985.6µs → 新 2749.4µs（1.086×），M=15360 旧 2867.5µs →
新 2985.6µs（0.960×）。**如实结论：不支持泛称 GPU 加速**——同步
次数下降（profiler 证据）不等于 wall-clock 加速，该合成负载在 4060
上并非同步瓶颈。本报告不主张 2a 的 wall-clock 收益；2a 的主张限于
（a）逐行 DtoH 消除（结构事实，待 4060 复测 857362f 计数）、
（b）诊断值语义不变（数值等价测试）。**真正的开销主体收益是 2b 的
K→1**（见下）。

**2b 条件器前向次数 K→1（结构性事实，非测量）**：
修复前到期 update 的每个 microbatch 都调一次
`compute_condition_activation_facts`（accumulation=K 时 K 次，前 K-1 次
结果被覆盖丢弃）；修复后仅最后 microbatch 调 1 次（`runtime.py:1025`
门控）。非到期 update 前后都是 0 次。按生产
`config/train_g1.toml`（local_batch=20, accumulation=20）：每到期
update 省 19 次条件器前向（该前向与训练前向同规模）。

## 3. 精简或保留的文件及依据（执行单第 4 条）

| 文件 | 结论 | 依据 |
|---|---|---|
| `scripts/audit_training_metrics.py` (439 行) | **保留** | 有调用方：(a) 运营手册入口 `docs/handover-20261017.md:104`（对训练 metrics JSONL 窗口做只读审计）；(b) CI unit 门里的回归测试 `tests/unit/test_training_metrics_audit.py` 直接 import 它（脚本删除则测试红）。与 `check_configs.py` 无功能重复（审计对象是 JSONL 遥测，不是 TOML 图）。 |
| `scripts/check_configs.py` (227 行) | **保留** | 有调用方：CI contracts job（`ci.yml:92`）每 run 执行 `--config-root config`；回归测试 `tests/unit/config/test_config_closure.py` import 它。docstring 自述与 `load_config` 的边界（只做 extends 闭包+环检查，不做业务 schema）——无重复。 |
| `.github/workflows/ci.yml` (99 行) | **保留** | 三 job 职责不重叠：unit（CPU 行为门，含 4 个本分支测试文件）、static（ruff 全仓 + pyright 改动文件）、contracts（shell 语法 + 配置图 + gitleaks）。unit 与 static 的 `uv sync` 参数相同是刻意的（unit 的结构守卫要 import `train.runtime`，需要 train/eval extras；见 6a85df8），合并 job 会模糊故障定位。无可删步骤。 |

无精简。预期结论即"保留"，逐条依据如上。

## 4. 测试结果与剩余风险

### 4.1 已运行（worktree，主区 venv：Python 3.12.13 / torch 2.13.0+cpu /
pyright 1.1.414，`PYTHONPATH=src`）

- CI unit 门 4 文件 + 受影响的 loop/step 测试（7 文件集合，本地
  Windows）：`pytest tests/unit/test_training_metrics_audit.py
  tests/unit/config/test_config_closure.py
  tests/unit/config/test_condition_diagnostics_config.py
  tests/unit/train/test_condition_diagnostics_stream.py
  tests/unit/train/test_step.py tests/unit/train/test_g1_alpha_runtime.py
  tests/unit/train/test_irepa_lambda_binding.py`
  → **69 passed, 5 skipped**（skip 全为 LINUX_ONLY 标记，本机 Windows）。
  口径澄清（复审指出）：CI unit 门（Linux）实测是 **46 passed /
  0 skipped**（仅门内 4 文件，无 LINUX_ONLY 可跳；上一提交 857362f
  为 47/0，本提交门内 +1 新测试，以 CI 日志实测为准）；本行 69/5
  是本地 Windows 7 文件集合（含 5 个 LINUX_ONLY skip），此前交付
  消息误把该口径当 CI 口径，两者都是真实数字但范围不同。
- 全 `tests/unit`（排除 7 个既有环境缺依赖的收集错误文件：fcntl×6、
  onnxruntime×1，均与基线相同）→ **26 failed, 1308 passed, 80 skipped**。
  26 个失败逐一与主区基线（未改动的 6a85df8）比对：同测试同样失败
  （camera_viewport/camera_data/irepa_artifact/pe_spatial/benchmark_trace/
  compile_cache/worker_spawn，全部环境类，与本分支改动无关）。
- `ruff check .` → All checks passed。
- `pyright`（CI static 同款 7 个改动 py 文件集：stream/runtime/loop/
  harness + 3 个测试文件）→ **0 errors**。
  注：本地 pyright 对 `sampling.py:753/788` 报 4 个既有 artifact
  （CI 环境两次全绿 run 已证明其干净；本次改动未触及那些行），
  本地与 CI 的 pyright/torch 解析存在轻微漂移，以 CI run 为准。

### 4.2 明确未运行

- **DCU/GPU 测试**：本机无 GPU。`mark_g1_gpu` 等 GPU 测试未在本分支
  环境运行；LINUX_ONLY 测试（5 skipped）本机跳过。
  需 root 在 last42 真实 Linux + DCU 环境复验（锚点见第 1 节行号）。
- 未使用任何其他分支的测试结果充当本分支验收。

### 4.3 剩余风险

1. **2a 的 GPU DtoH 计数是分析值（5 次），未在 GPU 上复测**：
   本机无 GPU；4060 上的实际计数请复审用同一 profile 方法复测
   确认（机制：无逐行拷贝，仅 5 次整张量谓词/标量归约，见 §2）。
2. **等价测试的数值边界是定界，不是硬保证（复审独立复测证实）**：
   `abs=1e-6` 断言基于 torch 2.13.0+cpu 实测边界 6.6e-7，**实测余量
   约 1.52×**（不是 10×）；且复审在 torch 2.11 CPU、full-active
   B=20/M=15360 上实测 1.47e-6，**已超过 1e-6**——该断言不是跨
   torch 版本/形状/环境的通用上界。因此等价测试**会跨版本漂移变
   红，这是设计内的行为**：红了说明归约内核/边界变了，处理方式是
   在对应环境重测最差差值、重新定界并记录依据，**不静默放宽
   阈值**。GPU 分支的 1e-5 同理：DCU/GPU 实际边界未测量，保守占位。
3. **`torch.manual_seed` 的 torch-stub 缺口**（torch 2.13 的
   manual_seed 参数在 stub 里是 Unknown）：按项目既有 ignore 模式
   处理，若 CI 锁定的 torch 版本 stub 修复后这些 ignore 变成冗余，
   可顺手删除（无功能影响）。
4. **2b 的测试覆盖是"结构钉住 + 数值等价"组合**：
   `measure()` 无法在无 runtime fixture 的 tests/ 里行为级驱动
   （CUDA + 完整前向图），故"只在最后 microbatch 计算"由方法级
   源码守卫钉住（与既有 (a)/(b)/(c)/(d) 守卫同型，root 已认可该
   模式）；若 loop 未来引入 retry/跳 microbatch 语义，
   `microbatch_index == accumulation - 1` 的接线需同步复核。
5. **sampling.py 本地 pyright 4 个 artifact**（见 4.1 注）：以 CI 为准。
