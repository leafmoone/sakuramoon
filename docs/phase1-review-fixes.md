# Phase 1 审查后定向修复报告（feat/condition-observability）

对应 root 2026-10-17 执行单。工作区：隔离 worktree `D:/WORK/.sakuramoon-p1fix`
（detached HEAD @ 6a85df8）。基线：origin/dev + 本分支既有提交。

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

**2a `_slot_cosine` 去逐样本标量转换**
（`condition_diagnostics_stream.py:150`）：

- 数学不变：每行仍是 `tensor_cosine` 的**同一组原语调用**
  （`left[row].norm() * right[row].norm()` 分母 + 1e-12 阈值 +
  `torch.dot` 商），active 样本语义与零范数处理契约（任一行分母
  ≤1e-12 → `None`）不变。`tensor_cosine` 本体未动。
- 变化仅在管道：逐行结果累积进设备张量，**消除每样本一次
  GPU→Python 标量转换**（旧版 `float(value)` 逐行强制同步，DCU 上
  反复打断异步流水）。整次调用只有约 5 次整张量同步
  （2×isfinite + any() + isfinite(cosines) + 最终 float），与行数无关。
- 均值用 `cosines.double().mean()`：500 组随机输入下与旧版
  "逐行 float() + Python double 顺序累加"**逐位相同**（见下方等价回归测试）。

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
  (:189) —— 向量化结果与逐行参考实现（6a85df8 原实现，保留在测试里做
  参考）**逐位相等**（`==`，非 approx），20 组形状×尺度组合 + 分段
  active 子集。实现改错（换批量 GEMM 点积、换均值归约、改阈值语义）
  会红。
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

**合成基准（CPU，非生产）** —— Windows CPU 环境，`_slot_cosine` 同输入
对比，n=300 均值，B=20（生产 local_batch），值逐位相等：

| 配置 | 旧（逐行 float 转换） | 新（张量累积） | 说明 |
|---|---|---|---|
| B=20, M=64 | 549.8 µs/次 | 602.4 µs/次 | 0.91× |
| B=20, M=256 | 534.2 µs/次 | 601.7 µs/次 | 0.89× |

**CPU 上无收益甚至略慢，这是预期且如实记录**：CPU 路径上没有
device→host 同步壁垒，`float()` 很便宜，而新实现多几次整张量
`isfinite/any` 启动开销。2a 的收益机制在 GPU/DCU：旧版每行一次
`float(value)` = 每 slot 每到期 microbatch **B 次强制同步**打断异步
流水；新版固定约 5 次整张量同步，与 B 无关。**DCU/last42 上的真实
数字需训练机复验**（本机无 GPU，不冒称生产性能）。

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

- CI unit 门 4 文件 + 受影响的 loop/step 测试：
  `pytest tests/unit/test_training_metrics_audit.py tests/unit/config/test_config_closure.py
  tests/unit/config/test_condition_diagnostics_config.py
  tests/unit/train/test_condition_diagnostics_stream.py tests/unit/train/test_step.py
  tests/unit/train/test_g1_alpha_runtime.py tests/unit/train/test_irepa_lambda_binding.py`
  → **67 passed, 5 skipped**（skip 全为 LINUX_ONLY 标记，本机 Windows）。
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

1. **2a 的 DCU 收益未实测**：CPU 基准显示 0.9×（如实记录）；
   GPU 侧收益依赖"同步次数减少"机制，需在 last42 用
   `config/train_g1.toml` 量级（B=20, M≈slot 宽度）复测
   微基准 + 端到端到期 update 耗时。
2. **`torch.manual_seed` 的 torch-stub 缺口**（torch 2.13 的
   manual_seed 参数在 stub 里是 Unknown）：按项目既有 ignore 模式
   处理，若 CI 锁定的 torch 版本 stub 修复后这些 ignore 变成冗余，
   可顺手删除（无功能影响）。
3. **2b 的测试覆盖是"结构钉住 + 数值等价"组合**：
   `measure()` 无法在无 runtime fixture 的 tests/ 里行为级驱动
   （CUDA + 完整前向图），故"只在最后 microbatch 计算"由方法级
   源码守卫钉住（与既有 (a)/(b)/(c)/(d) 守卫同型，root 已认可该
   模式）；若 loop 未来引入 retry/跳 microbatch 语义，
   `microbatch_index == accumulation - 1` 的接线需同步复核。
4. **sampling.py 本地 pyright 4 个 artifact**（见 4.1 注）：以 CI 为准。
