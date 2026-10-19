# 条件观测最终定向收尾审查

## 基线、权限与范围

- 指定/实测远端 `feat/condition-observability` HEAD：`069b360da6f357210d3a106870851bf48764e3f6`，基线工作树干净；GitHub run `39225030862` 已通过。
- 从该提交创建新的独立 worktree `tmp/condition-observability-final` 和本地 `fix/condition-observability-final` 分支；不覆盖前两轮工作树。
- 核验时远端 dev 已从早先 `4a94dfe` 前进到 `6e57ed203eebf00679a86a197a45519c5c93b0ae`（独立 condition optimizer 工作）；目标分支未移动。本轮不 merge/rebase 该改动，不修改 main/dev，不将独立 optimizer 功能混入本分支。
- 全部开发/验证为本地 Windows/WSL CPU，不 SSH、不上传到训练主机、不操作生产训练进程；不读取密钥、实际模型/数据或生产 checkpoint。

## P1：已修复的重要行为问题

### P1-1：recorder 停用后上游继续做无消费者诊断

**根因：**原 production 在 recorder 构造前就按配置启用 composite capture/runtime cadence；构造失败返回 None 只撤掉 observer。运行中 sink 写入故障使 recorder disabled，但静态 `condition_grad_share=True` 和 runtime cadence 不随之失效。

**修复：**

1. `ConditionDiagnosticsRecorder.should_record(update)` 是无副作用的实时状态+cadence 判定，包含 disabled/closed。
2. production 在构造 runtime 前初始化 recorder 为 None；一个 rank-local closure 查询该 recorder，供 runtime activation 和每次 loop 的额外梯度诊断共用。构造失败、尚未构造、校准路径、运行中 I/O 停用和 close 都不能使消费者缺席时继续计算。
3. 删除 production 的静态 composite capture 开启。runtime 仅在 eligible 更新的最终 microbatch forward 前开启，计算原来的同 forward、pre-optimizer 标量事实，并在 finally 中关闭/释放输出引用。
4. 独立复核发现 standalone constructor/assembly 调用方可能预先开启 capture，第一次补丁的 false gate 未清除该状态。本轮追加 runtime 入口所有权规则：在 prepare/eligibility 前清掉预启用 capture；正常/off-cadence/nonfinal/live-false 和 prepare/forward 异常均不残留引用。公开只读 property/disable 操作为普通运行时属性，不进入 state_dict。
5. 仍只降级构造/写入处的普通 I/O 故障；结构性 shape/key/missing facts、模型 forward、loss、activation code、backward、optimizer 的真实异常不吞掉。

**证据：**新 `test_condition_diagnostics_lifecycle.py` 保留真实 tiny DenseDiT、TextConditioner、ConditionTokenEncoder、TrainableComposite、runtime.measure、flow objective、training loop、SingleGpuStep、激活事实和 recorder。只绕过 GPU-only constructor/本地资产准备，用计数 wrapper 委托真实实现；不是只检查 JSONL 停写。对 enabled/write-I/O/constructor-I/O/closed/no-consumer 与默认 off 比较 loss、每个参数、Torch RNG、scheduler 更新号和 checkpoint cadence，逐值相同。构造 I/O、写入 I/O 使用真实对应边界注入。forward/loss/facts/record 结构异常保留同一异常对象。

### P1-2：malformed JSONL 被 CLI 成功退出掩盖

- 审计报告 **schema 4** 新增独立 `file.status=ok|malformed_rows|container_error`；window presence/coverage、冲突 update 排除和缺失区间语义不变。
- JSON 语法错误及非 object JSON 值使 CLI 返回 **5**，保留有效前后行统计、总 malformed 数量和前 50 条 line/error 详情；全坏文件不冒充成功空审计。
- 全文件 malformed 无法可靠归属 update，不能用缩窄窗口把它隐藏。合法 object 但没有整数 successful_update 继续按既有 `rows_without_update` 信息计数，不新增不兼容的分类。
- 先输出 stdout/`--out` 报告，随后按 **container 1 > window conflict 4 > malformed 5** 返回；参数错误 2 和输出写入失败 3 的既有优先级不变。
- 外部 schema-3 消费者需要接受 schema 4/exit 5。`file.status=ok` 不是宣称 window 无冲突或训练完整，仍必须看 window/status/excluded。

独立复核：audit 41 用例通过，覆盖 stdout/文件、坏行类型、58 条错误/50 条详情上限、有效尾行、冲突/容器/输出失败优先级及无整数 update 的既有约定。

## P2：已修复的开销与数值边界

### P2-1：log_every_n > 1 仍每步做两次额外 union norm 扫描

原 bool 为启动时快照，故即使非记录更新也做 pre/post 两次条件侧 union 扫描。现在 loop 用 `successful_updates + 1` 实时查询同一 eligibility，向 SingleGpuStep 传递本步 bool。默认 bool=False 与显式 bool=True API 仍兼容；后者是独立调用方明确要求每步扫描，不被误当 recorder 自动模式。

**不删除**原主指标的 encoder/global projection/growth 两项四次 mandatory norm 扫描，不改 clipping、optimizer.step、LR 或归一化；仅门控新增 stream 的 union 两次扫描。已记录更新仍在 sample-mean scale 后、clip 前后取相同集合，激活仍是最终 microbatch 的 pre-step 值。

### P2-2：masked NaN/Inf 乘零仍污染 RMS

`NaN * 0` 与 `Inf * 0` 不等于从样本集合中排除。`_masked_rms` 改为在平方前 `masked_fill(~keep, 0)`，保留 float32 promotion、相同 selected count、平方和/开根定义及 shape/empty 错误。不会用 nan_to_num 把有效位置的异常藏掉：selected NaN/Inf 仍非有限，实际 activation caller 按既有 advisory 规则记录 None。

数值测试在 CPU 复现修复前 8 个失败；修复后覆盖 masked NaN/±Inf、selected 非有限、五种 dtype/strided 输入与 selected-fp32 公式逐值等价、空/全 false、实际 gather/activation caller。独立数值选择 24 passed；完整 stream 54 passed。

## 实测工作量（不是估算吞吐）

以同一 tiny real CPU chain、4 个成功更新、每更新 accumulation=2、cadence=2，对基线和候选分别计数 activation fact 调用、实际保留 condition output、optional union scans 和 mandatory scans。根目录外 harness 复用同一合成模型/输入，无生产资产。记录实测计数，不声称 GPU/compiled-DDP wall time 加速。

下表是上述四更新合计实测；constructor-I/O 行在比较 harness 中模拟 recorder 构造后为 None 的状态，真实构造异常本身由 lifecycle 测试注入。

| 场景 | 版本 | facts | condition capture | union scans | mandatory scans |
| --- | --- | ---: | ---: | ---: | ---: |
| 正常 cadence 2 | 基线 | 2 | 8 | 8 | 16 |
| 正常 cadence 2 | 本轮 | 2 | 2 | 4 | 16 |
| update 2 写入 I/O 故障 | 基线 | 2 | 8 | 8 | 16 |
| update 2 写入 I/O 故障 | 本轮 | 1 | 1 | 2 | 16 |
| recorder 构造失败/None | 基线 | 2 | 8 | 0 | 16 |
| recorder 构造失败/None | 本轮 | 0 | 0 | 0 | 16 |

相关测试还逐步断言：正常更新 1/3 的 optional work 为零；due 更新 2/4 一次 capture+facts、两次 union；更新 2 写入失败后 3/4 全部 optional work 为零；构造失败/closed/no-consumer 从首步起为零；mandatory 每步 4 次。没有把尚未运行的 GPU wall time 换算为性能数字。

## P3：验证边界、旧测试及残余风险

### 不能用选择性 CI 冒充全仓测试

本轮使用已有独立 WSL Python 3.12.14 / Torch 2.13 CPU、train/eval/dev 依赖，另按 pyproject 已声明的 data-tools 范围补装 onnxruntime 1.31.0（及 flatbuffers）；不改项目依赖范围。所有 Python 子进程启动前屏蔽 CUDA/HIP/ROCR，线程有界；全 CPU runner 设置 24 GiB 地址空间上限、禁 core、模型离线/无资产路径。

测试分别执行：

1. **完整默认 `tests/` CPU discovery**，不是选择性 CI；基线 **1675 passed / 122 skipped / 48 deselected**，本轮 **1732 passed / 122 skipped / 48 deselected**，均 0 failed/error，另有同类依赖/只读 NumPy/profiler 的 5 个警告。保留默认 GPU skip 的可见计数，不删 optimizer CPU 测试。
2. **前两阶段及新增生命周期 CI 集合**：本地 **372 passed，无 skip**，含 Guard Window gloo world **1/2/4**、sink/pub/stream/wiring/audit/config/merge。全仓 Ruff PASS，31 个 dev-diff+新增 Python 文件 Pyright **0 errors / 0 warnings**；真实配置图与逐文件 shell syntax PASS。
3. **checkpoint、loop、iREPA 关闭/零权重、condition regressions**：包含在完整默认 discovery 内，JUnit 记录 checkpoint 78 passed、train 279 passed；名称含 iREPA 的 185 passed（另10 skipped）、含 condition_diagnostics 的99 passed，这些集合互有重叠不能相加。Guard Window gloo `[ws1]`/`[ws2]`/`[ws4]` 均真实 passed。独立 reviewer 还运行 94 + 58 相关用例。
4. 默认 discovery 外的 `src/sakuramoon/tests/unit/` 六个历史文件额外单跑（importlib mode）：基线和候选均 **6 passed / 3 failed**，同三个旧失败，不能说“全仓所有测试通过”。

明确的非 CPU/资产排除：

- `tests/unit/checkpoint/test_irepa_artifact.py::test_irepa_projector_is_constructed_on_requested_device` 请求 cuda:0；
- `tests/gpu/irepa/test_irepa_training_graph.py` 五个无 CPU 防护的 capture/compile 函数（11 个参数化 case）；
- `tests/gpu/irepa/test_pe_spatial_geometry.py` 三个依赖真实 CUDA teacher 的函数（36 个 case）；两项 CPU bucket 断言仍执行；
- 共 **48 deselected**，不是通过。
- 另 whole-file ignore `tests/gpu/irepa/test_pe_spatial_reference.py`：有一项是需要真实教师权重的 CPU key-set 验证，其余需 GPU/权重。准确分类为 **资产/加速器不可执行排除**，不是宣称该文件没有 CPU 逻辑；未取用/下载真实权重。

默认 discovery 外的三个旧失败：

| 文件 / 测试 | 基线与候选相同原因 | 处置 |
| --- | --- | --- |
| src/sakuramoon/tests/unit/model/test_g1_function_equivalence.py::test_alpha_zero_is_exactly_function_preserving_for_old_slots | 旧 fixture 的 stable_slot_count=24 不满足当前 exclusive slot-id bound | 不改模型或恢复历史 fixture 来制造绿灯；当前 tests/unit/model 等价测试已正常运行 |
| src/sakuramoon/tests/unit/train/test_g1_alpha_runtime.py::test_training_loop_passes_one_canonical_alpha_to_each_update | 旧 lambda 只接收 batch，当前 loss_fn 需要 batch,index | 不放宽 loop API；当前 tests/unit/train 同名已维护测试正常运行 |
| src/sakuramoon/tests/unit/train/test_g1_forced_checkpoints.py::test_forced_checkpoint_reasons_are_durable_and_ordered | 旧调用缺少 terminal_update keyword | 不改 checkpoint 行为；当前 tests/unit/train forced checkpoint 覆盖正常运行 |

这些文件未被修改、未纳入默认 discovery，不是本轮新回归；不将其隐藏或计入“对应测试全绿”。

### 本轮没有验证的事项

- **没有 DCU/GPU 测试、生产训练、真实 checkpoint restore 或质量验证**；122 个硬件/资产正常 skip 与48个主动 deselect 均非通过。
- 没有 compiled-DDP 与启用 iREPA 同时开启 diagnostics 的部署级性能/正确性实证；CPU iREPA off/zero/模块合同和 gloo owner 统计不能替代这些。
- live gate 为单 rank 顺序训练边界，不宣称处理另一个线程在同一 forward 中途异步关闭 recorder 的并发事务。
- CPU 的精确参数/loss/RNG 等价不证明 accelerator 浮点轨迹逐 bit 相同；额外 bool/callback branch 的实际加速器编译成本未测。
- malformed detail 仍受 50 条上限，完整总数量保留；稀疏 audit 仍保留/排序已有 update，不宣称常数空间。
- 此报告的 schema 4 是离线审计报告 schema，**不是 checkpoint 或主 TrainingMetric schema 变更**。

## 完整 diff 的保护面核对

本轮只涉及诊断运行时 gating、masked RMS、离线审计、对应测试、CI 入口和报告。核对相对 `069b360` 的 diff：

- `config/`、Camera/data/conditioning/model/objective、optimizer/scheduler、checkpoint 源码均无变化；LR/BS/accumulation/主 loss/condition token 结构不变。
- `step.py` 只新增 capture 状态 property/disable；既有 step 数学、optimizer groups、clipping 与 mandatory scans 不变。
- runtime 的 loss 和 iREPA 计算原样位于 capture 的资源作用域中，没有修改公式。
- Guard Window stream、publication helper、owner/counter 定义无修改；gloo world 1/2/4 保留真实运行。
- 新状态都是普通 Python attrs/callable；不注册 parameter/buffer/module，不写 artifact document/state_dict，不改 checkpoint schema。

最终完整 CPU、Ruff、Pyright、CI 与 push 身份由交付说明及对应提交的 GitHub Actions 确认；本地证据保存在 `tmp/condition-observability-evidence/final-*`，不把运行产物提交到仓库。
