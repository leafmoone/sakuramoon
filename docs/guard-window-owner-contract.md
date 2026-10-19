# Phase 2A：CMuon Guard Window Telemetry — Owner 与计数契约

## 1. 范围与结论

只读审查 `feat/condition-observability` 的第一阶段验收基线
`e5515e75303c7bbcb5de24b16443bfd1c2533d5d`，共同 dev 基线为
`4a94dfe564e8ef439eafee71be64dcc5092962c9`。采用独立 worktree
`D:/WORK/sakuramoon-p2a`（detached HEAD）。本次仅提交本报告；不修改优化器、
checkpoint、配置、TrainingMetric、测试或 CI，不实现窗口写入器或合并器。
没有访问模型/数据或操作生产训练进程。

**不能把所有 guard 计数统一跨 rank 相加：**

- FP32 rescue 五个标量和 `rescue_by_role` 的**新事件**是 owner-local。
- base 的 `skip_*` 是各 rank 重复计算的决策计数；一致性必须验证，不能假定。
- `observations` 在 commit 后、postcommit invariant 检查前递增；不能无条件等同于
  trainer 的 successful-update 号或 `step()` 正常返回次数。
- checkpoint 只保存 rank0 的 owner-local 历史，各 rank 恢复同一文件后会复制这份历史。
  恢复后的 owner 绝对值不能直接求和；窗口须从各 rank **实际恢复后的本地 baseline**做差。

2A只建立契约与证据，是否进入2B由用户审核决定。

## 2. 源码证据

以下路径前缀为 `src/sakuramoon/`，行号相对上述基线。

| 对象 | 位置 | 语义 |
|---|---|---|
| owner映射 | `optim/guarded_canonical.py:105-118,243-244` | FNV-1a 64 对 `f"{fqn}#chunk{chunk}"` 的UTF-8字节计算，再对world size取模；相同输入确定唯一owner |
| base skip | `optim/guarded_canonical.py:330-359` | 每rank遍历全部有梯度chunk；低信号分支在owner gate之前累加skip；仅未skip的NS受owner gate控制 |
| 信号/refs | `optim/guarded_canonical.py:300-324,562-579` | Nesterov信号与references各rank本地计算，没有信号/skip-mask共识 |
| base observations | `optim/guarded_canonical.py:526-603` | 参数commit/ref更新后加一，之后才进行可选参数invariant检查 |
| rescue active集 | `optim/fp32_rescue.py:239-240,290-304` | 不走base低信号skip，每个有梯度chunk都active；只有owner执行BF16 NS |
| rescue候选计数 | `optim/fp32_rescue.py:302-413` | owner在kernel调用前记attempt，按返回的数值verdict记接受/失败 |
| rescue observations | `optim/fp32_rescue.py:645-724` | commit后、postcommit检查前推进 |
| guard状态 | `optim/guarded_canonical.py:653-743`；`optim/fp32_rescue.py:925-955` | base保存skip/obs/refs，rescue子块保存五标量+role map；加载直接恢复 |
| 恢复身份校验 | `optim/guarded_canonical.py:697-713`；`checkpoint/load.py:899-927` | state与schema sidecar两层拒绝world size/owner mapping版本变化，要求显式迁移 |
| 生产保存/恢复 | `train/production.py:1193-1203,1430`；`checkpoint/save.py:236-244`；`checkpoint/load.py:1619-1637,1673-1679` | rank0写outer state，仅广播路径；每rank读取同一checkpoint，并将完整outer state交给load_state_dict，不保存各rank的独立计数历史 |
| 历史日志 | `train/production.py:565`；`optim/fp32_rescue.py:441-466,727-752` | 生产只给rank0设置stats logger；部分rescue汇总在最终verdict/commit之前输出 |

`guarded_canonical.py:322` 的“Single batched sync”指设备信号打包后通过`tolist()`
读到主机，不是distributed collective。不能把该注释当作跨rank一致性证据，
也不应误报成实现了错误的distributed sync。

## 3. 字段契约表

**O**=owner-local新事件；**R**=每rank复制的本地计数；**P**=当前进程辅助计数。
求和规则仅针对身份一致、覆盖完整的**同一窗口增量**，不授权相加恢复后的历史绝对值。
缺失字段/rank/来源不能默认补零。

| 字段 | 计数位置 | 覆盖范围/事件 | 跨rank规则与一致性 | checkpoint恢复 | 缺失处理 |
|---|---|---|---|---|---|
| `bf16_attempts` | `fp32_rescue.py:302-305` | **O**；owner调用BF16 kernel前+1，kernel抛异常也可已记attempt | 验证mapping/world/routing/同窗口完整rank集后对本地delta求和；各rank不要求值相等 | rank0绝对值被全部rank装载；其他rank旧历史不可重建 | 未知而非零；缺rank则全局不完整 |
| `bf16_safety_failures` | `fp32_rescue.py:357-364` | **O**；BF16返回非有限/超ceiling后进入救援时+1，不等于全部异常 | 同上；正常数值分支与FP32 attempts增量相等 | 同上 | 同上 |
| `fp32_attempts` | `fp32_rescue.py:363-365` | **O**；FP32调用前+1，kernel抛异常可无候选verdict | 同上 | 同上 | 不得用rescues+failures补齐异常attempt |
| `fp32_rescues` | `fp32_rescue.py:403-410` | **O**；接受FP32候选时+1，**不是整个step已commit次数** | 同上；仅完整数值verdict分支可用rescue+failure分解attempt | 同上 | 有局部rescue日志不证明update成功 |
| `fp32_rescue_failures` | `fp32_rescue.py:382-387` | **O**；FP32返回非有限/超ceiling，不包含所有kernel/通信/logger异常 | 同上 | 同上 | 未知不补零 |
| `rescue_by_role` | `fp32_rescue.py:411-413` | **O**；接受候选按role分桶 | 同上、按role合并delta；仅已知完整快照的未出现role可按稀疏零解释 | rank0整个map被各rank装载 | 缺map不是空map，不完整快照不能补role零 |
| `skip_total` | `guarded_canonical.py:348-355` | **R**；所有rank对全部有梯度chunk作本地低信号判定；rescue变体不走gate，新增delta为零但继承绝对值可非零 | **禁止求和**；校验同窗口各rank起点/终点/delta一致后保留一次；不一致失败 | base load `:730`直接恢复；部分缺scalar的loader默认零不是完整性证明 | 缺rank无法验证一致性，不能选rank0冒充完整 |
| `skip_by_role` | `guarded_canonical.py:352-354` | **R**；按role分桶 | 禁止求和；完整稀疏map规范化后全rank一致才保留一次 | base load `:731-734`装载 | 缺map不等于空map，不一致失败 |
| `skip_by_fqn` | `guarded_canonical.py:355` | **R**；按FQN分桶，多chunk可累加到同一FQN | 禁止求和；同routing/FQN身份且一致才保留一次 | base load `:735-738`装载 | FQN消失不能解释成零 |
| `observations` | base `:582`；rescue `:697` | **R**；commit/ref更新后、postcommit检查前+1 | 禁止求和；同身份全rank一致才保留一次；另带trainer起止update | base load `:739`恢复；可随旧checkpoint回退 | 缺失则身份不可确认；跨rank不一致则未来窗口校验必须失败 |
| `fp32_low_delta_rescues` / `fp32_low_delta_by_role` | `fp32_rescue.py:414-427,171-177` | **P/O**；低幅但接受的软救援 | 仅当前进程同身份窗口可按O规则处理 | **不持久化**，新实例归零 | 重启前历史未知 |
| `_steps_this_process` | `fp32_rescue.py:209-222` | **P/R**；进程内step调用，有限梯度检查之前+1 | 非全局update，不跨rank求和当训练步数 | 不持久化 | 不得据此推断缺失窗口 |

### 3.1 恢复的历史来源

保存的是rank0 outer state，其rescue计数只覆盖rank0拥有的chunk；恢复到W个rank后，
该历史复制W份，其他rank自己的旧历史不在文件内。即使绝对值相等，也不意味着O字段
变成了R字段；之后的新事件仍然owner-local。

后续2B/2C必须：保留实际装载的绝对值与来源；不改/清零原优化器计数；建立各rank
恢复后本地baseline；只对同身份end-start求delta；O字段在互斥与完整rank验证后合并新delta；
R字段校验一致后保留一次。不跨run、恢复边界、world、mapping、routing或计数器身份拼接。

## 4. 动态证据与证伪范围

### 4.1 既有测试

`tests/unit/optim/test_guarded_canonical.py:131-149` 的owner测试遍历 **1/2/4/8**，
实际运行通过；末尾`... or True`是不能证伪的旧断言，不作为world变化证据，本阶段不改测试。
另实际执行独立partition断言：对`probe0..3`各chunk0/1，在ws1/2/4中均验证
`sum(stable_owner(fqn,chunk,W)==r for r in range(W)) == 1`。得到owner向量：

```text
ws1: [0,0,0,0,0,0,0,0]
ws2: [1,0,0,1,1,0,0,1]
ws4: [3,0,2,1,1,2,0,3]
```

这不是均衡性的通用承诺；本夹具覆盖所有rank，用于实际step探针分配归属。

`tests/unit/optim/test_fp32_rescue.py:359-392` **已有**CUDA-only单rank checkpoint
roundtrip，覆盖五标量/role map/obs及缺rescue块的父类恢复。缺口是多rank读取rank0历史后的来源，
不是“完全没有roundtrip测试”。

### 4.2 CPU/gloo的可测边界

生产builder与SR RNG依赖CUDA，但真实构造器允许注入协作者，CPU不是完全不能验证guard。
探针采用真实类`__init__`/`step()`/`stable_owner`、真实gloo broadcast/all_reduce；
CPU AdamW与CPU-only SR adapter替代硬件依赖。健康组运行真实CPU NS；数值故障组注入
NS异常/非有限候选，另有logger/测试AdamW异常及合成初值/梯度不一致的时序负例。
这些组都不替换owner、计数或collective。失败探针阻止forensic/capsule写入，
**不验证forensic持久性，也不代表native TorchAO AdamW8bit/SR验收**。

独立复审（Python3.12.13/torch2.13.0+cpu）的夹具是4个BF16参数`probe0..3`，各[8,4]
拆2chunk，另1个FP32 AdamW参数，共8个NS输入。ws4 owner顺序
`[3,0,2,1,1,2,0,3]`，每rank2个。

| 场景 | 实跑观察（ws1/2/4，另注者除外） | 证伪意义 |
|---|---|---|
| base全active | 每rank NS=8/4/2；ws>1广播8次；参数全张量相等；obs1 | owner唯一且NS工作互斥 |
| base全skip | 每rank skip8、NS0、broadcast0；CMuon参数不变，momentum已变，refs冻结；AdamW1、obs1 | skip是R，跨rank求和会乘W |
| base半skip+其余NS非有限 | 每rank skip4已记；全rank安全异常、obs0、AdamW0、参数不变、momentum已变 | skip是决策计数，不是已commit事件；非全状态事务回滚 |
| rescue健康 | 每rank五元组(BFattempt,BFfail,FPattempt,rescue,FPfail)=(8/W,0,0,0,0)，obs1 | owner-local attempt覆盖 |
| BF16非有限/FP32救回 | 每rank(8/W,8/W,8/W,8/W,0)，obs1、参数一致 | rescue新事件互斥 |
| BF16/FP32均非有限 | 每rank(8/W,8/W,8/W,0,8/W)，obs0、AdamW0、参数不变 | failures是O且失败保留已发生候选计数 |
| ws4部分救回/部分失败 | 每rank(2,2,2,1,1)，但全step未commit、obs0 | rescues不是已提交更新数 |
| 单个非rank0 owner输入 | ws2 owner1、ws4 owner3；rank0计数零、owner attempts1，全rank更新、obs1 | rank0日志不是全局owner事件 |

### 4.3 复现来源、命令与作者复跑

独立复审最先以stdin运行分组探针，之后仅在仓库外提供公共夹具合并版：
`D:/WORK/sakuramoon-p2a-probe/reviewer_probe.py`。提供时仅`--help`检查过、未整套复跑；
**作者随后实际逐组运行该合并版**，不把最初的`--help`当测试结果。以下命令都在上述
基线worktree、现有venv下执行，输出仅stdout，探针不入git：

```bash
export UV_PROJECT_ENVIRONMENT=D:/WORK/Sakuramoon/.venv
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run --no-sync python -B D:/WORK/sakuramoon-p2a-probe/reviewer_probe.py --suite counters
uv run --no-sync python -B D:/WORK/sakuramoon-p2a-probe/reviewer_probe.py --suite contracts
uv run --no-sync python -B D:/WORK/sakuramoon-p2a-probe/reviewer_probe.py --suite negative
uv run --no-sync python -B D:/WORK/sakuramoon-p2a-probe/reviewer_probe.py --suite exceptions
uv run --no-sync python -B D:/WORK/sakuramoon-p2a-probe/reviewer_probe.py --suite skip-rate
```

作者复跑结果（退出码均0；负例的预期异常被探针捕获并单独报告）：

- `counters`：ws1/2/4各8种场景，共**24个场景结果**，真实gloo。
  全skip各rank `[8] / [8,8] / [8,8,8,8]`；rescue全救回各rank五元组分别
  `[8,8,8,8,0] / [4,4,4,4,0] / [2,2,2,2,0]`；全失败分别
  `[8,8,8,0,8] / [4,4,4,0,4] / [2,2,2,0,2]`，obs全0。
- `contracts`：ws1/2/4各7种场景，共**21个场景结果**。all-active/skip/mixed-failure、
  单个非rank0 owner、梯度分歧、AdamW异常和postcommit失败的观察与上表一致。
  ws2/4 postcommit failure实际obs全1、status全`CMuonSafetyError`；梯度幅度负例
  参数相等而momentum/refs不等。
- `negative`：独立ws2组5秒超时，一侧obs1/skip1/AdamW1，另一侧obs0/skip0/AdamW0；
  两侧输出预期的gloo通信RuntimeError；子进程回收，不复用坏process group。
- `exceptions`：ws1单chunk共**4个场景结果**，五元组依次为
  `(0,0,0,0,0)`、`(1,0,0,0,0)`、`(1,1,1,0,0)`、`(1,1,1,1,0)`，obs全0。
- `skip-rate`：三步实际为`(obs,skip,rate,eligible)` =
  `(1,8,1.0,8)`、`(2,16,1.0,8)`、`(3,18,3.0,2)`。

**这些是临时探针场景，不是新增48项pytest，也不在hosted CI中执行。**
临时脚本留在仓库外供当前机器复现；干净clone不包含它，也不能只凭命令路径宣称可永久复现。
后续持久化的核心回归测试属于2D，本次只提交勘察报告。

### 4.4 独立控制流/恢复探针（作者再次复跑通过）

另一个独立探针放在仓库外`D:/WORK/sakuramoon-p2a-probe/probe.py`，与上节8输入夹具不同：
本夹具为3个BF16参数、chunk数1/2/6、role为attention_q/ffn_in/adaln_shared，共**9个输入**。
保留真实构造器、step、owner、gloo、state_dict/load_state_dict；AdamW/SR为CPU测试适配器，
**NS为确定性合成kernel**、capsule发布为测试no-op，不声称该组验证原生NS或forensic耐久性。
作者实际复跑下列命令，退出0，7个worker退出码全0，无超时：

```bash
export UV_PROJECT_ENVIRONMENT=D:/WORK/Sakuramoon/.venv
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run --no-sync python -B D:/WORK/sakuramoon-p2a-probe/probe.py --world-sizes 1,2,4
```

| W | 每rank owner输入数 | 两步后owner attempts/rescues | 装载同一rank0状态后 | 再执行一步后 |
|---|---|---|---|---|
| 1 | `[9]` | `[18]` | `[18]` | `[27]` |
| 2 | `[4,5]` | `[8,10]` | `[8,8]` | `[12,13]` |
| 4 | `[1,3,3,2]` | `[2,6,6,4]` | `[2,2,2,2]` | `[3,5,5,4]` |

每rank全skip两步的`skip_total=18`、`observations=2`，`skip_by_role`为
`{adaln_shared:12, attention_q:2, ffn_in:4}`且`skip_by_fqn`跨rank相等；
owner kernel调用归属和role计数与独立FNV期望逐项比对。
装载试验是在内存中把rank0 outer state分发给各rank，再调用真实load；它模拟生产同文件
读取的**状态输入**，并非运行生产文件下载/保存路径。world/mapping/config/缺guard/缺cmuon
均被真实loader拒绝；状态往返相等。合成BF16失败/FP32失败的计数分支也通过。

特别是ws4恢复前总attempts=18，恢复后各rank都从2起步，再一步后的绝对值和为17，
既不是原历史18加新增9的27，也不能当完整历史；正确的新段delta是
`[3,5,5,4]-[2,2,2,2]=[1,3,3,2]`，和为9。

ws2/4的负控制将“owner取max代替sum”“复制skip/obs跨rank求和”“错位/round-robin owner”
代入期望断言，均触发拒绝；ws1下max与sum不可区分，不把该组叫作证伪。
本探针的梯度分歧只调用真实predicate，**没有执行混合broadcast schedule**；
真正执行该故障schedule的证据是上节reviewer独立5秒超时组。所有输出只作为临时测试证据，
不提交JSONL结果、合成诊断产物或脚本，不实现合并器。

### 4.5 一致性前提负例

独立复审运行了有意破坏DDP输入一致性的负例，**不代表正常DDP必然故障**：

- 同active分类、grad乘(rank+1)：ws2/4 step成功，最终参数相等，但momentum/refs不同。
  owner delta广播和参数指纹不能证明输入/optimizer-state一致。生产postcommit invariant
  仅对CMuon specs的RMS/absmax两个标量作MIN/MAX，不是全参数逐位验证，也不覆盖AdamW、
  momentum或refs；探针另行对完整合成CMuon张量作`torch.equal`比较。
- 不同skip分类：ws2单输入owner1，rank0 tiny grad→skip，rank1正常grad→active。
  独立gloo组5秒timeout复现broadcast参与不一致：一侧obs1、一侧obs0，最终通信RuntimeError，
  不是同步CMuonSafetyError。该负例必须隔离process group，不能死锁普通测试作业。

“所有rank执行计数语句”不能写成“实现保证这些计数永远一致”。未来必须实际校验，
本阶段不修改skip决策或增加生产collective。

### 4.6 observations与日志时序

precommit安全失败时obs不推进，但momentum/attempt/候选/skip可已变；postcommit invariant
失败时obs已加一且参数已提交。因此“任何异常都obs不变且全rank同步abort”不成立。
在ws1、单输入/单chunk异常探针中，BF16 kernel抛异常可留下(1,0,0,0,0)，
FP32 kernel抛可留下(1,1,1,0,0)，rescue logger抛可留下(1,1,1,1,0)，均可obs0。
`fp32_rescue_obs`在最终allreduce/commit前输出。在`stats_log_every_n=1`的探针中，
首个成功step先输出`fp32_rescue_obs=0`，再输出`observations=1`；生产rescue builder
默认周期100（`fp32_rescue.py:977-981`），可能没有对应的后一条。

旧skip_rate用累计skip_total除以`observations*本步n_inputs`，不是稳定窗口概率。
复审三步全skip、eligible=8/8/2时日志比例1/1/**3**；不能直接作为窗口发生率。
本阶段不增加机会数计数器。

## 5. checkpoint、回退与缺口

| 边界 | 现有行为 | 后续窗口契约（尚未实现） |
|---|---|---|
| 同checkpoint重启 | 全rank加载rank0历史，辅助计数重置 | 新恢复边界/本地身份，以实际装载快照为baseline，不累加历史绝对值 |
| 恢复旧checkpoint | 计数/obs可下降，load不要求单调 | 标记rollback/不可比较，新段，不用负数或零掩盖 |
| 父guard缺rescue块 | 新实例rescue保持零，base历史可非零 | 标明块的覆盖起点，不推断以前无rescue；复用旧实例load也不保证自动清零 |
| world size改变 | sidecar/state双层拒绝 | 不拼接；显式迁移不在本阶段 |
| mapping/routing变化 | mapping版本/world/routing/ref键集受校验 | 不复用旧owner证明；身份不匹配失败 |
| JSONL断档/漏rank | stats仅rank0且无可靠连续窗口身份 | 明确不完整，不补零；同身份端点差值不能恢复逐窗口记录 |
| 重复/冲突窗口 | 尚无生产window_id协议 | 后续显式检测；不静默重复累计，本阶段不实现合并 |
| 缺计数字段 | loader部分scalar默认零、部分map报错、rescue块可缺省 | loader兼容行为不是覆盖证明；unknown保留unknown |

## 6. 现有telemetry、calibration与消费者

- `telemetry/metrics.py:70,107-177,430-572`：schema **12**，无上述guard字段，JSON/W&B
  为显式键集合。本阶段不改schema；2B按用户要求采用独立per-rank guard流。
- `train/production.py:565`：仅rank0 stats logger。旧日志的owner字段只能标
  `rank0_owner_only`；skip/obs也只是一个rank样本、未经全rank一致性验证。
  覆盖须按字段说明，不能把全条记录伪装成完整多rank数据。
- `optim/guard_calibration.py:217-258`：影子标定对Nesterov RMS做MIN/MAX allreduce，
  spread非零失败，仅rank0写`guard-calibration-rank0.jsonl`；无生产NS/rescue，
  不能把其rows当生产attempt窗口。`production.py:1222-1244`禁止其发布checkpoint。
- `telemetry/metrics.py:575-644`：DurableJsonlSink复用点为mode0600/O_APPEND、严格JSON、
  fsync节奏、进程内锁。它不是多rank事务；短写异常可留半行，fsync失败不能证明耐久。
  后续2B须定义完整/正式边界，2C原子发布另行实现。
- 主metrics的audit脚本/回归测试不构成跨rank guard窗口消费者，不能用主流coverage
  宣称owner事件完整。本阶段未新增消费者。

## 7. 本次检查与未验收事项

Windows现有环境：Python3.12.13、torch2.13.0+cpu、Ruff0.16.10、Pyright1.1.414。
设置`UV_PROJECT_ENVIRONMENT=D:/WORK/Sakuramoon/.venv`、`PYTHONPATH=src`，未安装新依赖。

- `uv run --no-sync ruff check .`：**All checks passed**。
- CI同pathspec的18个Python文件，指定现有解释器Pyright：**0 errors**。
- 三个optim文件（guarded_canonical/fp32_rescue/fp32_rescue_f3）与四个CI门文件：
  **48 passed / 35 skipped**、3条第三方deprecation warnings；30项因CUDA/HCU缺失跳过，
  5项因Windows缺POSIX sink标志跳过。
- schema12指标回归 `tests/unit/test_irepa_metrics_v11.py`：**13 passed**、3条第三方warnings。
- config继承检查：15个TOML、1个声明外部父配置、`ok=true`；逐脚本`bash -n`通过。

以上均在本次2A checkout运行，不挪用上一阶段结果；本次提交的hosted CI另核对。

**未运行**：native Hygon DCU/DTK/RCCL多rank、GPU/NCCL多rank、TorchAO AdamW8bit/硬件SR、
真实训练模型/数据、生产checkpoint恢复、forensic耐久性。CPU/gloo不能冒充这些验收。
文档提交对生产训练开销变化为零，探针耗时不是训练吞吐基准。

## 8. 待用户审核

O字段仅同身份本地delta可在互斥与完整覆盖验证后求和；R字段必须校验全rank一致再保留一份。
恢复历史不能当新的全局累计值，缺口不能补零；attempt/候选接受/commit/obs须区分。

**本阶段提交并推送报告后暂停。未实现2B生产窗口遥测、2C合并器或2D最终验收。**
