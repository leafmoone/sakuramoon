# 最新 dev × Condition Observability 集成兼容性审核

## 首轮审查判定与授权边界（历史快照）

**GO — 源码合并及本轮可执行 CPU 兼容性审查范围。** 未发现新的 P1/P2 生产语义阻塞。此结论不是 GPU/生产部署批准，也不是声称所有历史测试或扩展 Pyright 为零错误。

所有冲突解决和新增测试仅在本地 `review/condition-observability-dev-integration` / `tmp/condition-observability-integration` 保留。执行了 `merge --no-commit --no-ff`，**未创建提交（含 merge commit）、未 push、未 PR、未修改 dev/main/原功能分支、未部署、未 SSH 或操作生产训练进程**。最终 HEAD 仍为 dev；MERGE_HEAD 保持功能分支，用户可直接审查未提交结果。

## 1. 实际身份、三方差异与隔离

| 身份 | 核验结果 |
| --- | --- |
| GitHub 仓库 | leafmoone/sakuramoon |
| 最新远端 dev | `6e57ed203eebf00679a86a197a45519c5c93b0ae` |
| 最新远端 feat/condition-observability | `70ab93f938bb8a565092c114c9dfa793b3be8fcc` |
| 唯一共同祖先 | `4a94dfe564e8ef439eafee71be64dcc5092962c9` |
| dev 独有提交 | 1：`6e57ed2 feat(optim): add config-driven condition clipping and learning rates` |
| 功能分支独有提交 | 26：从 `18ea327` 离线审计到 `70ab93f` 最终生命周期修复 |
| 相对共同祖先变更路径 | dev 20；功能分支 48；重叠 8 |

实测 SHA 与用户给定参考一致，不是直接假定。工作区实际 GitHub remote 名为 `github`（HTTPS URL 已核验）；fetch 显式写入 `refs/remotes/origin/dev` 和 `origin/feat/condition-observability`，避免改变原有 remote 配置。随后从最新 origin/dev 新建 review 分支/worktree，临时合并 origin/feat/condition-observability。首次 merge 因缺少 committer identity 在操作前拒绝，补用**单命令** identity 后执行；没有全局/仓库身份配置修改。

读取项目 AGENTS：保留旧 worktree/未提交修改；不读取 .env/秘密/真实模型数据；不创建模型输出 checkpoints 目录；不修改依赖范围；测试仅使用已有私有 WSL CPU 环境。另建 detached dev-baseline worktree，仅作对照；旧 worktree 均未 reset/checkout/stash/remove。

三方完整 patch、双向分叉历史、初始/最终路径清单在仓库外 `tmp/condition-observability-integration-evidence/`：`dev-from-base.patch`、`feature-from-base.patch`、`dev-vs-feature.patch`、`split-history.txt`、`identity.json`。不是只看冲突列表。

## 2. 所有冲突与逐处解决

重叠 8 文件：docs/config-and-resume.md、config/schema.py、perf/harness.py、telemetry/metrics.py、train/{step,loop,runtime,production}.py。其中前3个及 production 自动合并，后4文件共8处冲突。没有使用 ours/theirs、没有整文件复制覆盖。

| 文件 / 冲突块 | dev 真实语义 | 功能分支真实语义 | 采用的局部解决 |
| --- | --- | --- | --- |
| step.py / UpdateResult | condition_clip/rest_clip 分区 ClipResult | union pre/post 范数 | 四字段并存，定义不混合 |
| step.py / 构造签名 | config global_clip_norm + condition_optimizer | condition_grad_share bool | 三参数并存 |
| step.py / 初始化 | validate clip、DDP canonical selector、双分区 | validate diagnostic bool | 保留全部检查与唯一一次属性赋值 |
| step.py / clip 段 | independent 双组或 config-driven global clip | global 后额外 union post scan | 保留 dev 原分支；两条分支完成后统一 gated post scan，不恢复硬编码1.0 |
| step.py / result 构造 | partition results | diagnostic norm results | 全部传出 |
| loop.py / 构造签名 | clip/config 两控制 | bool/live callback gate | 三控制并存；保留 feature 双参 loss_fn/cadence 接线 |
| runtime.py / loop 构造 | 传 config clip+optimizer policy | 传实时 condition_grad_share | 三 kwargs 均保留 |
| metrics.py / as_json_mapping | 先建 payload，后附加可空 partition/LR 字段 | post-init Optional 类型收窄和直接 return | 保留类型收窄，但仍建 payload，再执行 dev 可选字段追加及 return |

自动合并复核：schema 同时保留 optimizer.condition、condition_diagnostics、guard_window；production 原样保留 dev split/LR 调度，以及 feature runtime gate、双 recorder observer 和嵌套 finally；perf harness 保留 dev clip 控制及 feature 双参 loss callback；配置/resume 文档双契约并存。

独立 reviewer 静态复核确认无重复定义/冲突标记；第二独立 critic 审查8个重叠文件和21个交叉 case，结论 CPU integration GO、无证据支持的新 P1/P2 blocker。

## 3. 指标集合与数学定义

设 `C` 是 optimizer 的配置前缀命中集合；`R` 是其余可训练参数；`U` 是诊断的固定 union（condition_tokens.* + dit.conditioner.*）。C/R 不交且覆盖全体 trainable；**U 不等于 C，也不受 parameter_prefixes 重定义**。默认 C=condition_tokens 时，U 中 conditioner 属 R，必须按 rest 系数裁剪。

- 所有梯度先按样本数归一化，诊断 pre 与 clip pre 都是 sample-mean 量纲。
- global：同一个系数作用于全部梯度；阈值0仅禁用裁剪，不禁归一化/finite 检查。
- independent：C/R 各自有限性检查、范数及系数；任一非有限在两组裁剪前拒绝并清理。summary pre/post 是各分区范数的 hypot；summary coefficient=total_post/total_pre 是摘要比例，**不是共同施加的系数**。any-clipping 标志由分区 coefficient 判断。
- 诊断 `condition_grad_norm_pre_clip/post_clip` 分别实际扫描 U 的前/后梯度，不以分区 coefficient 推算。`condition_grad_share_* = U_norm / total_norm`，**不是平方能量占比**；零 total 对应 null。独立裁剪的 hypot 总范数可作为对应分母，不混淆 pre/post。
- 主指标 `condition_pre_clip_grad_norm` 等字段代表 C；诊断流 `condition_grad_norm_pre_clip` 代表 U，名字相似但定义不同。

### 配置与实际行为兼容矩阵

| 配置 | 实际更新与梯度/系数 | 遥测与交叉证据 |
| --- | --- | --- |
| policy absent / independent_clipping=false | config-driven global clipping | global [0,.5] 实际 SGD 更新、union 范数/record；默认关闭跨进程轨迹对照 |
| global threshold=0 | 非零梯度保持，coefficient=1 | 实际满幅参数更新与 pre=post norm，record share 保持 |
| A only | 双分区各用自身阈值 | 非零 `(3,2,4)` 梯度精确更新、partition norms/coefs/hypot、U跨两组 pre/post |
| A threshold `(0,0)` | 两组均不裁，coef均1 | 非零梯度、满幅更新与两个记录 share |
| A threshold `(0,2)` / `(.5,0)` | 只裁一个分区 | 不对称参数更新与 union post share，不能按单一coef缩放U |
| B only | 保留 global clip；只拆 AdamW child 连续 membership runs | 原 LR suite 全矩阵；真实 CPU AdamW split+step；global/union 与 legacy一致 |
| A+B | 两组裁剪、condition AdamW rate 单独调度 | real guarded optimizer+runtime+loop+双流/中途I/O故障；真实 TorchAO state restore |
| 诊断关闭/构造失败/运行中停用 | 不变更任何以上训练语义 | count gate、参数/loss/state/RNG/scheduler/checkpoint cadence 精确相等 |

手算量纲用例：mean-scaled 梯度 `(condition_tokens, conditioner, other)=(3,2,4)`，total_pre=√29，U_pre=√13。以下为实际测试闭式断言及工具计算的参考值（非训练推荐）：

| C/R阈值 | C/R系数 | total_post | U_post | post share |
| --- | --- | ---: | ---: | ---: |
| .5 / 2 | 1/6 / 2÷√20 | 2.061553 | 1.024695 | .497050 |
| 0 / 0 | 1 / 1 | 5.385165 | 3.605551 | .669534 |
| 0 / 2 | 1 / 2÷√20 | 3.605551 | 3.130495 | .868243 |
| .5 / 0 | 1/6 / 1 | 4.500000 | 2.061553 | .458123 |

### LR 与恢复契约

Hybrid 的 base = scheduled base；CMuon=base×cmuon_multiplier；rest AdamW=base×adamw_multiplier；condition AdamW=base×adamw_multiplier×condition_multiplier。runtime 在 update_started 捕获**本次使用值**，而不是 scheduler 推进后的下一步 LR。重复设置从 base 重算，不复乘当前LR。非 hybrid `learning_rate` 保留原 effective AdamW 意义。

`optim/clip.py`、`optim/condition.py`、`checkpoint/load.py`、`telemetry/observer.py` 与最新 dev 内容一致（仅忽略 Windows EOL），没有重写 clip/LR/restore。主 TrainingMetric schema 保持12；诊断 stream schema及 Guard Window schema保持各自既有版本，离线 audit schema4不等于checkpoint schema。

新5项真实 TorchAO CPU恢复 case：生产 builder/split → quantized/plain moments + CMuon momentum + opaque SR bytes → torch.save → 真实 loader validate/remap → exact state compare；legacy→B、B→B、诊断开关跨恢复、float/tensor LR、新 WD/LR主导、scheduler更新不复乘；FQN扁平顺序/参数identity/BF16访问次序不变。neutral B rollback保留 tags/multiplier1；B→legacy仍拒绝。**没有执行 native SR next-step kernel**，opaque字节保持不等于GPU轨迹验收。

## 4. Guard Window、生命周期与默认关闭

Owner hashing (`fnv1a64-v1`)、路由、momentum、guard源码未改。observations/skip系列是复制统计；NS attempt/rescue系列按 stable_owner owner-local累加。跨rank合并保留rank0恢复历史来源和本窗口delta，不把 restored_rank0 owner历史重标为当前所有rank的local历史；schema1与窗口界限/尾窗/rollback规则未改。真实 gloo ws1/ws2/ws4 测试均运行。

新增 dual-stream 测试使用真实 guarded production builder、precision audit（矩阵BF16/敏感FP32）、routing、split、NS/guard/momentum、runtime、loop；只替换CUDA AdamW构造为真实 CPU AdamW、SR为有状态pass-through、冻结资产准备为合成输入。四更新accum2、diag stride2、A+B，两组每步确实裁剪。分别比较 off vs both-enabled、update2 diagnostic write-I/O、update2 guard write-I/O：每步loss/参数/moments/momentum/RNG/LR主指标逐值相同，scheduler[1,2,3,4]、checkpoint[2,4]、SR/AdamW step=4。两optimizer分支确有参数变化；没有拿no-op优化器绕过。

诊断错误不吞结构/模型/loss/optimizer异常。I/O停用互不传染，另一流仍按周期正常写；所有sink只close一次，fd经os.fstat证明释放；public重复close保持幂等。

### 实际工作量计数（四更新，非估算吞吐）

| 场景 | activation事实 | 保留condition输出 | optional union扫描 | mandatory norm扫描 |
| --- | ---: | ---: | ---: | ---: |
| 默认off | 0 | 0 | 0 | 16 |
| 双流正常 | 2 | 2 | 4 | 16 |
| update2诊断I/O停用 | 1 | 1 | 2 | 16 |
| update2 Guard I/O停用 | 2 | 2 | 4 | 16 |

逐更新断言：非记录步及诊断停用后的后续步骤全部optional=0；最终微批才capture/activation，finally后无output引用。未把完整套件的耗时差换算为训练性能，未测GPU吞吐。

另外两独立进程对 dev原代码与candidate的默认off路径：4更新accum2，实际A+B裁剪、LR warmup、61个参数张量（8 CMuon/53 AdamW）、6 AdamW child groups、56次真实NS kernel；递归比较 **2810 tensors / 554408 elements / 6021 scalar leaves**，raw bytes、dtype、顺序、状态均相同。candidate可选诊断构造、持久化、activation/capture/union、CUDA/collective sentinels全零。外部harness用原样AST提取builder/scheduler避开vendor资产import，替换边界同上，不冒充完整生产启动。

## 5. 必要修改与非必要修改控制

临时merge本身携带功能分支48路径的原有改动（含7个dev-tools executable bit修复）；完整delta见evidence patch/manifest。**本轮新增手工内容**只有：

1. 上述4个生产文件8处局部冲突union；没有新的生产功能、配置开关、通用保护框架。
2. `tests/unit/train/test_condition_clipping.py` 的loop loss callback增加microbatch形参（同一改动覆盖3个参数化case），保留全部数值断言。
3. `src/sakuramoon/tests/unit/train/test_g1_alpha_runtime.py` 同步双参callback，补Path/observation/generic/cast类型标注，不改loss或断言。
4. 新 `tests/unit/train/test_condition_optimizer_observability.py`：13项clip/diagnostic/实际CPU更新/record矩阵。
5. 新 `tests/unit/checkpoint/test_condition_observability_integration.py`：5项真实TorchAO恢复交叉。
6. 新 `tests/unit/train/test_condition_dual_stream_integration.py`：3项真实双流/故障/状态中立交叉，复用既有fixture而非复制完整模型和optimizer实现。
7. 本报告。原工作流不改，额外交叉case单独运行并也进入完整CPU discovery。

新fixture初测5失败：TorchAO subclass不支持deepcopy clone，及large敏感矩阵moment应依size/alignment量化；修正测试为torch.save/load快照及原生量化判据，未更改loader验证、数学容差、断言、skip/xfail或功能。dual fixture初始全FP32被真实precision audit拒绝，改用正确BF16投影精度，不关闭audit。外部parity harness的unusedvendorimport/合成输入dtype等bring-up失败均仅修在外部harness。

保护面核对：Camera/data/model/conditioning/objective、BS、训练配置、checkpoint schema、optimizer routing/step/momentum均无本轮手工修改；不引入新的collective。自动merge保持dev既有config-driven LR/clip，而不是把它还原成feature硬编码baseline。

## 6. 验证结果（同环境baseline parity）

WSL Ubuntu，Python3.12.14，Torch2.13.0+cpu，TorchAO0.16.0+cpu；已有私有venv、不sync改项目依赖；启动前CUDA/HIP/ROCR全mask，CPU线程有界、24GiB地址空间限额、禁core，资产离线。

| 检查 | 结果与准确范围 |
| --- | --- |
| 最新dev完整默认 tests/ 可执行CPU | **1512 passed / 122 skipped / 48 deselected**, 0failed/error，5warnings |
| 集成完整默认 tests/ 可执行CPU | **1905 passed / 122 skipped / 48 deselected**, 0failed/error，9warnings |
| 原CI选择性unit集合 | **372 passed**, 无skip；不能冒充全仓 |
| 新交叉集合 | **21 passed** = 13clip/diagnostic + 5restore + 3dual；也包含在1905中 |
| Guard Window CPU/gloo | ws1/ws2/ws4 三节点均真实passed，包含于上面集合 |
| 全仓Ruff | PASS |
| 精确CI改变Python文件scope | **36 files Pyright: 0errors/0warnings**，包含本轮新tests |
| 双方修改+冲突+新tests扩展scope | **45 files Pyright: 446errors**；详见下面parity，绝不称全绿 |
| 配置图 / shell syntax | PASS；16files/14refs/1external；逐个脚本bash -n |
| Gitleaks | 功能分支分叉提交历史与最终暂存diff扫描：无新增泄漏（最终结果见交付证据） |
| 额外历史src测试树 | dev/candidate均 **7passed/2failed**，如下 |
| GitHub集成CI | **未触发**：不push/noPR/no远端review提交；仅本地CI等价。不能复用两个父提交的绿灯冒充本集成CI |

完整CPU包含名称范围（集合重叠，不相加）：checkpoint115passed（另7skip）、train418passed（另7skip）、iREPA187passed（另10skip）、Camera149passed。输出9warnings中4个来自新tensor-LR拷贝fixture，其余与baseline相同类型的TorchAO弃用/NumPy只读/profiler提示，不是失败。

### 扩展Pyright精确差分

同一解释器，检查双方union中各父版本存在的文件：dev22files668errors，feature38files472errors，candidate45files446errors。candidate全部落在4个**内容未改**的dev文件：clip.py 7、test_fp32_rescue.py145、test_guarded_canonical.py291、test_irepa_metrics_v11.py3。

按 `(relative path, rule, full message, severity)` 多重集比较：相对dev有17条不同的旧test_fp32_rescue kwargs类型诊断、去掉239条；这17条已存在feature。同两条accepted-parent诊断并集相比**0new**；相对feature **0new/26removed**。没有为了零错误扩散修改四文件或加ignore。精确CI scope的36files为零，是独立、较窄的结果。

### 历史失败与CPU不可执行项

初次候选dev clipping tests 3失败与额外src alpha测试1失败均是单参loss callback撞已验收双参API，已最小同步；最终不残留新增失败。

额外 `src/sakuramoon/tests/unit/` 不在pytest默认testpaths中，但单独显式运行，dev/candidate同两失败：

- model/test_g1_function_equivalence.py::test_alpha_zero_is_exactly_function_preserving_for_old_slots：旧fixture stable_slot_count非exclusive bound。
- train/test_g1_forced_checkpoints.py::test_forced_checkpoint_reasons_are_durable_and_ordered：旧调用缺terminal_update。

未删除/排除该历史树来制造全绿；不是宣称“全仓全部测试通过”。

完整CPU runner沿用双方相同的安全不可执行选择，未修改测试自身skip/xfail：48 deselected = checkpoint请求CUDA projector 1 + GPU iREPA training graph五函数的11case + PE teacher geometry三函数36case。whole-file ignore `tests/gpu/irepa/test_pe_spatial_reference.py` 依赖真实teacher资产（含一个需资产的CPU key-set检查，其余GPU），不下载权重；其他GPU/资产测试正常122skip可见。所有排除均不计通过。

## 7. 剩余风险与最终状态

- **未运行DCU/GPU、RCCL/NCCL、compiled DDP、真实生产模型checkpoint恢复或生产训练。** CPU/gloo、真实CPU量化state恢复、opaque SR字节保持，均不能证明下一次GPU8bit/SR数值轨迹。
- CPUAdamW/pass-through SR用于step/neutrality测试，真实TorchAO quantized loader用于restore测试，不能将两种证据拼成“GPU下一步恢复通过”。
- 扩展Pyright446既有错误和额外历史2失败尚存，若要求无条件全树静态/历史全绿需单独清理授权；本轮不做无关精简。
- 编译开销/吞吐、真实encoder/data资产、同时启用iREPA+诊断+DDP的部署级行为未做硬件验收；Camera和iREPA可执行CPU回归通过不是这些验收的替代品。
- 本结论绑定上述两个实际SHA及当前本地解决结果；远端后续移动须重新审查，不能将报告用于覆盖未来dev。

**首轮终态：GO（源码与可执行CPU集成），硬件/部署未验收。** 当轮只保留未提交的隔离merge、测试及报告，不自动推送/PR/合并/部署/启动训练；完成后暂停等待用户审核。

## 8. 后续授权：review 候选固化与自身远端 CI

用户随后单独授权将本候选固化为独立 review 分支的双亲提交、普通推送该 review 分支，并等待该提交自己的 GitHub Actions。前文的“未提交/未 push/CI 未触发”是**首轮审查时点**，不是本阶段执行后的声明。正式 dev、main、原功能分支以及生产进程仍不在修改权限内。

固化前重新读取 AGENTS，核验 HEAD=`6e57ed203eebf00679a86a197a45519c5c93b0ae`、MERGE_HEAD=`70ab93f938bb8a565092c114c9dfa793b3be8fcc`；无 unmerged、unstaged 或 untracked 内容。完整暂存 patch 与首轮保留的验证 patch 逐字节相同，先备份为仓库外 `solidification/PRE_CHANGE.patch`，没有 abort/reset/stash。Fetch 后两条远端 HEAD 未移动，实际可推送 remote 仍为 `github`。

本阶段再次独立审查全部8个重叠文件的双方源码差异，而非仅复用 GO 报告，未发现 P1/P2 集成阻塞。只新增3个既有交叉测试文件的 CI 入口及 CPU 边界注释，完整保留原13个文件/372项用例；目标选择性 unit 总数为 **393 = 372 + 21**。未改任何生产源码、测试实现、断言、容差或 skip/xfail。

固化门：全仓 Ruff、精确 CI Pyright、配置图、逐文件 shell、393项 CI 选择性单测、完整可执行 CPU 套件以及最终 staged 敏感信息检查；扩展 Pyright 仅核对两父精确差分，不扩散处理旧问题。提交必须保持 dev 为第一父、功能分支为第二父。远端验收必须绑定 review 提交自己的 headSha，static/unit/contracts 全成功且 unit 日志证实三文件全部参与并通过。

固化前最终本地重跑：**393 passed / 0 skipped**（新增三文件分别13/5/3项，原372全部保留），完整CPU **1905 passed / 122 skipped / 48 deselected**，仍使用上文同一硬件/资产排除表；gloo ws1/ws2/ws4实际通过。全仓Ruff、36文件CI Pyright（0errors/0warnings）、配置图及shell通过。扩展45文件仍446errors，精确差分与§6一致（两父并集0new）。JUnit及完整消息差分保存在solidification目录。生产源码与上一轮已验证候选完全相同。

实际冲突解决可由 `git show --remerge-diff --ignore-space-at-eol review/condition-observability-dev-integration -- src/sakuramoon/train/step.py src/sakuramoon/train/loop.py src/sakuramoon/train/runtime.py src/sakuramoon/telemetry/metrics.py` 查看：以双方重新合并的冲突文本对照提交中的局部解法，而非用首父整分支 diff 冒充冲突解决。固化后同时在仓库外保留 `solidification/merge-remerge.patch`、完整双父 diff、提交身份及 CI 日志。

远端 run ID、提交 SHA、各 job 与日志证据以本阶段最终交付及仓库外 `condition-observability-integration-evidence/solidification/` 记录为准；在运行前不预写成功，也不拿父提交绿灯替代。本阶段仍不宣称 DCU/GPU、RCCL/NCCL、compiled DDP、真实生产恢复或 native SR 下一步轨迹通过。完成后再次暂停，不自动 merge 或部署。
