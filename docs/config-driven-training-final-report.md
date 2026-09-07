# SakuraMoon 配置驱动训练清理 — 最终报告

## 交付坐标

- **BASE**: `8593a684b2fbb9fe65971c6c69a12ccedaae8950`（dev，repo leafmoone/sakuramoon）
- **FINAL_HEAD**: `4ef62408`（本地，见下方提交列表）
- **BRANCH**: `refactor/config-driven-training`
- **WORKTREE**: come6 `/sakuramoon-runtime/sakuramoon-config-driven-training`
  （本地镜像 `D:\sakruamoon\config-cleanup`，bundle 同步，零 push）

### 提交列表（8593a68..HEAD，逻辑分组）

| commit | 内容 |
| --- | --- |
| `a336fca` | config: 执行接口改为 config 驱动（train/model.dit/distributed/optimizer；legacy stage 一次性翻译；strict schema；TOML 数值契约） |
| `dcff367` | model: 槽拓扑与增长 config/参数驱动（slots.py 派生、stable_slot_count 排他上界、new_slot_ids 差集） |
| `6d7bea9` | train: 删除历史 stage 图与 stage 门（stage.py 移除、terminal 实时读 max_updates、forced checkpoint live terminal） |
| `ad2db1a` | checkpoint: 结构性 resume 绑定（FQN 严格增/删集合、growth cutover 绑定、rank RNG 全 rank 重绑、旧 v3/v4/侧车兼容） |
| `23c6fb8` | sampling/eval/data/deploy: 其余锁绑定到 config 与启动器（profiles 用户自定、桶几何派生、部署硬编码移除、概念评估设备检查） |
| `6042c14` | chore: ruff 0.16.1 合规 |
| `3b2ed4f` | tests: 单元测试迁移到 config 驱动接口（flow 公式恢复 BASE 逐字 + 漂移测试套件重写） |
| `76bf0e0` | chore: shebang CLI 脚本可执行位（ruff EXE001） |
| `0984c38` | docs: 配置驱动接口与恢复契约（docs/config-and-resume.md） |
| `be1c1a5` | fix: 清除残留 stage/shape_count 日志读取 + pyright 漂移清理 |
| `4ef6240` | fix: timing 禁用时 per-update 循环计时器保持无事件（NoopPhaseTimer 保持）+ 最终报告 |

## REMOVED_POLICIES（已删除的历史策略/伪配置/硬编码）

1. **历史 stage 图**（`train/stage.py` 的 stage 表/门/预算图）— 删除；
   终点由 `train.max_updates` 实时读取（R14）。
2. **`[stage]` 运行时词汇** — 仅在配置读取边界翻译一次
   （`normalize_legacy_config`：resolution/local_batch/accumulation/max_updates→`[train]`，
   world_size→`[distributed]`，depth→`[model.dit]`，name→`run.label` 仅展示；
   冲突 fail-closed，废弃键一次性通知）。
3. **`model.dit` 固定槽拓扑假设** — 拓扑唯一来源 = `active_slot_ids`
   （缺省按 depth 派生历史非连续拓扑：16→16 槽、20→16∪{2,8,14,20}、24→0..23）；
   `stable_slot_count` 固定 24 锁删除，改为 `max(slots)+1` 排他上界校验。
4. **离线增长迁移脚本** `checkpoint/migrate_growth.py` — 删除（其
   `growth_migration.json` 侧车仍被 checkpoint 读取器接受）；增长切换改为
   **在线 config 驱动 resume**（`growth.enabled` + `new_slot_ids(target, source)` 差集 +
   恢复点锚定半余弦 ramp）。
5. **采样 profile 代码注册表** — 删除；`sampling.profiles` 为用户自定命名预设表，
   `sampling.profile` 按名选择；生成元数据要求显式 `noise_scale`/`t_eps`（来自 `[objective]`）。
6. **flow 目标锁常数** — `t_eps`/`p_std`/`noise_scale`/`noise_observation_boundary`/
   `guidance_scale` 从代码锁改为 `[objective]` 配置 + 范围校验；**公式逐字保留 BASE**
   （per-sample mean velocity loss、high-noise = t < boundary、clamp 400）。
7. **全局条件维度锁**（256/64/64/1024/6 固定 pin）— 改为正偶数/正数范围校验 +
   `final_modulation_size == 2*model_dim` 与槽数唯一性保留。
8. **桶 `shape_count` 锁** — 删除；形状集合由桶几何派生（空几何 fail-closed）。
9. **部署硬编码** — `training_stack.sh` 的 leaf10 主机名 pin（`REQUIRED_HOST_SUBSTRING`
   / `require_leaf_host`）删除；checkpoint 存储 probe 不再条件化；概念评估设备检查
   改为任意可见 CUDA 设备。
10. **LR 全局批缩放伪配置** — `[optimizer].lr_scaling`（`linear_global_batch` |
    `none`）显式驱动；`effective_global_batch()` 与 `scaled_learning_rate()` 为
    配置读取期派生量，`[train]` 内做全局批断言。
11. **iREPA 锁常数** — teacher_id/tap_slot/kernel/gamma/eps 从 Literal/固定点锁改为
    范围/奇偶/路径规范化校验；teacher 家族边界移到 asset 层（pe_spatial），
    tap_slot 活跃度在装配绑定期检查（R29）；路径可为绝对（R41 同类）。
12. **`os.environ` 部署残留与 stage 元数据门控** — resume 绑定中
    stage/world/resolution 降为信息性元数据（结构性绑定见下）。

## CONFIG_EXECUTION（配置执行契约）

- 顶层执行表：`[train] resolution/local_batch/accumulation/max_updates/activation_checkpoint_mode`
  （全局批为派生断言）、`[model.dit] depth+active_slot_ids`、
  `[distributed] backend/world_size`、`[optimizer] lr_scaling`。
- schema：pydantic v2 StrictModel（extra=forbid、strict、frozen）；TOML 数值契约：
  int≡float 合法书写，bool/str 拒绝；所有冲突在读取期 fail-closed。
- 路径语义：配置输入为可信路径（顶层绝对/相对均可、extends 相对所在文件、
  符号链接允许、环拒绝、钻石允许）。
- 21 个模板配置全部经 `load_config` 验证（base.toml 的 BENCHMARK 哨兵
  fail-closed 属预期行为）。

## RESUME（恢复绑定契约）

- **信息性**（不门控）：growth.stage/world_size/resolution、run label、分辨率。
- **结构性**（违规 fail-closed）：
  1. 槽拓扑：相等=普通 resume；严格子集=增长切换（要求 `growth.enabled=true`、
     源 ramp 完成、模块 new_slot_ids 与差集一致，新 ramp 自恢复点锚定）；
     相等+in-flight ramp 要求 `growth.enabled=true`。
  2. 身份完整性；设备 ordinal 每次重绑本 rank 设备（全 rank，含 rank-0 特例移除）。
  3. cadence 重绑当前 `full_every_updates`；确定性 ramp alpha。
  4. FQN 集合：新增 FQN 必须全部落在新槽前缀（`_verify_fqn_sets`），
     **不允许任何移除**；优化器 group 精确集合+顺序比对，diff ⊆ 新槽 FQN
     （`_verify_group_diff`，三处调用点统一 `new_fqns` 参数）。
  5. 预算：`train.max_updates` 实时读取，只允许延长 checkpoint terminal，
     收缩 ValueError（fail-closed）；零更新完成当 max_updates ≤ initial_update。
- 旧 v3/v4 checkpoint（无 new_slot_ids）、growth 侧车、`growth_migration.json`
  均可加载（可选键读者 + `stable_slot_count` 派生）。

## TESTS（come6 权威门，FINAL_HEAD）

- **pytest tests/unit**（come6，FINAL_HEAD）: **4 failed / 971 passed**
  （失败集 = BASE 基线的同 4 个 `test_cmuon_fp32_forensic.py` 预存环境失败，
  基线为 4 failed / 975 passed，无新增失败；测试数差 4 = 重写的
  symlink/绝对路径旧用例替换）。
- **ruff**: `All checks passed!`（src/tests/scripts，ruff 0.16.1）。
- **pyright**（config/checkpoint/model/train/objective 五核心目录）：
  **313 errors vs BASE 324**（去重后 195 vs 213）；逐条 diff 确认无新增
  错误类别——差异全部为：删除 migrate_growth.py（-22）、既有
  dict[str, object]/优化器构建器噪音的别名改名（ExactFloat→Float、
  depth→new_slot_ids）与新增的同类 dict 操作噪音（+6）。
  BASE 本身即非 pyright-clean（预存 324 条）。
- Windows 本机残留失败均为平台差异（`os.O_NOFOLLOW`/`os.O_DIRECTORY`/fcntl/
  symlink 缺失），已在 BASE 提交同环境复现确认非回归。

## 需求状态（对照附件规格 R01–R48）

| 主题 | 规格条目（按主题归并） | 状态 | 证据 |
| --- | --- | --- | --- |
| 执行接口 | [train]/[model.dit]/[distributed]/[optimizer] 目标接口；全局批仅断言；lr_scaling | DONE | `config/schema.py`、21 模板加载、`test_load.py` |
| 拓扑与增长 | active_slot_ids 唯一来源；stable_slot_count 派生；在线增长切换；离线迁移删除；侧车兼容 | DONE | `model/slots.py`、`model/growth.py`、`train/production.py` cutover、`test_growth.py` |
| stage 清除 | stage 图删除；legacy 一次性翻译；冲突报错；废弃键一次性通知；run label 仅展示 | DONE | `config/load.py normalize_legacy_config`、`test_transparent_white_spatial_combo_configs.py` |
| terminal | max_updates 实时读取；extend-only；shrink fail-closed；零更新完成 | DONE | `test_planned_updates_terminal.py` |
| resume 绑定 | 结构/信息性分层；FQN 增/删集合；无移除；rank 重绑；cadence 重绑；旧格式兼容 | DONE | `checkpoint/load.py _verify_fqn_sets/_verify_group_diff`、`checkpoint/test_load.py` |
| 优化器状态 | group diff ⊆ 新槽 FQN；三优化器家族统一 new_fqns 通道 | DONE | `_validate_*_optimizer_schema(new_fqns=…)` |
| 采样 | profiles 用户自定；显式 noise_scale/t_eps；locked 队列 image_count=60 不变量 | DONE | `sampling/profiles.py`、`test_heun.py`、`schema TrainingSamplingConfig` |
| flow 目标 | 公式逐字 BASE；锁常数→配置范围 | DONE | `objective/flow.py`（BASE diff 仅校验函数）、`test_flow.py` |
| 条件/桶/存储 | 维度 pin→范围校验；shape_count 删除；probe 无条件 | DONE | `conditioning/global_condition.py`、`data/buckets.py`、`storage.py` |
| iREPA | tap_slot 装配绑定（R29）；teacher 家族 asset 层；路径规范化可绝对（R41 类） | DONE | `config/schema.py IRepaConfig`、`config/assembly.py`、`test_irepa_config.py` |
| 评估路径 | output/concept 路径可绝对（R41）；prompt/validation 仓库相对 | DONE | `config/schema.py EvaluationEnabledConfig` |
| 部署 | 主机名 pin 删除；启动器只读 config | DONE | `scripts/training_stack.sh` diff |
| 兼容性 | 旧 v3/v4 checkpoint、growth 侧车、growth_migration.json 可加载 | DONE | `checkpoint/test_load.py` 兼容用例 |
| 交付物 | 7 示例配置；docs/config-and-resume.md；逻辑提交；本地 STOP 不 push | DONE | `config/*.toml`（7 迁移模板 + 14 历史模板共 21 全部可加载）、docs、10 提交 |

## FILES（64 文件：57 改 / 2 增 / 5 删）

核心源：`config/{schema,load,assembly}.py`、`model/{slots,growth,dit}.py`、
`checkpoint/{load,schema,artifact}.py`、`train/{production,runtime,step,preflight,sampling}.py`、
`objective/flow.py`、`conditioning/global_condition.py`、`data/{buckets,production}.py`、
`eval/{concept_suite,reconstruction,runtime}.py`、`sampling/{profiles,sampler,__init__}.py`、
`storage.py`、`telemetry/timers.py`、`cli/concept_eval.py`、`train/__init__.py`、
`scripts/training_stack.sh`、`config/*.toml`（7 迁移）。
新增：`docs/config-and-resume.md`、`src/sakuramoon/model/slots.py`。
删除：`checkpoint/migrate_growth.py`、`src/sakuramoon/tests/unit/checkpoint/test_g1_growth_migration.py`、
`src/sakuramoon/tests/unit/checkpoint/test_g1_optimizer_migration.py`、
`tests/unit/checkpoint/test_g1_growth_migration.py`、`tests/unit/checkpoint/test_g1_optimizer_migration.py`。
测试：22 个测试文件迁移（见 3b2ed4f diff）。

## LIMITS（限制与未尽项）

1. **2-DCU 小规模集成验证未执行**（规格允许仅测试临时模型；本轮保持
   零训练边界，未起任何真实训练进程）— 留给 reviewer 决策后的 GO。
2. 本机（Windows）无法验证 fcntl/symlink/O_NOFOLLOW/O_DIRECTORY 相关路径，
   以 come6（Linux/DTK）为权威。
3. `test_cmuon_fp32_forensic.py` ×4 为 come6 预存环境失败（BASE 基线同集），
   非本次改动引入，未修复（CMuon 数值安全边界禁止触碰）。

## BOUNDARIES（遵守的边界）

- 无生产/实验进程改动；无源 checkpoint/真实数据/模型资产修改；
- 无 W&B/hub 上传；无真实数据长训/canary；无 push/merge 到远端 dev；
- 无 Camera v2/新 AVIF-CG；无 joint-attention/核心算法改动；
- 无 CMuon 数值安全改动；无生产 recipe 默认值改动；
- flow 公式与 BASE 逐字一致（仅锁常数→配置范围校验）；
- 全部改动为本地逻辑提交 + 本地/内网 worktree 同步。

## VERDICT

**PASS（本地交付完成，待人工评审）**：come6 权威门 @ `4ef62408`：
pytest 4 failed / 971 passed（失败集 = BASE 基线同 4 个预存 cmuon 环境
失败）、ruff All checks passed、pyright 313（BASE 324，无新错误类别）；
21 模板全部可加载；恢复兼容性（v3/v4/侧车/growth_migration.json）与
增长切换契约由测试锁定；文档与示例配置齐备。

## NEXT

**STOP** — 本地交付完毕，等待用户评审。不 push、不 merge、不启动任何训练；
2-DCU 小规模集成验证待用户 GO 后执行。
