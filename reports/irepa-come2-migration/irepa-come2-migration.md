# iREPA come2 迁移报告（salt13 释放后的环境重建 + 重启就绪点重建）

- 日期：2026-09-06
- 执行机：come2（gateway `ssh -p 10608 root@ssh.zzai.scnet.cn`，内网 fabric 172.31.43.102，主机名 `crdnotebook-2096479592734380034-come2-94569`，Ubuntu 22.04.5，DTK 镜像预装，2×DCU 空闲，503G RAM，根盘 1.8T 可用 1.1T）
- 代码状态：`/sakuramoon-runtime/sakuramoon-irepa` @ **`393259e4a85528cefd67714eb5e60986ce43663b`**（分支 `fix/irepa-token-spatial-layout`，已 push origin github.com/leafmoone/sakuramoon；= 177c7a7 修复提交 + 393259e ruff 0.16.1 lint 修复）
- 结论：**come2 = iREPA 就绪机**；干净重启源 `migrated/ckpt_113400_irepa` 重建完成，**17 项审计 17/17 PASS**，与 salt13 原迁移产物语义等价（irepa_state 字节级一致、projector 位级一致）。

## 1. salt13 损失清单与恢复来源

| 丢失项（随 salt13 释放） | 恢复来源 | 状态 |
| --- | --- | --- |
| iREPA 修复提交 c3e00ed | 本地 `D:\sakruamoon\sakuramoon` 重建 | ✅ push 为 177c7a7（irepa.py sha256=f8d0856f… 字节级验证，见 reports/irepa-token-spatial-layout-fix.md） |
| DTK venv（3.2G） | salt15（172.31.33.90）fabric 字节级拉取 | ✅ 含 flash_attn DAS 匹配构建、torch 2.9.0/2DCU、ms-hub、pytest |
| model 资产 6.1G（clip/qwen/vae） | salt15 fabric 拉取 | ✅ |
| PE-Spatial teacher 资产（345,783,707B） | HF facebook/PE-Spatial-B16-512 下载 | ✅ SHA `86217607f0bb28c0adb5ac3f9b0608ae22f6fb634bf1c16b2316847e8148a2a5` PASS；asset.json 按校验器 9 字段手写；22 文件硬链接入 worktree（校验器拒符号链接） |
| `ckpt_113400_raw-113400-update-cadence` 原始 ckpt | **ModelScope hub** `leafmoone/sm_train_state`（g1/ 290 个 ckpt 归档，100000 起，含 113400 与 113600） | ✅ 14/14 文件字节级落盘（大小全对齐 hub 清单；COMPLETE="complete"；manifest 可解析） |
| `migrated/ckpt_113400_irepa` | come2 上重跑 `sakuramoon.checkpoint.migrate_irepa_checkpoint`（脚本在 repo；projector 初始化由 migration_seed 位级决定） | ✅ 见 §4 |
| p6b 训练/评估日志、launcher、eval 原始数据 | W&B 云端遥测（eff1000 全窗口）；hub eff1000 终端 114600（15/15 字节级） | ⚠️ 原始日志不可恢复；评估终报在本地 `D:\sakruamoon\eval-deliverables\verification-dossier-eff1000.md`（28 boards 在本地） |
| p6b 训练配置 TOML（irepa 段） | IRepaConfig schema 默认值 + eff1000 遥测形态重建（weight=0.5、ramp_in_updates=1000、无 ramp-out ⇒ λ 在首更新 113401 精确 0、1000u 线性到 0.5、114401-114600 保持） | ⚠️ 见 §5 重建说明 |
| eff1000 中间 ckpt 113700-114500、canary raw 113500/113600 | — | ❌ 不可恢复（hub 仅有 100 步粒度；113600 在 hub 但属 iREPA 污染线，禁用） |

## 2. 环境重建（byte-for-byte 纪律）

- venv：`/sakuramoon-runtime/sakuramoon-dtk-venv` = salt15 同源字节拷贝（3.2G）。venv 内脚本 shebang 混有 salt15 的 `/root/private_data/sakuramoon-dtk-venv`（NFS 源路径）→ come2 建两个兼容符号链接修复：`/root/private_data/sakuramoon-dtk-venv` 与 `/opt/sakuramoon-venv` → 运行时路径（0 字节，不占 NFS 配额）。
- 依赖补齐（venv 相对 salt13 的分叉）：`timm 1.0.29`（pyproject `timm~=1.0.16`，经 ai_proxy 10.16.1.51:3128 安装；不触碰 torch/flash_attn/triton 构建）；`onnxruntime` **不装**（HCU venv 惯例无此包，full pytest 惯例排除 2 个依赖文件：`tests/unit/data/test_deepghs_quality_pipeline.py`、`tests/unit/data/test_run_deepghs_quality_pipeline.py`）。
- 代码：`/sakuramoon-runtime/sakuramoon-irepa`（bundle 全历史 + fix 分支 refs，5,108,302B；checkout 177c7a7 → 本机 ruff 修复提交 393259e → bundle 反向 relay 回本地 → push origin）。
- 模型资产：`/sakuramoon-runtime/model`（6.1G）+ `model/pe_spatial_b16_512/`；`require_local_pe_spatial_teacher` 校验 **PASS**（22 文件硬链接）。
- DTK 铁律：torch 前必须 `source /opt/dtk/env.sh`（缺则 `ImportError: libgalaxyhip.so.5`）；本会话所有 torch 调用均遵守。

## 3. 测试证据（come2，venv python + `PYTHONPATH=src` + 已 source DTK env）

| 批次 | 结果 |
| --- | --- |
| 布局 oracle + 梯度（D=2560 生产契约、梯度落位、梯度零、投影等幂、布局门控、λ=0 位级、确定性、梯度范数）8 项 | 8/8 PASS |
| 对齐测试（含修正后 index-reference）10 项 | 10/10 PASS |
| `tests/unit/model` + `tests/unit/assets`（含 pe_spatial 资产契约） | **91/91 PASS**（102s） |
| ruff 0.16.1（4 个改动文件） | All checks passed（393259e 修复前 diag 脚本 5 项风格问题：4×RUF100 无用 noqa:E402 + 1×I001 import 排序，零运行时影响） |
| pyright（全仓基线对照 salt13=4939） | **4939 errors（delta 0）** |
| full pytest（tests/，排除 2 个 onnxruntime 文件） | **1020 passed / 5 failed / 2 skipped（996s）**：5 个失败全部为已知基线——`test_varlen_attention.py::…[host_metadata]`（HCU host_metadata 基线 bug）×1 + `test_cmuon_fp32_forensic.py`×4（F2 取证硬失败语义的已知差异，与 salt11 F3 基线 756P/4F 一致）；**0 新失败** |

## 4. 干净重启点重建（113400 → migrated/ckpt_113400_irepa）

- 源：hub 下载的 `g1/ckpt_113400_raw-113400-update-cadence`（14 文件，6.0G；trunk 2,088,736,024 + 1,104,529,088、optimizer.pt 3,143,794,315，全部字节级对齐清单）→ 移至 `/sakuramoon-runtime/p6b-source/ckpt_113400_raw-113400-update-cadence`（审计脚本 SOURCE 常量约定路径）。
- 迁移：`python -m sakuramoon.checkpoint.migrate_irepa_checkpoint --source-checkpoint … --destination /sakuramoon-runtime/p6b/migrated/ckpt_113400_irepa --config /sakuramoon-runtime/p6b/config/p6b_irepa_restart.toml --migration-seed 20260904`
  - dry-run 计划：source update 113400 / anchor **113401** / in_ch **2560** / seed **20260904** / +2 projector FQN（`irepa_alignment.projector.{weight,bias}`）/ 新增 AdamW 参数 2 —— 与 salt13 原迁移完全一致。
  - `irepa_state.json` = `{"migration_seed":20260904,"schema_version":1,"source_checkpoint_id":"raw-113400-update-cadence","source_update":113400,"start_successful_update":113401}` —— **与 salt13 原迁移字节级一致**。
- **17 项重启就绪审计（`scripts/diag_restart_readiness.py`）：17/17 PASS，0 FAIL**（`reports/irepa-come2-migration/restart-readiness-audit-come2.txt`）：trunk/rng/trainer_state 逐位一致、141 CMuon FQN 集合不变、AdamW=源+恰好 2 projector FQN、148 条源 AdamW state 重编号后位级一致、projector 无 AdamW state（pristine）、CMuon/sr_rng/transition 块位级一致、anchor 113401、projector 分片 = seed 20260904 确定性初始化（位级）、numel 17,695,488=2560*768*3*3+768。
- 重启设计约束不变：固定 iREPA 重训 = 本分支新代码 hash + 新 run_id/W&B 身份 + 新 output/ckpt 路径 + 独立时间线；从 113600/114600 续训禁止。

## 5. [irepa] 配置重建说明（重训前必读）

原 p6b 配置 TOML 随 salt13 丢失。迁移配置 `/sakuramoon-runtime/p6b/config/p6b_irepa_restart.toml` 由 repo 的 `config/train_s0.toml`（经 repo 自身 loader 解析 + dump 保证 schema 有效）生成，注入：① G1 真实 optimizer 衰减（matrix/sensitive weight decay = 0.0/0.0，取自 113400 的 resolved_config.toml）；② 最小 `[irepa]` 表（enabled/teacher_id/tap_slot=8/kernel=3/zscore/cosine，其余取 schema 默认：weight=0.5、ramp_in_updates=1000、无 ramp-out、gamma=0.6、eps=1e-6）。默认值与 eff1000 遥测形态自洽（λ 在 113401 精确 0、1000u 线性到 0.5、末 200u 保持）。迁移产物只依赖 optimizer 衰减 + irepa.enabled（in_channels 取自 checkpoint DiT hidden_size=2560），故迁移正确性不受配置重建影响。**重训 GO 前**：以 eff1000 W&B 遥测的 λ 曲线逐点核对重建配置的 ramp 参数（如需），并确认 data/eval 路径指向（validation cohort ~14GiB 不可再下载，须从 G1 宿主机 cohort 转移；G1 于 09-06 14:45 自 salt14 迁移至 come1 接管，cohort 源=come1）。

## 6. 遗留事项

1. ms-hub 0.4.0（come2 与 salt14 双侧复现）`download` 对 `leafmoone/sm_train_state` 全部文件 404，而 list API 正常；原始 curl + `Authorization: Bearer` 下载端点（`…/repo?Revision=master&FilePath=…`，大文件 302→CDN 需 `-L`）验证可用，本报告全部下载走此路径。**G1 发布器脚本的 verify 步骤（ms-hub download）可能同样受影响**——需向 G1 生产侧确认发布 verify 是否静默失败（G1 现宿主=come1，只读排查，不动生产）。
2. 重训数据：训练分片可再下载；validation cohort 须机间转移（源：come1）。
3. come2 辅助凭据文件：`/root/.salt14pw`、`/root/.salt15pw`、`/root/.hubenv14`、`/root/.askpass*`（均 0600）——按用户意愿保留或清理。
