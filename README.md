# SakuraMoon

动漫风文生图训练项目（DTK / Hygon DCU 生产）。包含：PackedDiT 模型、WebDataset
数据服务、生产训练循环、checkpoint 断点恢复、周期采样、FID/IS/KID/CMMD 与概念
条件评估、VAE 重建评估、checkpoint 发布（ModelScope），以及主机迁移运行手册。

- 配置与恢复的权威契约：`docs/config-and-resume.md`
- 相机取景策略：`docs/camera-viewport.md`

## 目录结构

```text
config/                     TOML 运行配置（extends 继承链，base.toml 为公共默认）
src/sakuramoon/
  model/                    PackedDiT 主干（block/attention/norm/mlp/输出头/深度增长）
  encoders/                 冻结编码器：Qwen3.5 2B 文本、Mage-VAE
  conditioning/             全局条件、条件 token、文本混合、打包、RoPE、相机取景
  data/                     WebDataset 管线、分片缓存、数据服务、caption 计划、验证集
  objective/                流匹配目标 + iREPA 辅助目标
  optim/                    AdamW8bit / CMuon 混合优化器、参数分组、梯度裁剪
  sampling/                 euler / heun_final_euler 求解器与命名 profile
  train/                    生产训练循环（fail-closed 生命周期、采样器、前探针）
  eval/                     FID/IS/KID/CMMD、概念套件、VAE 重建
  telemetry/                metrics.jsonl + wandb 队列 sink
  cli/                      各入口（train / data_service / generation_eval / ...）
scripts/                    training_stack.sh、发布器、验证集提示词构建器、审计脚本
dev-tools/                  CMuon 取证工具集
tests/unit/                 单测（Linux + DTK 下为权威门槛）
tests/gpu/                  真卡测试（单步、优化器、性能）
docs/ reports/              契约文档与审计/设计报告
```

## 模型架构

### 概览

| 组件 | 配置 |
|---|---|
| 主干 | `PackedDiT`（生产路径，1.57B 可训练参数，depth 20，bf16 下 3.19 GB）。名称为项目内部类名，技术本身是标准 varlen 无 padding 打包（flash-attn varlen / THD 做法，`PackedSequences` = flat tokens + cu_seqlens 边界），对照 `DenseDiT` 密集参考实现 |
| 文本编码器 | Qwen3.5 2B（24 层，混合线性/全注意力），bf16 冻结，逐 token 7 层捕获 |
| VAE | Microsoft Mage-VAE（DiCo 单步卷积架构），128 latent 通道、16× 下采样，bf16 冻结 |
| 目标 | 流匹配 JLT 采样 + x-prediction（模型预测干净 latent，损失在速度空间） |
| 辅助 | iREPA（512 run）：PE-Spatial-B16-512 冻结教师 + 3×3 投影器对齐 |
| 优化器 | CMuon 混合（256 与 512 生产 run 同为 `hybrid_cmuon_canonical_ns4_fp32_rescue`）：2D 投影走 Muon，其余参数走内层 AdamW8bit（8-bit 动量） |
| CFG | 2.9，在速度空间、各分支先 x→v 再做 CFG |

### DiT 主干（`model/dit.py`、`model/block.py`）

- hidden 2560，depth 20（按 4 块增量增长，稳定 slot FQN `slot_00…23` 支持跨深度
  续训），GQA 20 查询头 / 5 KV 头（原生，无 repeat），head_dim 128，SwiGLU MLP。
- **无 patch embedding**：每个 VAE latent 单元即一个 token（`[B,128,H,W] → [B,H·W,128]`
  → Linear 128→2560）。512² 图 = 32×32 = 1024 图像 token；256² = 256 个。
- 联合序列 = `[主 caption 变长文本 | 8 条件 token | H·W 图像 token]`，全双向注意力
  （仅 mask padding）。生产路径把 batch 内所有样本的联合序列拼成一条扁平变长
  张量（标准 varlen packing，`conditioning/packing.py` 的 `pack_sequences` →
  `PackedSequences`：flat tokens + cu_seqlens + 逐样本 span），零 padding、无逐样本
  mask，DAS FA2 varlen kernel 按边界切段——项目里真正不寻常的是打包内容本身：
  只有图像 token 带 2D RoPE 坐标（相机取景对其做全画布仿射），iREPA 的 slot-8
  捕获也走同一条 flat 路径。
- 2D RoPE：head_dim 拆 nope 32 + y 48 + x 48，theta 1000，cell 中心、面积归一的
  全画布坐标；文本/条件 token 坐标固定 (0,0)。
- 注意力输出内容门控：`attended * sigmoid(content_gate(tokens))`。
- 逐层条件 = 共享零初始化 AdaLN（6 通道：attn/MLP 各自的 scale/shift/gate），
  由一次前向的 `GlobalConditioner` 发射（timestep + size_scale + aspect 的 fp32 MLP，
  8 条件 token 均值池化后走零初始化残差投影）。
- 输出头：RMSNorm → 零初始化 fp32 Linear(2560→128)，预测干净 latent x。
- 生产 kernel：DAS FlashAttention-2 varlen（原生 20Q/5KV）。

### 文本与条件（`encoders/qwen.py`、`conditioning/`）

- Qwen 前向钩子捕获第 (2,4,8,12,16,20,24) 层隐状态 → `[B,L,7,2048]`；caption 序列化
  = 34 token 固定前缀 + 主 caption + 5 token 后缀 + 条件段（≤512 token，8 桶，
  dense 长度 98–546）。
- `TextConditioner`：7 层各自 fp32 RMSNorm → 共享投影 → 8 组门控层混合（fp32 门控，
  softmax over layers）→ 1 层双向无位置注意力细化 → 2560。每个主 caption token
  出一个 DiT token。
- `ConditionTokenEncoder`：8 个可学习查询 token 对条件段做交叉注意力；条件 dropout
  时整体换成可学习 null token（CFG 无条件分支由此训练）。
- 相机取景 `hdm_shifted_square_v2`（生产 p=1.0）：短边缩放到舞台正方形 R、长边
  量化、长轴均匀整数偏移，输出大画布的 R×R 裁剪。**不改序列结构**——只把图像
  token 的 RoPE 坐标经精确仿射映射进全画布帧。

### 目标与采样（`objective/flow.py`、`sampling/`）

- 时间采样 JLT：`t = sigmoid(N(0,1)·0.8 − 0.8)`；插值 `z_t = t·x + (1−t)·ε`。
- 模型预测 x；损失把预测与目标都换算成速度 `v=(x̂−z_t)/(1−t)`（clamp t≥0.05），
  fp32 均方误差。CFG 2.9 在速度空间合成。
- 求解器：`euler`（N 步）、`heun_final_euler`（2N−1 步）；时间网格线性 linspace，
  状态 fp32 积分。命名 profile：`preview`（euler 28，周期评估用）、`balanced`
  （heun 25，默认）、`reference`（heun 50）。
- iREPA：教师 PE-Spatial-B16-512（ViT 768，patch 16）冻结，原始 patch 特征做 fp32
  空间 z-score（γ=0.6）；学生在 DiT slot 8 后取图像 span，过 3×3 混合精度卷积
  投影器（bf16 权重 + fp32 bias，FQN 锁定 2 项），损失 = token 均 1−cosine，
  权重 0.5、1000 update 半余弦 ramp-in。

### 优化器

- 256 生产（cutover anchor 111500 起）与当前 512 run 同为 **`hybrid_cmuon_canonical_ns4_fp32_rescue`**
  （base 默认是 `torchao_adamw8bit`；生产链的 `train_g1_cmuon_production.toml` 覆盖整个
  `[optimizer]` 子树，512 toml 不覆盖 → 原样继承。部署 `resolved.toml` 已核实）：
- **Muon** 覆盖 2D 投影（141/289 参数 = 97.8% numel）：Nesterov 动量（μ 0.95，bf16 存储）、
  逐 chunk 五次 Newton-Schulz 正交化（4 步）、Moonlight 缩放 `α = lr·0.2·√max(d_out,d_in)`；
  guard 参考表（decay 0.999、floor 6.575e-7）+ FP32 owner-rank rescue
  （非有限/触顶 chunk 回退 fp32 路径）。
- 其余 148 参数（bias/norm/文本与条件编码器/输入输出投影）走内层 **AdamW8bit**：8-bit
  量化动量（block 256）、bf16 随机舍入（独立 SR CUDA RNG）、步进前有限梯度检查。
- LR：base 5e-5，按全局 batch 线性缩放（reference 256）：512 run ×480/256 = 9.375e-5；
  warmup 1000（256）/ 10000（512）；两组 weight decay 均为 0；梯度裁剪全局 L2
  max_norm 1.0（fp32 计算）。

### 数值策略

- bf16：所有 DiT 线性权重与激活、Qwen、VAE。
- fp32 敏感参数（永不做 bf16）：全部 1-D 参数（bias/norm/嵌入/门控）+ 整个
  `GlobalConditioner`/`FinalOutputHead` 投影 + 条件查询/null token + 文本混合门控；
  由参数审计强制（`optim/groups.py`）。
- fp32 计算岛：所有 RMSNorm 累加、全局条件前向、x→v 换算与速度损失、梯度范数/
  裁剪、采样状态、iREPA 目标与损失。
- 梯度裁剪：全局 L2，max_norm 1.0，fp32 计算。

## 运行环境（DTK 双卡）

- 代码仓库：`/root/private_data/sakuramoon`（NFS 家目录，配额有限，**只放代码与
  配置**，所有 dump/临时件用完即删）
- 运行根目录：`/sakuramoon-runtime`（本地盘）：`data/`、`cache/data/`、`model/`、
  `model-irepa/`、`output_model/`、`runs/`、`artifacts/` 都落在其中
- 数据队列状态：`config/data-service-mainset.json`（随项目在 NFS 上，**迁移必带**；
  不要放回 `/sakuramoon-runtime/cache/`，否则重建后从 cycle 0 重来）
- 两张 DCU：`ROCR_VISIBLE_DEVICES=0,1` / `CUDA_VISIBLE_DEVICES=0,1`
- 日志：`/root/sakuramoon-logs/`；密钥：`.env.training-stack.nul`（mode 600，模板
  `config/env.training-stack.example`；含 `MODELSCOPE_API_TOKEN`、`WANDB_API_KEY`）
- 任何 torch import 前先 `source /opt/dtk-26.04/env.sh`（否则 `libgalaxyhip.so.5` 失败）
- Python 3.12 + uv（`uv run --no-sync`）；torch 2.9.0 + DTK 26.04（生产 venv）

## 1. 数据服务（先启动）

独立拥有的分片供应服务：从 ModelScope 下载/预取 WebDataset 分片，经 unix socket
（`/run/sakuramoon/data-service.sock`）租给训练 rank，独占分片缓存与队列状态。

```bash
cd /root/private_data/sakuramoon
.venv/bin/python -u -m sakuramoon.cli.data_service \
  --config train_g1_512_all_bs80_irepa.toml \
  --config-root /root/private_data/sakuramoon/config \
  --root /sakuramoon-runtime
```

日志：`tail -F /root/sakuramoon-logs/data-service.log`

## 2. 训练（等数据服务 socket-ready）

生产上由 `scripts/training_stack.sh start` 一条命令管理三个进程（data / train /
publisher，flock 互斥，PID 文件在 `/run/sakuramoon/`）；手工等价命令：

```bash
cd /root/private_data/sakuramoon
.venv/bin/python .venv/bin/accelerate launch \
  --multi_gpu --num_processes 2 --num_machines 1 \
  --mixed_precision no --dynamo_backend no --main_process_port 29500 \
  -m sakuramoon.cli.train \
  --config train_g1_512_all_bs80_irepa.toml \
  --config-root /root/private_data/sakuramoon/config \
  --root /sakuramoon-runtime
```

- 训练**总是**从 `output_model/<run>/` 下数值最新的 COMPLETE checkpoint 恢复
  （目录名 `ckpt_<n>_raw-<n>-update-cadence` + `COMPLETE` 文件 + `manifest.json`）。
  `RESUME_CHECKPOINT=/abs/path` 指定根外迁移 checkpoint；全新启动需 `ALLOW_FRESH_START=1`。
- 完整 checkpoint 每 `[checkpoint].full_every_updates`（G1/512 = 100）update 原子写入，
  `slots=2` 轮转；半成品目录命名 `.ckpt_*.tmp`，永不被恢复/发布。
- 合并后的配置在启动时写 `<run_dir>/resolved.toml`（脱敏），resume 时与已发布
  版本 diff，结构冲突 fail-closed。

日志：`tail -F /root/sakuramoon-logs/train.log`

### 配置驱动训练与断点恢复

- `train.max_updates` 是**当前调用的绝对**成功更新终点，每个 resume 实时读取，
  双向权威：提高则延长，降低不回滚模型/优化器/计数。
- `max_updates ≤ checkpoint update` 时零新 update 干净退出（no-op）。
- 例（ckpt @130000，历史 terminal 168000）：`max_updates=135000` → +5000；
  `=120000` → +0。
- 契约与 worked example：`docs/config-and-resume.md`。

> ⚠️ 512 生产链的父配置 `train_g1_cmuon_production_camera_p100.toml` 只存在于部署
> 树（`/root/private_data/sakuramoon-g1/config/`），repo `config/` 里没有该文件，
> repo 中的 512 toml 因此无法离线直接 load。其内容（2026-09-18 在机核实）=
> camera p=1.0 块 + `concept_suite_enabled = false`（160000 训练内套件崩溃，套件
> 改离线跑），extends `train_g1_cmuon_production.toml`（repo 有，提供 CMuon 优化器
> 子树）；部署机 `max_updates = 200000`（repo 快照 168000）。以机器 config /
> `resolved.toml` 为准。

## 3. 发布 checkpoint 到 ModelScope

```bash
cd /root/private_data/sakuramoon
SOURCE_ROOT=/sakuramoon-runtime/output_model/g1 \
STATE_ROOT=/root/.sm-train-state-publisher \
REPO_PATH=g1 \
  bash scripts/publish_train_state.sh loop
```

- 每 `INTERVAL_SECONDS`（默认 600s）把 `SOURCE_ROOT` 下带 `COMPLETE` 的完整
  checkpoint 硬链接暂存后镜像到私有仓库，上传后远端回拉 `COMPLETE`/`manifest.json`
  sha256 复核。
- `STATE_ROOT` 必须在本地盘（`/root/...`）：硬链接暂存跨文件系统（NFS）会失败。
- ⚠️ 已知系统性问题：ModelScope 账号级 2000 commits/24h 滚动配额，本脚本的
  64 文件批量提交 + 每周期重提交变更文件会打满配额（429）。安全后果只是
  镜像延迟（blob 留在 LFS，worker 有 MAX_PENDING_TARS 闸门）；重建要求（事件
  驱动、单提交、持久 tracker）记录在项目 AGENTS.md。

## 评估

### 在线 FID/IS/KID/CMMD（`[evaluation]`，每 5000 update）

- 四项指标复用同一批 1000 张生成图（`sampling_profile="preview"`，euler 28 步）；
  真实侧从验证 cohort（512 run = `s0-validation-512-v3`，7 shards，10594 图）按
  评估器确定性后缀/顺序规则读取。
- 真实侧 Inception/CLIP feature 与 Inception logits 按（数据指纹 + 样本数 + 分辨率）
  **一次性**缓存到 `output_model/evaluation/g1/cache/real-features-*.pt`；prompt
  manifest SHA 或 cohort 变化时一次性重算（约 10.6k 图）。不匹配的缓存永不静默复用。
- 结果写 `output_model/evaluation/g1/step-<update>.toml` + `latest.toml`。
- ⚠️ `due()` 严格等于 `update % every_updates == 0`：**错过/失败的评估点永不重试**。

### 独立评估 CLI（补算/离线）

```bash
# FID/IS/KID/CMMD 补算（复用在线 TrainingEvaluator，不动训练）
.venv/bin/python -u -m sakuramoon.cli.generation_eval \
  --config train_g1_512_all_bs80_irepa.toml \
  --config-root /root/private_data/sakuramoon/config \
  --root /sakuramoon-runtime \
  --checkpoint /sakuramoon-runtime/output_model/g1/ckpt_<n>_raw-<n>-update-cadence

# 概念条件套件（120 concept × 双通路 text/condition，每 concept 5 图；
# 参考帖首用时从 Danbooru 下载到 concept-refs/ 缓存）
.venv/bin/python -u -m sakuramoon.cli.concept_eval \
  --config train_g1_512_all_bs80_irepa.toml \
  --config-root /root/private_data/sakuramoon/config \
  --root /sakuramoon-runtime \
  --checkpoint <ckpt_dir>

# VAE 重建（必须单卡可见；两个互斥确定性子集：recon FID/LPIPS/PSNR/MS-SSIM
# + real-real FID）
.venv/bin/python -u -m sakuramoon.cli.vae_reconstruction \
  --config train_g1_512_all_bs80_irepa.toml \
  --config-root /root/private_data/sakuramoon/config \
  --root /sakuramoon-runtime \
  --sample-count 512 --batch-size 16 --comparison-count 16
```

- 120-concept 全套件在训练内跑过一次 300s stall 崩溃（NCCL 看门狗；160000 那次是
  rank0 在 wandb submit 挂住 → rank1 在 complete barrier abort），所以 512 生产配置
  `concept_suite_enabled = false`，常规做法是离线走 `concept_eval` CLI。
- 训练内周期采样：每 1000 update，60 张锁定 cohort 图（+512 run 额外 12 张
  raw-cohort），`longitudinal_pin_update` 固定纵向锚点。

## 数据管线

### 语料与分片配对

- 语料 = ModelScope 私有仓库 `leafmoone/webdataset_danbooru_v3`（master），
  数据集 manifest（`data/dataset-manifest-webdataset-danbooru-v3.json`，自动
  初始化/刷新）：**10226 shards / 19.07 TB**，四个源 —— danbooru 4986 /
  zerochan 3000 / bangumi 2000 / gc5m 240（以部署 manifest 实测为准）。
- 每个 shard tar 内部统一扁平布局：`<source>/<id>.<ext>` + 同 stem 的
  `<source>/<id>.json` sidecar（原始语料的目录差异在入库流水线
  `scripts/deepghs_quality_pipeline.py` 发布时归一，训练侧只见统一布局）。
- **配对规则**：`webdataset.WebDataset` 流式读 tar，按 key（成员名去掉最后一段
  扩展名）把同 stem 的成员归成一个样本 dict；图片字段按
  `("jpg","jpeg","png","webp","avif")` 取唯一命中（双图 → multi-image 拒绝；
  无图有杂 payload → unsupported-image 跳过）。文件夹前缀与图片扩展名不影响配对。
- 元数据在各源 schema 间由 `metadata_adapter` + `parse_shard_metadata`
  （config `[data]` 字段映射）归一成 `OperationalMetadataRecord`，再由
  `parse_modelscope_caption_fields` 出 caption 面。
- 样本确定性身份 = `<shard-url>\0<key>` → 派生所有 dropout/裁剪骰子
  （`rng_identity`），换机重跑同一样本行为一致。

### 缓存与队列

- 缓存根 `cache/data/`（镜像 manifest 相对路径）；水位策略
  `low=192 GiB / high=256 GiB`，LRU(mtime) 驱逐，下载并发 8 流、预取前瞻 48。
  **cache 可重建**（data-service 从 ModelScope 经代理自补，60–129 MB/s）。
- 队列状态 `config/data-service-mainset.json`：`{cycle, rows[{path,status}]}`，
  每 cycle 顺序 = 排序后按固定种子 shuffle，跨进程/重启确定性一致；
  **迁移必带**（见下）。

### 验证集与提示词 manifest

- 验证 cohort：7 个分片（`VALIDATION_SHARD_COUNT = 7`），
  `data/validation-cohorts/<name>/` = 7 tars + `validation-selection.json` +
  `validation-prompts.json`（schema 4）+ build report。cohort 分片**不可重新
  下载**（冻结），迁移必带。
- `scripts/build_validation_prompts.py`：从 cohort tars 按评估器全局读序取图、
  生产解析器 + 零 dropout 构 caption plan、seed 44 洗牌取前 1000 例，canonical
  JSON 原子写出。**同时必须从 `--locked-source`（默认 legacy v2 manifest）追加
  8 个训练锁定条件案例**——`TrainingSampler`（`load_fixed_condition_pairs`）
  硬要求这 8 个固定 `validation-<md5>` 案例与 eval 读的是同一 manifest，
  缺失会在任何训练启动时 `fixed condition prompt manifest lacks the locked
  cohort` 崩溃（eval 只取前 sample_count 例，尾部不可见）。cohort 选择变化后
  用 `--force` 重建。

### Caption 计划

- 类别顺序 `tags → nl → condition`；`condition_mode = "artist_or_character"`
  （角色前缀 "style reference:" / "character identity:"）。
- Dropout：`all_condition=0.10`、tag 0.1、candidate_source 0.3、各 NL 分支 0.3
  （512 run 覆盖 `long_names=0.1`）；按样本 seed 抽取，tag shuffle 后去重
  （`_dedupe_*`，不扰动 RNG 流）。
- 文本预算：condition ≤512 token（8 桶 64–512），qwen dense 长度 98–546。
- 桶：基础面积 262144 px（512²），quantum 32，最大长宽比 4.0。

## 主机迁移运行手册（zzai pod → pod）

生产 pod 是**临时容器**：`/sakuramoon-runtime` 与 `/root`（NFS 家目录除外）随 pod
释放清空；`/root/private_data`（NFS 家目录）**跨 pod 存活**。迁移 = 把临时盘上
的东西搬到新机，NFS 上的免费继承。

### 标准热迁移流程（~30 min，训练不停）

1. **P1 静态（训练继续跑）**：内网 `tar | ssh`（源机生成 ed25519 key 授权目标
   :22，~180 MB/s）。成员 = `/sakuramoon-runtime/{sakuramoon-dtk-venv, model,
   model-irepa, cmuon-f1-emergency, torchinductor-cache, data, concept-refs,
   runs, artifacts}`（约 40G，几分钟）。**逐项核对目标侧体积**——tar 对不存在的
   成员静默跳过（`-C /` 里写 `/root/` 下的成员会 rc=2 全漏）。
2. **P2 根目录**：`/root/{start-gNN.sh, gNN-watchdog.sh, ckpt-mirror-gNN.sh,
   sakuramoon-logs}`；sed 替换主机名时**必须同时**做 `s/gNN/gNN2/g` 和
   `s/goodNN/goodNN2/g`（"goodNN" 不含子串 "gNN"，`REQUIRED_HOST_SUBSTRING`
   会漏改）；chmod +x 逐个确认。
3. **P3 热切（停机窗口内）**：等 checkpoint 边界停训练（kill watchdog → 组件 →
   touch `/run/sakuramoon/user-stop`），传 `/sakuramoon-runtime/output_model`
   （13–24G，~1 min），**目标侧逐文件 sha 验证最新 COMPLETE ckpt** 后才启动。
4. **P4 缓存**：可选后台流式传（254G 窗口）；可以中途砍——data-service 自补。
   ⚠️ **流式/partial 传完必须审计**：逐分片对 dataset-manifest 比大小
   （cache 根是 `cache/data/` 不是 `cache/`），坏分片会被 data-service 检出
   （"cached shard differs from manifest"）并拒租，训练 rank fail-closed 崩溃。
   审计脚本模式见 `/tmp/cache-audit.py`（good22）。

### 必带文件清单（缺一即崩或降级）

| 文件/目录 | 漏了会怎样 |
|---|---|
| `runtime/model-irepa/`（330M，iREPA 教师） | 每次启动 preflight `FileNotFoundError` 崩溃（512 run 必需；日志里的 "DataLoader worker SIGTERM" 只是 launcher 清尸，真 traceback 在它上面） |
| `runtime/cmuon-f1-emergency/`（7.9M） | 紧急回滚材料丢失 |
| `config/data-service-mainset.json` | 队列从 cycle 0 重来（epoch 顺序错位） |
| `data/validation-cohorts/` **两个** cohort（v2+v3，27G） | 评估真实侧/锁定案例源丢失，不可重下 |
| `data/validation-cohorts/s0-validation-512-v3/validation-prompts.json`（**1008 例版**，sha `599ca94a…`） | 任何训练启动 `lacks the locked cohort` 崩溃（见数据管线节） |
| 部署 config（含 `train_g1_cmuon_production_camera_p100.toml`） | 512 toml 的 extends 链断裂 |
| `.env.training-stack.nul`（3405B，mode 600） | 栈 FAST-FAIL |
| `model/qwen_3.5_2B/` + `model/vae/`（6.1G） | 编码器加载失败 |
| venv 3.2G（同路径 `/sakuramoon-runtime/sakuramoon-dtk-venv`，shebang 已指向该路径，原样搬） | 需要重建 + 修 shebang |
| 平台 ai_proxy（3 处：`.ai_user_info/ai_proxy`、`/etc/profile.d/zzai-ai-proxy.sh`、`.bashrc`） | 平台注入不可靠，缺了 ModelScope 下载断网；用 ASCII 版（平台原生 UTF-16 版 source 失败） |

### 启动 GUARD（迁移后必查）

1. `resolved.toml` 的 `[data.camera_viewport]` / `[evaluation]` 段与迁移前逐字一致；
2. sample-trace 出现 `camera=applied`（512 生产）；
3. 首个 update 出现、loss 在正常带（~0.52–0.57）；
4. 若出现 `lacks the locked cohort` = manifest 退化回 1000 例版，立即换回 1008 例版。

### 常见坑（历次踩过的）

- **pkill 自匹配**：远程 shell 的 cmdline 含模式串，`pkill -f` 会杀掉自己的
  plink 会话——用括号模式 `pgrep -f "[s]tart-gNN"`，且单独一条命令验证。
- **watchdog 先杀**：start 脚本会 `rm -f user-stop` 自动拉起整栈；停机顺序 =
  watchdog（按 PID）→ 组件 → touch user-stop。
- **DTK 环境**：任何 torch import 前 `source /opt/dtk-26.04/env.sh`
  （`libgalaxyhip.so.5`）。
- **NFS 家目录配额**（50G）：只放代码 + 最新 ckpt 镜像（~6G）；大 dump/clone
  是临时件，用完删（曾把 50G 写满导致 data-service 原子发布探针失败崩溃循环）。
- **plink 后台任务不可信**：本地 wrapper 提前 exit-0 但远端继续跑——长任务用
  单条有界同步等待（远端轮询 done-marker），并以远端地面真值（进程列表/ledger/
  done-marker）为准。
- **wandb**：崩溃重跑后从更早 ckpt 恢复会出现 "step less than current" 警告，
  重记覆盖，无数据损坏。

## 测试

```bash
source /opt/dtk-26.04/env.sh     # 必须先（DTK）
uv run --no-sync pytest tests/unit -q
```

- 权威门槛 = `tests/unit`（Linux + DTK）。基线：4 个既有环境性失败
  （`test_cmuon_fp32_forensic.py`），其余全过。
- `tests/gpu/` 需要真卡（单步、CMuon 双 rank、性能基准），不要裸 `pytest` 全树跑。
- Windows 本地跑单测会有 `os.O_NOFOLLOW`/`fcntl`/符号链接语义的平台差异失败，
  不算回归。
