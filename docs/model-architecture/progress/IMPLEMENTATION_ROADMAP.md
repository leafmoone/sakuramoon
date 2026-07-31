# SakuraMoon 详细实现工程路线图

状态：执行中。`R001` 已由 commit `473eea9` 完成；`R002` 已完成实现与主代理验收，等待 Foundation 包级审查。后续任务仍须按依赖顺序逐 ID 关闭。

独立审查记录：[AI/模型正确性审查](../reviews/ROADMAP/ai_review.md)；[Infra/性能审查](../reviews/ROADMAP/infra_review.md)。

## 1. 范围和执行边界

目标是在单机 4×RTX 5090 上，从零训练二次元垂类文生图模型。首版以 512 等效面积成品为目标，模型按 `16→20→24` 层增长，训练阶段固定为 `S0→S1→G1→S2→G2→S3`。768/1024 只作为 512 验收后的手工可选阶段。

本路线图覆盖：本地文档治理、Git、uv、配置系统、数据和缓存、Caption/Qwen/VAE、文本与 Style 条件、Single-stream DiT、目标函数与采样、optimizer/DDP/checkpoint、增长和 stage、W&B、FID/IS、profiling、故障注入及正式 canary。

当前已完成 Git 初始化、资产忽略边界、uv 安装和 Python 依赖锁。尚未执行数据集下载、模型加载、GPU kernel、训练代码或 canary；Notion 迁移与 MCP 移除已经完成。

## 2. 当前已核对的环境事实

| 项目 | 当前事实 | 路线图影响 |
|---|---|---|
| Git | 根仓已初始化；`R001` commit=`473eea9` | 后续任务保持逐 ID 原子 commit |
| uv | 0.12.0；Linux x86_64/Python 3.12 lock 已生成 | `R002` 的空环境重建证据见 `progress/environment-lock.md` |
| Python | 3.12.3 | 仅支持 Linux/Python 3.12，不做额外兼容 |
| PyTorch | 2.10.0+cu128；TorchAO 0.16.0；Triton 3.6.0 | import/CUDA 可见性已通过；kernel 仍由 `K001/T021/T040` 验证 |
| GPU | 当前只可见 1×RTX 5090，32607 MiB | 可完成 S0 和单卡 kernel 测试；S1 以后必须等 4 卡可见 |
| Qwen | `model/qwen_3.5_2B` 已下载；config 为 24L/2048 hidden | `T021` 从固定路径本地加载并验证；禁止下载、联网替换或 fallback |
| VAE | `model/vae` 已下载；MageVAE，128 latent channels、downsample 16、posterior mean | `T020` 从固定路径本地加载并完成 round-trip/质量验收；禁止下载或 fallback |
| 元数据 | `db/` 已准备，本地使用 | 不进 Git；仅在数据任务实际需要时按 schema 读取，不建立资产哈希清单 |
| 存储 | 工作区为 400 GiB NFSv3，当前约 363 GiB 可用；无 checkpoint 预留 | 达到 300 GiB cache 低限，但不是已验证 NVMe；正式 preflight 继续硬阻塞 |
| 参考工程 | `reference/` 仅供人工理解/对照，可完全不使用 | 根仓忽略；任何代码、测试、preflight、训练或运行时不得导入、执行或调用其中代码 |
| 凭据 | `.env` 已写 `MODELSCOPE_API_TOKEN`，权限 600 | `.env` 永不进 Git；resolved config、日志和 W&B 必须脱敏 |

目标配置仍要求 4×RTX 5090、14 vCPU、约 120 GiB host RAM 和足够 NVMe。正式四卡任务不得用当前单卡结果替代。

## 3. 本地文档和追踪规则

### 3.1 规范来源

| Requirement 前缀 | 本地文档 | 负责范围 |
|---|---|---|
| `C01-*` | `archive/notion/01-constraints-budget.md` | 项目约束、预算、总体验收 |
| `C02-*` | `archive/notion/02-image-mage-vae.md` | 图像处理、Mage-VAE、latent |
| `C03-*` | `archive/notion/03-text-encoder-and-input-protocol.md` | Qwen、framing、token 协议 |
| `C04-*` | `archive/notion/04-text-layer-aggregation.md` | 多层聚合、双向 adapter、Style |
| `C05-*` | `archive/notion/05-multimodal-sequence-and-rope.md` | modality、packing、RoPE |
| `C06-*` | `archive/notion/06-single-stream-dit.md` | DiT 主干、GQA、norm、gate |
| `C07-*` | `archive/notion/07-timestep-and-global-conditioning.md` | t/size/aspect 与 modulation |
| `C08-*` | `archive/notion/08-xpred-and-sampling.md` | x-pred、velocity loss、CFG、Heun |
| `C09-*` | `archive/notion/09-model-growth.md` | stable slots、增长和 state 迁移 |
| `C10-*` | `archive/notion/10-resolution-curriculum.md` | stage、bucket、课程和高分辨率 |
| `C11-*` | `archive/notion/11-data-and-cache.md` | 数据、dropout、缓存、恢复、验证隔离 |
| `C12-*` | `archive/notion/12-optimizer-and-training-system.md` | optimizer、DDP、checkpoint、性能和故障 |
| `OBS-*` | `current/observability-and-evaluation.md` | W&B、耗时、FID/IS、profile |

冲突时只允许以下优先级：`current/confirmed-decisions.md` > `current/open-items.md` > 本地补充决定 > 01–12 历史组件页。任何实现不得从历史候选或“推荐但未批准”段落自动生成配置。

### 3.2 机器可检查的追踪

`D001` 建立 `docs/model-architecture/progress/traceability.toml`。既有完整记录为兼容历史继续保留；向前执行只维护稳定 requirement ID、状态、实现路径和测试映射：

- `requirement_id` 与稳定 source/heading/node-kind 归属；既有 fingerprint 作为历史身份锚点，不随普通措辞更新。
- `status = planned|implemented|verified|blocked|superseded`。
- 配置键列表。
- 生产模块和 reference 模块列表。
- 测试、benchmark 和 artifact 路径。
- 既有实现 commit、证据和 review 不删除；新审查按任务包或高风险任务规则引用。
- 目标硬件等级：CPU、1GPU、4GPU。

`tools/verify_traceability.py` 必须双向检查：所有现行 requirement 都有实现/验证映射，所有关键模块和配置键也能反向找到文档依据。检查器先匹配既有 fingerprint，再只允许在相同 source/heading/node-kind 内按顺序吸收措辞漂移；条款增删、heading/source 迁移、历史 ID/fingerprint 改写仍失败。

## 4. 风险分级、里程碑审查和 Git 协议

2026-07-29 用户批准以下执行协议。它只减少代理调用、重复文档与重复测试，不改变架构、凭据安全、生产训练或真实硬件门槛。

1. 每个任务 ID 仍是独立实现与回滚单元。主代理创建 `progress/tasks/<TASK_ID>.md`，每个 ID 保持独立状态、diff、针对性测试和原子 commit。
2. 低中风险任务按包复用同一个实现代理：Foundation=`R002,D001,C001,A001`；Data=`D010-D016`；Encoders/Conditioning=`T020-T024`；Dense Model=`M030-M034`；Training Utilities=`T051-T053`。A001 只保留已完成的最小本地资产边界；A002 重型审计保持撤销。
   包级证据目录依次固定为 `reviews/FOUNDATION/`、`reviews/DATA/`、`reviews/ENCODERS/`、`reviews/DENSE/` 和 `reviews/TRAINING_UTILITIES/`。
3. 低中风险任务从逐任务审查改为包级里程碑审查。审查报告仍逐 ID 分别给出 AI/模型正确性与 Infra/性能结论；问题只修复受影响任务，不重跑无关任务。
4. `K001`、optimizer `T040`、DDP `T041`、checkpoint `T042`、growth/transition `T043`、训练 step `T050`、故障注入 `T054` 和所有正式 stage canary 保持单独实现、独立 AI reviewer、独立 Infra reviewer。
5. 每任务只跑针对性 unit、contract 和小型真实 GPU 测试；17x8 shape、100k 扫描、1,000-step canary 与完整恢复矩阵在对应里程碑集中运行一次。
6. metadata 扫描、下载校验、VAE 统计和 GPU canary 可在后台运行；仅当依赖已满足、接口已冻结、不写同一文件且不争用同一 GPU/NVMe 时并行，结果返回前不得关闭依赖项。
7. 新任务只维护稳定 ID、状态、实现路径、针对性测试和原子 commit。普通 CPU 任务不创建独立 `timing.json`；共同环境、资产和 benchmark 由任务包共享引用，`perf_baseline/after` 只为真实性能任务生成。
8. 生产 FA4 和完整 canary 前允许显式 dense SDPA reference 的 1-10 step 真实垂直 engineering smoke；不得把该结果作为 S0 放行证据。
9. 当前优先单卡相关任务。所有多卡实现、DDP/NCCL 验证和正式多卡 stage 暂不执行并保持 blocked/pending；单卡结果不得替代四卡证据。

开发耗时仅在需要比较资源或性能工作时写入 `progress/time-log.jsonl`；训练运行期 phase timing 另写 artifact/W&B。Git 跟踪的 task 与 test/timing/review JSON 由原子 commit 发布，不增加运行时事务 publisher。checkpoint、dataset manifest、validation exclusion、shard/cache 保留完整原子发布；可再生 image/metric scan 报告采用同目录临时文件、文件 `fsync` 与 `os.replace`。实现和审查代理不得创建 commit，最终验证与每 ID 原子 commit 由主代理完成。

## 5. 目标仓库和模块边界

```text
sakuramoon/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── config/
│   ├── base.toml
│   ├── train_s0.toml
│   ├── train_s1.toml
│   ├── train_g1.toml
│   ├── train_s2.toml
│   ├── train_g2.toml
│   ├── train_s3.toml
│   ├── train_h1.toml
│   ├── train_h2.toml
│   ├── eval.toml
│   ├── sample.toml
│   └── examples/all_options.example.toml
├── docs/model-architecture/
│   ├── current/
│   ├── archive/
│   ├── progress/
│   ├── reviews/
│   └── sources/
├── src/sakuramoon/
│   ├── config/{schema,load,resolve,redact}.py
│   ├── data/{modelscope,manifest,cache,metadata,validation}.py
│   ├── data/{caption,serialize,buckets,image_ops,pipeline,collate,state}.py
│   ├── encoders/{qwen,mage_vae}.py
│   ├── conditioning/{text_mixer,bidirectional,style_resampler}.py
│   ├── conditioning/{modality,packing,rope,global_condition}.py
│   ├── model/{norm,mlp,attention,block,dit,output_head,growth}.py
│   ├── objectives/{timestep,flow,loss,cfg}.py
│   ├── sampling/{heun,sampler}.py
│   ├── optim/{groups,adamw8bit,stochastic_rounding,clip}.py
│   ├── distributed/{ddp,global_mean,state_guard}.py
│   ├── checkpoint/{schema,save,load,migrate,pma}.py
│   ├── train/{step,loop,stage,preflight,failures}.py
│   ├── eval/{vae_reconstruction,prompts,generate,fid_is,quality}.py
│   ├── telemetry/{metrics,timers,nvtx,wandb_sink,profiler}.py
│   └── cli/{train,eval,manifest,benchmark,transition}.py
├── tests/
│   ├── unit/
│   ├── contracts/
│   ├── integration/
│   ├── gpu/
│   ├── distributed/
│   ├── checkpoint/
│   └── fault_injection/
├── tools/
│   ├── verify_traceability.py
│   └── verify_artifacts.py
└── artifacts/               # ignored; run manifests point here
```

模块之间只能通过显式 typed batch/state dataclass 交互。`data` 不导入 `model`；`encoders` 不知道 DiT；`objectives` 不负责 sampling；`checkpoint` 不包含训练策略；`telemetry` 不改变训练控制流。

## 6. 配置系统契约

### 6.1 文件和合并

- 使用 `config/` 单数目录和 TOML。
- `base.toml` 提供完整公共值；每个 `train_*.toml` 通过项目自定义的 `extends = ["base.toml"]` 覆盖单一 stage。
- 合并规则只允许表级递归合并和标量/数组整体替换，禁止数组按索引隐式合并。
- 使用 `tomllib + Pydantic v2 strict models`；schema `extra="forbid"`，语义字段没有代码默认值。
- 启动时输出完整 `resolved_config.toml`、schema version、所有输入文件 SHA-256 和 resolved hash。运行中只读取 resolved object。
- `all_options.example.toml` 必须列出所有键、合法范围、单位、是否固定、是否 benchmark 后填写和注释。示例不能被训练 CLI 接受为生产配置。
- 架构固定值仍写入 TOML用于溯源，但 schema 限制为批准值；修改必须先建立新决策记录。

### 6.2 必须覆盖的配置表

| 表 | 关键内容 |
|---|---|
| `run/paths/security` | run/stage/seed、目录、secret env 名称、脱敏规则 |
| `assets.qwen/assets.vae` | 固定本地路径、dtype、冻结策略 |
| `data.source/manifest/cache` | ModelScope dataset、revision、shard hash、LRU、worker、queue、恢复 |
| `data.validation/image/buckets` | 2,000 验证隔离、EXIF、no-upscale、retention、17 image buckets |
| `caption/dropout/text_buckets` | 类别顺序、NL 选择、全部 dropout、512 上限和八桶 |
| `model.text/style/dit/rope/condition/head` | 所有锁定 shape、层、gate、初始化、精度和 backend |
| `objective/timestep/sampling/cfg` | x-pred、JLT 参数、FP32 x-to-v、Heun-50、CFG 2.9 |
| `optimizer/scheduler/gradient` | TorchAO、参数组、SR RNG、WSD、clip、global mean |
| `distributed/checkpoint/growth/stage` | DDP、完整 sidecar、stable FQN、transition、预算 |
| `kernels/compile/profiling/failure` | FA4/SDPA policy、compile gate、profile 采样和硬失败 |
| `logging/wandb/timing/evaluation` | loss/grad/perf、local spool、FID/IS 周期和正式协议 |

全部 caption dropout 已由 D016 锁定为显式 TOML 固定值。`C002` 不再受用户 dropout 决定阻塞，但仍须独立完成 stage overlays 与 benchmark 后填写项。

## 7. Phase 0：仓库、环境、文档和配置

### R001：初始化 Git 与资产边界

- **对应文档：** `C12-*`、本路线图第 4–5 节。
- **实现路径：** 根目录 `.gitignore`、`README.md`、`AGENTS.md`、`docs/model-architecture/progress/asset-policy.md`。
- **动作：** `git init`；确认 `.env/model/db/data/cache/reference/checkpoints/wandb/profiles` 被忽略；本地模型、DB 和参考工程不进入 Git，也不建立逐文件资产清单。
- **验证：** `git status` 不出现任何 secret、模型权重、数据、嵌套 `.git` 或训练 artifact；secret scanner 对 tracked files 为零命中。
- **代理审查：** AI 侧检查文档优先级，Infra 侧检查大文件、凭据和可恢复性。
- **完成证据：** 初始 commit hash、tracked file manifest、ignore/secret test。
- **GPU：** 无。

### R002：uv 与依赖锁

- **对应文档：** `C11` 强制依赖、`C12` kernel preflight。
- **实现路径：** `pyproject.toml`、`uv.lock`、`docs/.../progress/environment-lock.md`。
- **动作：** 安装 uv；限定 Linux/Python 3.12；锁 PyTorch CUDA 12.8、TorchAO、Transformers、Diffusers、Safetensors、ModelScope、W&B、WebDataset、Pillow、einops、Pydantic、pytest、ruff、pyright。FA4/CuTeDSL、Triton、`causal_conv1d`、`fla` 使用明确版本或 Git SHA。
- **验证：** `uv sync --frozen` 可从空 `.venv` 重建；输出 driver/CUDA/NCCL/编译器环境报告。uv 不负责 driver/CUDA/NCCL，系统版本单独锁定。
- **性能门槛：** import 不算 kernel 通过，GPU 执行放到 `K001`。
- **完成证据：** lock hash、license list、fresh-env report。
- **GPU：** 无。

### D001：本地文档规范与追踪器

- **对应文档：** 全部 `C01–C12`、`OBS-*`。
- **实现路径：** `traceability.toml`、`tools/verify_traceability.py`、`AGENTS.md`、review/task 模板。
- **动作：** 为现行条款分配稳定 ID；显式标出历史取代项；建立 requirement→config→module→test→benchmark→artifact 映射。
- **验证：** 故意删除一条映射时检查器必须失败；archive 文件 checksum 不变；current 文档允许演进并写 changelog。
- **完成证据：** traceability report、文档 link check、两类 review。
- **GPU：** 无。

### G001：向前简化治理证据

- **依赖：** D015 原子提交完成；治理期间是唯一写任务。
- **实现路径：** `AGENTS.md`、现行审查条款、路线图、`traceability.toml`、trace verifier 与对应 contract tests。
- **动作：** 冻结既有 requirement ID/fingerprint/evidence/commit；新任务只维护稳定 ID、状态、实现路径和测试；落实包级/高风险审查边界与三类发布策略。
- **验证：** 纯措辞更新不改历史 fingerprint 仍通过；条款增删、heading/source 迁移、ID/fingerprint 改写仍失败；历史 evidence fixture 原样保留。
- **完成证据：** task 状态、针对性 test report 和原子 commit；不创建独立 timing 或逐任务 review 文件。
- **GPU：** 无。

### C001：严格 TOML schema、合并和脱敏

- **对应文档：** `C11` 配置契约、`C12` preflight、`OBS-*`。
- **实现路径：** `src/sakuramoon/config/*`、`tests/unit/config/`、`config/examples/all_options.example.toml`。
- **动作：** 实现 deterministic merge、strict types/ranges/cross-field validation、unknown-key hard fail、secret env resolution、resolved config/hash 和 W&B 脱敏。
- **关键测试：** 未知 key、缺 key、错误类型、越界概率、NL 五值不一致、固定架构值被改、非法 stage transition、secret 泄漏、include cycle。
- **性能：** 只在启动执行，不进入训练热路径。
- **完成证据：** golden resolved config、schema coverage=100%、redaction test。
- **GPU：** 无。

### C002：完整配置样例与 stage overlays

- **对应文档：** `C03–C12`、`OBS-*`。
- **实现路径：** `config/base.toml`、八个 `train_*.toml`、`eval.toml`、`sample.toml`。
- **动作：** 写入所有固定架构参数与详细注释；S0/S1/G1/S2/G2/S3 每次只改变一个主轴；H1/H2 `enabled=false`；batch/accumulation/checkpoint mode 等 benchmark 值显式存在。
- **阻塞项：** 用户填写除 `all_condition=0.10` 外的全部 dropout。正式值缺失时 stage config 必须不可运行。
- **验证：** 每个 overlay 解析后无隐式默认；diff checker 证明相邻 stage 只改变允许键；H1/H2 无法意外启动。
- **完成证据：** 8 个 resolved hash、stage-diff report、逐键注释审查。
- **GPU：** 无。

## 8. Phase 1：资产、数据、Caption 与图像管线

### A001：最小组件本地资产边界修复

- **状态：** 原 manifest/hash/capability/TOCTOU 方案保持撤销；本任务只修复现行固定本地路径检查的组件耦合。
- **实现路径：** `src/sakuramoon/assets/__init__.py`、Qwen/Mage-VAE loader 与组件级 unit 合同。
- **动作：** Qwen 与 Mage-VAE 各自只检查自身加载必需文件；显式全模型 preflight 可复用聚合检查。缺失时硬失败，禁止下载、联网替换和 fallback。
- **验证：** 仅准备 Qwen 文件时 Qwen 边界通过；仅准备 VAE 文件时 VAE 边界通过；两个真实 loader 都不得被无关组件缺失阻塞。
- **非目标：** 不恢复本地模型 manifest、bytes/SHA-256、repo/revision/license、capability、TOCTOU 或 hostile-local-environment 层，不重新成为数据或 encoder 主线的依赖阻塞。
- **GPU：** 无；真实 Qwen/VAE forward 与质量证据继续归 `T021`/`T020`。

### A002：本地资产边界简化（已撤销原方案）

- **状态：** A001 与 A002 原 manifest/hash/capability 和大型 AST scanner 方案已撤销；A002 只记录本次最小化，不再阻塞单卡主线。
- **保留规则：** Qwen/Mage-VAE 只从 `model/qwen_3.5_2B` 与 `model/vae` 加载；必需文件缺失时失败；禁止下载、联网替换与 fallback；项目代码不 import、执行或调用 `reference/`。
- **验证归属：** 普通本地路径/加载测试并入 `T020`、`T021` 和训练 preflight，不建立独立资产模块、inspection CLI 或全仓源码扫描器。
- **数据边界：** D010 的远端 WebDataset revision、bytes/SHA-256 与流式传输完整性继续保留，不属于本地模型资产审计。

### D010：ModelScope 不可变 dataset manifest

- **对应文档：** `C11` 数据范围、远端读取、恢复。
- **实现路径：** `data/modelscope.py`、`data/manifest.py`、`cli/manifest.py`。
- **动作：** 使用 `MODELSCOPE_API_TOKEN`，锁定 `leafmoone/webdataset_danbooru` revision；枚举 shard path/release/bytes/SHA-256/samples；下载到 `.partial`，校验后原子发布。
- **安全：** token 不得出现在命令行参数、exception、resolved config、W&B 或 manifest；鉴权失效明确停止。
- **验证：** 中断续传、checksum 错、revision 漂移、重复 ID、缺 shard、权限失败；完整 shard不得静默跳过。
- **完成证据：** immutable manifest hash、下载故障测试、source license/access 记录。
- **GPU：** 无。

### D011：Metadata schema、全局 ID 与验证隔离

- **对应文档：** `C11` 固定验证集；`open-items` 3.1。
- **实现路径：** `data/metadata.py`、`data/validation.py`、schema fixtures。
- **动作：** 解析所有 metadata 字段，建立约 11M 逻辑 ID 唯一性报告；按 `release × aspect bucket × caption availability` 固定抽取恰好 2,000 ID，生成独立 validation shard。
- **验证：** 在 shuffle buffer 前排除验证 ID；完整 dry run 的验证消费计数必须为 0；schema unknown/missing key hard fail。
- **完成证据：** train/validation manifest hash、分层分布、zero-leak report。
- **GPU：** 无。

### D012：单协调器 shard cache 与 shard-level 恢复

- **对应文档：** `C11` cache/LRU/at-least-once；`C12` 系统故障。
- **实现路径：** `data/cache.py`、`data/state.py`、cache benchmark CLI。
- **动作：** 单机唯一下载协调器、300–500 GiB 可配置 LRU、有界并发、checksum publish、completed/active shard state；完成 shard 不重读，active shard 从头重放。
- **验证：** 并发 rank 请求同一 shard、cache eviction、磁盘写满、进程中断、坏 shard、恢复 replay 计数。
- **性能门槛：** 冷缓存 2 小时数据供给≥12 samples/s、ready wait<2%、无 swap/无界 RSS；最终参数由 sweep 写回 TOML。
- **完成证据：** cache profile、故障注入、replay audit。
- **GPU：** CPU/网络/NVMe；最终可与 1GPU 消费者联测。

### D013：EXIF、no-upscale、17 buckets、resize/crop

- **对应文档：** `C02`、`C10`、`C11`。
- **实现路径：** `data/image_ops.py`、`data/buckets.py`、contract tests。
- **动作：** 单次 decode、EXIF→RGB、等比 cover resize、只缩小、可复现均匀 crop、retention≥0.80；512 规则生成 17 个近等 token buckets并按阶段缩放。
- **验证：** 所有 bucket 形状 golden；转置闭包；同 seed/pass/id crop bitwise 一致；源尺寸/crop offset只进 audit，不进模型；dimension mismatch>0.1% 停止。
- **数据扫描：** 全 manifest 报告 eligible、no-upscale reject、retention reject、每桶样本数；不重新做文本长度扫描。
- **完成证据：** bucket scan、crop golden、100k decode dimension report。
- **GPU：** 无。

### D014：Dropout、Caption 与结构化 serializer

- **对应文档：** `C03`、`C04`、`C11`。
- **实现路径：** `data/caption.py`、`data/serialize.py`、`tests/contracts/text_protocol/`。
- **动作：** 类别骨架 `nsfw→character→copyright→general→NL`；类别内确定性 shuffle；candidate 仅做跨四类 canonical 删除；tags `, `、NL `\n\n`；Artist 只构造末尾辅助 segment；同时输出 main/artist token indices和 mask。
- **长度：** condition max 512，桶 `[64,128,192,256,320,384,448,512]`，完整 dense 长度 `[98,162,226,290,354,418,482,546]`；先保协议边界和 Artist，再裁 NL/低优先 tags；不重扫。
- **验证：** tags-only/NL-only/mixed/empty/dropout/candidate/超长/oversized tag/suffix golden；禁止字符串回查 span；Artist 不进入 main indices；100k dry run 验证 all_condition `10%±0.5` 个百分点和各实际 dropout。
- **阻塞项：** 未决定 dropout 数值。
- **完成证据：** golden corpus、分布报告、tokenizer prefix/suffix 断言。
- **GPU：** 无。

### D015：有界 WebDataset pipeline 与 collate

- **对应文档：** `C11` producer/consumer；`C05` text dense + DiT varlen。
- **实现路径：** `data/pipeline.py`、`data/collate.py`、`data/state.py`。
- **动作：** CPU 完成 metadata/dropout/tokenize/decode/crop/bucket；初始每 GPU 2 persistent workers、每 rank 2 ready batches；只 decode 一次、tokenize 一次。
- **接口：** typed batch 明确 raw image、token ids/mask/indices、bucket、target H/W、audit metadata、sample ID 和 RNG identity。
- **验证：** 1/2/3 workers与 queue sweep；无 unbounded queue、重复 decode/tokenize、validation leak、跨 batch activation cache；resume 语义与 `D012` 一致。
- **完成证据：** pipeline contract、stage timing、CPU/RSS/pinned memory report。
- **GPU：** 最终 1GPU 联测。

### D016：锁定 caption dropout 配置合同

- **依赖：** D014 caption 协议、C001 strict schema 与 G001 向前治理规则已完成；不依赖模型 packing rereview。
- **实现路径：** `config/schema.py`、`config/examples/all_options.example.toml`、配置合同测试、现行决定与 trace registry。
- **动作：** 精确锁定 `all_condition/general/artist/copyright/nsfw=0.1`、`character=0.2`、`candidate_source=0.3`，五个 NL key 均为 `0.3`；字段仍全部必填且无代码默认值。
- **验证：** 示例 TOML 使用批准值；缺失、未知 key、整数冒充 float 或任一数值漂移均启动失败。100k 生产分布 dry run 单独保持 pending。
- **完成证据：** D016 task、针对性 CPU test report 与原子 commit；不回写 D014 历史证据，不创建独立 timing artifact。
- **GPU：** 无。

### D020：Caption component dropout hit telemetry remediation

- **依赖：** D014 caption 决策路径、D016 严格概率配置与 T051 固定 metrics key。
- **实现路径：** `data/caption.py`、`data/serialize.py`、`data/collate.py` 与针对性 contract tests。
- **动作：** 对 all-condition、四类 tag、Artist、candidate source 和五个 NL 分支保留逐样本独立 hit decision，经 serializer 传递并在 collate 聚合为 batch counts；禁止从最终空 body 反推。
- **验证：** 空源与重叠 dropout golden、固定 key 与 T051 metrics 对齐、DataLoader worker pickle round-trip。
- **完成边界：** CPU telemetry 代码可完成；100k 生产分布、最终空 body 率和 `all_condition=10%+/-0.5pp` 仍是独立 pending evidence。
- **GPU：** 无。

### D021：Trusted shard metadata pipeline remediation

- **依赖：** D010 immutable `ShardRecord`、D011 `parse_shard_metadata` 与 D015 local pipeline。
- **实现路径：** `data/pipeline.py`、`data/collate.py` 与 data/fault/GPU contract tests。
- **动作：** pipeline 显式接收 local path、`ShardRecord`、无默认 `MetadataFieldMapping` 与显式 nested metadata adapter；按 sample `__url__` 绑定 trusted record，durable 路径只从 coordinator manifest 取 record；sample JSON 不得提供或覆盖 release。
- **验证：** forged/missing raw release、alias/nested adapter、unknown URL、path-record mismatch、durable batch release；真实 ModelScope shard + 本地 Qwen/Mage + RTX 5090 smoke。
- **完成边界：** CPU/真实数据本地模型单卡代码完成；Data package rereview pending。正式 immutable manifest 与 production caption-availability mapping 仍 pending；D012/D015 durable two-worker 由 D022/D023 独立修复。
- **GPU：** CPU + 1GPU engineering smoke；多卡无。

### D022：Bounded multi-active shard state schema v3

- **依赖：** D012 manifest-bound durable state 与 D021 trusted shard boundary；D023 worker 调度尚未实现。
- **实现路径：** `data/state.py`、state unit contracts、D022 task/test evidence 与 trace registry。
- **动作：** schema v3 持久化显式 `worker_count` 与有界 `active_shards`；拒绝 worker topology drift 和 v1/v2；activation 在 fetch 前发布；cache fetch 保护全部 active shards；允许逐 shard complete。
- **恢复：** 每次 restart 对全部 active shards 精确累计 replay shards/samples；所有 recovered active 都成功重新 prepare 前禁止激活新 shard。
- **兼容边界：** 保留 D015 singleton lease 行为；本任务不修改 pipeline/collate 或实现两个 persistent workers，后者单独归 D023。
- **完成证据：** targeted state contracts、邻接 singleton pipeline/fault 回归、ruff、pyright 与 D022 test report；Data 包级复审仍 pending。
- **GPU：** 无。

## 9. Phase 2：冻结编码器与条件分支

### T020：Mage-VAE wrapper 与重建验收

- **对应文档：** `C02-*`、`open-items` 3.2。
- **实现路径：** `encoders/mage_vae.py`、`eval/vae_reconstruction.py`。
- **动作：** 只加载官方本地权重，`eval/inference_mode`，BF16 推理，posterior mean，输出 `[B,128,H/16,W/16]`；禁止额外 patchify和在线梯度。
- **验证：** shape/dtype/range、冻结零梯度、encode/decode、EXIF/RGB、50k–100k latent 统计；2,000 重建集满足 LPIPS/SSIM/人工严重错误门槛。
- **性能：** 按真实 bucket 测 encode latency/显存；记录与 DiT 串行时占比。
- **完成证据：** reconstruction report、latent stats、1GPU profile。
- **GPU：** 必须 1×5090。

### T021：冻结 Qwen wrapper 与七层输出

- **对应文档：** `C03-*`、`C04` hidden layer接口。
- **实现路径：** `encoders/qwen.py`、`tests/gpu/qwen/`。
- **动作：** 纯文本 Qwen、完整 24 blocks、一次前向、`output_hidden_states=true`、取 after `[2,4,8,12,16,20,24]`；`eval/inference_mode/use_cache=false`，无视觉路径、无生成、无 thinking。
- **验证：** checkpoint/tokenizer hash；prefix/suffix由 tokenizer实测；八 dense shape含空条件；冻结参数零梯度；输出 tuple-index 与 block 语义明确；fast kernel 未命中时 hard fail。
- **性能：** 八 shape warmup、latency/显存、padding占比；不得第二次 Qwen forward。
- **完成证据：** golden token/output metadata、kernel trace、1GPU benchmark。
- **GPU：** 必须 1×5090。

### T022：主文本多层聚合与双向 adapter

- **对应文档：** `C04` 主文本分支。
- **实现路径：** `conditioning/text_mixer.py`、`conditioning/bidirectional.py`。
- **动作：** 七个独立 RMSNorm，共享 `2048→1024`；8×128 group、per-token/per-group 7层 softmax；最深层 residual anchor；一个 NoPE 非因果 Attention-only MHA；输出 `1024→2560`。
- **验证：** main indices gather正确、padding不参与 key/query、无 Artist 因果泄漏；gate初始行为、权重和熵；gradient只进入 adapter；dense手工 reference对齐。
- **性能：** 无 layer-axis Transformer/FFN；检查 Python loop、launch 和中间 tensor，记录占 Qwen+condition wall-time。
- **完成证据：** output/gradient golden、mask suite、gate telemetry、profile。
- **GPU：** CPU reference + 1GPU BF16。

### T023：Artist Style Resampler 与 null tokens

- **对应文档：** `C04` Style 分支。
- **实现路径：** `conditioning/style_resampler.py`。
- **动作：** gather `[B,A,7,2048]`；共享 RMSNorm、learned layer embedding、artist×layer flatten、4 learned queries单层 cross-attention、residual MLP `1024→2048→1024`、输出 `4×2560`。
- **验证：** 四个 slot独立；无 Artist/artist dropout/all-condition均使用4个 learned null tokens且始终有效；main 输出不随 Artist segment改变；禁止单向量线性展开。
- **性能：** 变长 Artist mask、零长度边界、无第二次 Qwen/离线 cache。
- **完成证据：** leakage test、null-token test、slot diversity、profile。
- **GPU：** CPU reference + 1GPU。

### T024：Modality、packing、RoPE 与全局 size 接口

- **对应文档：** `C05-*`。
- **实现路径：** `conditioning/modality.py`、`packing.py`、`rope.py`。
- **动作：** 三类 learned modality embedding；`[valid text|4 style|image]`；varlen `cu_seqlens` 隔离；text/style坐标 `(0,0)`；image cell-center面积归一化坐标；head维 `32/48/48`，scale16/theta1000。
- **验证：** modality/packing/RoPE分别单测；不同样本不能attention；Q/K norm在RoPE前；频率共享；不得 repeat KV heads；dense reference同时屏蔽query/key并逐block清零。
- **性能：** 不把 varlen 转成大 dense；记录 packing CPU/GPU成本和 H2D。
- **完成证据：** coordinate golden、dense/varlen tensor contract、cross-sample isolation。
- **GPU：** CPU + 1GPU。

## 10. Phase 3：DiT Reference、目标函数和生产 Attention

### M030：全局条件与 modulation

- **对应文档：** `C07-*`。
- **实现路径：** `conditioning/global_condition.py`。
- **动作：** t 256维、size/aspect各64维固定 embedding；`384→1024→1024` SiLU；共享 `1024→6d` + per-block bias；独立 final `1024→5120`。
- **验证：** size/aspect公式、CFG两支相同；共享 projection和per-block bias均zero-init；final路径独立且zero-init；FP32敏感参数策略。
- **完成证据：** embedding golden、zero-init审计、gradient test。
- **GPU：** CPU + 1GPU。

### M031：Dense SDPA DiT primitives

- **对应文档：** `C06-*`。
- **实现路径：** `model/norm.py`、`mlp.py`、`attention.py`、`block.py`。
- **动作：** RMSNorm eps1e-6/FP32累计/BF16输出；20Q/5KV/head128原生GQA；Q/K head norm；content sigmoid gate；condition residual gates；SwiGLU6912；所有规定 projection bias=false、dropout=0。
- **验证：** shape/dtype/init、forward/backward/gradcheck小尺寸、padding query清零、无KV repeat、gate数学、reference attention。
- **性能：** reference优先可读；先记录 baseline kernel/launch/显存，不做提前融合。
- **完成证据：** dense golden、parameter policy、baseline profile。
- **GPU：** CPU small + 1GPU真实 shape。

### M032：16/20/24 stable-slot DiT 与 output head

- **对应文档：** `C06`、`C07` head、`C09` stable slots。
- **实现路径：** `model/dit.py`、`output_head.py`、`growth.py`。
- **动作：** 固定24 slot canonical FQN、active slot ids；hidden2560；image span conditional RMSNorm；zero-init `Linear(2560,128,bias=true)`；初始化任意输入 `x_pred=0`。
- **验证：** 参数量约1.85B–1.90B；16/20/24只改变active slots；旧FQN稳定；text/style输出不进入head；model-only artifact可推理。
- **性能：** 对256/512真实shape测内存模型，决定 checkpoint mode候选但不在本任务优化。
- **完成证据：** schema/parameter count、zero-output test、stable-FQN manifest。
- **GPU：** 必须1GPU；24L/512只做可承受smoke，不宣称4卡性能。

### M033：Flow、loss、CFG 与 Heun sampler

- **对应文档：** `C08-*`。
- **实现路径：** `objectives/*`、`sampling/*`。
- **动作：** JLT `P_mean=-0.8/P_std=0.8`、noise_scale1、t_eps0.05；`z_t=t*x+(1-t)*epsilon`；网络x-pred；以 `d=max(1-t,0.05)` 对 clean/prediction 分别做 FP32 x-to-v，loss 为 `MSE(x_pred,x)/d^2`；per-sample mean后global mean；velocity CFG2.9。采样 profile 由 M034 独立实现。
- **验证：** t=0 noise/t=1 clean；`x_pred=x` 在 t=0.99 为零 loss；inverse-square 最大权重400；high noise `t<0.95`、low noise `t>=0.95`；cond/uncond分别x-to-v后CFG；旧 objective raw config hash 禁止 resume，model-only 仅标记 `pre_fix` 推理；DDP global mean留到 `T041`。
- **完成证据：** math golden、finite-difference solver tests、sampling determinism。
- **GPU：** CPU + 1GPU integration。

### M034：三档采样 profile 与生成 provenance

- **对应文档：** `C08-007`。
- **实现路径：** `config/schema.py`、`sampling/*`、`eval/spec.py`、`cli/eval.py`。
- **动作：** 固定 `preview=Euler-28/28 NFE`、`balanced=Heun-25 + final Euler/49 NFE`、`reference=Heun-50 + final Euler/99 NFE`；三档仅允许 linear time、FP32 state、noise scale 1、t_eps 0.05、x-pred 与全区间 CFG 2.9。TOML 必须显式选择 profile，不允许提供 NFE 或构造未验证组合；正式评估显式绑定 `reference`。
- **验证：** solver/steps/NFE registry golden；Euler/Heun NFE、endpoint 与 determinism；unknown/missing profile、用户提供 NFE、profile 漂移全部硬失败；resolved config、评估与生成 metadata 记录完整采样身份，旧 objective model-only 推理标记 `pre_fix`。
- **边界：** 不加入 er_sde、DPM++、beta57、resolution shift、多卡路径或正式质量长跑。
- **完成证据：** schema/profile contracts、finite-difference/determinism、短单卡真实 solver smoke、Dense Model 包级 AI/Infra 审查。
- **GPU：** CPU + 1GPU integration；不得外推四卡结论。

### K001：FA4 varlen BF16 GQA 生产后端

- **对应文档：** `C05`、`C06`、`C12` kernel规范。
- **实现路径：** `model/attention.py` backend、`tests/gpu/fa4/`、kernel benchmark。
- **动作：** 安装并锁定 FA4/CuTeDSL；真实执行20Q/5KV varlen forward/backward；禁止静默 fallback和KV复制；dense SDPA只作显式reference/fallback配置。
- **正确性：** 固定batch比较 output/loss/gradient/update；容差来自同backend重复control p99；覆盖所有17 image shapes、最短/最长文本、空条件和跨样本隔离。
- **性能：** warmup/compile与steady-state分开；报告tokens/s、launch、gap、allocated/reserved和error；未快于有效reference或不稳定则阻塞生产。
- **审查：** 独立AI reviewer + 独立Infra reviewer。
- **完成证据：** kernel trace、before/after、数值报告、版本SHA。
- **GPU：** 必须1GPU；4GPU整合留到 `T041`。

## 11. Phase 4：Optimizer、DDP、Checkpoint 与增长

### T040：参数精度、TorchAO AdamW8bit 与 SR RNG

- **对应文档：** `C12-B/D`。
- **实现路径：** `optim/groups.py`、`adamw8bit.py`、`stochastic_rounding.py`、`clip.py`。
- **动作：** 大矩阵param/grad BF16；RMSNorm、gate、标量、style/null、condition、小head FP32；矩阵decay0.01，其余0；单个AdamW8bit `lr=2e-5, betas=(0.9,0.95), eps=1e-8, block_size=256`；BF16 SR；FP32 global clip1.0。
- **验证：** canonical FQN逐参数审计dtype/group/state class/bytes/order/step；optimizer SR RNG与训练RNG隔离；zero-grad与lazy state规则；nonfinite不提交step。
- **canary：** 1,000-step FP32-parameter reference vs mixed+SR，validation loss EMA回退≤3%，无NaN/Inf/state分叉。
- **审查：** 独立AI reviewer + 独立Infra reviewer；optimizer launch占比>5%时必须评估批处理/融合但不得改语义。
- **完成证据：** parameter audit、canary曲线、state hashes、profile。
- **GPU：** 必须1GPU，后续4GPU重复状态一致性。

### T041：单卡训练语义与四卡 DDP

- **对应文档：** `C12-A/B/F`。
- **实现路径：** `distributed/*`、`train/step.py`。
- **动作：** S0原生单卡；S1起同机4卡DDP；冻结Qwen/VAE在DDP/optimizer/checkpoint外；严格global sample mean；各rank训练RNG独立，optimizer SR RNG共同。
- **验证：** 单卡合并batch reference与4卡DDP的loss/gradient/update；model/moments/per-param step/SR state全rank hash一致；无跨rank sample attention；任一rank故障全部退出。
- **性能：** NCCL P2P、reduction/wait/overlap、bucket设置、GPU idle；不允许缩到3卡继续。
- **审查：** 独立AI + Infra reviewers。
- **完成证据：** 1GPU/4GPU equivalence、NCCL profile、state guard报告。
- **GPU：** 当前1GPU只能完成前半；最终必须4×5090。

### T042：Raw checkpoint、恢复与 model-only artifact

- **对应文档：** `C12-E/F`。
- **实现路径：** `checkpoint/schema.py`、`save.py`、`load.py`、`pma.py`。
- **动作：** canonical-FQN sharded Safetensors model；完整TorchAO optimizer sidecar；trainer/data/growth/RNG state；checksum/manifest/temp dir/原子commit/`COMPLETE`；raw/model-only/PMA/release分kind。
- **周期：** 每1,000 successful updates或6小时先到者；finalize、pre/post growth、ramp中点/结束、pre-decay强制保存；保留最近2份滚动raw与所有accepted stage raw。
- **验证：** save→fresh process load→next step对齐uninterrupted；缺失/bitflip/错误ID/dependency hash在forward前失败；checkpoint失败保留上一完整点；model-only不依赖续训sidecar。
- **审查：** 独立AI + Infra reviewers。
- **完成证据：** round-trip diff、故障矩阵、disk/timing报告。
- **GPU：** 1GPU先行，4GPU sharded state必须复验。

### T043：Stage transition 与 16→20→24 增长

- **对应文档：** `C09-*`、`C10` stage顺序、`C12`迁移。
- **实现路径：** `model/growth.py`、`checkpoint/migrate.py`、`train/stage.py`、`cli/transition.py`。
- **动作：** 两次各均匀插入4个随机新slot；旧FQN/state原样保留，新optimizer state空；growth alpha固定半余弦0→1，计划updates的2%，限制1,000–5,000 successful updates；新stage/pass/seed从完整manifest重开。
- **验证：** alpha=0新旧函数等价；new-slot allowlist；无copy/moment copy/LR multiplier；ramp中点/结束checkpoint恢复；失败回滚pre-transition raw且不自动重试。
- **控制：** resume只允许同topology；transition只接受唯一前序；训练程序只写`stage_ready=true`，用户手工finalize/启动。
- **审查：** 独立AI + Infra reviewers。
- **完成证据：** FQN/state migration report、ramp曲线、rollback test。
- **GPU：** 1GPU数学验证；4GPU G1/G2正式canary。

## 12. Phase 5：训练循环、可观测性、评估和性能

### T050：训练 loop、preflight 与硬失败语义

- **对应文档：** `C12-F/G`、全部上游contract。
- **实现路径：** `train/step.py`、`loop.py`、`preflight.py`、`failures.py`、`cli/train.py`。
- **动作：** 成功update作为唯一scheduler/growth/checkpoint计数；microbatch/accumulation；global finite检查和clip；stage预算；no-force preflight；nonfinite/OOM/schema/kernel/backend异常同步停止。
- **preflight：** config/hash、资产、dataset revision、GPU/driver/NCCL/NVMe、冻结零梯度、parameter schema、17 image shapes、八文本shape、zero-update loss、optimizer step、sample、checkpoint round-trip。
- **验证：** 不自动减batch、改accumulation、改backend、改world size、改LR、跳坏shard或从PMA恢复；异常写诊断包且不推进successful update。
- **完成证据：** preflight_report.json、train-step golden、failure-state tests。
- **GPU：** 1GPU完整；4GPU在S1 gate。

### T051：低开销 timing、NVTX、W&B 和本地 durable metrics

- **对应文档：** `C12-G`、`OBS-*`。
- **实现路径：** `telemetry/metrics.py`、`timers.py`、`nvtx.py`、`wandb_sink.py`。
- **动作：** 本地JSONL先落盘、异步W&B默认启用；记录loss、pre/post clip grad norm、clip fraction、LR、nonfinite、dropout命中、samples/tokens/FLOPs、显存、queue、阶段时间。
- **phase：** cache/tar/JSON/caption/tokenize/decode/EXIF/crop/bucket/H2D/Qwen/VAE/condition/DiT fwd/loss/bwd/DDP/clip/optimizer/zero-grad/sample/checkpoint。
- **正确性：** GPU用CUDA events，CPU用monotonic clock；不得每段synchronize；W&B网络失败只进入本地重传队列，不改变训练。
- **性能门槛：** 常驻计时开销<1%；超过则降低采样率但不能删除关键phase。
- **完成证据：** timing schema、overhead benchmark、断网恢复与redaction test。
- **GPU：** 1GPU + 4GPU分别测。

### T052：VAE/Prompt/FID/IS 评估系统

- **对应文档：** `C01`能力优先级、`C02`重建、`C08`正式sampling、`OBS-*`。
- **实现路径：** `eval/*`、`cli/eval.py`、`config/eval.toml`。
- **动作：** checkpoint-driven evaluator；固定prompt/condition/seed/size；正式Heun-50、CFG2.9；raw latest/PMA-10/accepted对比；实现VAE、FID、IS和人工质量索引。
- **调度示例：** 每10,000 successful updates趋势评估10k样本；stage end正式50k样本；全部可配置并经benchmark后修订。
- **可复现：** 锁feature extractor库/模型/preprocess/real-stat hash/prompt manifest/checkpoint/seed/IS splits；趋势和正式artifact分kind。
- **判定：** FID/IS不单独放行；tag控制、审美、NL、构图和严重伪影仍必须人工/任务指标验收。
- **性能：** evaluator GPU占用、训练暂停和99 NFE成本单列；不得无记录争抢训练GPU。
- **完成证据：** deterministic rerun、real-stat manifest、metric artifact、成本报告。
- **GPU：** 必须1GPU；正式调度需与4GPU训练资源方案一并批准。

### T053：Profiler 与端到端 benchmark harness

- **对应文档：** `C12-C/F/G`、`open-items` 6.1/6.2。
- **实现路径：** `telemetry/profiler.py`、`cli/benchmark.py`、benchmark configs。
- **动作：** warmup首次compile与steady-state分开；每候选100 warmup+500 measured，最终24L/512至少1,000 measured；PyTorch Profiler/Nsight Systems抽样，Nsight Compute只看已证实热点。
- **报告：** step p50/p95/p99、phase占比、samples/s、image/text tokens/s、DiT FLOPs/s、GPU active/idle、kernel launch/gap、NCCL、CUDA allocated/reserved、host/pinned RAM、checkpoint摊销。
- **优化门槛：** coarse phase或可融合小kernel累计>5%必须给出优化/保留理由；regional compile只有端到端稳态≥3%且无recompile/fallback才启用。
- **公平性：** before/after同checkpoint、数据序列、shape分布、batch、硬件、软件锁；不能靠减少token/功能或未披露增显存换速度。
- **完成证据：** perf_baseline/after、trace索引、resolved hash。
- **GPU：** 1GPU harness；最终4×5090。

### T054：系统与训练故障注入

- **对应文档：** `C11`恢复、`C12-F`故障矩阵。
- **实现路径：** `tests/fault_injection/`、故障驱动CLI。
- **注入：** 下载中断、截断shard、token失效、checksum错、worker退出、磁盘写满；microbatch/DDP reduction/optimizer/checkpoint各阶段杀进程；nonfinite、OOM、SR RNG分叉、NCCL rank failure。
- **验收：** 只恢复上一`COMPLETE`；所有rank同步停；完成shard不重读、active shard从头；不得自动更改batch/backend/world size/optimizer/LR/checkpoint频率。
- **审查：** 独立AI + Infra reviewers。
- **完成证据：** 故障矩阵逐项pass、replay计数、恢复parent ID。
- **GPU：** 1GPU子集；完整DDP/NCCL必须4GPU。

## 13. Phase 6：Stage 配置填充与正式 canary

### S000：目标机容量与 stage overlay 填充

- **对应文档：** `C10-D/E`、`C12-F`。
- **依赖：** `T053` benchmark完成，dropout已决定。
- **动作：** 为S0/S1/G1/S2/G2/S3填写local/global batch、accumulation、checkpoint mode、valid samples/data passes、actual DiT FLOPs、successful updates、checkpoint slots和wall-time预测；H1/H2继续disabled。
- **验证：** 相邻stage只变一个主轴；transition前序唯一；配置无placeholder、隐式默认或未知key；每个resolved config有hash。
- **完成证据：** stage budget表、resolved configs、容量审查。
- **GPU：** 1GPU填S0；S1以后必须4GPU benchmark。

### S001：S0 单卡 16L/256 canary

- **进入门槛：** 全部P0 contract、Qwen/VAE、dense/FA4、optimizer 1,000-step canary、单卡checkpoint/fault tests通过。
- **执行：** zero-update→1 step→200 updates→1,000 successful updates→耐久窗口，保持真实在线数据/Qwen/VAE/DiT/optimizer。
- **验收：** loss/grad/clip稳定；无NaN/OOM/swap；checkpoint fresh-load next-step对齐；数据/验证隔离；W&B与本地metrics一致；用户手工批准S0。
- **证据：** stage report、AI/Infra review、固定sample、timing/profile、accepted raw checkpoint。
- **GPU：** 当前1×5090可执行。

### S002：S1 四卡 16L/256 canary

- **唯一变化：** world size 1→4，新stage/pass/seed。
- **执行：** 4GPU preflight、NCCL/P2P、global mean/state hash、200–1,000 successful updates、checkpoint restore与rank failure。
- **验收：** all-rank model/moments/step/SR hash一致；训练RNG rank-local；数据供给≥12 samples/s、wait<2%；无三卡fallback。
- **GPU：** 必须4×5090；当前环境不可关闭此任务。

### S003：G1/S2/G2/S3 顺序放行

每个stage单独创建实现任务和审查任务，不能复用上一stage结论：

| Stage | 唯一主要变化 | 附加验收 |
|---|---|---|
| G1 | 16→20L | alpha=0等价、growth ramp、mid/end checkpoint、post-ramp稳定 |
| S2 | 256→512 | 17 buckets、none/alternating/all checkpoint比较、完整吞吐/显存 |
| G2 | 20→24L | 重复完整growth协议，512下稳定和恢复 |
| S3 | 24L/512收尾 | 1,000-update endurance、PMA稳定窗口、正式质量与恢复 |

生产硬门槛：数据≥12 samples/s；完整20/24L 512四卡≥6 samples/s；低于4停止，4–6只优化；每卡峰值≤27.2 GiB；ready wait<2%；无OOM/swap/nonfinite自动续跑或state分叉。达到门槛只写`stage_ready=true`，仍由用户手工finalize。

### S004：H1/H2 可选高分辨率

- **H1 768：** 仅从已接受512 raw手工启动，独立数据覆盖、显存和吞吐benchmark。
- **H2 1024：** 优先从accepted H1进入；若从S3直升必须新决策。
- **保持关闭：** iREPA、FSDP2、在线EMA、latent/text cache和复杂solver，除非profile或硬门槛触发重新决策。

## 14. 验证层级和最小测试矩阵

| 层级 | 每任务要求 | 不可替代的证据 |
|---|---|---|
| Static/CPU | ruff、pyright、unit、schema/golden、determinism | 不能证明GPU kernel正确 |
| 1GPU smoke | load、forward、backward、update、峰值显存 | 不能证明四卡DDP/NCCL |
| 1GPU numeric | reference vs production output/loss/grad/update | 不能只检查import或shape |
| 1GPU perf | warmup、steady profile、17×8 shape边界 | 不能外推四卡吞吐 |
| 4GPU correctness | global mean、state hashes、resume、rank failure | 必须真实4×5090 |
| 4GPU endurance | 500–1,000 measured updates、fault、checkpoint | stage间不能复用 |
| Quality | 固定prompts、VAE集、FID/IS、人工tag/美学/NL | loss或FID单项不能放行 |

所有数值容差必须先通过同backend重复运行建立control p99，再登记到测试配置；不得临时放宽直到通过。

## 15. 执行顺序和阻塞关系

```text
已完成：Notion迁移/MCP移除/.env安全边界
  ↓
R001 → R002 → D001 → C001
                         ↓
D010 → D011 → D012 → D013 → D014 → D015 → G001 → D016
                                              ↓
T020 + T021 → T022 + T023 → T024
  ↓
M030 → M031 → M032 → M033 → M034 → K001
  ↓
T040 → T041 → T042 → T043 → T050 → T051/T052/T053 → T054
  ↓
C002/S000 → S001 → S002 → G1 → S2 → G2 → S3
  ↓
可选 H1 → H2

A001：只保留已完成的最小本地资产边界；A002 重型审计保持撤销，不参与上述依赖链。
```

允许并行的只有不写同一文件且接口已冻结的任务，例如T020与T021、T022与T023、T051与T052。实现和审查不得并行；上游contract未验证时不得提前写生产优化。

## 16. 开工前仍需用户决定

D016 已关闭唯一 caption 架构输入阻塞项。批准值为 `all_condition/general/artist/copyright/nsfw=0.1`、`character=0.2`、`candidate_source=0.3`，五个 NL key 均为 `0.3`；后续仍需 100k 生产 dry run 验证实际分布，但它不再是数值决策。

FID/IS的10k-update/10k趋势样本和stage-end/50k正式样本是新增示例配置，不是既有绝对质量阈值。首次accepted baseline后再决定是否建立数值放行阈值，不能在实现前杜撰。

## 17. 路线图完成定义

只有当以下项目全部存在时，才可从“规划”进入“实现”：

- 本路线图通过独立AI与Infra审查，差异已修正。
- `R001` 建立Git基线，secret和大资产不进入历史。
- `D001` 建立可机检的requirement追踪。
- D016 已将用户 dropout 决定绑定到严格配置并通过针对性测试。
- 当前单卡/未来四卡资源边界被明确，不把1GPU证据外推为4GPU结论。
