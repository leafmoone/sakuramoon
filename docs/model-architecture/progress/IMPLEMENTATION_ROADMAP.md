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
| 存储 | 工作区为 400 GiB NFSv3；D026 探测时可用 383,347,851,264 bytes | 用户已批准显式 server-backed 模式；锁定 mount identity、原子发布、小型有界 cache、三份实测 checkpoint 余量和 `/run` 本地 IPC |
| 参考工程 | `reference/` 仅供人工理解/对照，可完全不使用 | 根仓忽略；任何代码、测试、preflight、训练或运行时不得导入、执行或调用其中代码 |
| 凭据 | `.env` 已写 `MODELSCOPE_API_TOKEN`，权限 600 | `.env` 永不进 Git；resolved config、日志和 W&B 必须脱敏 |

目标配置仍要求 4×RTX 5090、14 vCPU、约 120 GiB host RAM 和足够的受治理 server-backed storage。正式四卡任务不得用当前单卡结果替代。

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
| `run/paths/storage/security` | run/stage/seed、目录、server-backed mount/runtime identity、容量、secret env 名称、脱敏规则 |
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
- **当前边界：** 全部 dropout 与 Text/Style 决策均已锁定；C002 只负责 production config、overlays 与 assembly binding。S000 benchmark 值继续使用 loader 硬拒绝的显式 sentinel，不得以 synthetic validation identity 冒充生产 stage 放行值。
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
- **现行覆盖：** 本任务的 300–500 GiB 本地 cache 假设已被 D026 server-backed 决策取代；D012 历史实现与证据不回写。
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

### D023：Parent-coordinated persistent two-worker shard pipeline

- **依赖：** D022 schema v3 multi-active state 与 D021 trusted `ShardRecord`/metadata/collate 合同。
- **实现路径：** `data/pipeline.py`、`data/collate.py`、D023 multiprocessing/fault contracts、task/test evidence 与 trace registry。
- **动作：** 父进程独占 state/cache coordinator，先激活并 prepare shard，再通过每 worker 容量 1 的输入队列交给精确两个 persistent DataLoader workers；ready output 与 completion channel 显式有界。
- **完成语义：** worker 正常耗尽且父进程收到 ordered done 与 completion 后才逐 shard complete；worker 异常、`os._exit` 或父迭代器提前关闭均保留 active，restart 从 shard 起点重放。
- **恢复：** worker topology 必须精确匹配 schema v3；所有 recovered active 全部重新 prepare 后才能激活新 shard；全部 active 继续防 cache eviction。
- **完成证据：** 两个不同 worker ID/PID 跨多 shard 复用、真实 worker exit、父 close、精确 replay、reprepare barrier、bounded channel、ruff/pyright 与 D023 test report；Data 包级复审仍 pending。
- **GPU：** 无；production cold-cache throughput/RSS/ready-wait sweep 保持 pending。

### D024：Process-isolated dataset supply service

- **依赖与覆盖边界：** 复用 D010 完整 shard 校验/原子发布、D012 LRU、D022 schema-v3 active/completed/replay 基础语义和 D023 persistent worker completion 合同。D024 只替换生产 ownership：训练父进程不再拥有 cache、tar 顺序或 shard state，改由单机唯一独立 data service 持有；不得回写 D022/D023 历史证据，也不得改变 trusted `ShardRecord`、validation exclusion、caption/image/collate 或 whole-shard at-least-once 语义。
- **独立进程：** 新增独立启动、独立存活的 data-service CLI/process。它独占 ModelScope 网络、token 解析、`.partial`、bytes/SHA-256、fsync/原子发布、cache catalog、LRU/eviction、`mainset` 和 active/completed/replay state。trainer 不得 spawn、restart、stop 或以内联路径替代该 service；service 不可用时 preflight/训练硬失败，禁止退回训练进程或 DataLoader worker 下载。
- **`mainset` 合同：** service 每轮从 immutable training manifest 读取全部 tar path，每个恰好一次，生成新的随机排列并原子持久化 `mainset_id`、manifest identity、shuffle identity、精确 ordinal 顺序和逐行状态。只有 service 能推进或恢复 cursor；trainer、checkpoint、stage、resolution/model growth、worker topology 和 resume 请求都不得携带、选择或改写 tar order/position。
- **供给协议：** service 严格按当前 `mainset` ordinal 下载、校验、原子发布并租约保护 tar；在 worker 消费 active `A/B` 时并发准备后续 `C/D/E...`，内部保持有界 verified lookahead。它通过本机有界 IPC 向轻量 client 只发已经完整验证的 immutable `ShardRecord + absolute local path + lease identity`；训练侧只按 service 给出的顺序消费，不枚举、选择、下载、哈希、扫描或清理 shard。
- **完成与轮换：** DataLoader worker 只读本地 tar、产生 batch 和 ordered done；父进程 client 只转发 normal-exhaustion completion ACK，不解释或修改 state。service 收到匹配 lease/worker identity 的 ACK 后才逐 tar complete；worker/service/client/trainer 异常、断连、中断或 ACK 缺失时保持 active，service restart 从 tar 起点 replay。只有当前 `mainset` 的全部 tar 已下载、验证、按序供给且所有 outstanding lease 正常完成后，service 才可原子关闭并删除旧表、创建下一份完整 manifest 随机 `mainset`；仅下载完但仍有 lease 时不得删表或供给下一轮。
- **checkpoint 解耦：** service state 独立持久化且绝不快照、复制或引用到 raw checkpoint。trainer resume 只恢复训练与 optimizer state，然后连接 service 并消费 service 当前给出的 tar；人工暂停/续训、stage、分辨率和模型增长都不要求恢复数据位置、同一 tar 或同一 next batch。production checkpoint schema 的独立 remediation 由 T044 关闭，D024 不修改 T042 历史证据。
- **有界与空间控制：** 实际接通严格 TOML `data.cache.download_concurrency`，新增无默认值的 `data.cache.verified_shard_lookahead`；service 内部 in-flight download、verified-ready shard、lease output 与 ACK channel，以及 DataLoader worker input/ready batch/completion channel 分别有界。每个下载按 manifest bytes 预留空间，quota 同时计算 published shard、in-flight reservation 与 manifest-owned `.partial`；active lease 全部防 eviction，inactive lookahead 可按 LRU 淘汰，不得预下载整个 dataset。
- **关键路径隔离：** trainer 和 DataLoader worker 禁止 import/call transport、`fetch_dataset_shard`、SHA、cache eviction 或 partial cleanup。已验证 cache hit 的重新校验、启动 orphan `.partial` 清理和所有完整文件扫描都只在 service 内完成；shard IPC 每个 lease/完成各一次，绝不按 sample/batch/update 往返。训练热路径只能在 lookahead 耗尽时等待 descriptor，不能同步执行任何下载/校验工作。
- **配置清理：** 现有 `range_workers` 不得继续作为无效果的必填字段；本任务必须依据固定 revision 的真实 ModelScope 合同，要么在 service 内实现有界 Range/断点续传及精确重组校验，要么通过受治理配置变更删除该字段。两条路径都必须保持完整 tar 校验后才可发布 lease。
- **性能门槛：** 启动成本与稳态分开报告；正式计时前先由 service 填满配置的 verified lookahead。稳态 cold-cache 并发下载必须达到数据供给 `>=12 samples/s`、ready wait `<2%`、无 swap/无界 RSS/quota 越界，并与相同 workload 的 fully-cached control 比较 trainer step p50/p95/p99、GPU active/idle 和 DataLoader batch latency；超出预先登记的 same-backend control 波动即不放行。若共享 CPU/server-backed storage 无法隔离到该门槛，必须调整并显式锁定 service CPU/I/O/concurrency 配置或使用独立存储，不能声称“零影响”。
- **验证：** 用独立真实 service process 和可控慢 transport 证明 worker 消费 `A/B` 时 `C/D/E` 已 ready/in-flight，trainer 进程没有网络/SHA/cache stack；验证每轮全 manifest permutation、每 tar 恰好一次、持久化 ordinal、IPC 身份/容量、精确并发/lookahead/bytes 上界、active 不 eviction、inactive 可 eviction、cache hit、损坏/截断/中断/ENOSPC/orphan partial、service/trainer/worker 分别退出、ACK 丢失、recovered-active barrier、mainset 收尾与下一轮原子创建。另验证 trainer checkpoint/restart 不读取或改变 service cursor。随后做 bounded 真实网络/server-backed storage/1GPU overlap smoke；两小时冷缓存 sweep 在 Data 里程碑集中执行一次。
- **完成边界：** D024 保持独立 task、diff、测试、trace、证据和原子 commit。D024 完成后只启动一个新的 DATA 包级 reviewer；D024 与该 reviewer 关闭前，`T044/T050` 不得关闭 production resume/data-to-train 接线，`S000` 不得锁定 production service/download/lookahead/worker/queue 参数。
- **GPU：** service 本身仅 CPU/网络/server-backed storage；使用 1GPU consumer 做 bounded 隔离/吞吐 smoke，不做长跑或正式 stage。

### D025：Governed production data assembly

- **依赖：** D024 service client、D023 exact two-worker contract 与 D021 trusted shard/parser/collate boundary。
- **动作：** 生产 factory 是 loader controls 与 parser/exclusion policy 的唯一来源；签发 process-local accepted batch-stream handle，绑定 resolved config、manifest/service session、worker topology 与精确 factory identity。普通 iterator 或调用者构造的 `TrainingBatch` 不能进入 production T050。
- **验证：** callable spawn-serializability 硬失败；真实 AF_UNIX service → 两个 spawned workers → accepted stream；正常完成逐 lease ACK，提前 close/worker failure 保持未确认 lease active 并在 restart 从 shard 起点 replay。
- **完成边界：** 独立 task/diff/test/evidence/commit；D023/D025/D026 完成后才启动新的 DATA package rereview。

### D026：Governed server-backed storage and host-local IPC

- **依赖：** 用户批准当前 NFSv3 server-backed 模式；复用 D024 persistent mainset/cache 与 T044 实测 raw checkpoint bytes，不修改其历史证据。
- **动作：** 严格 `[storage]` 无默认配置锁定 NFS filesystem/source/version/hard mount、实际 reserve、三份 checkpoint 与 atomic probe；cache 取消 300 GiB 下限但保持显式 bounded high/low。AF_UNIX socket 与 singleton lock 固定在非 NFS 的 `/run/sakuramoon/`。
- **验证：** 全部 persistent paths 同一 mount identity；同目录 write/file-fsync/replace/directory-fsync/readback；free space 覆盖 cache high-watermark + `3 × measured raw checkpoint` + reserve；身份、容量、probe、runtime path 漂移全部硬失败。
- **完成边界：** targeted CPU 与当前真实 NFS probe；不运行长跑、正式 stage 或多卡。D026 结论并入新的 DATA package rereview。

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
- **历史完成范围：** canonical-FQN sharded Safetensors model；完整TorchAO optimizer sidecar；trainer/data/growth/RNG state；checksum/manifest/temp dir/原子commit/`COMPLETE`；raw/model-only/PMA/release分kind。该实现与证据保持不回写，其中 data-state sidecar 已被 D024/T044 新决定取代，不能直接作为后续 production schema。
- **周期：** 每1,000 successful updates或6小时先到者；finalize、pre/post growth、ramp中点/结束、pre-decay强制保存；保留最近2份滚动raw与所有accepted stage raw。
- **验证：** save→fresh process load→next step对齐uninterrupted；缺失/bitflip/错误ID/dependency hash在forward前失败；checkpoint失败保留上一完整点；model-only不依赖续训sidecar。
- **审查：** 独立AI + Infra reviewers。
- **完成证据：** round-trip diff、故障矩阵、disk/timing报告。
- **GPU：** 1GPU先行，4GPU sharded state必须复验。

### T043：Stage transition 与 16→20→24 增长

- **对应文档：** `C09-*`、`C10` stage顺序、`C12`迁移。
- **实现路径：** `model/growth.py`、`checkpoint/migrate.py`、`train/stage.py`、`cli/transition.py`。
- **动作：** 两次各均匀插入4个随机新slot；旧FQN/state原样保留，新optimizer state空；growth alpha固定半余弦0→1，计划updates的2%，限制1,000–5,000 successful updates。transition 不控制 data-service `mainset` 或 tar cursor，service 继续当前代次与顺序。
- **验证：** alpha=0新旧函数等价；new-slot allowlist；无copy/moment copy/LR multiplier；ramp中点/结束checkpoint恢复；失败回滚pre-transition raw且不自动重试。
- **控制：** resume只允许同topology；transition只接受唯一前序；训练程序只写`stage_ready=true`，用户手工finalize/启动。
- **审查：** 独立AI + Infra reviewers。
- **完成证据：** FQN/state migration report、ramp曲线、rollback test。
- **GPU：** 1GPU数学验证；4GPU G1/G2正式canary。

### T044：Service-decoupled raw checkpoint 与恢复合同

- **依赖与范围：** 依赖 T042 已完成 raw/model-only/PMA 基础和 D024 冻结的 service client 合同。本任务只去除 checkpoint 对 data state 的所有 schema/API/manifest 绑定并复验恢复；不实现 data service，不回写 T042 历史 task/review/evidence。
- **实现路径：** `checkpoint/schema.py`、`save.py`、`load.py`、resume/preflight 接线、targeted CPU/1GPU checkpoint tests、T044 task/review evidence 与 trace registry。
- **动作：** production raw 只保存 model parameters、完整 TorchAO optimizer state、scheduler/growth、trainer 与 successful-update/sample counters、训练 RNG、optimizer-SR RNG、resolved config 和 checkpoint identity；明确拒绝 `mainset_id`、tar cursor/order、active/completed/replay、cache/lease、prefetch/queue 或任意 opaque data sidecar。既有含 data state 的旧 schema 必须在 forward 前按显式版本合同拒绝或经受治理迁移，禁止忽略未知 sidecar 后静默加载。
- **恢复验证：** save→fresh process load 完整恢复训练与 optimizer state；使用同一显式固定输入 batch 比较 uninterrupted 与 resumed 的 output/loss/all-gradient/clip/update/optimizer state/RNG。随后单独证明 resume 连接 D024 service 当前 cursor，既不请求旧数据位置，也不要求 live next batch 相同。
- **审查：** checkpoint 高风险边界，保持独立实现、独立 AI reviewer、独立 Infra reviewer、独立 diff/test/trace/evidence/原子 commit。
- **GPU：** targeted CPU + 1GPU；4-rank sharded restore 仍保持 blocked，不能由单卡证据关闭。

## 12. Phase 5：训练循环、可观测性、评估和性能

### T050：训练 loop、preflight 与硬失败语义

- **对应文档：** `C12-F/G`、全部上游contract。
- **数据依赖：** D024、新的 DATA 包级复审和 T044 独立双审必须先关闭；正式 CLI 先完整恢复 checkpoint 中的训练/optimizer state，再从 resolved config 连接并鉴别已运行的 data service、构造轻量 data client、persistent workers 与 batch consumer。trainer 禁止读取/恢复 service cursor，禁止构造 transport/cache、直接加载 shard state 或保留测试专用内联下载组装。
- **实现路径：** `train/step.py`、`loop.py`、`preflight.py`、`failures.py`、`cli/train.py`。
- **动作：** 成功update作为唯一scheduler/growth/checkpoint计数；microbatch/accumulation；global finite检查和clip；stage预算；no-force preflight；nonfinite/OOM/schema/kernel/backend异常同步停止。
- **preflight：** config/hash、资产、dataset revision、GPU/driver/NCCL、server-backed mount/capacity/atomic publication 与 host-local IPC、冻结零梯度、parameter schema、17 image shapes、八文本shape、zero-update loss、optimizer step、sample、checkpoint round-trip。
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
- **验收：** 只恢复上一`COMPLETE`训练/optimizer state；所有rank同步停。data service 独立保证完成shard不重读、active shard从头，trainer restart 不恢复其位置；不得自动更改batch/backend/world size/optimizer/LR/checkpoint频率。
- **审查：** 独立AI + Infra reviewers。
- **完成证据：** 故障矩阵逐项pass、replay计数、恢复parent ID。
- **GPU：** 1GPU子集；完整DDP/NCCL必须4GPU。

## 12.5 Phase 5.5：人工训练边界与离线逐部性能优化

本阶段不拥有训练生命周期，也不会在单卡实现刚完成时自动启动。必须先完成 Data、Encoders/Conditioning、K001、Dense Model、optimizer、checkpoint、训练 loop、telemetry 和单卡故障子集的当前 1GPU 实现与复审并填好 `S000` 单卡配置；随后由用户手工启动未优化 eager `S001`、观察训练是否正确，并自行决定 checkpoint 与停止时机。只有用户明确提供可完整续训的 raw checkpoint、确认 canonical 训练已停止并释放任务所需 GPU/NVMe 资源后，才执行 `P060-P067`。所有优化保持独立 task、diff、测试、证据和原子 commit；每个 task 的允许路径必须包含 `docs/model-architecture/progress/traceability.toml`，但只更新实际受影响的稳定 requirement ID。实现与审查不得并行，每个 task 都需要独立 AI/模型正确性和 Infra/性能审查。

人工控制与训练状态保护协议固定如下：

- 用户用生产 resolved config 和真实在线 Data/Qwen/VAE/conditioning/DiT/loss/backward/clip/optimizer 路径手工启动训练；zero-update、首个 successful update 和有界稳定窗口只提供 loss/grad/clip、OOM/swap、数据队列与 telemetry 证据，不触发自动暂停或优化。用户选择用于优化的 successful update 边界并记为 `N`。
- 用户可以选择一份既有 `COMPLETE` raw checkpoint，或在 chosen update `N` 手工请求 T044 修订后的 production raw 协议保存；程序不得替用户选择 `N`、自动停止 trainer 或改写周期 checkpoint 频率。checkpoint 必须覆盖 model parameters、TorchAO optimizer state、scheduler/growth state、trainer 与 successful update/sample counters、训练 RNG、SR RNG、resolved config 和 checkpoint identity；不得包含任何 data-service state，model-only 或 PMA checkpoint 不得作为优化断点。
- 用户手工停止 canonical trainer 并明确释放资源后，`P060-P067` 才能启动。优化任务不得向训练进程发送 signal/控制命令，不得自动 pause/resume，不得改写 canonical checkpoint、训练输出目录、data/cache state、W&B run 或 durable metrics；若训练仍占用同一 GPU 或 NVMe 路径，优化任务必须等待，不能抢占资源。
- `P060-P067` 的 eager/candidate 测试只使用 checkpoint `N` 的只读身份和任务私有副本；测试 update、compile cache、profile、trace 和临时输出全部进入任务允许的隔离路径，不计入或回写正式训练状态。
- `P067` 只交付 accepted optimization manifest、兼容性证据和人工续训命令所需的 resolved identity，不执行正式 update。由用户决定是否采用优化结果以及何时手工 fresh-load checkpoint `N`；采用时从 `N+1` 继续，不采用时以 eager 路径从同一 `N+1` 继续。successful update、sample、scheduler/growth 和 RNG 状态不得重置、跳步或重复记账；data service 保持独立，不被优化流程回退或改写。
- 优化若改变 parameter layout、canonical FQN、optimizer state 或 checkpoint schema，必须提供显式、原子的版本迁移，并证明旧 checkpoint 导入、round-trip 和 eager/optimized `N+1` 对齐；否则拒绝该优化，不能牺牲既有训练状态。

工期按 profiler 实际热点和保留候选计算，不要求为了“完成优化”实现无收益 kernel：

| 范围 | 预计工程时间 | 适用条件 |
|---|---:|---|
| eager 启动证据与 `COMPLETE` checkpoint 验收 | 0.5-1 个工作日 | 仅为技术验收工作量；实际启动、checkpoint 和停止时机由用户决定 |
| 只完成全链路审计，少量或没有候选被保留 | 7-10 个工作日 | profiler 未发现足够热点，候选以正确性/无收益证据关闭 |
| 常规选择性 compile、算子、fusion、kernel 与 pipeline 优化 | 15-25 个工作日，约 3-5 周 | 预计路径；包含 targeted 1GPU 正确性、性能测试和独立复审 |
| 深度自定义 Triton/QKV/optimizer kernel | 25-40 个工作日，约 5-8 周 | 仅在热点与端到端收益证明值得实现时进入 |

任务级计划量为 `P060` 1-2 天、`P061` 2-4 天、`P062` 2-5 天、`P063` 3-6 天、`P064` 3-6 天、`P065` 2-4 天、`P066` 2-4 天、`P067` 2-3 天；这些范围不是必须全部相加，profile 无收益的候选应以证据关闭。GPU 排队、上游修复、依赖构建和四卡门槛不包含在上述单卡工期内。

共同正确性门槛固定如下：

- before/after 使用上述同一 `COMPLETE` eager checkpoint `N`、resolved config、显式固定 correctness batch/shape 序列、batch/accumulation、RNG 状态、软件锁和硬件；禁止把 live service tar 连续性作为正确性前提，也禁止通过减少 token、关闭功能、改变精度合同或未披露增加显存换速度。真实 service 只用于单独的端到端 overlap/吞吐比较。
- 先以 eager 和 same-backend repeat 建立预先登记的数值容差，再比较模块 output、per-sample/mean loss、全部 canonical-FQN 参数梯度、clip coefficient、一次 optimizer update、optimizer state、checkpoint round-trip、fresh-process resume next step；不得在看到失败后放宽容差。
- 涉及 varlen/attention/routing 的 task 额外验证 malformed boundary 硬失败、跨样本隔离、全部有效 token 路由和无 per-block D2H；涉及 custom kernel 的 task 必须覆盖真实 RTX 5090 forward/backward，不能用 import、shape 或 mock 代替。
- 正确性、失败语义、checkpoint 兼容性、无 silent fallback、无 measured-window compile/recompile/fallback 任一失败时，不运行或不采信性能结论。单项优化只有在可复现 component gain 且端到端无 p50/p95/p99、显存、RSS/swap 或 checkpoint 摊销回退时才能保留。
- 每项只运行 targeted CPU/1GPU 验证；17 image buckets × 8 text shapes、完整组合 benchmark 和恢复矩阵只在 `P067` 集中运行一次。1GPU 结果仍不得关闭 4GPU DDP/NCCL、正式 stage 或 regional compile 的四卡放行门槛。

### P060：冻结 eager 正确性 oracle 与端到端热点基线

- **依赖：** `D024`、`T020-T024`、`K001`、`M030-M037`、`T040-T044`、`T050-T054` 的当前 CPU/1GPU 实现与必要复审完成；`S000` 单卡 resolved config 已冻结，用户已手工启动并检查 `S001`、指定 `COMPLETE` raw checkpoint `N`，fresh-process 恢复门槛通过，并明确确认 canonical trainer 已停止且优化资源可用。
- **动作：** 以 T051/T053 计时和 profiler 合同冻结 eager baseline；逐 phase 记录 data wait、H2D、Qwen、VAE、conditioning、packing/RoPE、DiT forward、loss/backward、clip、optimizer、checkpoint 的 p50/p95/p99、GPU active/idle、kernel launch/gap、allocated/reserved 和 host/pinned RSS。
- **正确性：** 发布不可变 workload identity、eager output/loss/all-gradient/update/resume oracle 和 same-backend repeat p99；本任务不引入优化代码。
- **完成证据：** `perf_baseline.json`、热点排序、trace 索引、正确性 oracle、候选/不优化理由矩阵。

### P061：Regional `torch.compile` 候选

- **范围：** 只编译 profiler 证实的稳定 tensor regions，优先重复 DiT block 的 norm/modulation、gate/residual、SwiGLU pointwise 与 output head；Data、Python packing、packed-entry boundary 验收、checkpoint 和故障控制保持 eager。FA4 作为显式 opaque/custom-op 边界，不由 Inductor 重写。
- **验证：** graph break、guard、dynamic varlen、growth alpha、stride、autograd 和 cache 逐项检查；warmup 后 measured window 必须零 compile、零 recompile、零 fallback。Inductor cache 使用任务允许的 NVMe 路径，不依赖 `/tmp`，不进入 Git。
- **放行：** 先通过共同 output/loss/all-gradient/update/resume gate；只有端到端稳态提升 `>=3%` 才保留候选。单卡任务不得把 `compile.regional_enabled` 改为生产开启；正式启用仍等待 hash-bound 4GPU correctness、DDP 和 resume 证据。
- **完成证据：** eager/compiled golden、compile counters、graph-break/recompile 报告、`perf_baseline.json`/`perf_after.json` 和 retained/rejected 结论。

### P062：算子选择、GEMM 与数据布局优化

- **范围：** 审查 Q/K/V/content-gate projections、SwiGLU gate/up/down、condition projections、contiguous/cast、中间 tensor 生命周期和 grouped/packed GEMM；优先使用锁定 PyTorch/cuBLASLt/Triton 能复现的正式 API。
- **约束：** 不改变 hidden/intermediate/head 数、GQA、bias/dropout、参数精度或计算顺序。QKV/QKVG 参数打包若改变 canonical FQN、optimizer state 或 checkpoint schema，必须作为本 task 内显式迁移协议并证明旧 checkpoint 导入、round-trip 和 next-step 对齐；没有收益证据时保持原布局。
- **验证：** 每个候选分别比较 forward、loss、全部输入/参数梯度、一次 update、allocated/reserved、kernel 数和 gap；不得把 dense reference 或 KV repeat 引入生产路径。
- **完成证据：** operator/layout matrix、GEMM trace、checkpoint compatibility report、before/after benchmark。

### P063：RMSNorm、modulation、SwiGLU 与 residual fused operators

- **范围：** 依次评估 `FP32 RMSNorm + sample modulation gather + affine`、`gate gather + growth + residual`、`SiLU(gate) * up`、content gate multiply 和 final-head norm/modulation 融合。先验证 P061 自动 fusion；只有 profiler 仍显示热点时才实现本地 Triton/custom op。
- **正确性：** 保留 RMSNorm FP32 累计/BF16 输出、`eps=1e-6`、scale/shift/gate 顺序、growth switch 和 padding/varlen 语义；custom op 必须有明确 backward、非连续/边界 shape contract、grad reference 和异常硬失败，不允许 forward-only 加速。
- **验证：** 逐 fused operator 做 eager golden、forward/backward、全参数梯度、update、same-backend repeat、峰值显存和 kernel launch/gap；再做多 block 小型 smoke。
- **完成证据：** fused-op contract matrix、Triton/Inductor trace、numeric report、before/after benchmark。

### P064：Q/K Norm、2D RoPE 与 FA4 周边 kernel 优化

- **范围：** 评估 Q/K FP32 RMSNorm、`32/48/48` NoPE/y/x split、共享 frequency rotation、contiguous materialization和 FA4 前后 content-gate epilogue的融合；不得重新实现或替换已锁定 FA4 attention 核心。
- **正确性：** 固定 Q/K norm-before-RoPE、V 不归一化、20Q/5KV、head_dim 128、native GQA、BF16、非因果、`pack_gqa=true`、跨样本隔离和 accepted boundary identity。任何 upstream kernel/库必须固定版本、wheel hash和可治理 commit/license provenance，禁止静默 fallback。
- **验证：** dense numerical reference、真实 FA4 output/loss/all-gradient/update、malformed/mutated boundary negative、17×8 之外的 targeted 极值 shape、多 block timing/memory/profile；禁止 per-block D2H。
- **完成证据：** kernel contract/provenance matrix、FA4 reference comparison、profiler kernel/gap 和 before/after benchmark。

### P065：Gradient、finite/clip 与 optimizer fused update

- **范围：** 评估 multi-tensor FP32 finite+sumsq+global norm、一次 coefficient scale、TorchAO AdamW8bit update kernel grouping和 SR RNG 状态切换开销；优化目标以 P060 中真实端到端占比为准，不从 isolated zero-gradient timing 外推。
- **正确性：** 保留 global sample mean、FP32 norm、clip=1.0、nonfinite硬失败、BF16 matrix/FP32 sensitive parameter policy、256-block 8-bit moments、stochastic rounding、参数分组/FQN、attempted/successful update计数和 serialized bitwise next-step合同。
- **验证：** 所有参数 finite/norm/clip、全部 optimizer state bytes/class、held-out EMA ratio、one-step update、checkpoint/resume RNG/state；单卡仅关闭本 task，四 rank global mean/state equality仍属于 T041/S002。
- **完成证据：** multi-tensor golden、optimizer state/update report、RNG/resume report、before/after profile。

### P066：Data、Qwen/VAE、conditioning、packing 与 H2D overlap

- **范围：** 逐 phase 复审 D024 process-isolated data service/read-ahead/IPC、D023 worker/queue、cold-cache NVMe、pinned memory/nonblocking H2D、Qwen length buckets、VAE、Text/Style heads、packing/coordinates/sample IDs和安全的跨 batch CPU/GPU overlap；默认同GPU Qwen/VAE/DiT串行，只在 profiler 证明收益且显存有界时评估 stream overlap。P066 不负责补做 D024 核心能力。
- **约束：** 不增加跨 batch text/latent/activation cache，不改变 caption/dropout、在线 Qwen/Mage-VAE、D024 `mainset`/lease/worker_count/state/cache语义、batch/accumulation、token上限或 backend；queue、prefetch、compile cache 和临时 buffer始终有界。
- **验证：** service `mainset` 顺序/lease/replay、训练 RNG/validation exclusion、worker exit/restart、Qwen七状态、VAE posterior mean、conditioning output/grad、packing隔离和固定-batch end-to-end output/loss/update保持一致；报告 ready wait、H2D、RSS/swap、显存和吞吐。
- **完成证据：** phase review matrix、cold/warm-cache report、overlap timeline、correctness/fault regressions和 before/after benchmark。

### P067：单卡优化组合验收与逐部独立复审

- **动作：** 从 `P061-P066` 只组合已单独通过正确性和性能门槛的变体；逐项消融 compile、operator/layout、fused op、kernel、optimizer和pipeline收益，检测交互回退，未达门槛的候选保持关闭并记录理由。
- **集中验证：** 从同一 checkpoint `N` 的独立副本和同一显式固定 correctness batch 一次完成17 image buckets × 8 text shapes、真实 Qwen/VAE/conditioning/DiT/loss/backward/clip/optimizer/checkpoint、故障恢复子集、fresh-process resume next step和 eager-vs-optimized `N+1` output/loss/all-gradient/update/state；真实 D024 service 另做不要求数据位置连续的端到端供给/overlap benchmark，随后运行 T053 公平端到端 benchmark。
- **审查：** 每个 `P061-P066` 独立 AI/Infra 结论必须已关闭；`P067` 再由新的独立 AI reviewer 和 Infra reviewer 检查组合正确性、数值容差、checkpoint兼容、无 silent fallback/recompile、内存边界和端到端收益。
- **输出：** 生成唯一 accepted 1GPU optimization manifest，锁定启用/禁用项、source/config/build hash、kernel provenance、compile counters、数值报告和 `perf_baseline.json`/`perf_after.json`；发布可供用户手工从 checkpoint `N` 进入 update `N+1` 的 resume handoff，但不启动训练，且不得外推为4GPU结论。

## 13. Phase 6：Stage 配置填充与正式 canary

本阶段所有训练启动、暂停、finalize、恢复、扩模型和扩分辨率都由用户手工决定。训练程序只执行用户明确选择的 resolved config、checkpoint 与唯一合法 transition，达到门槛时只写证据和 `stage_ready=true`；不得自动进入下一 stage。

### S000：目标机容量与 stage overlay 填充

- **对应文档：** `C10-D/E`、`C12-F`。
- **依赖：** `T053` benchmark harness完成，dropout已决定；本任务先为 eager `S001` 启动冻结单卡配置，不等待 `P060-P067`。regional compile 若缺少四卡证据仍保持关闭。
- **动作：** 为S0/S1/G1/S2/G2/S3填写local/global batch、accumulation、checkpoint mode、valid samples/equivalent data passes、actual DiT FLOPs、successful updates、checkpoint slots和wall-time预测；equivalent data passes 只按样本暴露量换算，不对应或重置 service `mainset` 代次；H1/H2继续disabled。
- **验证：** 相邻stage只变一个主轴；transition前序唯一；配置无placeholder、隐式默认或未知key；每个resolved config有hash。
- **完成证据：** stage budget表、resolved configs、容量审查。
- **GPU：** 1GPU填S0；S1以后必须4GPU benchmark。

### S001：S0 单卡 16L/256 canary

- **进入门槛：** 全部P0 contract、Qwen/VAE、dense/FA4、optimizer 1,000-step canary、单卡checkpoint/fault tests通过。
- **执行：** 用户手工启动 eager zero-update、首个 successful update 和有界稳定窗口；用户选择 update `N` 的 `COMPLETE` raw checkpoint并手工停止。`P060-P067` 完成后只提供验收结果，由用户决定是否以及何时手工从同一 checkpoint 的 update `N+1` 恢复，再继续到累计200 updates、累计1,000 successful updates和耐久窗口。全程保持真实在线数据/Qwen/VAE/DiT/optimizer，优化测试产生的 update 不计入正式训练。
- **验收：** loss/grad/clip稳定；无NaN/OOM/swap；checkpoint fresh-load next-step对齐；数据/验证隔离；W&B与本地metrics一致；用户手工批准S0。
- **证据：** stage report、AI/Infra review、固定sample、timing/profile、accepted raw checkpoint。
- **GPU：** 当前1×5090可执行。

### S002：S1 四卡 16L/256 canary

- **唯一变化：** world size 1→4；训练 stage/RNG 按 transition 合同迁移，data service 不新建或重置 `mainset`。
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
| 1GPU optimization | eager vs candidate output/loss/all-grad/update/resume + before/after | 正确性先于性能；组合证据不能替代逐项消融 |
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
D010 → D011 → D012 → D013 → D014 → D015 → G001 → D016 → D020 → D021 → D022 → D023 → D024 → D025 → D026 → DATA package review
                                                                                                      ↓
T020 + T021 → T022 + T023 → T024
  ↓
M030 → M031 → M032 → M033 → M034 → K001
  ↓
T040 → T041 → T042 → T043 → T044 → T050 → T051/T052/T053 → T054
  ↓
C002/S000 → 用户手工启动S001 eager → 用户选择COMPLETE raw checkpoint N并手工停止
  ↓
P060 → P061 → P062 → P063 → P064 → P065 → P066 → P067
  ↓
P067只交付优化manifest与续训证据 → 用户手工决定并从同一checkpoint恢复N+1
  ↓
累计200/1,000 updates/耐久窗口 → 用户逐stage手工finalize/启动S002 → G1 → S2 → G2 → S3
  ↓
可选 H1 → H2

A001：只保留已完成的最小本地资产边界；A002 重型审计保持撤销，不参与上述依赖链。
```

允许并行的只有不写同一文件且接口已冻结的任务，例如T020与T021、T022与T023、T051与T052。`P061-P066` 为了保留逐项可归因的 baseline 默认按编号串行；只有 P060 热点矩阵证明写路径、benchmark identity和GPU资源互不冲突时才可并行只读预审。实现和审查不得并行；上游contract未验证时不得提前写生产优化。

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
