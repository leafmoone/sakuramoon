Here is the result of "view" for the Page with URL https://app.notion.com/p/3acae967ecf28174bbaafbd053308860 as of 2026-07-29T06:42:43.823Z:
<page url="https://app.notion.com/p/3acae967ecf28174bbaafbd053308860" icon="📋">
<ancestor-path>
<parent-page url="https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff" title="新模型架构决策中心"/>
</ancestor-path>
<properties>
{"title":"待做与待确认清单 v2（会话重整）"}
</properties>
<content>
<callout icon="📋" color="yellow_bg">
	**用途：**本页是会话重整后的唯一开放事项与执行清单。已明确批准的架构不再要求重复确认；checkbox 只有在附带 artifact、报告或明确决策记录后才能关闭。
</callout>
现行方案：<mention-page url="https://app.notion.com/p/3acae967ecf281d69eb2c080768c6cd1"/>
# 0. 状态和完成证据
- **待用户决定：**会改变输入协议或训练分布，证据准备完成后由用户明确选择。
- **P0：**正式 S0 之前必须关闭。
- **P1：**可开发 reference，但长期四卡训练前必须关闭。
- **P2：**不阻塞首版 512。
- 每项只附与风险相称的直接证据，例如针对性测试、真实运行结果、benchmark、profile、checkpoint ID 或决策记录；不为本地已准备资产强制生成 manifest/hash，“代码已写”本身仍不是完成证据。
- 低中风险实现由里程碑包共享 `ai_review.md` 与 `infra_review.md` 并逐 ID 给出结论；高风险任务和正式 stage canary 仍分别产出双审。普通 CPU 任务不要求独立 timing artifact，真实性能变更仍须提供同配置 before/after 数据。
# 1. 仅剩的待用户决定
## 1.1 Dropout 数值
- [x] 固定 `general=0.1`、`artist=0.1`、`character=0.2`、`copyright=0.1`、`nsfw=0.1`、`candidate_source=0.3`。
- [x] 固定 `long_names=0.3`、`long_no_names=0.3`、`short_vibes=0.3`、`nl2=0.3`、`nl3=0.3`；五项逐项存在且数值相同。
- [ ] 用至少 100k 样本 dry run 验证 `all_condition=0.10` 命中率为 `10%±0.5` 个百分点，并分别报告最终空 body 率和各 component 命中率。
- [x] 所有字段无代码默认值；缺失、未知 key、错误类型或任一批准值漂移时启动即失败。
# 2. 已关闭，不再重复确认
- [x] 官方 Mage-VAE、posterior mean、在线编码、原生 128ch latent、无额外 patchify、严格 no-upscale。
- [x] 指定 ModelScope Qwen checkpoint、Krea 2 式手工 framing、无 thinking、无有效 EOT、空 body CFG 模板。
- [x] tags 用 `, `，NL 用 `\n\n`；主文本类别骨架为 `nsfw→character→copyright→general→NL`；candidate 是删除掩码。
- [x] Artist 只进入在线构造的辅助 segment和4-token style分支；serializer记录segment/token indices，同一次Qwen前向，主文本不消费Artist tokens，不做离线style cache。
- [x] 文本预算固定为 `text_condition_max=512` 与 `[64,128,192,256,320,384,448,512]` 八桶；完整Qwen长度为 `[98,162,226,290,354,418,482,546]`，不因Artist路径变化重扫。
- [x] 主文本 7 层 grouped gated mixing + 一个双向 Attention-only block；style 为 4-query Resampler。
- [x] `[text|4 style|image]`、三类 modality embeddings、面积归一化 2D RoPE、25/37.5/37.5 分配。
- [x] DiT `d=2560`、20Q/5KV、head_dim=128、SwiGLU=6912、16→20→24、约 1.85B–1.90B。
- [x] condition `384→1024→1024`、共享 `6d` + per-block bias、独立 final modulation、128ch x-pred head。
- [x] JLT x-pred/velocity-space loss、velocity-space CFG 2.9、Heun-50 + final Euler、PMA-10 仅评估/发布。
- [x] S0→S1→G1→S2→G2→S3、随机新 slots、固定半余弦 alpha、17 buckets、transition 人工启动。
- [x] S0 单卡原生、S1 起 DDP、TorchAO AdamW8bit、raw checkpoint 分层和生产吞吐/显存门槛。
# 3. P0：数据、VAE 与输入协议
## 3.1 Manifest、验证隔离和数据扫描
- [ ] 生成不可变训练 manifest，记录 repo/revision/path/release/bytes/SHA-256/samples，并验证约 11M 个逻辑 `id` 全局唯一。
- [ ] 生成恰好 2,000 个唯一 id 的 `validation_manifest.jsonl` 与独立 validation shard；完整训练 dry run 中这些 id 的消费计数必须为 0。
- [ ] 对全部 metadata 执行 256/512 的 17-bucket 分配扫描，报告每桶样本数、eligible、no-upscale reject、retention reject 与 retention 分位数。
- [ ] 对至少前 100k 实际解码样本核对 EXIF 后尺寸；`dimension_mismatch>0.1%` 时暂停正式训练。
- [ ] 固化 caption serializer golden cases：tags-only、NL-only、tags+NL、empty、candidate 命中、所有 dropout、截断和 suffix 保留。
- [ ] 断言训练输入中 `<think>`/`</think>` 次数为 0，padding mask 为 0，prefix/suffix token 数与锁定 tokenizer 一致。
## 3.2 VAE 重建与 latent 统计
- [ ] 固化 2,000 张 VAE 重建集：1,600 分层随机 + 400 风险覆盖；其中 1,500 走 512 路径、500 覆盖高分辨率/极端比例。
- [ ] 完成颜色范围、EXIF、RGB、resize/crop、no-upscale、posterior mean 和 decoder round-trip 单元测试。
- [ ] 达到 LPIPS/SSIM/严重错误/细节损失全部门槛，并保存逐类人工审查结果。
- [ ] 对 50k–100k 个训练 crop 统计 latent 全局/逐通道 mean/std、P1/P50/P99、绝对最大值与 BF16 安全性。
- [ ] 验证同一 seed/pass/id 的 crop 完全可复现，crop offset 和源尺寸绝不进入模型输入。
## 3.3 独立数据供给 service 与 `mainset`
- [ ] D024 实现独立启动/存活的单机唯一 data service，使 ModelScope 网络、token 解析、`.partial`、bytes/SHA-256、原子发布、cache/LRU/eviction 和 active/completed/replay state 全部只存在于 service；用进程边界与调用栈测试证明 trainer/DataLoader workers 没有这些路径，service 不可用时硬失败且无内联 fallback。
- [ ] service 为每轮持久化一张 `mainset`：严格绑定 immutable training manifest，包含全部 tar path 且每个恰好一次，记录新的 shuffle identity、精确 ordinal 顺序和逐行状态；验证 trainer、checkpoint、stage/resolution/model growth、worker topology 与 resume 都不能传入或改写 tar order/cursor。
- [ ] 按 `mainset` 顺序实现有界并发 download/verify/publish、verified lookahead、eviction lease 和本机 IPC；trainer 消费 `A/B` 时 service 准备 `C/D/E...`，所有 download/ready/lease/ACK、worker input/output、ready batch 与 completion channel 均有显式容量，active lease 不 eviction，磁盘预留和 quota 计入 published、in-flight 与 `.partial`。
- [ ] 验证 normal-exhaustion completion ACK 才逐 tar 完成；worker/service/client/trainer exit、断连或 ACK 丢失保留 active 并从 tar 起点 replay。只有当前 `mainset` 全部 tar 已下载、验证、供给且所有 outstanding lease 完成后，才原子删除旧表并创建下一份全 manifest 随机 `mainset`；崩溃不得丢失旧表或提前供给下一轮。
- [ ] T044 从 production raw checkpoint schema、manifest 和 resume API 中移除全部 data-service state；checkpoint 仍须完整保存并恢复 model、TorchAO optimizer、scheduler/growth、trainer counters、训练 RNG、optimizer-SR RNG、resolved config 与 identity。fresh-process next-update 正确性使用显式固定输入 batch，禁止要求或伪造 live tar/batch 连续性，并为既有含 data sidecar 的旧 raw schema 提供明确拒绝或受治理迁移合同。
- [ ] 在真实独立 service、真实多进程 DataLoader 和 1GPU consumer 上完成 cold/warm-cache overlap、worker/service/trainer fault 与人工 checkpoint resume smoke；达到 `>=12 samples/s`、ready wait `<2%`、无 swap/无界 RSS/quota 越界，并证明下载/校验不会让 same-backend fully-cached trainer step p50/p95/p99 超出预登记波动。
# 4. P0：模型 reference 与正确性
## 4.1 文本与 style
- [ ] 实现结构化 serializer 输出：主文本、Artist辅助segment、`main_token_indices`、`artist_token_indices`、attention mask和condition length；禁止通过token字符串回查span。超限时必须先保留协议边界和Artist segment，再裁NL/低优先级主文本tags。
- [ ] Artist segment必须位于所有主分支消费token之后；验证主文本聚合没有Artist token或因果泄漏，style分支只gather Artist span。
- [ ] 实现 Qwen 7 层显式 tuple-index 映射、per-layer RMSNorm、共享 2048→1024、8-group token-dependent mixing 和深层残差锚点。
- [ ] 实现一个 NoPE 非因果 Attention-only MHA block，正确屏蔽 padding；验证冻结 Qwen 零梯度、无视觉路径、无 KV cache。
- [ ] 实现完整 style 路径和 4 个 learned null tokens；验证 4 queries 产生独立 slots，不由单向量线性展开。
## 4.2 Packing、RoPE 与主干
- [ ] 分开实现并测试 modality embedding、packing 和 RoPE 三步；按样本保存 text/style/image span 与 latent H/W。
- [ ] 实现 `[valid text|4 style|image]` varlen、`cu_seqlens` 隔离、image-span gather；dense reference 同时屏蔽 padding query/key 并逐 block 清零。
- [ ] 验证面积归一化 cell-center 坐标、`32/48/48` RoPE、`position_scale=16`、`theta=1000` 与 QK-RMSNorm-before-RoPE。
- [ ] 实现 16/20/24 stable slots、20Q/5KV 原生 GQA、content gate、condition residual gates、SwiGLU 6912 和 RMSNorm 精度规则。
- [ ] dense SDPA 与 FA4 varlen 的 output/loss/gradient/update 在登记容差内一致；禁止 KV head repeat 或跨样本 attention。
## 4.3 Condition、Head 与目标
- [ ] 实现 `t/size/aspect→condition_hidden`、共享 `1024→6d` 和 per-block `6d` bias；验证两者都 zero-init。
- [ ] 实现独立 `condition_hidden→SiLU→Linear(1024,5120)` final modulation；测试它不复用 block `6d`。
- [ ] 实现 image-only conditional final RMSNorm 与 zero-init `Linear(2560,128,bias=true)`；初始化时任意输入的 `x_pred` 为 0。
- [ ] 用 golden test 固化 `z_t=t*x+(1-t)*epsilon`、`t=0` noise、`t=1` clean，并禁止把 clean endpoint 写作 `z0`。
- [x] 验证严格 JLT FP32 `v_target=(x-z_t)/d`、`v_pred=(x_pred-z_t)/d`、inverse-square clamp 最大权重 400、per-sample/global mean、`t<0.95`/`t>=0.95` 观测分桶，以及 cond/uncond 各自 x-to-v 后才做 CFG；采样 profile 的 solver/NFE 合同由 M034 单独关闭。
# 5. P0：环境、Optimizer 与 Checkpoint Canary
## 5.1 环境和 kernel preflight
- [ ] 锁定 driver、CUDA、PyTorch、TorchAO、FA4/CuTeDSL、Triton、causal_conv1d、fla、ModelScope Hub、Safetensors 和 NCCL 版本。
- [ ] 在 RTX 5090 实际执行 FA4 varlen BF16 20Q/5KV forward/backward、Qwen DeltaNet fast kernel 和 fused SwiGLU；仅 import 成功不算通过。
- [ ] 检查 4×32GB GPU、NCCL P2P、14 vCPU、120 GB RAM、网络凭据与 NVMe quota；cache 高水位之外至少容纳 3 份实测 full raw checkpoint。
- [ ] resolved config 不得含 `REQUIRED_AFTER_BENCHMARK`、未知 key 或隐式默认回填；preflight 不提供绕过硬项的 force 开关。
## 5.2 TorchAO 和精度 canary
- [ ] 逐 canonical FQN 审计 dtype、decay group、TorchAO state class/bytes、parameter order、step 与隔离 optimizer SR RNG。
- [ ] 完成 1,000-step FP32-parameter reference 对 mixed BF16/FP32 + stochastic rounding；validation loss EMA 回退≤3%，无 NaN/Inf或状态分叉。
- [ ] 验证四 rank 的 model、moments、per-parameter step 与 SR RNG hash 一致，同时训练 RNG 保持 rank-local。
- [ ] 验证 strict global sample mean 与单卡合并 batch reference 一致，FP32 global clip=1.0。
## 5.3 Raw checkpoint
- [x] T042 历史 schema 已实现 canonical-FQN sharded Safetensors model、完整 optimizer sidecar、trainer/data/growth/RNG state、checksum、manifest、临时目录和 COMPLETE；其中 data state 是已被 D024/T044 取代的旧生产合同，历史实现与证据保留，去除工作由 3.3 的新条款关闭。
- [x] T042 已在其历史范围完成 save→fresh process load→next step 与缺失/损坏 sidecar 硬失败；D024/T044 仍须按 3.3 使用固定外部 batch 复验训练/优化器恢复，并明确排除 live-data continuity。
- [ ] 普通 resume 只接受相同 topology；transition 只接受配置列明的唯一前序。模型目录去掉续训 sidecar 后仍能独立推理。
- [ ] raw、model-only snapshot、PMA 与 release artifact 使用不同 kind/目录；PMA 绝不作为 resume 或 growth 输入。
# 6. P1：目标机 Benchmark 与显式 Stage 配置
## 6.1 数据路径
- [ ] 冷缓存连续 2 小时达到≥12 samples/s，ready-queue wait\<2%，无 host swap、无界内存或缓存 quota 越界。
- [ ] 比较每 rank 1/2/3 workers、多个有界 queue depth、下载并发、Range workers、300–500 GiB quota 高低水位并锁定最小稳定配置。
- [ ] 分段记录 cache wait、tar read、JSON/caption、tokenize、decode、EXIF、resize/crop、bucket wait、H2D、Qwen、VAE 与 queue depth。
## 6.2 训练路径
- [ ] 对 16/20/24层与 256/512 的真实路径分别 benchmark local batch、global batch、accumulation、checkpoint none/alternating/all。
- [ ] 完整 20/24层 512 四卡路径达到≥6 samples/s、每卡≤27.2 GB、data wait\<2%；低于4停止，4–6只优化。
- [ ] 比较 FA4 varlen 与 dense SDPA reference，报告 samples/s、image/text tokens/s、DiT FLOPs/s、峰值显存和数值误差。
- [ ] regional compile 默认关闭；仅在正确性、DDP、resume 通过且稳态端到端提升≥3%时启用。
- [ ] 报告 step p50/p95/p99、GPU active/idle、kernel launch/gap、DDP wait、optimizer、host/pinned RAM 和 checkpoint 摊销开销。
## 6.3 Stage overlays 与预算
- [ ] 生成 base + S0/S1/G1/S2/G2/S3 六份显式 overlays，以及默认禁用的 H1/H2 模板和 resolved-config hash。
- [ ] benchmark 后填写每 stage 的 valid samples/equivalent data passes、DiT FLOPs、successful updates、batch/accumulation、checkpoint slots 和 wall-time 预测；equivalent data passes 只由样本暴露量换算，不对应、重置或选择 service `mainset` 代次。
- [ ] 实现 drain/finalize、`stage_ready` report 和 transition preflight；训练程序不得自动改变 world size、深度、分辨率、LR 或数据混合。
- [x] 已取代：transition 不再以新的 stage/pass/seed 重置 tar 顺序；固定验证集保持不变，独立 service 继续当前持久化 `mainset`，该替代合同由 3.3 的 D024 项实现和验证。
# 7. P1：深度增长、恢复与故障注入
## 7.1 两次增长
- [ ] 实现 final 24-slot stable FQN 和 active slot ids；两次增长分别只允许预定义 new-slot allowlist。
- [ ] 验证旧参数/optimizer state完整保留，新 slot 随机初始化、alpha=0、optimizer state为空，且无 copy/moment copy/LR multiplier。
- [ ] 在 alpha=0 做新旧模型函数等价；在 ramp midpoint、alpha=1 各保存并恢复 checkpoint。
- [ ] 执行 1,000–5,000 update 固定半余弦 ramp 与 post-ramp 稳定窗口，记录 loss、grad norm、throughput、memory 和 alpha。
- [ ] 失败时写诊断包并回滚 pre-transition raw checkpoint，不自动重试增长或改变 ramp。
## 7.2 故障注入
- [ ] 注入下载中断、截断 shard、鉴权失效、cache checksum失败；坏 shard不得发布，完整 shard不得静默跳过。
- [ ] 在 microbatch、DDP reduction、optimizer step 与 checkpoint各阶段杀进程；只允许恢复上一份 COMPLETE。
- [ ] 注入 nonfinite loss/gradient、OOM、SR RNG分叉、NCCL rank failure、worker failure 与磁盘写满；所有 rank 同步停机。
- [ ] 验证完成 shard不重读、活跃 shard从头重放，replayed shards/samples可审计；不得自动改 batch、backend、world size、optimizer、LR 或 checkpoint频率。
# 8. P1：质量验收和生产放行
- [ ] 固定 prompt/seed/尺寸，正式使用 Heun-50、CFG=2.9；快速预览不进入验收。
- [ ] 同时比较 raw latest、同稳定窗口 PMA-10 与 accepted checkpoint，检查 tag 控制、人物/服装/颜色/姿态/镜头、NL 跟随、构图、细节和严重伪影。
- [ ] 每 stage 同时满足数据暴露、DiT FLOPs/updates、最近1,000 successful updates稳定性、吞吐/显存和恢复门槛。
- [ ] 每个 stage 保存 `ai_review.md`、`infra_review.md`、`perf_baseline.json`、`perf_after.json`、`stage_performance_report.json` 和 profiler trace索引。
- [ ] 只有用户手工批准后才 finalize/transition；训练程序只提供 evidence 和 `stage_ready=true`。
# 9. P2：首版 512 之后
- [ ] H1 24层/768：仅在512成品接受、原生数据覆盖和独立 benchmark通过后手工启用。
- [ ] H2 24层/1024：优先从已接受H1进入；若从S3直升，必须重新审批 transition。
- [ ] FSDP2：只有DDP无法满足27.2 GB、目标 batch或6 samples/s硬门槛时重新开决策。
- [ ] latent/text cache：只有在线Qwen/VAE或数据供给被 profile证明为瓶颈后评估。
- [ ] 在线FP32 EMA：只有PMA工具链失败且有明确收益需求时评估。
- [ ] iREPA、编辑视觉分支、额外结构技巧和复杂solver保持关闭。
# 10. 文档与配置清理
- [ ] 用户阅读 v2 两页后，将现行方案标记为实现统一入口。
- [ ] 在组件06把FSDP2正文明确标为被组件12覆盖；在组件09把bitsandbytes/FSDP2迁移说明改为引用组件12。
- [ ] 把组件04和决策中心残留的“约1B”改为1.85B–1.90B。
- [ ] 把组件08中PMA可作下一stage初始化的历史表述收窄为只用于评估/发布。
- [x] 已取代：组件11的 stage data pass 不再作为实现来源；现行 transition 与 data-service `mainset` 以本页 3.3 和现行决定为准，archive 保持只读且不回写。
- [x] 组件03/05已写入 condition 512、8个长度桶和不因Artist变化重扫的最终决定；目标机只继续验证padding/varlen执行路径。
- [ ] 原组件页保留论证历史，但所有仍可被误当成配置来源的旧段落必须带“已取代”及现行方案链接。
# 11. 来源
- <mention-page url="https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff">新模型架构决策中心</mention-page>
- <mention-page url="https://app.notion.com/p/3aaae967ecf281ba8f73fac2f9e4c4f3"/>
- <mention-page url="https://app.notion.com/p/3aaae967ecf2816096f0ea37634a2f7e"/>
- <mention-page url="https://app.notion.com/p/3aaae967ecf281fba3cfe0f5dc53fece"/>
- <mention-page url="https://app.notion.com/p/3abae967ecf28104ad74d1b843728be4"/>
- <mention-page url="https://app.notion.com/p/3abae967ecf281de8dafc6dbecd04fe6"/>
- <mention-page url="https://app.notion.com/p/3abae967ecf281809823e98b5d91222e"/>
- <mention-page url="https://app.notion.com/p/3abae967ecf2815b84fbda3c707728c2"/>
- <mention-page url="https://app.notion.com/p/3abae967ecf28167a869fb61c5ff0e96"/>
- <mention-page url="https://app.notion.com/p/3abae967ecf28103be8feea5e27f21e1"/>
- <mention-page url="https://app.notion.com/p/3abae967ecf2816da90ccbee372b27c4"/>
- <mention-page url="https://app.notion.com/p/3aaae967ecf281db800cfb1d6545f880"/>
- <mention-page url="https://app.notion.com/p/3abae967ecf281ebadadd176e1b492db"/>
</content>
</page>
