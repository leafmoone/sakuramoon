Here is the result of "view" for the Page with URL https://app.notion.com/p/3abae967ecf281ebadadd176e1b492db as of 2026-07-28T12:01:23.531Z:
<page url="https://app.notion.com/p/3abae967ecf281ebadadd176e1b492db">
<ancestor-path>
<parent-data-source url="collection://69ca66ff-43e7-4128-bb7b-9f3751506705" name="组件决策记录"/>
<ancestor-2-database url="https://app.notion.com/p/250be554eacc40219065073dfcf66fd7" title="组件决策记录"/>
<ancestor-3-page url="https://app.notion.com/p/3aaae967ecf281c8a10bf797b47982ff" title="新模型架构决策中心"/>
</ancestor-path>
<properties>
{"date:决定日期:is_datetime":0,"date:决定日期:start":"2026-07-28","url":"https://app.notion.com/p/3abae967ecf281ebadadd176e1b492db","决策编号":"ARCH-12","序号":12,"影响":"高","标签":["训练","系统"],"状态":"待验证","组件决策":"12 优化器与训练系统"}
</properties>
<content>
<callout icon="✅" color="green_bg">
	**训练系统决定已批准，待目标机实现验证。** 12-A～12-G 已全部闭合：单卡原生→四卡 DDP、TorchAO AdamW8bit mixed BF16/FP32、同步 stochastic-round RNG、FA4 varlen、完整 block checkpoint、可回退 regional compile、WSD LR、原子 raw checkpoint、FQN 增长迁移、生产故障矩阵，以及双角色开发审查与全路径性能工程规范均已锁定。
</callout>
# 组件边界
本组件负责把已批准的模型和数据架构落实为可长期运行的训练系统：optimizer 后端、单/多卡并行、参数与归约精度、activation checkpoint、梯度累积、分布式 checkpoint、1→4 卡迁移、深度增长状态迁移及恢复验证。本组件不重新决定模型宽深、loss、采样器、caption 协议或分辨率课程。
# 已继承约束
- 最终可训练参数约 1.85B–1.90B，深度按 16→20→24 增长；稳定 slot FQN 和 growth migration 语义继承组件 09。
- 当前 optimizer 选择为 AdamW8bit。旧参数 optimizer state 必须保留，新增 slot state 从零初始化；不得复制 moments 或静默重置旧 state。
- S0 为单卡 16层/256；S1 起为同机四卡。所有 stage 由用户手动切换，启动器只做 preflight。
- 冻结 Qwen3.5-2B 与官方 Mage-VAE 在每个 rank 各自复制，使用 BF16、`eval()` 与 `inference_mode()`；不参与 DDP/FSDP、optimizer 或模型训练 checkpoint。
- 训练 checkpoint 使用模型产物与续训 sidecar 分层、checkpoint ID 一致、临时目录写入和 `COMPLETE` 原子提交；PMA 只离线合并模型权重，不承担恢复。
- 冷缓存数据管线必须独立持续供给至少 12 samples/s；完整 20/24层、512 等效面积训练的四卡聚合吞吐必须至少 6 samples/s，ready queue 为空导致的 GPU 等待低于 2%。具体 global/local batch 和累积步数由目标机 benchmark 填入 stage config。
# 当前官方兼容性核对
## PyTorch FSDP2 / DCP
PyTorch `fully_shard` 会把参数原位转换为 DTensor，并要求 optimizer 在 DTensor parameters 上构造；参数、梯度与 optimizer state 按 data-parallel mesh 分片。官方建议对 Transformer blocks 自底向上应用 `fully_shard`，FQN 保持不变；模型与 optimizer 的分布式保存/加载通过 DCP state-dict API 完成。
- [PyTorch FSDP2 fully_shard](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
- [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch DCP recipe](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)
## bitsandbytes AdamW8bit
bitsandbytes 官方文档定义了 block-wise 8-bit optimizer state，但未承诺 FSDP2/DTensor optimizer 兼容。上游 issue #1633 中维护者明确说明当时没有 FSDP optimizer 支持；截至当前仍为开放 feature issue。社区候选 patch 能绕过 DTensor local-shard kernel 错误，但尚未验证与非 DTensor AdamW8bit 更新的严格数值等价，也不是可锁版本的正式发布接口。
- [bitsandbytes AdamW8bit](https://huggingface.co/docs/bitsandbytes/en/reference/optim/adamw)
- [bitsandbytes issue #1633](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1633)
# 12-A：并行基线
## A. 单卡原生 + 四卡 DDP，TorchAO AdamW8bit，最终决定
- S0 使用普通单进程模型与 TorchAO AdamW8bit；四卡阶段只在同一可训练 composite module 外增加 PyTorch DDP。
- DDP 仅包含 DiT、文本多层聚合/适配器、style 分支、全局条件和 output head；冻结 Qwen/VAE 保持在 wrapper 外，每 rank 一份。
- 四卡每 rank 保存完整可训练参数、gradient 和 AdamW8bit state；梯度累积的非最终 microsteps 使用 `no_sync()`，只在 optimizer update 前执行一次梯度归约。
- baseline 使用 `find_unused_parameters=false` 与 `gradient_as_bucket_view=true`；每个作业的 active slots 固定，growth 通过独立 G1/G2 作业完成。
- rank 0 写一份 replicated model/optimizer checkpoint；各 rank 额外保存自己的 RNG 与必要 data topology state。1→4 卡 transition 显式加载同一 model/optimizer state，再新建四卡 data pass。
优点：TorchAO AdamW8bit 处理普通本地 tensors，并为 BF16 parameters 提供 stochastic-round 写回；单卡与四卡 optimizer 语义一致，增长迁移与 checkpoint 调试也比 DTensor 路径简单。
代价：每卡仍复制全部训练状态并对完整 gradient all-reduce；最终 24层/512 的显存和通信吞吐必须实测。
## B. FSDP2 + bitsandbytes AdamW8bit 社区 patch
可以减少每卡状态，但依赖未合并 monkey patch 和未完成的数值验证；还会把 optimizer、增长迁移与 DCP 正确性同时放在非官方路径上。不进入 baseline，也不实现静默 fallback。
## C. FSDP2 + 受支持的 optimizer 后端
DDP 无法通过硬门槛时才重新讨论，不预先批准，也不静默切换：
- 第一候选是已批准的 [TorchAO AdamW8bit](https://docs.pytorch.org/ao/stable/generated/torchao.optim.AdamW8bit.html) + FSDP2，因此不再更换 optimizer 后端；但 DTensor、checkpoint、增长迁移、目标 PyTorch/TorchAO 版本和数值收敛仍须重新验证。
- 第二候选是 FSDP2 + torch fused AdamW。兼容路径更常规，但 optimizer state 不再是 8-bit，会改变显存预算和既有 optimizer 决定。
## D. 四卡仍各自独立训练
没有同步梯度，不是单一 global model；不采用。
# 粗略显存边界
最终约 1.9B 参数使用 FP32 parameters/gradients + 两份 8-bit moments 时，DDP 每卡静态三项约 19.0 GB，最终阶段很难留出足够 activation 空间。若大矩阵改为 BF16 parameters/gradients，静态三项约 11.4 GB；但只有带 stochastic-round parameter 写回的受支持 optimizer 才进入生产候选。少量数值敏感参数保留 FP32，真实开销由启动时逐参数审计确定。除此之外仍要容纳冻结 Qwen/VAE、activation、communication bucket、CUDA workspace 和临时张量；32 GB 卡继续执行 27.2 GB 的 85% 峰值门槛。
# 12-A 最终决定
采用 A：单卡原生训练，四卡 DDP，使用 TorchAO AdamW8bit；冻结 Qwen/VAE 不进入 DDP。只有在最终 24层/512 的真实路径上满足以下条件才锁定生产运行：
- 峰值显存不超过 CUDA 报告物理显存的 85%，且连续运行无逐步增长。
- 能使用配置要求的 local microbatch/gradient accumulation，不因 OOM 退化到无法满足 global batch。
- 冷缓存数据供给达到至少 12 samples/s；完整 512 训练四卡聚合达到至少 6 samples/s。通信等待与数据等待分别可观测，ready queue 空等待低于 2%。
- checkpoint/resume、单卡→四卡和 16→20→24 optimizer-state 迁移测试全部通过。
若任一硬门槛失败，停止并重新决定受支持的 FSDP2 组合；优先评估 TorchAO AdamW8bit，其次评估 torch fused AdamW。不得自动切换 optimizer，也不得安装 bitsandbytes monkey patch。
# 12-B：混合精度与全局 loss 语义
## 核对结论
“BF16 混合精度”和“所有可训练参数只存 BF16”不是同一件事。常规 AMP/FSDP mixed precision 会在 forward/backward 使用 BF16，但仍为 optimizer 保留全精度参数；这是为了避免低学习率阶段的小更新在 BF16 写回时停滞。
bitsandbytes 的 8-bit AdamW kernel 确实实例化了 BF16 parameter/gradient 类型，内部用 FP32 计算 moments 与更新；但最终直接把结果确定性转换回参数 dtype，没有独立 FP32 master weight，也没有为 parameter 写回执行 stochastic rounding。以权重约 `0.1` 为例，BF16 相邻值间隔约 `4.88e-4`，小于约 `2.44e-4` 的确定性更新可能完全不改变该权重；学习率衰减后风险更明显，AdamW weight decay 也会受相同舍入影响。
TorchAO AdamW8bit 则显式支持 `bf16_stochastic_round=True`：更新在 FP32 中计算，再按低 16-bit 概率随机写回 BF16。其官方测试包含小学习率下 BF16 stochastic-round 与 FP32 mixed-precision reference 的一致性检查。随机舍入不保存 FP32 master copy，但能避免确定性舍入造成的系统性小更新丢失。
- [PyTorch FSDP mixed precision](https://docs.pytorch.org/docs/stable/fsdp.html)
- [bitsandbytes BF16 optimizer kernel](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/csrc/kernels.cu)
- [TorchAO AdamW8bit implementation](https://github.com/pytorch/ao/blob/main/torchao/optim/adam.py)
- [TorchAO stochastic-round implementation](https://github.com/pytorch/ao/blob/main/torchao/optim/quant_utils.py)
- [TorchAO low-bit optimizer tests](https://github.com/pytorch/ao/blob/main/test/test_low_bit_optim.py)
## A. FP32 parameters + bitsandbytes AdamW8bit
数值风险最低，但最终 1.9B DDP 每卡仅 parameters、gradients、两份 8-bit moments 就约 19.0 GB；再加冻结 Qwen/VAE、activation 和运行时缓存后，很难在 32 GB 卡上保留 15% 余量。保留为短程正确性 reference，不推荐作为最终生产配置。
## B. 全部 BF16 parameters + bitsandbytes AdamW8bit
静态三项约 11.4 GB，kernel 技术上可运行；但 parameter 写回是确定性 BF16 cast，没有 master weight 或 stochastic rounding。对于三到六个月、学习率逐步衰减的从零训练，weight stagnation 风险不可接受，不作为生产候选。
## C. BF16 大矩阵 + 少量 FP32 敏感参数 + TorchAO AdamW8bit stochastic rounding，推荐候选
- attention/MLP projection、文本适配器与 style MLP 等大矩阵参数以 BF16 存储，gradient 也为 BF16。
- RMSNorm/QK-RMSNorm scale、bias/标量、AdaLN/门控、learned style/null tokens、timestep/尺寸条件敏感参数及最终小型 output head 保持 FP32。具体列表按稳定 FQN 写入配置并在启动时打印参数量与字节数，不允许按运行时启发式静默改变。
- optimizer 使用 `torchao.optim.AdamW8bit(..., bf16_stochastic_round=True)`。由 bitsandbytes 切换到 TorchAO 及 moments 量化实现的变化已经批准；PyTorch/TorchAO 版本必须锁定。
- 大矩阵仍按两份约 1 byte/parameter 的 moments 粗估，parameters + gradients + moments 接近 11.4 GB，外加少量 FP32 敏感参数；比方案 A 节省约 7.6 GB，也免去 FP32 parameter 的 BF16 autocast weight cache。
- TorchAO 只有在本地 tensor 至少 4096 elements 且 element count 可被 `block_size=256` 整除时才使用量化 state，否则回退为普通 tensor state。preflight 必须逐参数实例化并汇总真实 optimizer-state dtype/bytes，任何大参数回退都视为配置错误。
- stochastic rounding 会调用当前 CUDA 默认 RNG。DDP 下不得直接使用各 rank 彼此不同的训练 RNG，否则相同 FP32 更新会随机写成不同 BF16 参数，模型副本将逐步分叉。实现必须隔离两类状态：各 rank 独立的 training RNG；全 rank 完全相同的 `optimizer_sr_rng`。每次 optimizer step 前暂存本 rank training CUDA RNG、装入共同 SR state，step 后取回推进后的共同 SR state并恢复 training RNG；checkpoint 保存一份共同 SR state与各 rank training RNG。
## 数值敏感路径
- timestep、noise 和 x-pred/noisy-latent 目标在 FP32 中生成；model output 在 residual/MSE 前转回 FP32。
- 每样本先对有效 latent/channel elements 求 mean，再进入 batch 聚合，使不同宽高比分桶的单样本权重一致。
- loss、mask 后 reduction、gradient global norm、自定义 RMSNorm/QK-RMSNorm 统计量以及 timestep/尺寸标量归一化使用 FP32；large linear/attention matmul 使用 BF16。
- BF16 baseline 使用 `GradScaler(enabled=False)`；FP16 不作为自动 fallback。
## DDP 与严格全局 mean
- DDP 按 parameter dtype 建 bucket：BF16 大矩阵 gradients 以 BF16 all-reduce，FP32 敏感参数 gradients 以 FP32 all-reduce；启用 `gradient_as_bucket_view=True`，避免 gradient 与 communication bucket 长期双份占用。
- 一个 optimizer update 内，每个 microbatch backward 使用“各样本 mean loss 的本地 sum”；非最终 microsteps 使用 `no_sync()`。
- 最终 backward 后，DDP 已除以 world size `W`。所有 rank 以 INT64 all-reduce 得到有效全局样本数 `N`，再把所有 gradients 乘以 `W/N`，之后以 FP32 累计 global norm、clip 并 step；结果等价于严格的全局样本 mean。
- loader 必须保证每 rank 每次 update 参与相同 collective 次数且至少有一个有效样本；坏图在成 batch 前跳过。
- [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
## 最小验证，不做大规模消融
1. 在目标 32 GB RTX 5090 环境锁定匹配的 PyTorch/TorchAO 版本；当前工作环境未安装 bitsandbytes/TorchAO，且只暴露约 3.26 GiB VRAM，不能代表生产机。
2. 先做 planned initial/min LR 的 optimizer microtest，验证 stochastic rounding 的均值无偏、bitsandbytes 确定性 BF16 的 update-loss 比例，以及 state 实际字节数。
3. 使用同一 S0 16层/256 checkpoint、固定数据序列，各运行一个短 canary：FP32 parameters + TorchAO AdamW8bit reference；方案 C mixed parameters + 同一 TorchAO AdamW8bit stochastic-round。只比较 1000 optimizer updates，不扩展为多组消融。
4. 每层记录 parameter/gradient/update RMS、nonzero update ratio、finite 状态、固定验证 batch loss、峰值显存和 samples/s；方案 C 不得出现持续 dead/stagnant layer，验证 loss EMA 不得相对 reference 持续恶化超过 3%。
5. 在 canary 中间保存 checkpoint；恢复 model、optimizer、data state、各 rank training RNG 与共同 `optimizer_sr_rng` 后，下一 optimizer step 必须与未中断分支一致。四卡连续至少 1000 updates 后，抽查所有 trainable parameters、moments 和共同 SR state 在各 rank byte-identical；任一分叉都禁止进入完整 S1。通过后才允许进入完整 S0/S1。
## 不采用
- 不为 bitsandbytes 维护自定义 stochastic-round CUDA patch。
- 不在 GPU 保存完整 FP32 shadow/master weights；这会基本抵消 BF16 的显存收益。
- 不用 CPU master weights 做每 step 往返，网络/PCIe 开销不适合该长期训练。
# 12-B 最终决定
批准方案 C，并同步修订 12-A：保持单卡原生 + 四卡 DDP 与 AdamW8bit，但 optimizer 后端由 bitsandbytes 改为 TorchAO；使用 BF16 大矩阵、FP32 敏感小参数和 `bf16_stochastic_round=True`。loss、目标、归一化统计和 global gradient norm 使用 FP32，GradScaler 关闭。
该方案把静态训练状态控制在约 11.4 GB + 少量 FP32 开销。进入完整 S0 前必须通过上述 1000-step canary；BF16 all-reduce、真实 optimizer-state bytes、stochastic RNG checkpoint/resume 和目标机吞吐都是硬门槛，不通过则停止并重新讨论。
# 12-C：Activation Checkpoint、FA4 与 compile
## 已继承的 attention 后端
组件 06-E 已批准的后端不重新选择：
- 生产 attention 使用 FlashAttention-4 CuTeDSL `flash_attn_varlen_func`，flattened BF16 Q/K/V + CUDA INT32 `cu_seqlens`，原生 `20Q/5KV` GQA、`head_dim=128`、`causal=false`。
- PyTorch SDPA `enable_gqa=True` 只作为 dense correctness/debug fallback；production backend 不可用时必须显式失败或由用户指定 fallback，不能静默改变。
- FA4 保持独立 backend 边界。不得为了 `torch.compile(fullgraph=True)` 改成 padding dense attention、重复 K/V heads 或取消 varlen。
- [FlashAttention-4](https://github.com/Dao-AILab/flash-attention#flashattention-4-cutedsl)
- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
## A. 完整 DiT block checkpoint，推荐
checkpoint 边界固定为完整 DiT block；不拆成 attention/MLP 内部小片段，也不 checkpoint 冻结 Qwen/VAE。
- 在 model loop 中函数式调用 checkpoint，保留原始 block module 与稳定 slot FQN；不使用会给 parameter name 增加 wrapper 前缀的封装。
- 显式使用 `use_reentrant=False`、`early_stop=True`、`determinism_check="default"`。debug smoke 可设 `debug=True`，生产为 false。
- block 内 attention/residual/MLP dropout 和 stochastic depth 已全部为 0，因此 eager baseline 使用 `preserve_rng_state=False`；TorchAO stochastic rounding 位于 optimizer step，不在被 checkpoint 的 forward 内。
- PyTorch 在 `torch.compile` 下会无条件保存 checkpoint RNG，即使传入 false；这是已知行为，不能把 eager 与 compile 的这点开销假定为一致。
- [PyTorch activation checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html)
## Stage 配置
不根据运行时空闲显存自动改变 checkpoint；resolved stage config 保存明确的 active slot ID 列表：
- S0/S1/G1 的 256 阶段默认 `activation_checkpoint=none`。
- S2/G2/S3 的 512 阶段以“隔一个 active block checkpoint 一个完整 block”为首个 benchmark 候选；如果 none 已满足 local microbatch、global batch、27.2 GB 峰值和吞吐门槛，则正式配置保持 none。
- 只有 512 仍无法过显存门槛，或未来手工启用 H1/H2 768/1024 时，才测试每个 block checkpoint。
- 每次增长或阶段切换都在 overlay 中列出 checkpoint slot IDs；不能因 active depth 变化自动翻转旧 block 的 checkpoint 状态。
按 forward 约 `F`、backward 约 `2F` 粗算，checkpoint 一半 blocks 的理论训练 FLOPs 增量约 `0.5F/(3F)=16.7%`，全部 blocks 约 `33.3%`；实际吞吐由 FA4、GEMM、DDP 与数据流水共同决定。
## B. 不采用的 checkpoint 方案
- 不使用 reentrant checkpoint。
- 不对每个 attention/MLP 子算子分别 checkpoint；保存输入更多、调用更碎，复杂度高于收益。
- 首版不使用 selective activation checkpoint、memory-budget API 或 offload-to-CPU activation。
- 不把 activation checkpoint 当作 OOM 后的静默重试。任何频率变化都由 stage benchmark 后写回配置并重新启动。
## C. torch.compile 边界
首次正确性与 S0 canary 使用 eager DiT + FA4，不把 compile 设为启动依赖。通过 eager、checkpoint on/off、DDP 和 resume 后，才启用一个可回退的 regional compile benchmark：
- compile 作用于 DDP 内部的 trainable module/repeated DiT block region，先 compile inner module，再构造 DDP；不 compile DDP wrapper。
- 使用 repeated-block regional compilation，避免把 16/20/24 层全部内联成一个超大 graph。目标版本可优先评估 `torch.compiler.nested_compile_region`；若版本不支持，使用等价的 per-block regional API。
- 初始配置 `dynamic=True, fullgraph=False`；只把 packed `total_tokens` 维设为 dynamic，hidden/head dimensions 和每个 stage 的 local microbatch 保持静态。
- FA4 varlen 调用保留显式 graph boundary，由 CuTeDSL 自己管理 kernel；compile 只优化 norm/modulation、QKVG projection、content gate、output projection、SwiGLU 与 residual 等纯 tensor 区域。
- 首版关闭 CUDA Graph capture 与 max-autotune，避免动态 varlen、checkpoint recompute 和 DDP 同时引入额外状态；后续只能作为独立 benchmark 打开。
- TorchAO optimizer step 使用其自身的编译路径，不与 model forward/backward 包进同一个 compiled train-step graph。
- 冻结 Qwen/VAE 默认不 compile；它们的优化不得阻塞 DiT 训练。
- [PyTorch compile placement](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.where_to_apply_compile.html)
- [PyTorch regional compilation](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)
- [nested_compile_region](https://docs.pytorch.org/docs/stable/generated/torch.compiler.nested_compile_region.html)
## compile cache 与回退
- resolved config/run manifest 保存 compile enabled、backend、mode、dynamic policy、PyTorch/Triton/CUDA/FA4 版本；compile artifact 不进入模型 checkpoint。
- 使用 `TORCH_LOGS=recompiles,graph_breaks` 等等价计数完成 17 image buckets + 实际文本长度分布 warmup。warmup 后仍持续 recompile、达到 recompile limit 或局部 graph 静默退回 eager 时，benchmark 判失败并显式关闭 compile。
- compile 关闭后必须能从同一 checkpoint 正常恢复；模型/optimizer state schema 不得依赖 compiled wrapper。
- regional compile 稳态端到端提升低于 3% 时保持 eager，避免为三到六个月训练引入没有实际收益的故障面。
## 正确性与性能验收
1. 同一固定 batch 比较 checkpoint none/alternating/all 的 output、loss、parameter gradients 和一次 optimizer update；覆盖空条件、最长文本、非方形 bucket、microbatch=1 与 16/20/24 active depth。
2. 验证 checkpoint forward/recompute 不读取可变 Python 全局状态；growth alpha、slot mask、backend selector 在一次 forward/backward 内不可变化。
3. 比较 eager 与 regional compile 的 loss、gradient、FA4 packed 边界和 checkpoint resume；不能只比较最终图片。
4. 分别记录 none/alternating/all 的 peak allocated/reserved VRAM、samples/s、image tokens/s 和 checkpoint recompute 时间，只在满足 27.2 GB、目标 batch与完整 512 训练至少 6 samples/s 的候选中选吞吐最高者；数据管线 12 samples/s 门槛单独验收。
5. compile 首次时间与 warm-cache 稳态分开报告；训练启动 preflight 先遍历 stage shape warmup，不把首次编译误算为数据等待。
# 12-C 最终决定
批准 A + C 的受控组合：完整 block non-reentrant checkpoint、stage overlay 显式选择频率；256 默认关闭，512 先测 none 与 alternating，只有必要时全开。生产 attention 继续 FA4 varlen，SDPA 仅作显式 dense fallback。compile 默认关闭，稳定后只做 inner-module regional dynamic benchmark，收益低于 3% 就保持 eager。
# 12-D：AdamW8bit 参数组、裁剪与 LR
## 参考边界
- JLT 的 x-pred 训练参考使用 AdamW、`betas=(0.9,0.95)`、global grad clip 1.0、warmup 后 constant/cosine，并按参数维度排除 norm/bias 的 decay。本项目沿用其 beta2 与 clip，而不沿用按 global batch 自动线性放大 LR。[JLT main_jit.py](https://github.com/akatsuki-neo/JLT/blob/main/main_jit.py)
- SANA 1.5 1.6B AdamW 配置使用 `lr=2e-5`、1000-step warmup、constant LR、clip 0.1 和 zero weight decay；其模型/批量不同，只把 `2e-5` 作为本项目保守起点。[SANA 1.5 AdamW config](https://github.com/NVlabs/Sana/blob/main/configs/sana1-5_config/1024ms/Sana_1600M_1024px_AdamW_fsdp.yaml)
- 组件 09 已批准：旧 parameter state 完整保留，新 slot state 从零开始；新旧层同 LR，无 layer-wise decay 或新层 multiplier；growth alpha 是 schedule state，不进入 optimizer。
## A. 单个 TorchAO AdamW8bit
所有可训练参数由一个 `torchao.optim.AdamW8bit` 管理，固定：
- `lr_peak=2e-5`
- `betas=(0.9,0.95)`
- `eps=1e-8`
- `amsgrad=false`
- `block_size=256`
- `bf16_stochastic_round=true`
不启用 foreach/fused bitsandbytes 路径，不嵌套第二个 optimizer。TorchAO 当前会把 param-group LR 转为 FP32 tensor；scheduler 必须对现有 LR tensor 执行 `.fill_(new_lr)`，不能用 Python float 替换，否则 optimizer step 应拒绝启动。
- [TorchAO AdamW8bit source](https://github.com/pytorch/ao/blob/main/torchao/optim/adam.py)
## 参数组
### decay，`weight_decay=0.01`
显式模块角色为 Linear/Conv 的 trainable weight matrices：
- DiT Q/K/V/content-gate/output projections。
- SwiGLU gate/up/down。
- 输入/输出 projection、condition/modulation projection。
- 文本多层聚合 adapter、token refinement、style attention-pooling/MLP 中的矩阵权重。
### no_decay，`weight_decay=0`
- 所有 RMSNorm/QK-RMSNorm weights。
- 所有 bias、标量、1D scale/shift/gate parameters。
- learned style slots、null style tokens、其他 learned global/token embeddings，即使 tensor rank 为 2。
- growth alpha 不属于本组：它是组件 09 的 FP32 schedule state，完全不传给 optimizer。
分组以模块 role + stable FQN allowlist 为准，不能只用 `ndim>=2` 猜测。preflight 要求每个 `requires_grad=True` parameter 恰好属于一个 group，冻结 Qwen/VAE 为零个 group；打印每组 FQN、dtype、parameter count、预计/实际 state bytes。任何大矩阵未量化、重复、遗漏或意外进入 no_decay 都拒绝启动。
## B. LR：一次 warmup + stable + 最终 decay，推荐
使用成功 optimizer update 计数驱动的 WSD，不按 epoch/wall time，也不在 stage transition 重启：
1. **Warmup：** 只在首次 S0 从零训练执行 2000 个成功 updates，从 0 线性升至 `2e-5`。
2. **Stable：** 从 S0 延续穿过 S1/G1/S2/G2 和 S3 前段，保持 `2e-5`。world size、global batch、分辨率和 active depth 变化都不自动线性缩放 LR。
3. **Decay：** 只有 24层/512 已完成 G2 ramp 和稳定窗口、用户明确启动 final-decay overlay 后，才从独立 pre-decay raw checkpoint 做半余弦下降至 `2e-6`。长度为 `clamp(ceil(0.10×planned_total_successful_updates), 10000, 50000)`，在总预算填入后固化。
4. decay 一旦开始不自动回升；若需要延长 peak-LR 训练，从 pre-decay checkpoint 扩展 stable segment，而不是在 decay 分支上 warm restart。
scheduler state 保存 global successful-update count、segment、segment start、warmup/decay length、peak/floor LR 和公式版本。数据跳过、nonfinite skip、验证、只 forward 和失败回滚不推进 scheduler。
选择 WSD 的原因是总训练持续三到六个月、stage 由用户手工切换且最终预算需 benchmark 后填写；stage-local cosine/warm restart 会在增层和升分辨率时同时改变第二个轴，并破坏旧 moments 的连续语义。[TorchTitan](https://github.com/pytorch/torchtitan)
## C. 全局 gradient clipping
- 固定 global L2 `max_norm=1.0`，不做 per-layer、per-branch 或 adaptive clipping。
- 顺序固定：完成所有 microbatch backward → 最终 DDP reduction → 按 `W/N` 恢复严格 global sample mean → FP32 累计全部 trainable gradients 的 global norm → 检查 finite → clip → optimizer step。
- mixed BF16/FP32 gradients 的平方和统一转 FP32 累计；用同一 clip coefficient 原位乘回各 dtype gradient。四卡 all-reduce rank-local finite flag 和 norm maximum，保证任一 rank 异常时所有 rank 同步跳过。
- 记录 pre-clip norm、clip coefficient、post-clip norm、被裁剪 update 比例和各 major module 的 norm；clip 是最后保护，不是掩盖过高 LR。长期频繁裁剪进入 12-F 故障门槛。
- nonfinite 使用 hard failure/同步 skip 语义，不允许 `error_if_nonfinite=false` 后继续把坏值交给 optimizer。[PyTorch clip_grad_norm_](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
## D. zero-grad 与 state 初始化
- 每个成功/跳过 step 后统一 `zero_grad(set_to_none=True)`；gradient accumulation 中间不得清 gradient。
- 每次 TorchAO `optimizer.step()` 都在共同 `optimizer_sr_rng` guard 中执行；所有 rank 的 parameter 遍历顺序、存在 gradient 的 FQN 集合和随机数消费量必须一致。step 后先校验共同 SR state hash，再恢复各自 training RNG。该 guard 只包 optimizer step，不包 noise/timestep、Qwen/VAE 或 DiT forward。
- from-scratch state 由 TorchAO 在第一次存在 gradient 的 step 懒初始化；preflight 后执行一次不更新参数的 state-layout dry run，核对量化条件和字节数。
- growth transition 中旧 FQN 的 moments、quantization metadata 和 per-parameter step 原样加载。新增 slot FQN 在 `post_growth_pre_update` checkpoint 中显式列为 uninitialized state，moments 语义为 0、step=0；第一次有效 gradient 到来时由锁定版本的 TorchAO 创建，不复制相邻层 state。
- scheduler 不因新参数 step=0 而重启；新增层与旧层共享当前 global LR，实际接入强度只由组件 09 的 half-cosine alpha ramp 控制。
- optimizer state load 后、任何 backward 前检查 group FQN/hash、dtype、shape、block size、state class 和 old/new allowlist；`dropped_old_fqn` 必须为 0。
## 不采用
- 不做 LR linear scaling、layer-wise LR decay、style/text branch multiplier 或新层高 LR。
- 不按 stage 重做 warmup/cosine restart。
- 不使用 per-parameter clip、AGC、gradient noise 或自动 LR backoff。
- 不在异常时重置全部 optimizer moments。
# 12-D 最终决定
批准 A+B+C+D：TorchAO AdamW8bit 固定 `2e-5 / (0.9,0.95) / eps 1e-8`；矩阵 weight decay 0.01、敏感参数不 decay；2000-step warmup 后跨阶段 stable，最终由用户单独启动 0.1× floor cosine decay；严格 global mean 后做 FP32 global norm clip 1.0。增长时旧 state 保留、新 state 置零、LR/scheduler 不重启。
# 12-E：Checkpoint 文件契约与迁移
## A. 目录、提交与一致性
每个 raw checkpoint 是一个不可拆分目录 `ckpt_<successful_update>_<checkpoint_id>/`，内部严格分为：
- `model/`：只含可训练模型产物、模型配置与冻结组件引用，可脱离续训 sidecar 用于推理。
- `train_state/`：optimizer、scheduler/growth、RNG、数据位置与训练计数，仅用于续训。
- `manifest.json`：checkpoint ID、parent ID、checkpoint kind、successful/attempted update、stage、active slots、world/worker topology、schema 版本、代码/配置/依赖 hash，以及每个文件的 bytes 与 SHA256。
- `COMPLETE`：最后写入的提交标记；恢复器只扫描有该标记且 manifest/files 全部校验通过的目录。
checkpoint 只允许在完整 optimizer update 或同步 skip 结束并 `zero_grad(set_to_none=True)` 后提交，不保存 microstep gradients。所有 rank 先停在同一 barrier，将文件写入同一文件系统的临时目录；rank 0 完成 checksum/manifest，执行 flush/fsync 后原子改名并最后发布 `COMPLETE`。首版使用同步保存；异步 snapshot 不进入 baseline，避免三套大状态同时占用 host RAM和出现快照时序歧义。
## B. 独立模型产物
- 对未包装的 trainable composite module 导出 canonical FQN state dict，不保留 `module.`、compile wrapper 或其他运行时前缀。
- 权重使用 Safetensors，按不超过 2 GiB 分片，并生成 index JSON；保留每个 parameter/buffer 的原始 BF16/FP32 dtype，不在保存时转换精度。
- `model/config.json` 保存架构 schema、active slot 顺序、当前 growth alpha、RoPE/attention/text/style 接口和推理所需协议。
- 冻结 Qwen3.5-2B 与官方 Mage-VAE 权重不在每个 checkpoint 重复复制；配置记录不可变 repo/revision、配置 hash、tokenizer revision 与 caption/chat framing。模型目录独立于 trainer sidecar，但加载时必须解析这些已锁定依赖。
- raw、PMA 和发布 artifact 使用不同目录与 artifact kind；PMA 不覆盖 raw，也不得被恢复器选中。
## C. TorchAO optimizer sidecar，推荐
DDP 下 optimizer state 在四个 rank 完整复制；baseline 由 rank 0 保存一份 CPU full state：
- 保存 `optimizer.pt`，内容只能是 `optimizer.state_dict()` 的 tensors/primitive containers，不 pickle 整个 optimizer 或 model object；加载使用 `torch.load(..., weights_only=True, map_location="cpu")`，并先导入锁定版本 TorchAO 以注册 `OptimState8bit` safe global。
- 每个 param group 显式携带与 parameter ID 同序的 stable `param_names`；另存 `optimizer_schema.json`，记录 FQN→saved ID、shape、dtype、group、state class、block size、step 与 state tensor metadata/hash。
- TorchAO 的 `OptimState8bit` tensor subclass、codes、scale、qmap 和 step 原样保存，不先反量化成 FP32；大约 3.8–4 GiB 的复制 optimizer state 允许使用一个 ZIP64 PyTorch 文件。
- 恢复器不依赖 parameter object identity 或偶然遍历顺序：先按 FQN/schema 验证，再重建当前 parameter-ID state dict。相同拓扑要求 exact FQN set；增长拓扑只允许显式 new-slot allowlist 没有 state。
- baseline 不使用 DCP。DCP 适合真正分片的 FSDP2 state 和 world-size reshard；当前 DDP 没有 state 分片，且增长需要 partial optimizer load。若 12-A 的 DDP 硬门槛失败并正式改为 FSDP2，再单独切换和验证 DCP，不能让两种格式静默互换。
TorchAO 当前 `OptimState8bit` 已注册 safe global，并实现 DCP 所需 tensor-subclass 操作；这支持上述普通 state-dict 保存，也为未来 FSDP2 留出迁移路径。[TorchAO OptimState8bit](https://github.com/pytorch/ao/blob/main/torchao/optim/subclass_8bit.py) [PyTorch DCP](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)
## D. 续训 sidecar
- `trainer_state.json`：global successful/attempted updates、stage、累计有效样本/FLOPs、WSD segment/LR、clip/nonfinite counters、resolved config/code/dependency hash。
- `growth_state.json`：generation、active/new slots、growth seed、alpha 公式版本、R/k 与 transition checkpoint IDs。
- `data_state.json`：完全继承组件 11 的 shard-level at-least-once 状态，不复制进 optimizer/model。
- `rng/rank-<r>.safetensors` 与配套 JSON：各 rank 的 Python/NumPy/Torch CPU/CUDA training RNG；`rng/optimizer_sr.safetensors` 只保存一份全 rank 共同 SR RNG。
- GradScaler 已关闭，不创建空 scaler 文件。compile cache、FA kernel cache、DataLoader queue 和 sample shuffle buffer 不保存。
普通同拓扑 resume 必须恢复每 rank training RNG；1→4 transition 不伪装成逐随机数续跑：四个 training RNG 由 `run_seed + target_stage + rank` 确定性重新派生，但共同 `optimizer_sr_rng` 从单卡 checkpoint 原样继承并复制给四个 rank。
## E. 单卡→四卡迁移
1. S0 写出并验证完整 raw checkpoint；整个目录先 stage 到本机 NVMe，四 rank 只读已通过 SHA256 的本地副本，避免同时从远端读取。
2. 每 rank 以同一 stable FQN 顺序构建 16层 raw module 与 TorchAO optimizer，加载同一 model/optimizer sidecar；scheduler/successful update 不重启。
3. 模型加载后再应用可选 inner regional compile，最后构造 DDP。DDP 构造前后都校验 trainable parameter hash，四 rank 必须一致。
4. 共同 SR RNG 延续，training RNG 按新拓扑派生；组件 11 的旧单卡 data pass 明确结束，以新 topology seed 开始 S1 data pass并记录 discontinuity。
5. 先做 zero-update forward/loss 检查，再做一次 optimizer update；四 rank 的模型、moments、per-parameter step 与共同 SR state 必须一致。通过后执行组件 09 规定的 200–1000 update S1 稳定窗口。
每 rank 从同一本地文件独立加载会增加一次启动时内存/磁盘读取，但 120 GB host RAM 足够，且 OS page cache 会复用数据；首版不实现复杂的 rank-0 optimizer-state 广播。
## F. 16→20→24 增长迁移
1. 从 transition 前 raw checkpoint 构建目标 active slots；新增 slot 只由 manifest 中的 `growth_seed` 初始化，alpha=0。
2. 模型 load 不使用宽泛 `strict=False`：missing keys 必须精确等于本次新增 slot allowlist，旧 FQN 任一 missing/unexpected/shape-dtype 变化都失败。
3. optimizer adapter 以 stable FQN 把所有旧 state 原样映射到当前 parameter ID；target param groups包含新旧参数，但新增 FQN 的 `state` entry 为空。旧 state 的 codes/scale/qmap/step hash 必须不变，`dropped_old_fqn=0`。
4. scheduler、共同 SR RNG 和旧 training counter 继续；新层 step=0、moments 未初始化，第一次有效 gradient 时由锁定 TorchAO 版本懒初始化。
5. 在任何更新前写 `post_growth_pre_update` raw checkpoint，执行组件 09 的 alpha=0 等价性和 resume 测试，再开始 ramp。
## 验收
- 同拓扑 save→fresh process load 后，下一 update 的 model、optimizer、LR、共同 SR RNG、loss 和 data accounting 与未中断分支一致。
- 1→4 与两个 growth transition 均输出机器可读迁移报告；reused/new/dropped FQN、group hash、state class/bytes 与 checkpoint parent chain 可审计。
- 对半写目录、缺文件、单文件 bit flip、错误 checkpoint ID、错误 TorchAO/PyTorch 版本和 model/optimizer 混装执行故障注入，恢复必须在任何训练 forward 前失败。
- full save 与 load 的 host RAM、NVMe bytes/s、暂停时长和文件总量进入 12-F；不得为缩短暂停而跳过 checksum 或 atomic commit。
# 12-E 最终决定
批准 A+B+C+D+E+F：模型用 canonical-FQN sharded Safetensors；DDP 的一份完整 TorchAO optimizer 用带 FQN schema 的安全 state-dict sidecar；训练元数据、data state 与 RNG 分文件；同步原子提交。1→4 复制完整 state 并新开 data pass，增长以 FQN 只迁移旧 state、新 slot 留空。DCP 只在未来正式切到 FSDP2 时启用。
# 12-F：生产门槛、故障策略与启动矩阵
## 已统一的两类吞吐
组件 01 与组件 11 的数值用途不同，现统一为：
- **数据供给门槛：** 冷缓存、真实 ModelScope→本地 cache→解码/tokenize/crop 流水线独立连续 2 小时，至少 12 samples/s；完整训练中 ready queue 为空造成的 GPU 等待低于 2%。
- **完整训练门槛：** 四卡 20/24层、512 等效面积，包含网络读取、在线 Qwen、在线 Mage-VAE、DiT forward/backward、DDP reduction 与 optimizer step，稳态聚合至少 6 samples/s。低于 4 samples/s 直接停止并重新讨论架构/预编码/缓存；4–6 只允许继续优化，不能进入长期生产。
- 17 个 512 buckets 的 image tokens 仅约 ±1.6%，samples/s 可作为主门槛，同时必须报告 image tokens/s、实际 DiT FLOPs/s 和各阶段耗时，防止用更轻的样本混合虚增吞吐。
来源：[01 约束、预算与验收标准](https://app.notion.com/p/3aaae967ecf281ba8f73fac2f9e4c4f3) · [11 数据与缓存管线](https://app.notion.com/p/3aaae967ecf281db800cfb1d6545f880)
## A. 启动 preflight，任一失败即拒绝启动
1. resolved config 中不得存在 `REQUIRED_AFTER_BENCHMARK`、未知 key 或代码默认回填；stage、source checkpoint、允许前序、active slots、world size、bucket 列表、batch/accumulation、checkpoint slots、LR segment 和数据 revision 均须明确。
2. 锁定并记录 NVIDIA driver、CUDA、PyTorch、TorchAO、FA4/CuTeDSL、Triton、`causal_conv1d`、`fla`、ModelScope Hub 和 Safetensors 版本；FA4 varlen、Qwen DeltaNet fast kernel 与 BF16 支持必须实际执行一次，不能只检查 import。
3. 校验 GPU 数量/型号/32 GB 物理显存、14 vCPU、120 GB host RAM、本地 NVMe quota/空余量、NCCL P2P、网络凭据及数据不可变 revision。NVMe 可用空间至少覆盖 cache 高水位之外的 3 份实测 full raw checkpoint。
4. 构建 trainable composite 后审计 stable FQN、active slots、parameter dtype、decay/no-decay group、TorchAO state class/bytes、冻结 Qwen/VAE 零梯度和共同 SR RNG guard；所有 rank 的 schema/hash 必须一致。
5. 对当前 stage 的全部 17 image shapes、最短/最长文本和空条件各执行 forward/backward；至少完成一次 global-mean、clip、optimizer step、model sample、checkpoint save→fresh load→下一 step。
6. resume 与 transition 是不同命令：普通 resume 要求相同 topology；S1/G1/S2/G2/S3 只能接受配置列明的唯一前序，不允许一次同时改变 world size、深度、分辨率、数据混合或 optimizer。
preflight 输出 `preflight_report.json`，包含逐项 pass/fail 和证据 hash；不得提供 `--force` 绕过硬项。
## B. 性能与稳定性 benchmark
### 通用协议
- 先遍历所有 image/text shape，完成 kernel/allocator/可选 compile warmup；首次编译时间单列，不计入稳态吞吐。
- 每个候选配置至少运行 100 个 warmup successful updates + 500 个 measured successful updates；最终 24层/512 候选执行至少 1000 个 measured updates。验证和 checkpoint 暂停单列，但在线数据、Qwen、VAE、DiT、DDP 与 optimizer 必须计入 step throughput。
- 每 rank 记录 CUDA peak allocated/reserved、host RSS、pinned RAM、step p50/p95/p99、Qwen/VAE/DiT/DDP/optimizer/data-wait 时间和 bucket 分布。
- shape warmup 后任何 rank 峰值不得超过 27.2 GB；最后 500 updates 的 CUDA reserved 增长不得超过 256 MiB，训练进程 aggregate RSS+pinned RAM 不得无界增长或增加超过 2 GiB。不得出现 OOM、host swap 或 allocator 持续爬升。
- 只有同时满足目标 global batch、显存、完整训练 6 samples/s、data wait \<2% 的候选才能比较 none/alternating/all activation checkpoint；选择其中端到端吞吐最高者。
- regional compile 仍只在稳态提升 ≥3% 且无持续 recompile/graph fallback 时启用。checkpoint 的同步暂停按保存间隔摊销后不得超过训练 wall time 的 5%。
### 正确性控制
- eager + SDPA dense reference、eager + FA4 production、checkpoint on/off 和可选 regional compile 使用固定 batch 比较 output/loss/gradient/update；容差由同 backend 重复执行的 control p99 建立，不跨 backend 要求 bitwise。
- DDP global mean 与单卡合并 batch reference 比较；共同 SR RNG 下四 rank 的 model、moments、step 与 SR state hash 必须一致。
- 12-B 的 1000-step FP32-vs-mixed canary、验证 loss EMA ≤3% 回退界限和 resume-next-step 检查是进入正式 S0/S1 的前置硬门槛，不被本节 500-step 性能 benchmark替代。
## C. 运行时数值故障
- 任一 rank 出现 non-finite loss/gradient/norm：所有 rank 同步放弃该 optimizer step，scheduler/growth/PMA 计数不推进，写诊断包并正常终止整个作业。不得跳过后自动长期继续；人工定位后只能从最近完整 raw checkpoint 恢复。
- 稳定段以最近 1000 个 successful updates 建 loss/grad-norm rolling baseline。单点超过 baseline p99 的 3 倍记为警告；超过 10 倍，或连续 3 个 update 超过 3 倍，立即停止并回滚。growth ramp 继续使用组件 09 的同一规则。
- warmup 与 growth ramp 外，rolling 1000-step clip fraction \>5% 警告；\>20% 暂停作业并检查 LR、loss scaling 与数据异常。clip 不能作为自动 LR 调节器。
- 任一 trainable major matrix 持续没有有效 update、参数/optimizer state 跨 rank hash 分叉、共同 SR state 不同、冻结模块出现 gradient，均立即停止。
- 固定验证 loss 连续 3 次相对最近 accepted checkpoint 回退 \>3%，或 tag 控制发生明确回退时，不自动回滚正在运行的作业，但取消 `stage_ready` 并要求人工决定；Tag 仍是最高质量门槛。
## D. 数据与系统故障
- schema key 缺失/未知、validation ID 进入训练、caption 协议或 tokenizer revision/hash 不符，启动前硬失败。
- 单图解码/格式损坏单独计为 `corrupt_sample_skip`；rolling 100k 样本 \>0.1% 警告，\>1% 停止。严格不放大、`aspect_retention_reject`、正常条件为空等预期行为不得混入 corruption 分母。
- `dimension_mismatch>0.1%` 继续按组件 11 立即暂停；token overflow/truncation 使用扫描后写入配置的显式上限，运行值超过上限即取消 stage readiness并排查。
- 完整 shard 在有界重试后仍下载/校验失败、ModelScope 鉴权失败、cache quota 失控、host swap、worker 反复退出或 ready-queue 空等待 ≥2%，均停止作业；不得静默跳过整个 shard。
- 一个 rank 崩溃或 NCCL collective 失败时，其余 rank 全部退出，不能缩成三卡继续。恢复只接受上一份 COMPLETE checkpoint；未完成 shard 按组件 11 从头重放。
## E. Checkpoint 周期与保留
- 正常 full raw checkpoint 在“距上次 1000 个 successful updates”或“距上次 6 小时”两者先到时，于下一个 step 边界同步保存；stage finalize、pre-growth、post-growth-pre-update、ramp midpoint/end 和 pre-decay 额外强制保存。
- 始终保留最近 2 份滚动 full raw、每个已接受 stage 的最终 raw，以及当前 transition 的前后 checkpoint。只有新 checkpoint 通过 fresh-process restore + next-step probe 后，才允许删除更旧滚动副本。
- PMA-10 的十份输入使用 model-only snapshots，不复制 optimizer/RNG/data sidecar；PMA artifact 与 raw retention 分开。阶段边界和异常窗口不得混入同一 PMA window。
- checkpoint 失败、磁盘不足、checksum/atomic rename/COMPLETE 任一步失败时保留上一份完整 checkpoint并停止作业，不为腾空间自动删除最后可恢复点。
- `latest` 只可作为便利指针；恢复器必须按 manifest parent chain 与 COMPLETE 校验选取，不信任文件名或修改时间。
## F. 必做故障注入
在完整 S0 前与四卡 S1 前分别执行，S2 额外重复数据/显存相关项：
1. 在 microbatch、DDP reduction、optimizer step 前后杀死进程；不得提交跨 update 的混合状态。
2. 在 model、optimizer、rank RNG、data state 写入中途杀死 rank 0；半成品必须被忽略。
3. 对已完成 checkpoint 的每类文件做缺失、bit flip、错误 checkpoint ID 与错误 dependency hash；必须在任何训练 forward 前失败。
4. 中断下载、截断 cache shard、撤销 token；分别验证续传/重下、坏 shard 不发布和鉴权明确终止。
5. 注入 nonfinite gradient、OOM、一个 rank 的 SR RNG 分叉、NCCL rank failure、DataLoader worker failure与磁盘写满；不得自动改 batch、backend、world size、LR 或 checkpoint 策略。
6. 强制恢复验证：已完成 shard 不重读，活跃 shard 从头重放，`replayed_samples` 与报告一致；模型/optimizer/RNG/data parent checkpoint ID 完全相同。
## G. Stage 启动矩阵
<table fit-page-width="true" header-row="true">
<tr>
<td>Stage</td>
<td>唯一主要变化</td>
<td>进入长期运行前的附加硬项</td>
</tr>
<tr>
<td>S0：1GPU 16L/256</td>
<td>from scratch</td>
<td>optimizer microtest；1000-step FP32/mixed canary；单卡 checkpoint/resume；至少前几个有效 data passes 由配置固化</td>
</tr>
<tr>
<td>S1：4GPU 16L/256</td>
<td>world size 1→4</td>
<td>共同 SR RNG与跨 rank state一致；global mean；新 data pass；200–1000 successful updates</td>
</tr>
<tr>
<td>G1：4GPU 20L/256</td>
<td>16→20</td>
<td>FQN state迁移；alpha=0等价；post-growth checkpoint；R ramp + post-ramp窗口</td>
</tr>
<tr>
<td>S2：4GPU 20L/512</td>
<td>256→512</td>
<td>17 buckets；不放大/裁剪；none vs alternating checkpoint；数据12/训练6 samples/s；27.2 GB</td>
</tr>
<tr>
<td>G2：4GPU 24L/512</td>
<td>20→24</td>
<td>重复完整growth协议；512下的ramp稳定性与checkpoint恢复</td>
</tr>
<tr>
<td>S3：4GPU 24L/512</td>
<td>完整深度收尾</td>
<td>1000-update endurance；最终吞吐/显存/恢复/固定集验收；PMA window只能从稳定段建立</td>
</tr>
<tr>
<td>H1/H2：768/1024</td>
<td>可选更高分辨率</td>
<td>默认disabled；512成品已接受后重新benchmark并单独批准</td>
</tr>
</table>
每个 stage 达到预算与硬门槛后只写 `stage_ready=true` 和 `stage_report.json`；仍由用户手动 finalize/切换。preflight 可以拒绝错误切换，但不能自动启动下一阶段。
## 不采用
- 不在 OOM 后自动减 batch、增加 accumulation 或改变 checkpoint 频率。
- 不在 FA4/compile/fast Qwen kernel 失败时静默 fallback。
- 不自动重置 optimizer、降低 LR、跳过坏 shard、缩小 world size或从 PMA 恢复。
- 不以单个 training loss 下降代替固定集、tag 控制、吞吐和恢复验收。
# 12-F 最终决定
批准 A–G。该矩阵把完整训练 `6 samples/s` 与数据供给 `12 samples/s` 分开；生产硬门槛为 27.2 GB、目标 batch、ready-queue wait \<2%、无 OOM/swap/状态分叉、checkpoint 可恢复。任何自动 fallback 均关闭；异常先同步停机，stage 只由用户手动切换。
# 12-G：双角色实施审查与性能工程规范
## A. 每个实现单元必须经过两种专业角色
本要求适用于组件 01～12 的每个实现单元、每个训练 stage、每次 kernel/backend 变更和每个性能优化 PR。开发与验收必须分别以两种专业角色执行：
- **AI/模型开发者：** 负责数学目标、tensor shape/dtype、条件语义、梯度路径、随机性、数值稳定性、训练/推理一致性、增长与 checkpoint 语义，以及 reference 测试。
- **Infra/性能开发者：** 负责时间/空间复杂度、GPU/CPU/网络/NVMe 流水线、kernel 选择与安装、算子融合、launch/sync 开销、DDP/NCCL、显存生命周期、可观测性、故障恢复和生产容量。
- 两种角色都必须参与开发和审查。若由同一个人或同一个 AI agent 承担，必须执行两次独立 review 并输出两份记录，不能用一份“已自检”同时代替。
- 实现只有在两类 review 均通过、正确性测试和性能证据齐全后才可标记完成；任何一方可以因语义风险或性能退化阻止进入下一 stage。
## B. 固定开发顺序
1. 先实现最小可读 reference/eager 路径，建立固定输入下的 output、loss、gradient、update 和 checkpoint baseline。
2. 在真实 stage shape、文本长度、microbatch 和目标 kernel 上记录优化前 profile，不根据直觉宣称瓶颈。
3. Infra review 根据 wall-time 占比、GPU timeline、kernel launch、memory 与 I/O 数据选择热点；AI review确认优化不改变模型数学和训练语义。
4. 只优化被 profile 证明的重要路径；完成后用同一输入/config/version 复测正确性和性能，保存 before/after 报告。
5. 正确性 regression 或稳态提升低于测量噪声的优化不得进入生产。涉及 optimizer、attention 语义、精度、batch/global-mean、RNG、checkpoint schema 或 stage 顺序的变化必须重新请求架构批准。
## C. Kernel、复杂度与 GPU 利用率
- 生产环境必须安装、锁版本并在启动时执行验证已批准的高性能依赖：FA4/CuTeDSL、Triton、TorchAO、`causal_conv1d`、`fla` 和 NCCL；仅能 import 但实际走慢速 fallback 视为失败。
- 保持预期的渐近复杂度：数据路径对 samples/tokens 线性；除已批准 attention 外不得引入额外二次扫描。禁止重复 tokenization、重复图像解码、重复 VAE/Qwen forward、反复遍历完整 manifest 或为统计重新物化大 tensor。
- 禁止在 sample、token、head、parameter 或 block 热路径中保留可批量化的 Python 串行循环；优先使用 batched/packed tensor 运算、原生 GQA、varlen、融合 kernel 和异步流水。
- 不得为了接口方便重复 K/V heads、把 varlen 转成大面积 dense padding，或把 GPU tensor搬回 CPU处理。热路径禁止无必要的 `.item()`、`cpu()`、全局 `synchronize()`、逐 tensor host round-trip 和小文件同步 I/O。
- 对 RMSNorm/QK norm、RoPE、AdaLN/modulation、QKVG projection、SwiGLU、residual、loss reduction、gradient norm/clip 与 optimizer step 检查 kernel 数量和中间 tensor。能稳定融合且有实测收益时优先采用官方 fused kernel、`torch.compile`/Triton 或定制算子。
- TorchAO 当前按 parameter 执行 optimizer update；若 optimizer 的 Python/launch 串行开销超过稳态 step time 的 5%，必须单独评估兼容的批处理/融合实现，但不得静默更换 optimizer、state 量化或 stochastic-round 语义。
- 自定义 Triton/CUDA/CuTeDSL 算子必须同时提供易读 reference、forward/backward 对照、BF16/FP32 与边界 shape 测试、非连续输入约束、数值容差、benchmark 和 fallback policy。没有正确性证据不得仅凭速度进入生产。
## D. 全路径时间记录
### 常驻低开销计时
每个训练 update 必须按统一命名记录以下 wall-time 区间：
- ModelScope/cache wait、tar read、JSON/caption、tokenize、decode、EXIF、resize/crop、bucket wait、pin/H2D。
- Qwen forward、VAE encode、condition/style aggregation。
- DiT forward、loss、backward、DDP reduction/wait、global-mean/finite/clip、optimizer step、zero-grad。
- validation/sample、checkpoint CPU snapshot、serialize、checksum、fsync/commit 和总暂停。
CPU/I/O 使用 monotonic high-resolution clock；GPU 异步区间使用 CUDA events。使用 NVTX range 标记 stage、microbatch、Qwen、VAE、DiT block region、attention、MLP、DDP、optimizer 和 checkpoint，不能通过每段强制 `cuda.synchronize()` 获得看似准确但破坏性能的时间。
### 采样式算子 profile
- 常驻计时只保留 coarse phase，目标额外开销 \<1%；若超过则降低采样频率，不得删除关键阶段。
- 每个 stage 在 warmup 后抽样固定 update，使用 PyTorch Profiler/Nsight Systems 记录 GPU active/idle、kernel launch count、平均/分位 kernel duration、inter-kernel gap、CPU launch thread、NCCL overlap、Tensor Core/内存带宽指标。
- Nsight Compute 只针对已由 timeline 证明的重要 kernel，避免全程采集拖慢训练。
- DiT 内部至少可在采样模式下分解 QKVG projection、RoPE、FA4、output projection、norm/modulation、SwiGLU、residual 与 checkpoint recompute；不要求生产每 step 为每个 block创建 CUDA event。
### 输出
- 每 rank 写结构化 `timing.jsonl`，rank 0 聚合 `stage_performance_report.json`。
- 报告必须包含 p50/p95/p99 step time、各 phase 时间与占比、samples/s、image/text tokens/s、DiT FLOPs/s、CUDA allocated/reserved、host/pinned RAM、GPU active ratio、kernel launches、DDP/data idle、checkpoint amortized overhead。
- timing schema、clock 类型、采样率和 profiler 开关写入 resolved config/checkpoint manifest，使不同实现的对比可复现。
## E. 性能审查门槛
- before/after benchmark 必须使用同一 checkpoint、数据序列、bucket/text 分布、batch/accumulation、硬件与软件锁；首次编译和稳态分开报告。
- 任一 coarse phase持续占 step time \>5%，或一组可融合的串行小 kernel 累计超过 5%，Infra review 必须给出优化、保留理由或明确 blocker；不能只报告总 samples/s。
- 优化后必须同时报告吞吐、峰值显存、GPU active、launch/gap 和正确性；不接受用更高显存、改变有效 batch、减少文本/图像 token 或关闭功能换取未注明的速度提升。
- 每个 stage 的 `stage_report.json` 必须引用 AI review、Infra review、profile traces 和 before/after performance report。S0/S1/S2/G1/G2/S3 均不能复用另一 stage 的性能结论。
- 仍以 12-F 的数据 12 samples/s、完整 512 训练 6 samples/s、27.2 GB、data wait \<2% 为最低生产门槛；达到最低值不代表停止优化，长期高占比热点仍需处理。
## F. 必交付审查产物
每个实现里程碑至少保存：
- `ai_review.md`：数学/语义、shape/dtype、梯度、数值、RNG、checkpoint 与 reference 测试结论。
- `infra_review.md`：复杂度、kernel/依赖、GPU timeline、显存、I/O/DDP、故障恢复、热点与优化取舍。
- `perf_baseline.json`、`perf_after.json`、`stage_performance_report.json` 和必要的 profiler trace 索引。
- operator/kernel 单测与 benchmark、完整 dependency lock、resolved config/hash。
缺少任一必要产物时，相关实现保持“待验证”，不得宣称完成或进入长期训练。
# 12-G 最终决定
批准以上 A–F 作为实现阶段的强制工程规范：每个部分均以专业 AI/模型开发者和专业 Infra/性能开发者两个角色开发与独立审查；优先安装和验证高性能 kernel，减少串行小算子、host sync 与不必要的数据移动；所有关键阶段提供低开销计时和采样 profiler，用实测 wall-time 指导提速。
# 12 最终摘要
- **并行：** S0 单卡原生；S1 起同机四卡 DDP。冻结 Qwen/VAE 每 rank BF16 推理，不进入 DDP/optimizer/checkpoint。
- **精度与 optimizer：** 大矩阵 BF16、敏感小参数 FP32；单个 TorchAO AdamW8bit，`bf16_stochastic_round=true`，全 rank 共用隔离的 optimizer SR RNG。
- **训练计算：** FA4 varlen 生产 attention；完整 block non-reentrant activation checkpoint 按 stage 显式配置；regional compile 默认关闭，实测收益 ≥3% 才启用。
- **更新语义：** strict global sample mean、FP32 global clip 1.0；`lr=2e-5`、2000-step 一次 warmup、跨 stage stable、最终手工启动半余弦 decay 至 `2e-6`。
- **恢复：** canonical-FQN sharded Safetensors 模型 + 一份完整 TorchAO optimizer sidecar + 独立 trainer/data/RNG state；临时目录、checksum、manifest、原子提交和 `COMPLETE`。
- **迁移：** 1→4 复制完整 state并新开 data pass；16→20→24 只迁移旧 FQN state，新 slot state 留空、alpha=0 后按组件 09 ramp。
- **生产门槛：** 数据供给 ≥12 samples/s、四卡完整 512 训练 ≥6 samples/s、每卡 ≤27.2 GB、数据等待 \<2%，无 OOM/swap/nonfinite 自动续跑或 rank state 分叉。
- **实施审查：** 每个实现和 stage 必须分别通过 AI/模型开发者与 Infra/性能开发者 review，提供 timing、profile、kernel benchmark 和 before/after 性能证据；达到最低吞吐不等于可以忽略明显热点。
- **当前状态：** 决策闭合但尚未在目标 4×RTX 5090 机器完成 1000-step canary、stage benchmark、checkpoint round-trip、双角色审查与故障注入，因此标记为“待验证”而非“已接受”。
</content>
</page>
