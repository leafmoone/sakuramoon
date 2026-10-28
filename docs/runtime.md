# 运行环境与部署

支持的解释器范围是 Python 3.11–3.12。训练环境和推理环境分别准备：基础依赖足以运行独立推理，`train` / `eval` extras 提供数据、优化器与评估工具；CUDA/DTK 的底层 wheel 必须与硬件匹配。

## NVIDIA / WSL2

先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)安装与驱动及 GPU 架构匹配的 torch。不要把其他项目的虚拟环境整体覆盖成这份项目的依赖。示意流程：

```bash
uv venv --python 3.12
source .venv/bin/activate
# 在此安装官方提供、适配本机硬件的 torch wheel。
uv pip install -r pyproject.toml
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m sakuramoon.cli.infer --help
```

需要训练或评估工具时，分别使用 `uv pip install -r pyproject.toml --extra train` 和 `--extra eval`。它们不是运行单张推理图的前置条件。准备完成后可用 `uv run --no-sync`，避免运行时重新解析并替换硬件相关依赖。

实际 NVIDIA 验证使用 torch 2.13.0 + CUDA 13.2、Transformers 5.14.1、RTX 5090 D；这是一次已验证组合，不代表所有允许的 torch 版本均已实测。Transformers 5.15 已改变项目使用的 Qwen 内部接口，因此当前依赖限制为 5.14.x。

`flash-attn-4`、CUTLASS DSL、quack 和原生 CUDA causal-conv1d 都不是当前 SDPA 推理的必需依赖。Qwen 卷积兼容入口优先使用已安装的 FLA；没有 FLA 时使用 PyTorch。可选 `cuda-fast` extra 只面向需要自行验证 FLA 的 CUDA 用户。

## Hygon DCU / DTK

沿用厂商匹配的 torch、Triton、DAS FlashAttention 和 FLA 组合。不要在现有 DTK 环境运行无差别 `uv sync` 或安装 NVIDIA torch wheel。读取模型前先加载厂商运行库环境，例如：

```bash
export DTK_ROOT=/path/to/dtk
source "$DTK_ROOT/env.sh"
source /path/to/existing-dtk-venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python -c 'import torch; print(torch.__version__, torch.version.hip); print(torch.cuda.device_count())'
python -m sakuramoon.cli.infer \
  --root /path/to/runtime --checkpoint /path/to/selected-checkpoint \
  --prompt "A moonlit garden, anime illustration." \
  --attention-backend flash --output-dir outputs/dcu
```

`libgalaxyhip.so.5` 找不到，通常说明厂商环境尚未加载；不是模型文件丢失。`torch.cuda.is_available()`、`cuda:0` 在 HIP 环境中也是正常 API。

不要在显存已被生产训练占满时再加载完整推理模型。先检查设备空闲内存，在独立进程和输出目录中验证；不要为了试推理而擅自停训练或更改训练的配置/依赖。

## 路径与状态

代码仓库和运行资产根目录可以分离：训练/数据 CLI 使用 `--root` 解析相对运行路径，配置树用 `--config-root` 单独指定；推理用 `--root` 寻找冻结编码器。资产、缓存和 checkpoint 不应提交到 Git。

| 内容 | 建议位置 / 约束 |
|---|---|
| 代码与维护文档 | Git 工作目录 |
| 冻结模型 | 运行根的 `model/qwen_3.5_2B`、`model/vae` |
| 原始模型产物 | 运行根的 `output_model/` |
| 数据服务队列 | `data/data-service-mainset.json`，需要随续训状态保留 |
| 数据服务 socket / lock | 运行根的 `runs/`；会在入口处解析成绝对路径 |
| 生成图片 | `outputs/`，或仓库外的显式目录 |
| stack 日志 / PID | 默认 `${RUNTIME_ROOT}/logs`、`${RUNTIME_ROOT}/runs/stack` |
| 优化器故障材料 | 默认当前工作目录下的 `artifacts/`；可通过已有构造参数或 `SAKURAMOON_FORENSIC_DIR` 指定 |
| 逐样本跟踪 | 默认关闭；显式设置 `SAKURAMOON_SAMPLE_TRACE_PATH` 才写入 |

不要把旧队列路径的默认值变化当成一次无状态升级。已有训练必须保留原队列路径或迁移队列文件，否则可能重新开始分片顺序。历史 checkpoint 的恢复仍应使用训练时的 resolved config 和状态，按[恢复契约](config-and-resume.md)做显式迁移。

云平台的系统盘、持久盘和共享盘语义不同。尤其容器系统盘往往随实例释放丢失，而共享目录可能有配额；应向平台确认并分别安放大权重、缓存和最小恢复材料，不依据目录名称猜测持久性。

## 训练进程与可选发布

`scripts/training_stack.sh` 从自身位置推导代码根，默认把运行根设为代码根；生产部署可以显式覆盖：

```bash
PROJECT_ROOT="$PWD" \
RUNTIME_ROOT=/path/to/runtime \
VENV_ROOT=/path/to/dtk-venv \
CONFIG_NAME=train_g1.toml \
WORKLOAD_ENV_FILE=/path/to/protected-workload-env.nul \
bash scripts/training_stack.sh start
```

这是生产管理命令，需要事先完成数据、配置、环境和恢复点准备，不是新机器的零配置启动示例。先核对 workload 环境文件的生成方式和权限。

模型上传默认关闭。只有操作者明确提供 `PUBLISH_ENABLED=1`、`REPO_ID` 和所需 token 时才启用 publisher。其他发布脚本同样要求显式目标仓库，不再默认写入原作者的账户。脚本沿用调用者的代理和认证环境，不自动执行平台私有 profile 文件。

历史 Danbooru 加工工具 `run_deepghs_quality_pipeline.py` 的写入模式还要求显式设置 `SAKURAMOON_DATASET_REPO_ID`；源/目标 revision 可分别用 `SAKURAMOON_DATASET_SOURCE_REVISION`、`SAKURAMOON_DATASET_TARGET_REVISION` 指定。其设备列表、batch size、下载并发和超时按 CLI 参数生效，不再锁死为两卡固定吞吐设置。

该工具的 ONNX 元数据分类需要额外的 `data-tools` 依赖组（`uv pip install -r pyproject.toml --extra data-tools`），并按其运行模式安装训练/数据下载依赖；普通图片生成不需要 ONNX Runtime。

快照发布工具还需要系统提供 `tar`、`zstd`、`split` 和 `flock`；这些工具不是普通推理的依赖。训练 stack 保留调用者的 HOME/USER 身份，不强制使用 root，也不强制设置第三方 Hugging Face 镜像地址。

快照脚本不再包含用户级 Codex 配置，并排除 `.env*`、私钥常见命名、`.ssh`、`.codex`、日志、core dump 和临时文件。这些排除规则不是对任意目录内容的秘密扫描；选择快照源时仍应只包含预期的项目资产。迁移脚本默认只恢复，设置 `START_TRAINING=1` 才会启动训练；发现已有 trainer 时拒绝恢复。

## 本地只读审计 CLI

在已有环境的仓库根目录运行；以下仅为用法示例，不会启动训练。metrics 路径和闭区间端点应按待审计的本地文件选择：

```bash
PYTHONPATH=src uv run --no-sync python scripts/check_configs.py --config-root config
PYTHONPATH=src uv run --no-sync python scripts/audit_training_metrics.py /path/to/metrics.jsonl \
  --start 100 --end 200
```

`check_configs.py` 检查配置继承图，`config/external-parents.toml` 声明的外部父节点不读取、不遍历；它不验证业务 schema 或运行资产。`audit_training_metrics.py` 接受 `.jsonl` / `.jsonl.gz`，默认统计 `main_loss`，可用 `--loss-field` 指定字段；缺失行只表示遥测缺口，不能推断该 update 未执行。两者默认向 stdout 输出 JSON；可选 `--out` 会写出文件，metrics 审计禁止将其指向输入。检查退出码和错误输出，不能仅凭生成了 JSON 就认定审计通过。

历史用法及当时的验证数字见提交 `b15d7eb` 的原路径 `docs/handover-20261017.md`；它们不是当前检查结果。

## 验证范围

```bash
# 已准备的 Linux 环境；GPU 测试按硬件可用性单独选择。
PYTHONPATH=src uv run --no-sync ruff check src
PYTHONPATH=src uv run --no-sync pyright --pythonpath "$(command -v python)"
PYTHONPATH=src uv run --no-sync pytest tests/unit/model tests/unit/conditioning tests/unit/checkpoint
```

部分历史 `tests/unit` 用例实际要求 CUDA 设备，目录名本身不保证 CPU-only。DCU 厂商运行时在隐藏全部设备时也可能初始化失败，不要把这种启动失败当成断言失败，更不要在繁忙生产容器中反复重跑。

仓库已有严格类型检查欠账；没有通过的全库检查不能表述为通过。当前 NVIDIA 证据覆盖真实 checkpoint 推理、注意力隔离/梯度和完整组合模型的小规模前后向；未覆盖 NVIDIA 多卡 DDP、生产 CMuon 长训、最小显存边界或跨厂商逐像素一致性。

旧的 `test_cmuon_fp32_forensic.py` 中仍有按“FP32 below-floor 必须硬失败”编写的断言；当前 F3 逻辑对有限的 below-floor 更新采用软接受，相关 F3 用例已在 NVIDIA 上通过。旧断言与当前语义的冲突应单独维护，不能标作环境性失败或全套测试通过。

## Optional equivalent fusion validation

`[kernels] equivalent_fusions = false` is the default. Opting in enables lazy
`torch.compile(backend="inductor", fullgraph=True)` boundaries for QK RMSNorm →
learned gamma → existing 2D RoPE, block RMSNorm + timestep modulation, and gated
residual additions. No attention algorithm or GEMM boundary changes. Packed
sample gathers are inside the graph, so the compiler can avoid full modulation
copies; allocation behavior must still be verified on the target compiler.
Shared/sample gradients use autograd's reductions. Outer model compilation inlines
the same operations rather than nesting cached compiler wrappers.

These are real compiled boundaries, not merely module regrouping. They retain
FP32 statistics and position arithmetic, the NoPE/Y/X split, independent Q/K head
counts and contiguous RoPE outputs. Explicit casts preserve the reference source
operations, **not** a bitwise BF16/FMA/reduction-order guarantee. No automatic
fallback or claimed CUDA/DTK compatibility: an unsupported compiler fails visibly.
Configure before DDP/outer compile; enabling is not saved as architecture metadata.

For a prepared environment, the asset-free CLI uses dimensions from the specified
resolved/inherited config and synthetic sample lengths. It checks every output and
input/parameter gradient, prints the FX graph/compiler kernel metric, and records
separate warmed reference/compiled profiler traces:

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONPATH=src \
uv run --no-sync python -m sakuramoon.cli.validate_fusions \
  --config train_g1_hybrid_cmuon_ns4_core.toml --device cpu --dtype float32 \
  --sample-lengths 5 3 7 --warmup 3 --iterations 5 --explain \
  --trace-dir /tmp/sakuramoon-fusion-traces
# On the verified experiment backend, choose its actual device and sample lengths.
# BF16 tolerances must be assessed, not silently relaxed to force acceptance:
OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONPATH=src \
uv run --no-sync python -m sakuramoon.cli.validate_fusions \
  --config "$EXPERIMENT_CONFIG" --config-root "$EXPERIMENT_CONFIG_ROOT" --device "$EXPERIMENT_DEVICE" --dtype bfloat16 \
  --sample-lengths 31 17 9 --atol 0.06 --rtol 0.06 --explain \
  --trace-dir /tmp/sakuramoon-bf16-traces
```

`TORCH_LOGS=output_code` additionally exposes generated kernels for allocation and
rounding inspection. `generated_kernel_count` is a compiler metric; CPU ATen event
counts are **not GPU kernel launches**. Device trace `kernel` events are counted
only on a CUDA-compatible profiler runtime. This is a primitive benchmark, not a
full training-step result. CPU peak GPU memory is reported unavailable.

For complete-step timing, the existing synthetic training harness remains the
entrypoint. Compare separate otherwise-identical isolated configs with final norm
removed in both and only `kernels.equivalent_fusions` changed. Its profiler stage
is separate from timed updates:

```bash
PYTHONPATH=src uv run --no-sync python scripts/benchmark_training_perf.py \
  --config "$EXPERIMENT_CONFIG" --config-root "$EXPERIMENT_CONFIG_ROOT" --repository-root "$PWD" \
  --output-root "$ISOLATED_BENCH_OUTPUT" --gpus 1 \
  --warmup-updates 5 --measure-updates 10 --profiler
```

For real DDP use that script's documented `accelerate launch` two-GPU command and
`--gpus 2`; keep the experiment activation-checkpoint and compile settings. Actual
attention vendor libraries/assets are required. CPU primitive checks do not accept
the real packed attention, GPU/DCU DDP or end-to-end performance. Keep the reference
path unless the verified target demonstrates an actual benefit.

### BF16 cast policy and separate controls

Compiled fusion explicitly requests Inductor `emulate_precision_casts=true`.
Without it, a compiler can elide BF16 downcast/upcast pairs, including products
before the shared norm-weight gradient reduction. A near-cancellation regression
at hidden size 2560 and packed lengths 31/17/9 rejects that behavior without
loosening the original error bound. Unsupported vendor options fail visibly.

For outer regional compilation, the selected mode's actual options are retained
and the precision option is added. This affects the **whole compiled block**, not
only the new primitives; it may change unrelated intermediate rounding relative
to the previous compiled reference. It does not promise bitwise equality or a
speedup. Keep these controls distinct when testing the real backend:

| Control | `equivalent_fusions` | `torch_compile_enabled` | `torch_compile_emulate_precision_casts` |
| --- | --- | --- | --- |
| Eager reference | false | false | false |
| Existing compiled reference | false | true | false |
| Precision-policy-only control | false | true | true |
| Fused candidate | true | true | true (also enforced by fusion) |

All three kernel flags default false; the existing compiler remains unchanged
when fusion and the explicit precision-policy control are both disabled. Do not
silently change the reference policy to claim parity. Compare forward/backward,
actual DDP/activation-checkpoint behavior and full-step timing before adoption.
`validate_fusions --only packed_norm_modulation` can replay one failing primitive
with the same seeded inputs as the full invocation; omitted cases are printed as
unvalidated, not passed. Run remaining cases before accepting the complete path.

### Experiment source reconciliation

The supplied experiment source used earlier optimizer-construction and inline
observer forms, removed the optional diagnostics/guard APIs, and added an opt-in
condition-gradient spike capture. The merged code retains current GitHub
consolidation, type/device guards and default-off diagnostic APIs; it does not
claim those retained APIs were experiment additions. The extra spike capture is
retained through `SAKURAMOON_CONDITION_SPIKE_LOG`, requiring an absolute external
JSONL path, rank zero and the canonical independently clipped condition-token
partition. It records scalar pre-clip facts, not weights or gradients. No
variable means no capture. The additive clipping callback runs only after both
partitions pass finite checks and before either gradient set is scaled; it does
not change clipping thresholds, gradient values or default behavior. The supplied
five-file snapshot did not include the experiment clipping implementation, so
this callback's tested timing contract is not a claim of byte-identical source.
