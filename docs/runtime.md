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
