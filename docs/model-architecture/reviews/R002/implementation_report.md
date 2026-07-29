# R002 实现报告

## 范围

R002 建立 Linux x86_64/Python 3.12 的 uv lock、CUDA 12.8/PyTorch ABI 组合、依赖许可证清单、cache-warm 空 `.venv` 恢复和空 cache 源码冷重建证据。没有读取 `.env`、下载数据集、加载 Qwen/VAE、执行 GPU kernel 或修改训练语义。

## 实现摘要

- 安装并固定 uv 0.12.0；`pyproject.toml` 和 `uv.lock` 限定 Linux x86_64/Python 3.12。
- 采用 PyTorch `2.10.0+cu128` / TorchAO `0.16.0` / Triton `3.6.0`，避免 torch 2.11 对 causal-conv wheel 的额外源码耦合。
- 精确锁定 FA4 `4.0.0b24`、CuTeDSL `4.6.0.dev0`、Quack `0.5.3`、FLA `0.5.2` 和 Transformers `5.14.1`。
- 移除全局 prerelease 选择，并直接锁定稳定的 tokenizers、hf-xet 与 sentry-sdk；FA4/CuTeDSL/apache-tvm-ffi 的预发布状态属于明确上游 ABI 约束。
- 为 D001/C001/A001 增加 markdown-it-py、tomli-w、DuckDB、PyArrow 和锁定 Node wheel 的 pyright extra。
- 遵循 wheel-first：先验证 causal-conv 上游 wheel；仅因其 filename/METADATA 版本不一致破坏 uv 重建后，切换到 immutable source commit 编译。

## AI/模型正确性自检

- 未写入任何训练参数或未决 dropout；`all_condition=0.10` 之外的值仍未决。
- 版本组合满足 Qwen3.5 的 causal-conv/FLA import 依赖，但 import 不代表 Qwen fast kernel 已执行。
- 未同时安装传统 flash-attn 2.x；FA4 namespace 不与旧实现混装。
- GQA `pack_gqa`、FA4 backward、TorchAO optimizer 和 causal-conv 数值路径全部保留给对应真实 GPU 任务。

## Infra/性能自检

- 最终 lock 能在保留 `cache/uv` 时从空 `.venv` frozen sync，也能在隔离空 uv cache/venv 中重新取得锁定依赖并从固定 commit 编译唯一源码依赖。
- CXX11 ABI 与 torch wheel 均为 True；源码 commit、toolchain、build variables、extension bytes/hash 已记录。
- `pyproject.toml` 的 uv cache 路径和运行时 `uv cache dir` 均为 `cache/uv`；冷重建使用独立 ignored 临时 cache/venv，没有改变项目 `.venv` 或 `cache/uv`。
- 系统 NCCL 与 PyTorch wheel NCCL 分开记录，未把单卡 import 外推为四卡结论。
- 工作区为 NFSv3 而非已验证 NVMe；容量只达到 cache 低限且没有 checkpoint 预留，后续 preflight 继续硬失败。

## 代理交接

实现代理完成 lock 和 fresh sync 后因模型容量错误退出；主代理保留其工作，修正传递预发布 pins，重跑最终空 `.venv` sync、导入与系统验收。Foundation Infra 预审后，独立冷重建代理使用隔离空 cache/venv 完成 frozen sync、固定源码编译与 import 验证；完整约 5 MB 日志仅保留在 ignored 临时目录，没有复制进 Git。

## 空 cache 源码冷重建

- 隔离 cache 和 venv 初始均为空；`uv 0.12.0` 使用 `/usr/bin/python3`、`--frozen`、`--no-python-downloads` 完成 94 个 distributions 的安装。Bash `time -p` 记录 real/user/sys 为 `2002.709/1706.820/66.070` 秒。
- `causal-conv1d==1.6.2.post1` 从完整 commit `4f6ae4e26ae5fe8af9372f8d312ab25cc4595223` 编译。工具链为 CPython 3.12.3、GCC/G++ 13.3.0、CUDA 12.8.93、CMake 3.31.6、Ninja 1.11.1、glibc 2.39、Torch 2.10.0+cu128；CXX11 ABI=True，build vars 为 `CAUSAL_CONV1D_FORCE_BUILD=TRUE`、`CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE`、`MAX_JOBS=4`。
- 冷构建 extension 为 270,042,496 bytes、SHA-256 `0a216693f3733015216ac936d183cf7ce8e9298c8e880d3ad81ddccc68232903`；cache-warm 项目 extension 为 270,041,048 bytes、SHA-256 `51655c843d2f8228884007b178b42c89eb0dd052495e68029602e39986a93b0e`。绝对构建路径进入未剥离 debug 信息并改变 ELF BuildID，因此两者非 bit-identical；这不阻塞可冷重建结论，但禁止宣称逐字节可复现。
- 22 个关键 imports 全部通过，测试环境显式禁用 CUDA 可见性并设置 Hugging Face/Transformers/W&B offline。未执行 kernel，没有访问模型、数据、DB、reference 或 `.env`。
- 冷构建开始时的 `pyproject.toml` 为 2084 bytes、SHA-256 `77ef980615f6a501330a6943bf2995632fdb7583a4eaf5747a7f08574566bd9a`；当前文件为 2501 bytes、SHA-256 `b718f6c8c235af6df19d7fa10cd1f49348907844ad3ed5ffff65716d5dc87fc2`。差异仅为工具入口。对 `[project]`、`[dependency-groups]` 和完整 `[tool.uv]` 生成排序、紧凑 canonical JSON 后，两者均为 1678 bytes，SHA-256 均为 `7f10b80856f5e6d20b6637c416f753980a7347a3c3e64c601c289b0afb3035cf`。

## Foundation Infra 预审修复

- 预审复现了两个工程入口缺陷：默认 pytest 在 src layout 下 collection 时报 `ModuleNotFoundError: sakuramoon`，默认 Pyright 因缺少扫描边界而会遍历 ignored 大目录并超时；C001 的旧 Pyright 证据还依赖未跟踪的临时配置。
- `pyproject.toml` 现为 pytest 固定 `pythonpath = ["src"]`。默认 `uv run --frozen pytest -q` 不设置手工 `PYTHONPATH` 即完成 collection 和测试。
- Pyright 的 tracked 配置固定 strict 模式、Python 3.12、项目 `.venv`、`src` extra path，并将分析入口限定为 `src/sakuramoon`、`tests`、`tools`。同时显式排除 `.venv`、本地模型/DB/数据/reference、cache、notebook checkpoint 和训练运行产物，避免扫描大资产或 ignored 生成文件。
- 默认 Pyright verbose 仅发现代码入口内的 25 个 Python 文件，其中包含共享工作树中并行 A002 正在新增的合同测试；ignored `src/sakuramoon/config/.ipynb_checkpoints/` 文件已被排除。Pyright 仍从锁定 `.venv` 解析第三方类型信息，但不会把 `.venv` 当作项目源码遍历。
- 该修复只改变工具配置与 R002 证据。没有修改依赖、`uv.lock`、训练语义、资产边界或其他任务实现。
- 修复后的第一次默认全量测试通过 125/125。一次验收复跑期间，D001/A002 的并行工作树继续变化，`tests/unit/docs/test_verify_traceability.py::test_reverse_module_and_config_inventory_is_required` 因先返回 config inventory 错误而非该测试期望的 module inventory 错误失败；结果为 124 passed、1 failed。该文件不属于 R002 允许路径且差异明确来自其他任务，因此 R002 未修改它。A002 提交后，主代理在稳定工作树重跑默认 pytest，最终 125/125 通过；默认 Pyright、Ruff 与 `uv lock --check` 也通过。

## Traceability 登记

- 以 A002 commit `0f3e181` 提交后的 registry revision 7 为基线，只做一次 7→8 递增；现行 source revision、changelog、fingerprint 和既有 requirement ID 均未改写。
- 新增 CPU `environment_lock` profile，唯一 owner 为 R002。该 profile 不映射训练运行时配置、生产模块、reference 模块或 runtime benchmark；pytest 代码面作为工具入口回归，正式证据只指向 `progress/environment-lock.md` 与 `reviews/R002/**`。
- `OPEN-049` 改为 `implemented`，commit 绑定为 `task:R002`，并逐项登记 pyproject、lock、Python version、环境锁、traceability、task、测试/实现/时序报告、许可证、cache-warm/cold-rebuild 证据和 time log。它仍未标为 `verified`，独立 Foundation AI/Infra 复审继续 pending。
- `OPEN-050` 的 FA4/Qwen/TorchAO 真实 kernel 执行和 `OPEN-051` 的四卡/NCCL/NVMe 容量门槛完全不变，仍为 `planned`。
- D001 isolated-repo fixture 仅增加 OPEN-049 已绑定的 R002 tracked paths，使 registry 正例能在隔离副本验证实现路径存在；checker 语义和其他测试断言均未改变。最终 live checker 为 221/221、0 errors，targeted checker tests 36 passed，完整测试 159 passed，Ruff 与 strict Pyright 通过。

## 结论

R002 的依赖锁、cache-warm fresh-venv、空 cache 源码冷重建、默认工程入口与 OPEN-049 追踪登记针对性验收通过，状态进入“等待 Foundation 包级独立复审”。Bit-for-bit 构建、kernel、四卡、真实模型和正式训练门槛未关闭。
