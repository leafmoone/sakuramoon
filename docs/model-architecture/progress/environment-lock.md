# R002 Environment Lock

状态：依赖锁已锁定；已验证 CPU cache-warm 空 `.venv` 恢复、空 uv cache 源码冷重建与单卡可见性。不包含 GPU kernel 放行，也不宣称源码构建 bit-for-bit 一致。

## 支持边界

| 项目 | 锁定值 |
|---|---|
| OS/kernel | Ubuntu 24.04 userspace；Linux 5.15.0-173-generic x86_64 |
| glibc | 2.39 |
| Python | CPython 3.12.3；仅 `>=3.12,<3.13` |
| uv | 0.12.0；binary SHA-256 `b6e3cb5b4858d920c63e1d88e31a7a4d8f567073ee4e5e4a1889f93984dc28ea` |
| compiler | GCC/G++ 13.3.0；CMake 3.31.6；Ninja 1.11.1 |
| CUDA toolkit | 12.8.93 |
| GPU/driver | 1×NVIDIA GeForce RTX 5090, 32607 MiB, CC 12.0；driver 580.105.08 |
| NCCL | system 2.25.1+cuda12.8；PyTorch wheel `nvidia-nccl-cu12` 2.27.5 |
| CPU/RAM | 128 logical CPU；约 1.0 TiB RAM；无 swap |

uv 只锁 Python wheel 和 Python extension ABI，不管理 host driver、CUDA toolkit、system NCCL、compiler 或 mount。任何 host stack 变化都必须重新运行 preflight。

## 关键 Python ABI

| 组件 | 版本/来源 |
|---|---|
| torch | `2.10.0+cu128`，官方 cu128 index |
| torchao | `0.16.0` |
| triton | `3.6.0` |
| transformers | `5.14.1` |
| flash-linear-attention | `0.5.2` |
| flash-attn-4 | `4.0.0b24` |
| nvidia-cutlass-dsl | `4.6.0.dev0` |
| quack-kernels | `0.5.3`，避免与 FA4 的 CuTeDSL exact pin 冲突 |
| causal-conv1d | `1.6.2.post1`，Git commit `4f6ae4e26ae5fe8af9372f8d312ab25cc4595223` |
| tokenizers / hf-xet / sentry-sdk | `0.22.2` / `1.5.2` / `2.66.1`，显式稳定版 pins |

`pyright[nodejs]` 锁定 `nodejs-wheel-binaries 24.16.0`；host PATH 中的 Node 18.19.1 不作为 pyright 可重建性依赖。

## Wheel 优先与源码例外

所有可用依赖优先使用 lock 中带 URL、bytes 和 SHA-256 的上游 wheel。`causal-conv1d` 的候选上游 wheel 为：

- filename: `causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`
- URL: `https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`
- bytes: `193835395`
- SHA-256: `c16c1c48d4fa63415cc797e02d69f97248c57c04627d99e394d5bb0ef266e288`

uv 可下载该 wheel，但拒绝把 filename 的 local version 与内部 `METADATA` 的 `1.6.2.post1` 视为同一 distribution，导致每次 sync 重新准备。该 wheel 因可重建性不合格而未进入最终 lock；没有回退到临时 pip。

源码例外固定 tag commit、PyTorch runtime match、CXX11 ABI=True、CUDA 12.8、GCC 13.3 和 `MAX_JOBS=4`。cache-warm 项目 extension 位于 ignored `.venv/lib/python3.12/site-packages/`，为 `270041048` bytes，SHA-256 `51655c843d2f8228884007b178b42c89eb0dd052495e68029602e39986a93b0e`；uv 的项目缓存由 `pyproject.toml` 固定为 ignored `cache/uv`，`uv cache dir` 也解析为该路径。两者都不进入 Git。源码默认编译 `sm_75/80/87/90/100/120`；这只是构建兼容性，不是 kernel 正确性或性能证据。

## 空虚拟环境恢复证据（cache-warm）

最终命令：

```bash
uv venv --clear .venv --python /usr/bin/python3
time -p uv sync --frozen --python /usr/bin/python3
```

执行时删除并重建了 `.venv`，但保留了 `cache/uv`。结果：从空 `.venv` 安装 94 个 distributions；`uv sync` real time 24.81 秒、user 0.13 秒、sys 1.76 秒；无下载、无重新编译，`causal-conv1d` 复用了 uv 缓存中的构建结果。NFS 上清理旧 `.venv` 使完整命令 wall time 为 77.3 秒。`uv lock --check` 解析 95 个 lock package entries 并通过。

关键 Python/extension 导入全部通过：torch、torchao、transformers、diffusers、safetensors、modelscope、wandb、webdataset、Pillow、einops、pydantic、triton、causal_conv1d、causal_conv1d_cuda、fla、flash_attn namespace、cutlass、quack、duckdb、pyarrow、tomli_w、markdown_it。Torch 报告 CUDA 12.8、1 个设备、RTX 5090、CC 12.0。导入期间仅出现 TorchAO docstring 的上游 `SyntaxWarning`；没有静默 fallback 证据。

这些结果证明 frozen lock 可以在保留项目 uv cache 时恢复到空 `.venv`，并证明 extension 可加载和 GPU 可见。后续独立空 cache 冷重建见下一节。FA4、FLA、causal_conv1d、TorchAO 的真实 forward/backward/update 仍分别由 `K001/T021/T040` 关闭。

## 空 cache 与空 venv 冷重建证据

使用 ignored 隔离根目录中的空 `uv-cache/` 与空 `venv/`，执行：

```bash
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=<isolated>/venv UV_CACHE_DIR=<isolated>/uv-cache uv sync --frozen --python /usr/bin/python3 --no-python-downloads -vv
```

命令 exit 0，安装 94 个 distributions；Bash `time -p` 记录 real/user/sys 为 `2002.709/1706.820/66.070` 秒。`causal-conv1d` 从 commit `4f6ae4e26ae5fe8af9372f8d312ab25cc4595223` 重新编译，使用 CPython 3.12.3、GCC/G++ 13.3.0、CUDA 12.8.93、CMake 3.31.6、Ninja 1.11.1、glibc 2.39、Torch 2.10.0+cu128、CXX11 ABI=True，以及 `CAUSAL_CONV1D_FORCE_BUILD=TRUE`、`CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE`、`MAX_JOBS=4`。

冷构建 extension 为 `270042496` bytes、SHA-256 `0a216693f3733015216ac936d183cf7ce8e9298c8e880d3ad81ddccc68232903`、ELF BuildID `0ca5b226d68788d717040e68c0953837b277f838`。cache-warm extension 的 BuildID 为 `bf94bdd2e03281c4d3ffd3f9628531893b3635b1`。两份 `.so` 因绝对构建路径进入未剥离 debug 信息而非 bit-identical；这证明固定依赖可从空 cache 重建，但不证明逐字节可复现或 kernel 正确性。

22 个关键 imports 全部通过；验证显式设置 `CUDA_VISIBLE_DEVICES=''`、Hugging Face/Transformers offline 与 W&B offline，未运行 GPU kernel。隔离重建没有改变项目 `.venv` 或 `cache/uv`，没有访问模型、数据、DB、reference 或 `.env`。完整约 5 MB 日志和本地校验清单仅保留在 ignored 隔离目录，不进入 Git；tracked 证据为 `reviews/R002/cold-rebuild-report.json`。

冷重建启动与当前 `pyproject.toml` SHA-256 分别为 `77ef980615f6a501330a6943bf2995632fdb7583a4eaf5747a7f08574566bd9a` 与 `b718f6c8c235af6df19d7fa10cd1f49348907844ad3ed5ffff65716d5dc87fc2`。对 `[project]`、`[dependency-groups]`、完整 `[tool.uv]` 生成排序紧凑 JSON 后，两份 canonical lock input 均为 1678 bytes，SHA-256 `7f10b80856f5e6d20b6637c416f753980a7347a3c3e64c601c289b0afb3035cf`；中途加入的 pytest/Pyright 配置不改变依赖重建结论。

上段的“当前”仅指 cold-rebuild evidence 捕获时点，不是滚动工作树身份。提交 `9775755a24a6f3bd55e1e35562b3602a0bf968bb` 随后加入精确 `uv==0.12.0` 工具门槛；该提交后的 `pyproject.toml` 为 2531 bytes、SHA-256 `0fa1c0488ce0acbb573340f81f017a88a0fff2c5e530babf79f9efc31f6bbea3`，canonical lock input 为 1708 bytes、SHA-256 `a1a3559fcdf9454bd7a04a14664718a46d0f9b15e321202123a35bfe4df28d5d`。`uv.lock` 仍为 75101 bytes、SHA-256 `ee6a52d796e029a9a19db1e59011f8a801f3ea3b451f3a70b0190679dc2244ef`；依赖、source 与 build variables 未改变。不可变的时点映射见 `reviews/R002/post-remediation-binding.json`，既有 cold-rebuild artifact 保持原样。

## 存储与正式训练阻塞

工作区 mount 是 NFSv3 PVC（`local_lock=none`），总量 400 GiB，验收时约 363 GiB 可用。容量已达到 300–500 GiB cache 区间的低端，但它不是已验证本地 NVMe，且尚未在 cache 高水位之外预留或实测 3 份 full raw checkpoint 空间。因此：

- 此项不伪装为 R002 依赖锁失败；
- `D012/T042/T050` 必须重新验证 throughput、跨进程协调、atomic rename/fsync、eviction 和 checkpoint 实际 bytes；
- 正式数据/训练 preflight 在真实 NVMe 路径与 checkpoint 余量确认前保持硬阻塞，无绕过开关。
