# R002 实现报告

## 范围

R002 建立 Linux x86_64/Python 3.12 的 uv lock、CUDA 12.8/PyTorch ABI 组合、依赖许可证清单和空环境重建证据。没有读取 `.env`、下载数据集、加载 Qwen/VAE、执行 GPU kernel 或修改训练语义。

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

- 最终 lock 能从空 `.venv` frozen sync，且第二次缓存命中未重新构建源码。
- CXX11 ABI 与 torch wheel 均为 True；源码 commit、toolchain、build variables、extension bytes/hash 已记录。
- 系统 NCCL 与 PyTorch wheel NCCL 分开记录，未把单卡 import 外推为四卡结论。
- 工作区为 NFSv3 而非已验证 NVMe；容量只达到 cache 低限且没有 checkpoint 预留，后续 preflight 继续硬失败。

## 代理交接

实现代理完成 lock 和 fresh sync 后因模型容量错误退出；主代理保留其工作，修正传递预发布 pins，重跑最终空环境 sync、导入与系统验收，并完成本报告。没有重复执行已完成的冷缓存构建。

## 结论

R002 实现与针对性验收通过，状态进入“等待 Foundation 包级审查”。kernel、四卡、真实模型和正式训练门槛未关闭。
