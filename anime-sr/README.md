# anime-sr — AnimeSR-Mage-UFlow（4× 二次元盲超分）

SakuraMoon 仓库的独立子树，实现 v2.0 执行计划书
（`docs/plan-v2.0.md`，2026-08-26 冻结）：冻结 Mage-VAE latent 空间上的
残差流匹配 1-step 超分。与 T2I 主线完全隔离（§8 独立代码库原则）。

## 文档（先读）

| 文档 | 内容 |
|---|---|
| `docs/plan-v2.0.md` | **权威规格**（2320 行完整计划书，唯一 source of truth） |
| `docs/design.md` | 架构/数学/里程碑/决策日志（浓缩版） |
| `docs/data-contract.md` | 张量契约、manifest、eligibility、验证集 |
| `docs/degradation-v1.md` | 5 profile 退化链 + codec bank |
| `docs/evaluation-protocol.md` | 评测集、指标、Faithful 硬门槛、M0 门 |
| `docs/evidence/m0-bakeoff-2026-08-25.md` | M0 VAE bake-off 实测证据（PASS） |
| `docs/superseded/rfmsr-design-v1.md` | 旧 RFMSR 0.3B+GAN 设计（已废弃） |

## 目录

```
anime-sr/
├── config/                    # 全部训练参数（仓库规则：参数只在 TOML）
│   ├── base.toml              #   冻结核心（结构/流/桶/推理，§21）
│   ├── stage1_flow.toml       #   M4 Phase I（10M 曝光，flow-only）
│   ├── stage2_faithful.toml   #   M5 Phase II（2M 曝光，one-step 保真）
│   ├── smoke.toml             #   M3 15-20M 冒烟模型
│   ├── benchmark.toml         #   M4 硬件基准（§15.2）
│   └── data.toml              #   M1 数据/过滤/退化（§10/§11）
├── src/anime_sr/
│   ├── vae/                   # 冻结 Mage-VAE：mage_vae_impl.py（vendor 源）
│   │                          #   + mage.py（FrozenMageVAE 包装：确定性
│   │                          #   encode / decode_with_grad / SHA256 指纹）
│   ├── config/                # pydantic schema + TOML 深度合并 loader
│   ├── model/                 # U-Flow 主干 + pixel 条件编码（M3 起）
│   ├── flow/                  # LQ-centered 残差流匹配（M3 起）
│   ├── data/                  # M1 数据管线 + 退化算子
│   ├── train/                 # M4/M5 训练循环
│   ├── eval/                  # M0 ceiling 与 §16 评测
│   └── cli/                   # CLI 入口（benchmark/train/eval）
├── tests/                     # 配置 schema + VAE 包装测试
└── docs/
```

## 快速上手

```powershell
# 本地（Windows，RTX 4060 可用；全局 python 即 torch 2.11 + cu128）
cd D:\sm_data\sakuramoon-dev
$env:PYTHONPATH = "anime-sr/src"
python -m pytest anime-sr/tests -v          # 13 项测试（VAE 测试需本地权重，缺失自动 skip）

# 加载配置（base + 阶段 overlay）
python -c "from anime_sr.config import load_config; `
  print(load_config('anime-sr/config/base.toml','anime-sr/config/stage1_flow.toml').flow.pred)"
```

- Mage-VAE 权重**不入库**：本地 HF 缓存
  `C:\Users\PC\.cache\huggingface\hub\models--mage-flow-community--Mage-Flow\snapshots\*\vae\diffusion_pytorch_model.safetensors`；
  测试用 `ANIME_SR_VAE_PATH` 指定。缺失路径 → 直接报错（仓库规则）。
- 训练机：sakrua2（双卡 Hygon DCU，DTK 26.04，venv `sakuramoon-dtk-venv`），
  与 SakuraMoon 主线同机隔离运行。

## 里程碑状态（§13）

| M | 状态 |
|---|---|
| M0 VAE ceiling | **PASS**（08-25 bake-off，见 evidence；2000 真实+500 合成 harness 待建） |
| M1 数据管线 | 未开始 |
| M2 pixel baseline | 未开始 |
| M3 冒烟 | 未开始（`config/smoke.toml` 已备好） |
| M4 Phase I | 未开始（`config/stage1_flow.toml` 已备好） |
| M5 Phase II | 未开始（`config/stage2_faithful.toml` 已备好） |

## 纪律

- 本目录内：训练参数只进 `config/*.toml`；权重/密钥不入库；CLI 打印
  traceback；改动后跑 Ruff + Pyright + 单元测试。
- 计划书 §17.3 的发布 manifest SHA256 要求对 anime-sr 子树有效（用户冻结
  决策，优先于仓库「不维护哈希」通则）。
