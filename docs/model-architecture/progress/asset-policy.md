# SakuraMoon 最小资产边界

状态：2026-07-30 按用户决定撤销 A001/A002 的本地资产身份、capability 和全仓执行边界系统。

## 本地模型

- Qwen 固定从 `model/qwen_3.5_2B/` 加载，Mage-VAE 固定从 `model/vae/` 加载。
- 加载前只检查固定目录和组件实际需要的文件是否存在；真实格式、结构和张量兼容性由对应模型加载与小型真实测试验证。
- 不维护 `assets/manifest.toml`，不扫描或记录模型文件 bytes/SHA-256，不验证本地 repo/revision/license，不建立 verified capability、identity registry、TOCTOU 或防伪造层。
- 禁止自动下载、缺失补下载、联网替换、默认 cache 替换或任何模型 fallback；必需文件缺失或真实加载失败时直接报错。

## 本地数据与网络数据

- `db/`、用户已准备的本地数据、模型权重和缓存继续由 Git 忽略，不做逐文件资产身份审计，也不进入仓库。
- 训练数据按既定 WebDataset 网络流式链路读取。远端 shard 的长度/摘要验证只用于发现网络传输截断或损坏，不用于重新确认用户已准备的本地模型或数据库身份。
- 凭据只按环境变量名引用；不得读取、记录或提交 `.env` 内容。

## 参考目录

- `reference/` 只供人工理解和对照，可以完全不使用，并继续由根仓忽略。
- 生产代码、测试、preflight、训练和运行时不得 import、执行或调用其中代码。
- 不再维护参考仓库 origin、commit、license hash 或覆盖动态 Python 语义的全仓 AST fact algebra；代码审查和直接 import/path 合同足以执行该边界。

## Git 边界

`.env`、私钥、`model/`、`db/`、`data/`、`cache/`、`reference/`、checkpoint、W&B、profile、trace 与训练产物不得进入 Git。小型源码、配置、schema 和明确需要的测试 fixture 可以正常提交，不要求为普通任务生成无意义的资产证明文件。
