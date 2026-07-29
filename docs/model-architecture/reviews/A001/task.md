# A001 审查任务

规范任务文件：`docs/model-architecture/progress/tasks/A001.md`。本任务属于 Foundation 包，包末逐 ID 审查。

实现范围：严格 asset manifest/schema、只读 preflight、Qwen/VAE config 契约、DB 元数据边界和参考仓库身份锁。独立 reviewer 必须确认 blocked 资产不会读大 payload 或绕过硬失败，并区分静态 config 声明与 T020/T021 的真实模型 load/round-trip。
