# A001 审查任务

规范任务文件：`docs/model-architecture/progress/tasks/A001.md`。本任务属于 Foundation 包，包末逐 ID 审查。

实现范围：严格 asset manifest/schema、单一 runtime readiness、显式 selected-DB 审计、独立 reference 审计、Qwen/VAE config 契约、DB 元数据边界和参考仓库身份锁。独立 reviewer 必须确认 runtime 不依赖 ignored reference/optional DB，blocked 或 size-drift 资产不会读取大 payload，binding 无公开绕过路径，检查后漂移会硬失败，并区分静态 config 声明与 T020/T021 的真实模型 load/round-trip。复审还必须逐一验证两个 capability 的所有公开 consumer 不能被实例属性遮蔽、实例/类调用都执行相同 gate、字段 mutation 与 runtime/DB graft 均 fail-closed，以及 `required_for_runtime` 对 `true` 和所有非-bool 等值输入严格拒绝。
