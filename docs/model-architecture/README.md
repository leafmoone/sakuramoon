# SakuraMoon 模型架构文档

本目录是模型架构、开放事项、实施路线图和验证证据的本地唯一入口。自 2026-07-29 完成迁移后，不再依赖 Notion MCP 读取或维护这些内容。

## 当前有效文档

- [确定方案](current/confirmed-decisions.md)：实现时唯一允许引用的架构基线。后续明确修订覆盖历史方案。
- [待做与待确认](current/open-items.md)：尚未关闭的决定、验证项和工程门槛。
- [可观测性与评估](current/observability-and-evaluation.md)：W&B、耗时、FID/IS 与评估调度的本地补充决定。
- [迁移报告](MIGRATION_REPORT.md)：Notion 导出的范围和完整性验证。
- [SHA-256 校验和](SHA256SUMS)：原始导出、现行入口和会话原始资料的内容校验。

## 目录约定

- `archive/notion/`：Notion 原始导出的只读档案，保留论证历史与取代关系，不作为训练配置来源。
- `current/`：现行方案和开放事项，后续直接在本地维护。
- `sources/`：会话或附件形式的原始资料，只用于追溯。
- `reviews/`：后续每个实现任务的 AI 正确性与 Infra 性能审查。
- `progress/`：路线图、任务状态、耗时和证据索引。

## 维护规则

1. 架构或协议变化必须先更新 `current/`，再修改代码和配置。
2. 每条实现要求必须能追踪到配置键、模块文件、测试、benchmark 和证据产物。
3. `archive/` 内容不可修改；修订通过新的本地决策记录完成。
4. 不在文档、TOML、日志、W&B 或 Git 中保存访问令牌。ModelScope 凭据只从 `MODELSCOPE_API_TOKEN` 环境变量读取。
5. 代码实现、AI 审查和 Infra 审查未全部通过前，任务保持“待验证”。
