# Notion 到本地文档迁移报告

迁移日期：2026-07-29

## 范围

已归档 16 个 Notion 实体：

- 1 个架构决策中心。
- 12 个组件页面（01 至 12）。
- 2 个会话重整 v2 页面：确定方案与待做清单。
- 1 个组件决策数据库的 schema/元数据；数据库中的 12 个组件正文已分别归档。

同时归档了会话附件 `early-architecture-conversation.txt`，用于追溯早期方案。该附件在复制前未发现 `ms-...` 形式的 ModelScope token。

## 本地路径

| 内容 | 本地位置 | 用途 |
|---|---|---|
| 当前确定方案 | `current/confirmed-decisions.md` | 实现唯一架构基线 |
| 当前开放事项 | `current/open-items.md` | 决定与验证清单 |
| Notion 原始页面 | `archive/notion/*.md` | 只读历史档案 |
| 早期会话资料 | `sources/early-architecture-conversation.txt` | 原始证据 |
| 内容校验 | `SHA256SUMS` | 迁移完整性检查 |

## 完整性检查

- 原始导出文件数：16。
- 组件编号覆盖：01–12，无缺号。
- 每个导出文件均包含完整的 `</page>` 或 `</database>` 结束标记。
- 未发现 MCP 输出截断标记。
- 导出内容引用的 16 个内部 Notion 实体均存在对应本地文件。
- 页面中未发现需要另行下载的 Notion file/image attachment 标记。
- 两份 `current/` 文档与对应 v2 原始导出的 SHA-256 完全一致。
- 所有迁移内容已写入 `SHA256SUMS`。

## 切换规则

迁移完成后，从 Codex 用户配置中移除 Notion plugin/MCP 配置。该操作只断开本地 MCP 集成，不删除 Notion 工作区中的原页面。后续决策、路线图、实现记录和审查证据只维护在本目录。

已验证 `/root/.codex/config.toml` 中不存在 Notion plugin/MCP 配置。当前已启动的 Codex 会话可能仍保留启动时加载的工具注册，但本任务不再调用；重启 Codex 后配置移除完全生效。
