# D024 DATA 包级 Infra/性能复审

审查对象：D024 remediation commit `35e7e6f35d7936fbdc71b948d6db8ef5d706902b`。本文件是追加的不可变复审证据；既有 DATA 报告保留原样。

结论：**PASS（已实现 CPU/有界 1GPU scope）**。

- service ownership lock 先于 stale-socket/bind 获取；竞争失败路径保留 winner socket，避免跨进程破坏 singleton。
- real consumer contract 使用真实 service/client/worker 进程边界和有界 IPC；不以 in-memory fake client 代替隔离证明。
- bounded worker input/output、ready/completion、lease/ACK 和 active eviction protection 保持成立；异常、提前关闭和 ACK 缺失继续 whole-shard replay。

独立验证：D024 CPU service/collate/fault `16 passed`（15.21s，9 warnings）；RTX 5090 consumer `2 passed`（5.55s）；Ruff 通过；Pyright `0 errors, 0 warnings`；trace `235/235`、0 errors。

生产冷缓存网络/NVMe throughput、ready wait、RSS/swap/quota、worker/queue sweep、两小时窗口和四卡/DDP/NCCL 仍 pending/blocked，未由本次 1GPU smoke 外推。
