# D024 DATA 包级 AI/模型正确性复审

审查对象：D024 remediation commit `35e7e6f35d7936fbdc71b948d6db8ef5d706902b`。本文件是追加的不可变复审证据；既有 DATA 报告保留原样。

结论：**PASS（已实现 CPU/有界 1GPU scope）**。

- cache-root singleton lock 与 mainset path 解耦；同 cache 不同 mainset 的第二 service 被拒绝。
- 并发 binder loser 不会删除 winner socket；socket unlink 仅作用于本 server 成功绑定且 inode 仍匹配的路径。
- RTX 5090 smoke 真实经过 spawned `DataServiceServer`、AF_UNIX `DataServiceClient`、两个 persistent workers、四次 lease/ACK 与 CUDA consumer，并观察到首轮 mainset rotation。
- D021 trusted `ShardRecord`、metadata mapping、validation exclusion、caption/image/collate 与 D022/D023 whole-shard replay 语义未改变。

独立验证：D024 CPU service/collate/fault `16 passed`；RTX 5090 consumer `2 passed`；Ruff 通过；Pyright `0 errors, 0 warnings`；trace `235/235`、0 errors；remediation diff check 通过。

此结论不关闭 immutable production manifest、live ModelScope/cold-cache、两小时吞吐、RSS/swap/quota、四卡/DDP/NCCL、长跑或正式 stage 门槛。
