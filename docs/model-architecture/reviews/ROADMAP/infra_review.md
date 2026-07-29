# 路线图 Infra/性能独立审查

审查来源：独立 infra roadmap 子代理与 requirements reviewer 子代理的只读环境/文档核对。最终文件由主代理按问题清单逐项核对。

## 结论

状态：通过，以下强制修订已经进入路线图。

- 配置目录按用户要求统一为 `config/` 与 `train_s0.toml` 等TOML，历史 `configs/train/*.yaml` 不再使用。
- 当前根目录非Git仓库、uv不在PATH、Python 3.12.3、当前只有1×RTX 5090均列为显式前置事实。
- `.env`、模型、DB、dataset/cache、checkpoint、W&B、profile和嵌套参考仓库已进入Git忽略边界。
- uv只锁Python依赖；driver/CUDA/NCCL/编译器另存环境报告。
- 数据12 samples/s、完整512四卡6 samples/s、每卡27.2 GiB、data wait<2%的生产门槛保持不变。
- 全路径低开销计时、NVTX、W&B本地持久化、before/after公平benchmark和开发任务耗时分别规划。
- FID/IS使用checkpoint驱动的显式evaluator，记录GPU占用和训练暂停，不静默争抢四卡训练资源。
- 禁止FA4/Qwen fast kernel/compile静默fallback，禁止通过减token、关功能或未披露增显存换吞吐。

## 残余风险

- 参考仓库暂按根仓忽略并由asset manifest锁commit；若未来需要修改参考代码，必须另做vendor/submodule决定。
- 当前无法确定四卡NCCL和NVMe容量，相关stage配置必须在目标机benchmark后填写。
- 评估50k样本、Heun-50/99 NFE成本很高，正式间隔需由实测修订但必须显式写回TOML。

