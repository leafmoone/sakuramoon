# 路线图 AI/模型正确性独立审查

审查来源：独立 requirements reviewer 子代理；审查对象为本地 v2 决策、开放清单、用户新增要求和路线图约束。最终文件由主代理按问题清单逐项核对。

## 结论

状态：通过，以下强制修订已经进入 `progress/IMPLEMENTATION_ROADMAP.md`。

- 明确 `current/confirmed-decisions.md` 优先于历史组件候选，防止旧的 d=2048、约1B、Artist主caption、FSDP2、bitsandbytes和PMA transition表述回流实现。
- 每个任务映射到稳定 requirement ID、配置、模块、测试、benchmark和artifact。
- Artist只进入Style分支；文本上限512和八桶固定且不重扫。
- x-pred、FP32 x-to-v、velocity CFG和Heun-50/99 NFE分别列为数学golden test。
- 除all_condition=0.10外的dropout保持用户决定门槛，生产配置不得编造值。
- FID/IS被标记为新增趋势/回归指标，不取代tag控制和人工质量验收，也不杜撰绝对阈值。
- 单卡证据不能外推四卡；S1以后明确阻塞于真实4×5090。
- 实现代理与审查代理分离；每项同时检查模型语义和Infra影响。

## 残余风险

- 当前可见GPU只有1张，DDP、NCCL、四卡SR state一致性和完整512吞吐仍不可验证。
- Dropout数值尚未决定，`C002/D014/S000`保持阻塞。
- FA4、TorchAO和Qwen fast kernel版本只能由后续真实执行确定。

