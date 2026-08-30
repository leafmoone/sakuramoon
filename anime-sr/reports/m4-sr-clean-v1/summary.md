# M4-1024 HR-GT quality pool statistics (SR-clean-v1)

- population: **9430** 1024-eligible train images (frozen eligibility table, raw v2 labels)
- shards scanned: shard-000000.tar=9844, shard-000001.tar=9839, shard-000002.tar=9901
- coverage gaps (population ids missing in raw): 0; webp-size fallbacks: 0
- cross-check vs frozen table: label mismatches {}, eligibility re-derive mismatches 0, tar-webp dim mismatches 0/9430

## Pool sizes

| pool | n | share of full |
| --- | ---: | ---: |
| P0_priority | 1808 | 19.2% |
| P1_sr_clean_v1 | 6286 | 66.7% |
| P2_sr_clean_wide | 7440 | 78.9% |
| P3_lineart_extra | 1103 | 11.7% |
| P4_rough_extra | 1065 | 11.3% |

## 6M repetition intensity

| pool | n | mean exposures/source | sum crop-positions (stride64) | mean exposures/possible-crop proxy |
| --- | ---: | ---: | ---: | ---: |
| P0_priority | 1808 | 3319 | 302143 | 19.86 |
| P1_sr_clean_v1 | 6286 | 955 | 994631 | 6.03 |
| P2_sr_clean_wide | 7440 | 806 | 1159533 | 5.17 |
| full_eligible | 9430 | 636 | - | - |

## Diffs

- P1 - P0: +4478 (quality: {'good': 2200, 'normal': 2278})
- P2 - P1: +1154 (quality: {'low': 862, 'worst': 292})

## Clean-score (correlation only, no gating)

- Spearman(quality ordinal, score) = -0.11243154636962455
- polished mean 0.3609680078791865 (n=7615) vs non-polished 0.37780010798898056 (n=1815)

## Verdict

推荐：P2（SR-clean-wide，7440 张 = 78.9% eligible 全集）

原因（N / 6M exposures / 内容覆盖 / quality 构成 / crop flexibility / clean-score 相关性）：
- N：P2 = 7440，是「polished + illustration/bangumi/comic」内可定义的最大标签池；P1 = 6286（3.48× strict priority 1808），P0 单独 6M 需 3319 exposures/source（5.3× 全集速率），过拟合风险高，不作 6M 候选。
- 6M exposures：P2 = 806 exposures/source；按 stride-64 可能 crop 位计 = 5.17 次/可能 crop（P1 = 6.03，P0 = 19.86）——P2 的 crop 重复强度与全集同档，无重复过训信号。
- 内容覆盖：覆盖 1024-eligible 全集的 78.9%；排除的 21.1% 是 aux 类（rough 1065 / monochrome+line-art / 3d 渲染），其高频纹理统计与「完成的涂色动画」SR 目标域不同（rough=未完成渲染，3d=不同纹理域）——若部署需要线稿覆盖，用随附 P3 manifest（1103 张，monochrome∪comic、非 corrupt、非 not_painting）作为显式附加候选，不混入主池。
- quality 构成：P2 = P1（polished+core+tier≥normal：masterpiece 306 + best 721 + great 781 = 1808，恰等于 strict priority P0；good 2200 + normal 2278）+ low 862 + worst 292。P1 分类 = illustration 6126 / bangumi 62 / comic 98。纳入 low/worst 的依据：唯一保真度代理 clean_score 与 quality tier 不相关（Spearman −0.112，且符号为负——画评越高质量分反而略低，进一步证明画评≠保真度），按主观画评剔除 low/worst 没有可测量的保真度收益，只会损失 18% 的 HR-GT。
- crop flexibility（P1 实测，P2 同档）：webp 真实尺寸 min_dim 中位 1536、max≤2048；stride-64 crop 位分布 ==1: 38 / >1: 6248 / >16: 5642 / >64: 4538 / >256: 899——99.4% 的图 >1 个 crop 位，crop 多样性充足。
- clean-score 相关性：不可用（近常数 0.34–0.40，Spearman −0.112）⇒「clean pool」只能由 Danbooru 标签结构（quality×completeness×classification）定义，clean_score 永不作 hard gate（保持 -1.0）。

操作后果（供 6M GO 决策）：6M 语料从当前 9430（1c74230 冻结的 19/60/21 自然组成）切到 P2(7440) = 重建 slot map = 数据侧重启动，必须在 6M GO 前拍板；候选清单见 sr-clean-v1-sample-ids.txt（P1 全集 6286）与 P2 = P1 ∪ {low/worst 1154}（p2_low_worst_samples.txt 为抽样）。若维持 FULL(9430) 不改，本报告的池定义与 manifest 仍作为 6M 后的 Stage 数据候选存档。
