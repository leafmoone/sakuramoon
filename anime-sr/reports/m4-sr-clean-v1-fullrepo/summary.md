# Full-repo streaming scan — SR-clean label statistics (v2 repo)

- shards: **2964**; json members: 12826239; webp headers: 10114939; unresolved json: 2192685
- nsfw distribution (scanned): {'nsfw': 1085484, 'sfw': 8507426, 'questionable': 1040644}
- population: **3519472** 1024-eligible train images (+113691 validation)
- webp-size fallbacks to meta dims: 540240
- stream cost: 12833246 range requests, 105.1 GB transferred (transient 8KB windows; only json+headers useful; zero disk writes)

## Per directory

| dir | shards | images | unresolved | webp members | 1024-eligible train |
| --- | ---: | ---: | ---: | ---: | ---: |
| data/1_2024 | 800 | 7956611 | 11251 | 7956611 | 2533584 |
| data/1_2025.9 | 405 | 1714713 | 10374 | 1714713 | 856535 |
| data/2026.7 | 384 | 440077 | 0 | 440077 | 9547 |
| data/2_2026.1 | 517 | 543779 | 1 | 3538 | 271849 |
| data/artstation-2D | 857 | 2170961 | 2170961 | 0 | 0 |
| data/background-2D | 1 | 98 | 98 | 0 | 0 |

## 未解析 json 分解（schema 说明）

| dir | unresolved | 说明 |
| --- | ---: | --- |
| data/artstation-2D | 2170961 | artstation 架构：id 为空（无 danbooru post id）+ 无 quality/anime_completeness/anime_classification 字段 + 图像成员为 .jpg（无 webp）→ 无法建索引（0 eligible 是「不可索引」而非「无好图」；标签/尺寸本身可读，若需利用要单独打标） |
| data/1_2024 | 11251 | malformed / 维度为 null（21626/10655180 = 0.20%，与生产语料同档） |
| data/1_2025.9 | 10374 | malformed / 维度为 null（21626/10655180 = 0.20%，与生产语料同档） |
| data/background-2D | 98 | 1 片 98 图，量级可忽略，未单独探测 schema |
| data/2_2026.1 | 1 | malformed / 维度为 null（21626/10655180 = 0.20%，与生产语料同档） |

- webp 尺寸回退：540240 张（tar 内无配对 .webp 成员或头不可读 → 用 meta 尺寸；几乎全部位于 2_2026.1，该目录图像为原始 png/jpg 无缩放，meta 尺寸=实际尺寸（抽验 3/3）→ 无高估）
- data/2_2026.1：仅 3538/543779 图成员为 webp，其余为原始 png/jpg（无缩放步骤）；抽验 meta 尺寸=实际文件尺寸（2_2026.1 3/3）→ eligible 判定可靠，但该目录与 danbooru webp 语料不是同一图像管线
- artstation-2D 的 2170961 张图 json 可读（tags+尺寸齐全）但不可索引；若未来要利用需单独打 danbooru 风格标签并赋 id

## Pool sizes

| pool | n | share of full |
| --- | ---: | ---: |
| P0_priority | 672314 | 19.1% |
| P1_sr_clean_v1 | 2301219 | 65.4% |
| P2_sr_clean_wide | 2756861 | 78.3% |
| P3_lineart_extra | 430609 | 12.2% |
| P4_rough_extra | 416118 | 11.8% |

## 6M repetition intensity (reference; 6M was the M4 run size)

| pool | n | mean exposures/source | sum crop-positions (stride64) | mean exposures/possible-crop proxy |
| --- | ---: | ---: | ---: | ---: |
| P0_priority | 672314 | 9 | 274340598 | 0.02 |
| P1_sr_clean_v1 | 2301219 | 3 | 780848939 | 0.01 |
| P2_sr_clean_wide | 2756861 | 2 | 894865545 | 0.01 |
| full_eligible | 3519472 | 2 | - | - |

## Diffs

- P1 - P0: +1628905 (quality: {'good': 811582, 'normal': 817323})
- P2 - P1: +455642 (quality: {'worst': 116386, 'low': 339256})
