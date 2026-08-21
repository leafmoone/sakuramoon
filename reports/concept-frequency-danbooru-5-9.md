# Concept frequency statistics — Danbooru v2 (5.9 metadata)

Generated from the precomputed counts in `leafmoone/dan_5_9_metadata`
(DuckDB, 14.1 GB; persistent copy at ModelScope `leafmoone/docker_tmp`
`tools/dan_5_9.db`). Re-run with
`python scripts/concept_frequency_stats.py --db <path> --out out.json`.

## Source

| | |
|---|---|
| images (`metadata` rows) | 11,268,846 |
| tags (`tag_stats` rows) | 964,790 (artist 543,881 / character 317,867 / general 103,042) |
| artist tag occurrences | 11,018,945 |
| character tag occurrences | 18,668,240 |

Percentiles below are **descending-rank** (`p10` = count at the 10th
percentile from the top; `p50` = median).

## artist (543,881 concepts, mean 20.3 occurrences)

| p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|
| 39 | 10 | 3 | 1 | 1 | 1 | 1 |

| ≥10 | ≥20 | ≥50 | ≥100 | ≥500 | ≥1000 | ≥5000 | ≥10000 |
|---|---|---|---|---|---|---|---|
| 140,680 (25.9%) | 90,051 | 45,192 | 23,302 | 2,420 | 593 | 8 | 1 |

Coverage of total artist occurrences: top-20 0.98%, top-100 2.83%,
top-1,000 12.1%, top-5,000 29.3%.

Top-30: dairi (18,655), ebifurya (6,487), crote (6,434), bkub (5,676),
ruu (tksymkw) (5,623), hammer (sunset beach) (5,584), itomugi-kun (5,278),
haruyama kazunori (5,262), kou hiyoyo (4,517), yaegashi nan (4,480),
kanon (kurogane knights) (4,467), mizuki hitoshi (4,360), rebecca
(keinelove) (4,344), kouji (campus life) (4,333), inoino (4,041), tani
takeshi (3,853), naga u (3,817), ojipon (3,753), hara (harayutaka)
(3,666), pageratta (3,572), matsunaga kouyou (3,472), hiroki (yyqw7151)
(3,312), onikobe rin (3,278), blade (galaxist) (3,177), bow (bhp)
(3,172), tony taka (3,171), a1 (initial-g) (3,097), futa (nabezoko)
(3,053), ixy (3,012), hana kazari (2,971).

## character (317,867 concepts, mean 58.7 occurrences)

| p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|
| 54 | 13 | 3 | 1 | 1 | 1 | 1 |

| ≥10 | ≥20 | ≥50 | ≥100 | ≥500 | ≥1000 | ≥5000 | ≥10000 |
|---|---|---|---|---|---|---|---|
| 94,335 (29.7%) | 61,765 | 33,684 | 20,661 | 5,862 | 3,132 | 497 | 159 |

Coverage of total character occurrences: top-20 5.34%, top-100 12.9%,
top-1,000 38.9%, top-5,000 65.3%.

Top-30: hatsune miku (135,838), hakurei reimu (94,376), kirisame marisa
(82,664), flandre scarlet (59,642), remilia scarlet (59,488), izayoi
sakuya (51,339), artoria pendragon (fate) (44,698), komeiji koishi
(42,168), kochiya sanae (39,731), konpaku youmu (39,560), cirno (38,998),
admiral (kancolle) (38,716), alice margatroid (38,651), patchouli
knowledge (37,395), yakumo yukari (36,856), sensei (blue archive)
(34,540), shameimaru aya (33,083), reisen udongein inaba (30,155), komeiji
satori (29,792), fujiwara no mokou (29,738), akemi homura (28,946),
kaname madoka (28,683), hong meiling (27,269), saigyouji yuyuko (27,097),
inubashiri momiji (25,323), kagamine rin (24,669), scaramouche (genshin
impact) (24,030), yakumo ran (22,140), kaenbyou rin (22,086), konpaku
youmu (ghost) (22,075).

## Key findings

1. **The tail is extreme for both types.** Median frequency is 3 for
   both artist and character; ≥75% of all concepts have count 1. This is
   a classic power law, so aggregate loss will not reveal concept
   learning — the 120-concept benchmark must be tiered.
2. **artist is far more diffuse than character.** 543,881 artist
   concepts vs 317,867 character, but artists have *fewer* total
   occurrences (11.0M vs 18.7M): mean 20.3 vs 58.7. top-1,000 artists
   cover only 12.1% of artist occurrences, while top-1,000 characters
   cover 38.9%.
3. **Character concentration is dominated by famous franchises** (Touhou,
   Fate, Blue Archive, kancolle, Genshin) — top-100 characters alone
   cover 12.9% of all character occurrences.
4. Only 1 artist (dairi, 18,655) and 159 characters reach 10,000
   occurrences.

## Pools for the 120-concept benchmark (v1, before update 70k)

| tier | artist pool | character pool |
|---|---|---|
| high (≥1000 / ≥5000) | 593 | 497 |
| mid (≥100) | 23,302 | 3,132 |
| tail (≥10) | 140,680 | 94,335 |

A 120-concept draw can therefore be stratified, e.g. high-frequency
characters (497 candidates), high-frequency artists (593), mid-frequency
(≥100), and long-tail (≥10), with exact per-tier quotas and seed
selection to be decided when the benchmark v1 composition is specified.
