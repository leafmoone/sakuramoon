# SakuraMoon Camera — MBS 100U Canary Run Evidence

**Window:** U118100 → U118200 (exactly 100 successful updates, hard stop at U118200)
**Reviewed launch SHA:** `9a8810dfe535b54f1c0a393952c483cd29b6875e`
**Evidence branch:** `camera-v2-mbs-canary-100u-evidence`
**Generated:** 2026-09-08 02:39 UTC (post-run audit)

## Verdict

| Dimension | Result |
|---|---|
| Training stability (§23) | **PASS** |
| MBS intervention actually exercised (§15/§22) | **FAIL** — eligible = selected = applied = 0 for all 100 updates |
| Overall | **PARTIAL** — stability PASS, exposure FAIL |
| Vertical asymmetry fixed | **NOT YET DETERMINED** (§24: this round establishes stability, exposure, and terminal checkpoint only) |

The §15 early-activity gate (cumulative `camera_mirror_applied > 0` by U118103) was **not met** and its
prescribed STOP was not executed: the monitoring session (dd17abfe) died at ~U118122 (03:24, liangshen
preset `maxTokens=1024` reasoning-only turn death), before the gate could trigger. The run then completed
the authorized 100 updates unmonitored for its final ~18 minutes. All terminal gates (§17–§20, §27) pass
and were verified post-run.

## Root cause of zero mirror exposure (evidence-based)

Dataset × policy interaction at 256px:

- 40,000 logical samples → 10,194 camera-selected (p=0.25) → 8,140 camera-applied
  (vertical 5,620 / horizontal 2,520).
- Severity histogram over applied camera rows: **lt2 = 3,399 / 2to4 = 1,892 / ge4 = 329**.
- Every row with `latent_center_shift ≥ 2.0` (`min_latent_shift`) is **horizontal** — excluded by the
  vertical-only policy. Vertical-applied rows (near-square sources in this danbooru bucket) get long-axis
  shifts < 2.0 latent cells under zoom 1.10–1.50.
- Therefore `eligible = 0`, and with `pair_probability = 1.0`: `selected = applied = 0`,
  `physical_views == logical_samples` (40,000) in every record.
- The canary thus ran as a **camera-only control**; the mirror path was numerically idle
  (loss = 0.5·L_orig + 0.5·L_mirror degenerated to L_orig only; no extra views).

This is a property of the intervention definition × dataset, not a training fault: all conservation
identities held exactly, nonfinite = 0, and camera telemetry behaved normally.

## Launch

| Item | Value |
|---|---|
| Host | come3 (`crdnotebook-…-come3-31863`), 2× Hygon DCU (device name "BW"), DTK torch 2.9.0 |
| Config | `config/train_g1_camera_v2_p25_mirror_canary.toml` (unchanged from reviewed SHA) |
| World size / devices | 2 / 0,1 |
| Optimizer | `hybrid_cmuon_canonical_ns4_fp32_rescue` |
| LR | 0.000156249996508 (= 5e-05 × 800/256 linear-global-batch; constant for all 100 updates) |
| Logical global batch | 400 (effective_batch; stage config global_batch = 800 is the LR-scaling basis) |
| Stop cap / start / target | 118200 / 118100 / 118200 |
| Camera | `hdm_shifted_square_v2`, p=0.25, zoom 1.10–1.50 |
| Mirror | `vertical_mirror_pair_v1`, min_latent_shift=2.0, pair_probability=1.0, pair_weight=1.0 |
| Publisher isolation | interval 86400s, repo path `experiments/camera-v2-mbs-canary-118100-118200`, **zero uploads**, no last-published file, production `s0` untouched |

## Timeline (local, 2026-09-08)

- 02:55 attempt 1 start → preflight FAIL (`require_local_inception_weights`; 2 inception pth missing on the fresh host). Zero state change; stack stopped.
- 03:02 repair: both inception weights copied from come4, sha256-verified (95,628,359 B / 108,949,747 B).
- 03:03 attempt 2 start OK (data=74505, publisher=74597, train=74743; 2/2 ranks; resume U118100).
- 03:03:06 publisher first cycle: "no complete checkpoint tree" (expected); sleeps 86400 s.
- 03:24 monitoring session dd17abfe died at ~U118122 — **monitoring gap from here on**.
- 03:42:39 clean exit after update 118200 (stop cap; no U118201; NCCL teardown clean).
- ~10:00 post-run audit (this report); stack stopped; runtime state migrated to come7 (come3 pending release).

## Stability numbers (100 records, schema v12)

| Metric | Value |
|---|---|
| Updates | 118101…118200, contiguous, no duplicates/missing, count = 100 |
| nonfinite total | 0 |
| total loss first / mean / final | 0.571557 / 0.567765 / 0.554264 (min 0.534436, max 0.624876) |
| grad norm pre/post max | 0.121891 / 0.121891 (means 0.046476 / 0.046476; clip_fraction max 0.0) |
| samples/sec median (p10/p90) | 51.52 (49.45 / 53.47) |
| update wall s median (p10/p90) | 15.53 (14.97 / 16.19) |
| HCU mem peak allocated / reserved | 12.69 GiB / 62.49 GiB |
| DDP/rank failure | none |
| optimizer / CMuon hard failure | none |
| LR / effective_batch / growth_alpha | constant (1.5625e-04 / 400 / 1.0) |

## Mirror exposure aggregate (100 updates)

| Quantity | Value |
|---|---|
| logical samples | 40000 |
| camera selected / applied | 10194 / 8140 |
| vertical applied | 5620 |
| eligible / selected / applied | 0 / 0 / 0 |
| degraded (selected − applied) | 0 |
| extra physical views | 0 |
| physical / logical | 1.0000 |
| applied / logical | 0.0000 |
| severity lt2 / 2to4 / ge4 | 3399 / 1892 / 329 |
| original start/center/end | 0 / 0 / 0 (no pairs formed) |
| mirror start/center/end | 0 / 0 / 0 (no pairs formed) |

Per-record invariants (all 100): `applied ≤ selected ≤ eligible ≤ vertical_applied ≤ logical`,
`extra_views == applied`, `physical == logical + extra_views`, `selected == eligible`, `degraded == 0`.

## Terminal state

- Final ckpt: `/sakuramoon-runtime/output_model/g1_camera_v2_p25_mirror/ckpt_118200_raw-118200-update-cadence` — COMPLETE (`complete\n`), trainer successful_updates = 118200,
  stage budget 58000→168000 untouched, cadence last = 118200, reason = `update-cadence` (not stage-finalize).
- U118201: no log line, no checkpoint, no trainer state > 118200.
- Train processes after completion: 0. Stack stopped (publisher + data service).
- Source ckpt `/sakuramoon-runtime/output_model/g1_camera_v2_p25/ckpt_118100_raw-118100-update-cadence` immutable: manifest sha256 `821a8c12546a094865972fef0c23152e9754e01f82eed8353f7248b92076b892` before == after.

## Hashes

| Artifact | sha256 |
|---|---|
| reviewed code SHA | `9a8810dfe535b54f1c0a393952c483cd29b6875e` |
| source ckpt manifest (U118100) | `821a8c12546a094865972fef0c23152e9754e01f82eed8353f7248b92076b892` |
| final ckpt manifest (U118200) | `f014827fb441aca8780a5bb710eeb2631a40aac112ed3aa6b07b6e3b65d657f7` |
| metrics.jsonl | `5fb691f39f754e265102a405b9d63dac6f23865e3d4fc579204ab6f6fe21c126` |
| train.log | `8fb5150a59e0d48d0bc836a2ac9c396b9ae77464bbc76b2277962aa7993e79a1` |

## Git

- `git diff 9a8810dfe535..HEAD -- src config scripts tests` = **EMPTY** (no code/config changes in the evidence branch).
- Evidence commit: `docs: record MBS 100U canary evidence` (exact 3-file stage).

## Scientific interpretation

- **Training stability: PASS.** The stack, optimizer (CMuon hybrid rescue), data path, telemetry,
  checkpointing and governed stop cap all behaved exactly as reviewed under real 2×DCU training.
- **MBS intervention actually exercised: NO** (0/40000 mirror views).
- **Vertical asymmetry fixed: NOT YET DETERMINED.** No claim may be made from this run's loss alone.
- Decision for the next external-review round: whether to (a) adjust the eligibility policy
  (e.g., min_latent_shift threshold, orientation scope) and re-canary, or (b) run the frozen same-source
  causal audit on U118200 as a camera-only control terminal.

## Authorization state

additional training started = NO · U118201 = NO · P50 = NO · production changed = NO · FID run = NO (per §25).

**NEXT: HARD STOP → external review of `camera-v2-mbs-canary-100u-evidence` → then decide same-source U118200 causal audit.**
