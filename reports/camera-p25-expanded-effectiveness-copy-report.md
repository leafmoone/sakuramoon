== SakuraMoon Camera Viewport v2 · P25 Expanded Effectiveness ==

DESIGN
  checkpoints = PRE=U116100 MID=U117100 POST=U118100
  prompts = 24
  legacy prompts = validation-fac64aa1d, validation-c17a74b54, validation-fe7761d38, validation-d985f354c
  expanded prompts = 20 (5 strata x 4: single_close_upper/single_full_body/multi_character/scenery_environment/mixed_other)
  seeds = 3 per prompt (v0=bank/legacy seed, v1/v2=splitmix64-derived; master 20260905)
  training-coupled points = 12 (4 zoom series x center/left/right)
  identity points = 1 (legacy_center_z1)
  images expected = 2808
  images actual = 2808
  protocol frozen before render = True
  prompt selection source = repo evaluation prompt bank (audited; deterministic stratified selection)
  cherry-picked = NO

INTEGRITY
  missing = 0
  corrupt = 0
  duplicate logical keys = 0 (verified: no_duplicate_logical_keys)
  legacy protocol reproduction = 0/52 byte-exact under new 12-case batch context; 52/52 byte-exact under matched 4-case batch context (BIT_EXACT_REPRODUCTION=PASS under matched context; root cause = batch-size-dependent HCU numerics, deterministic; old/new statistics NOT mixed)
  render manifest hash = see audit.json provenance.render_manifests (18 chunk manifests, sha16)

PRIMARY STATISTICS
  inference unit = prompt (top cluster) -> seed -> paired camera points
  bootstrap = hierarchical 10000 (primary) + prompt x seed iid (secondary)
  bootstrap seed = 20260905

SHIFT
  PRE direction = 0.6059
  MID direction = 0.6259
  POST direction = 0.5937
  POST-PRE delta = -0.0122
  95% hierarchical CI = [-0.0796, 0.0561]
  sign PRE/MID/POST = 0.837 / 0.842 / 0.837
  response error PRE/MID/POST = 0.2415 / 0.2417 / 0.2524
  low-PRE subgroup delta = 0.0126
  mid-PRE subgroup delta = -0.0590
  high-PRE subgroup delta = 0.0097

ZOOM
  PRE log error = 0.1917
  MID log error = 0.1926
  POST log error = 0.1766
  POST-PRE delta = -0.0151
  95% CI = [-0.0403, 0.0121]
  monotonicity PRE/MID/POST = 0.859 / 0.748 / 0.764
  Spearman PRE/MID/POST = 0.2416 / 0.2419 / 0.2710
  low-PRE subgroup delta = -0.0099
  high-PRE subgroup delta = -0.0066

CONTENT
  CLIP PRE = 0.9116
  CLIP MID = 0.9171
  CLIP POST = 0.9129
  POST-PRE delta = 0.0013
  95% CI = [-0.0051, 0.0073]
  PE content PRE/MID/POST = 0.8852 / 0.8911 / 0.8866
  systematic regression = NO
  affected strata = none

LEGACY4 REPLICATION
  original-seed result = bank seed v0, per-case content delta case1..4 (full per-seed table in audit.json legacy4_replication): -0.001, -0.085, -0.020, 0.000
  new-seed result = case2 v1/v2 mean content drop: -0.005
  case2 content regression = YES (rule: all-seed mean < -0.02): all-seed mean -0.032; bank seed v0 -0.085 (first audit 0.781->0.70); new seeds v1/v2 = 0.017 / -0.027 — seed-dependent
  right-edge case2/3 regression = YES (rule: any legacy4 all-seed right-edge mean < -0.05). All-seed mean case1..4 = -0.175, 0.178, -0.217, -0.630; bank seed v0 case1..4 = -0.113, -0.373, -0.697, -0.080 (case2/3 worst on v0, as in first audit); new seeds (v1/v2 mean) case1..4 = -0.206, 0.454, 0.023, -0.905 (worst on new seeds: case4) — seed-specific, not structural
  classification = REPRODUCED

EDGE SYMMETRY
  PRE left-right = 0.0757 (L=0.6438 R=0.5681)
  MID left-right = 0.0134 (L=0.6326 R=0.6193)
  POST left-right = 0.0887 (L=0.6381 R=0.5493)
  POST systematic right regression = NO
  affected prompt count = 12
  affected strata = see audit.json edge_asymmetry

CENTER / EDGE
  center zoom = log-err 0.1917 -> 0.1766
  edge zoom = right-edge dir 0.5681 -> 0.5493
  interaction = see audit.json center_edge (per-z center_zoom_err vs edge_dir_cos, PRE/MID/POST)
  old regressed points replicated = coupled_z1.4142_center=no, coupled_z1.5_center=no, coupled_z1.4142_right_edge=YES, coupled_z1.225_right_edge=YES, coupled_z1.1_right_edge=YES

CEILING
  low PRE prompt count = 8
  low PRE POST gain = 0.0126
  mid PRE POST gain = -0.0590
  high PRE POST gain = 0.0097
  ceiling hypothesis = SUPPORTED (high-PRE group capped: no >0.02 gain; low-PRE not clearly larger)

PROMPT CONSISTENCY
  positive = 14
  stable_good = 2
  null = 2
  tradeoff = 2
  negative = 4

TRAJECTORY
  monotonic_gain = 3
  early_gain_hold = 1
  late_gain = 7
  no_gain = 0
  non_monotonic = 13
  regression = 0

OUTLIERS
  strongest gains = validation-06d95#v2:coupled_z1.4142_right_edge, validation-033c6#v0:coupled_z1.5_right_edge, validation-d985f#v0:coupled_z1.225_left_edge
  strongest regressions = validation-0cc20#v0:coupled_z1.1_left_edge, validation-02376#v2:coupled_z1.4142_right_edge, validation-02b45#v0:coupled_z1.1_right_edge
  largest content drops = validation-c17a7#v0:coupled_z1.5_center, validation-c17a7#v0:coupled_z1.1_center, validation-c17a7#v0:coupled_z1.225_left_edge
  largest asymmetries = validation-0cc20#v2 z1.5, validation-d985f#v1 z1.4142135623730951, validation-d985f#v2 z1.1

BASE QUALITY CONTEXT
  FID 53.5928 -> 52.8045
  KID 0.021868 -> 0.020761
  IS 4.3949 -> 4.6909
  concept 0.043610 -> 0.044143
  interpretation = NO-REGRESSION CONTEXT ONLY; NOT CAMERA-EFFECT EVIDENCE

EXPANDED EFFECT CLASSIFICATION
  one of:
    CLEAR_POSITIVE
    NULL_EFFECT
    TRADEOFF
    NEGATIVE
  => NULL_EFFECT

P25 EFFECTIVENESS
  PASS / WEAK / FAIL = WEAK

EXPOSURE HYPOTHESIS
  more camera exposure justified by evidence = NO

RECOMMENDATION
  one of:
    P25_PRODUCTION_CANDIDATE
    LONGER_P25_OR_P50_EXPERIMENT_JUSTIFIED
    KEEP_P25_OFF_PRODUCTION
    REVIEW_CAMERA_SAMPLING_BALANCE
    REVIEW_CAMERA_MECHANISM
  => KEEP_P25_OFF_PRODUCTION

P50
  authorized = NO
  locked = YES

PRODUCTION
  changed = NO

GIT
  src changed = NO
  tests changed = NO
  config changed = NO
  commit = NO
  push = NO

NEXT
  STOP AT USER GATE

== END ==
