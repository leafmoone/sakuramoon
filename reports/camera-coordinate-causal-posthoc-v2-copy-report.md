== SakuraMoon Camera Coordinate Causal - Posthoc V2 Review Handoff ==

BASE
  repository = leafmoone/sakuramoon
  prior branch = camera-v2-causal-posthoc-review
  prior SHA = 0caf0a156f45eefc0986829005e49e59a7301082
  new branch = camera-v2-causal-posthoc-v2-review
  new base = 0caf0a156f45eefc0986829005e49e59a7301082

COMMITS
  tooling = @POSTHOC_V2_TOOLING_HEAD@
  evidence = @POSTHOC_V2_EVIDENCE_HEAD@
  remote SHA = @POSTHOC_V2_REMOTE_SHA@
  pushed = @PUSHED@
  force push = NO

IMMUTABILITY
  final_snapshot = 5/5 unchanged
  historical causal reports = 5/5 unchanged
  expanded reports = 6/6 unchanged
  V1 posthoc reports = 4/4 unchanged
  V1 posthoc tooling = 3/3 unchanged
  src diff = @SRCDIFF@
  config diff = @CONFIGDIFF@

REPLAY
  historical replay = @REPLAY@
  raw causal crosscheck = EXACT (9 M_ + 3 arms x S_ bit-equal to audit.json)

V1 FALSE-POSITIVE ISSUE
  P1 cap = 1e-6 (V1 used it as a hidden-state gate; V2 keeps it diagnostic-only)
  P1 observed max rel (V1) = see camera-coordinate-causal-posthoc-review.json (committed)
  P0 observed max rel (V2) = 5.563e-04
  V1 hidden state result = True (driven by the cap)
  reason V1 blocked = single worst repeat outlier > 1e-6 mapped to hidden_state_detected

MICROPROBE V2
  units = ? camera + ? ordinary
  checkpoints = {"PRE": {"path": "/sakuramoon-runtime/output_model/g1/ckpt_116100_raw-116100-update-cadence", "update": 116100, "alpha": 1.0}, "MID": {"path": "/tmp/camera-coordinate-causal/ckpts/MID/ckpt_117100_raw-117100-update-cadence", "update": 117100, "alpha": 1.0}, "POST": {"path": "/sakuramoon-runtime/output_model/g1_camera_v2_p25/ckpt_118100_raw-118100-update-cadence", "update": 118100, "alpha": 1.0}}
  timesteps = strata (0,2): {}
  immediate repeat bitexact rate = 0.9363839285714286
  repeat jitter present = True
  repeat jitter mean = -7.956266580593018e-07
  repeat jitter p95 = 8.940696716308594e-06
  repeat jitter p99 = 9.687691926956211e-05
  repeat jitter max = 0.00019669532775878906 (rel 0.0005562964412705204)
  coordinate mutation = False
  parameter mutation = False
  persistent buffer mutation = False
  RNG consumption = True
  order-dependent drift = False
  accumulation slope = P1 4.0275820841391853e-07 CI [-8.583000938718524e-07, 1.7560193858419843e-06] / P2 -4.047372688849794e-08 CI [-6.574254560594763e-07, 6.213143933564423e-07]
  accumulation CI = see p1/p2 c_slope ci95 above
  hidden mutable state detected = False

NUMERICAL FLOOR
  camera SAME PRE = mean -5.693e-08 CI [-2.651e-07, 1.502e-07]
  camera SAME MID = mean 4.837e-09 CI [-2.222e-07, 2.361e-07]
  camera SAME POST = mean 2.181e-08 CI [-2.059e-07, 2.605e-07]
  camera SAME POST-PRE = 7.874e-08
  CI = [-2.281e-07, 3.955e-07]
  ordinary SAME = PRE -2.675e-07 / MID 2.936e-07 / POST -8.877e-08
  same/causal scale ratio = 2.006e-04

CAUSAL V2
  OPPOSITE raw PRE/MID/POST = 1.706e-03 / 2.025e-03 / 2.099e-03
  OPPOSITE adj PRE->MID = 3.195e-04 [1.801e-04, 4.570e-04]
  OPPOSITE adj MID->POST = 7.303e-05 [1.502e-05, 1.310e-04]
  OPPOSITE adj PRE->POST = 3.925e-04 [2.410e-04, 5.408e-04]
  IDENTITY raw PRE/MID/POST = -1.722e-03 / 2.232e-03 / 2.249e-03
  IDENTITY adj PRE->MID = 3.954e-03 [3.691e-03, 4.223e-03]
  IDENTITY adj MID->POST = 1.710e-05 [-3.931e-05, 7.330e-05]
  IDENTITY adj PRE->POST = 3.971e-03 [3.695e-03, 4.245e-03]
  SHUFFLED raw PRE/MID/POST = 1.284e-03 / 1.916e-03 / 2.040e-03
  SHUFFLED adj PRE->MID = 6.311e-04 [2.904e-04, 9.819e-04]
  SHUFFLED adj MID->POST = 1.246e-04 [6.924e-05, 1.812e-04]
  SHUFFLED adj PRE->POST = 7.557e-04 [4.043e-04, 1.112e-03]

POINT ESTIMATE
  exact observed mean used = YES
  bootstrap only for CI = YES
  invariant across bootstrap seed = YES (regression-locked)

LOSS PREFERENCE
  OPP trend = MONOTONIC_GAIN
  SHUFFLED trend = MONOTONIC_GAIN
  IDENTITY trend = PLATEAU
  classification = POSITIVE_LOSS_PREFERENCE_LEARNING

PREDICTION DISPLACEMENT
  OPPOSITE PRE/MID/POST = 0.0605 / 0.0428 / 0.0430
  IDENTITY PRE/MID/POST = 0.0681 / 0.0557 / 0.0557
  SHUFFLED PRE/MID/POST = 0.0631 / 0.0442 / 0.0445
  classification = LOSS_ALIGNMENT_GAIN_WITH_REDUCED_PREDICTION_DISPLACEMENT

PLANNER BALANCE
  offset sampler implementation = k = random.Random(offset_seed).randrange(available + 1), k in 0..available uniform (camera_viewport.py @ b2443af)
  conditional symmetry verified = True (static: P(k)=P(available-k) exact for uniform sampler)
  universal observed START/CENTER/END = 692 / 670 / 686
  universal expected = see offset-balance.json planner.blocks.all
  vertical TOP/CENTER/BOTTOM observed = 521 / 524 / 512
  vertical expected = see offset-balance.json planner.blocks.vertical
  exposure count imbalance = False

VERTICAL GEOMETRY
  TOP n = 521
  BOTTOM n = 512
  zoom SMD = 4.090e-02
  latent shift SMD = 3.575e-02
  aspect SMD = 4.143e-02
  available SMD = 4.143e-02
  retention SMD = 3.815e-02
  largest imbalance = main_tokens (SMD 5.257e-02)

REWEIGHTED EFFECT
  common geometry cells = 11
  TOP ESS = 518.6386154029349
  BOTTOM ESS = 510.2715302086059
  TOP adjusted effect = 1.825e-03 CI [0.0014895396485939993, 0.0021860739800730494]
  BOTTOM adjusted effect = -6.443e-04 CI [-0.0009951498707519459, -0.00028926600606747605]
  negative persists = True

MATCHED EFFECT
  pairs = 512
  post-match max SMD = 1.103e-02
  TOP-BOTTOM paired difference = 2.437e-03 CI [0.0019195196646251134, 0.002968733798843459]
  result = see matched.top_minus_bottom

CONTINUOUS OFFSET
  Spearman = -3.129e-01 CI [-0.3650075099400405, -0.2593693676734586]
  decile trend = d0:2.53e-03 d1:2.38e-03 d2:9.20e-04 d3:7.04e-04 d4:1.93e-04 d5:2.51e-05 d6:-3.29e-04 d7:-4.85e-04 d8:-3.44e-04 d9:-1.16e-03
  extreme mirror bins = [0.0,0.1)vs[0.9,1.0):1.87e-03 [0.1,0.2)vs[0.8,0.9):1.31e-03
  interpretation = monotone-ish decline from TOP to BOTTOM

CONTENT / SHARD
  routing covariates = AVAILABLE
  shard imbalance = False
  raw caption metadata available = NOT_AVAILABLE
  notable interactions = see offset-balance.md section 7

OFFSET BALANCE CLASS
  EFFECT_PERSISTS_AFTER_GEOMETRY_BALANCE

NUMERICS V2
  NUMERICS_V2_CLEAN = True
  repeat jitter warning = True
  hidden mutable state = False

POSTHOC V2 VERDICT
  POSITIVE_LOSS_PREFERENCE_LEARNING

RECOMMENDATION
  REVIEW_VERTICAL_END_SUPERVISION_FIRST (case B)

BEHAVIOR
  expanded behavior = NULL_EFFECT / WEAK (committed)
  recomputed = NO

VALIDATION
  original tests = @ORIG_TESTS@
  V1 posthoc tests = @V1_TESTS@
  V2 tests = @V2_TESTS@
  total = @TOTAL_TESTS@
  skips = @SKIPS@
  xfails = @XFAILS@
  replay = @REPLAY@
  ruff = @RUFF@
  py_compile = @PYCOMPILE@
  diff check = @GITDIFF@

SECURITY
  secret hits = @SECRETS@
  model/checkpoint staged = NO
  dataset staged = NO
  binary evidence staged = NO

AUTHORIZATION
  longer P25 started = NO
  P50 started = NO
  production changed = NO

EXTERNAL REVIEW TARGET
  branch = camera-v2-causal-posthoc-v2-review
  exact SHA = @POSTHOC_V2_REMOTE_SHA@

NEXT
  HARD STOP
  SEND EXACT SHA TO EXTERNAL REVIEWER
  EXTERNAL REVIEW MUST HAPPEN BEFORE ANY TRAINING GO

== END ==
