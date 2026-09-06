== SakuraMoon Camera Coordinate Causal - Posthoc Review Handoff ==

BASE
  repository = leafmoone/sakuramoon
  prior review branch = camera-v2-causal-audit-review
  prior exact SHA = 254e098e8941a4f47111d48fd37b858167116529
  new branch = camera-v2-causal-posthoc-review
  new branch base = 254e098e8941a4f47111d48fd37b858167116529

COMMITS
  tooling = dac6a6248e3f4d00fde0bc67ef13d3bbddf8bd03
  evidence = @POSTHOC_EVIDENCE_HEAD@
  remote SHA = @REMOTE_SHA@
  pushed = @PUSHED@
  force push = NO

HISTORICAL IMMUTABILITY
  final_snapshot 5/5 hashes unchanged = 5/5
  historical causal reports unchanged = 5/5
  expanded reports unchanged = 6/6
  src diff = EMPTY (git diff 254e098..HEAD -- src config)
  config diff = EMPTY

HISTORICAL VERDICT ISSUE
  historical verdict = INCONCLUSIVE
  same_maxabs gate = max(M_SAME_maxabs over ck) < 1e-6 forced INCONCLUSIVE
  actual same maxabs = 9.16e-05 / 1.57e-04 / 8.12e-05 (PRE/MID/POST ordinary)
  reported harness numerics flag = derived from determinism probe only (mismatch confirmed)
  mismatch confirmed = YES

NUMERICAL FLOOR
  camera SAME PRE = mean -5.693e-08 [-2.631e-07, 1.509e-07]
  camera SAME MID = mean 4.837e-09 [-2.166e-07, 2.390e-07]
  camera SAME POST = mean 2.181e-08 [-2.040e-07, 2.559e-07]
  abs p95 PRE = 0.000e+00 ; abs p99 PRE = 2.985e-05 ; maxabs PRE = 1.859e-04
  abs p95 MID = 0.000e+00 ; abs p99 MID = 3.409e-05 ; maxabs MID = 4.311e-04
  abs p95 POST = 0.000e+00 ; abs p99 POST = 3.499e-05 ; maxabs POST = 3.122e-04
  camera SAME POST-PRE = 7.874e-08 [-2.323e-07, 3.898e-07]
  ordinary SAME replication = -2.675e-07 [-1.051e-06, 5.346e-07] / 2.936e-07 [-4.596e-07, 1.212e-06] / -8.877e-08 [-8.025e-07, 6.303e-07]
  microprobe status = OK
  P1 pure-repeat pred bit-exact rate = 0.9557291666666666; max |loss delta| rel = 0.00022787900491375157
  SAME-vs-CORRECT directional drift = False; accumulation ratio = 2.6437847866419295
  hidden_state_detected (fail-closed) = True

RAW CAUSAL
  M_OPPOSITE PRE/MID/POST = 0.001706 / 0.002025 / 0.002099
  M_IDENTITY PRE/MID/POST = -0.001722 / 0.002232 / 0.002249
  M_SHUFFLED PRE/MID/POST = 0.001284 / 0.001916 / 0.002040

NOISE-ADJUSTED
  OPPOSITE PRE->MID = 0.000319 [0.000186, 0.000456]
  OPPOSITE MID->POST = 0.000073 [0.000014, 0.000132]
  OPPOSITE PRE->POST = 0.000392 [0.000244, 0.000544]
  SHUFFLED PRE->MID = 0.000629 [0.000281, 0.000973]
  SHUFFLED MID->POST = 0.000125 [0.000068, 0.000182]
  SHUFFLED PRE->POST = 0.000753 [0.000396, 0.001115]
  IDENTITY PRE->MID = 0.003952 [0.003694, 0.004222]
  IDENTITY MID->POST = 0.000017 [-0.000039, 0.000073]
  IDENTITY PRE->POST = 0.003969 [0.003702, 0.004250]

LOSS PREFERENCE
  classification = BLOCKED_NUMERICS
  primary arms = OPPOSITE, SHUFFLED (supportive: IDENTITY)
  trend = OPPOSITE:MONOTONIC_GAIN IDENTITY:PLATEAU SHUFFLED:MONOTONIC_GAIN

PREDICTION DISPLACEMENT
  OPPOSITE relRMS PRE/MID/POST = 0.060505 / 0.042835 / 0.043014
  IDENTITY relRMS PRE/MID/POST = 0.068053 / 0.055694 / 0.055724
  SHUFFLED relRMS PRE/MID/POST = 0.063050 / 0.044200 / 0.044546
  classification = LOSS_ALIGNMENT_GAIN_WITH_REDUCED_PREDICTION_DISPLACEMENT

OFFSET
  universal START (n=692) adjusted POST-PRE = 0.001560 [0.001277, 0.001860]
  universal CENTER (n=670) adjusted POST-PRE = 0.000090 [-0.000047, 0.000221]
  universal END (n=686) adjusted POST-PRE = -0.000493 [-0.000779, -0.000213]
  horizontal END (n=174) adjusted POST-PRE = -0.000076 [-0.000499, 0.000321]
  vertical END (n=512) adjusted POST-PRE = -0.000634 [-0.000982, -0.000293]
  physical-axis convention verified = True (camera_viewport.py @ b2443af)
  systematic END regression = True
  systematic physical-right regression = see horizontal END row
  systematic physical-bottom regression = see vertical END row

BEHAVIOR
  expanded behavioral verdict = NULL_EFFECT / WEAK (committed)
  recomputed = NO

NUMERICS
  NUMERICS_CLEAN = False
  verdict uses same boolean as report = YES
  SAME diagnostic vs causal effect scale = 0.00020092189071057868

POSTHOC VERDICT
  BLOCKED_NUMERICS

RECOMMENDATION
  FIX_AUDIT_HARNESS_BEFORE_TRAINING

VALIDATION
  replay = @REPLAY@
  existing pytest = @EXISTING_PYTEST@
  new pytest = @NEW_PYTEST@
  total tests = @TOTAL_TESTS@
  skips = 0
  xfails = 0
  ruff = @RUFF@
  py_compile = @PYCOMPILE@
  git diff check = @GITDIFF@

SECURITY
  secrets = @SECRETS@
  model/checkpoint staged = NO
  dataset/cache staged = NO
  large binary staged = NO

AUTHORIZATION
  longer p25 started = NO
  p50 started = NO
  production camera changed = NO

EXTERNAL REVIEW TARGET
  branch = camera-v2-causal-posthoc-review
  exact SHA = @REMOTE_SHA@

NEXT
  HARD STOP
  SEND EXACT SHA TO EXTERNAL REVIEWER
  DO NOT CONTINUE TRAINING

== END ==
