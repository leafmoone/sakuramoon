== SakuraMoon Camera Vertical Bottom / Same-Source Supervision Audit ==

BASE
  repository = leafmoone/sakuramoon (github)
  reviewed base = 34f646abdb1e64f45dfddd47cf7e8b9247a7a979
  branch = camera-v2-vertical-bottom-supervision-review
  tooling commit = 51cd85ba7549efe591210c7e833cfdc9c48dbd22
  evidence commit = <EVIDENCE_HEAD - the commit containing this file>
  remote SHA = <ls-remote read-back; == evidence HEAD>
  pushed = <recorded in session FINAL COPY after push>
  force push = NO

PAIR DESIGN
  candidate vertical units = 1030
  unique sources = 1030
  selected pairs = 512
  original START anchors = 259
  original END anchors = 253
  mild/medium/strong = 350/141/21
  pair manifest sha = 7f3a81b248473d0b0591377c1bcd9f7df353066c9716836dabe90d9861f56297
  source image availability = 512/512 extracted
  anchor reconstruction parity = 480/480 bitexact, worst maxabs 0.000e+00

GEOMETRY
  same source = YES
  same zoom = YES
  same full canvas = YES
  same |shift| = YES
  signed shift mirrored = YES (exact)
  TOP crop = (0, k_start, 256, k_start+256)
  BOTTOM crop = (0, k_end, 256, k_end+256)
  all pair invariants = True

CONDITIONING
  qwen states shared = YES
  text routing shared = YES
  condition routing shared = YES
  caption regenerated = NO

TIMESTEP / NOISE
  strata = [0.1388061520926247, 0.24819609741338816, 0.37948289980451305, 0.5560734456280833]
  same noise TOP/BOTTOM = YES
  same input across PRE/MID/POST = YES

NUMERICS PREREQUISITE
  V2 numerics clean = True
  hidden mutable state = False
  P3 state_history_effect = False
  P3 now wired into gate = True
  repeat jitter diagnostic = True

MIRROR ESTIMATOR FIX
  point definition =
    mean(TOP-BOTTOM)
  paired bootstrap = seed 20260907, n=10000, source-pair resampling
  old pooled-sign formula used = NO
  unequal-count regression = mean(low) - mean(high) = 2.25 - 0.0 = 2.25 (never a pooled sign mean)

TOP CAUSAL
  PRE M = 0.000389735 [5.94804e-05, 0.000724385]
  MID M = 0.00180776 [0.00148623, 0.00214115]
  POST M = 0.00211501 [0.00177065, 0.0024898]
  adjusted PRE->MID = 0.00141766 [0.00111898, 0.0017271]
  adjusted MID->POST = 0.000307636 [0.000111816, 0.000505215]
  adjusted PRE->POST = 0.0017253 [0.00136104, 0.00212742]
  CI = [0.00136104, 0.00212742]

BOTTOM CAUSAL
  PRE M = 0.00432636 [0.00383759, 0.00482942]
  MID M = 0.00367702 [0.00325391, 0.00412115]
  POST M = 0.00354669 [0.00311532, 0.00399634]
  adjusted PRE->MID = -0.000649035 [-0.000996853, -0.000310067]
  adjusted MID->POST = -0.000130995 [-0.000288734, 2.26394e-05]
  adjusted PRE->POST = -0.00078003 [-0.00115646, -0.000421724]
  CI = [-0.00115646, -0.000421724]

PAIRED TOP-BOTTOM
  PRE->MID gap = 0.0020667 [0.00155563, 0.00260406]
  MID->POST gap = 0.00043863 [0.000184761, 0.000705578]
  PRE->POST gap = 0.00250533 [0.00191989, 0.00316719]
  95% CI = [0.00191989, 0.00316719]
  inference unit = source pair

NATURAL vs SAME-SOURCE
  historical natural TOP = 0.00182477 [0.00148954, 0.00218607]
  historical natural BOTTOM = -0.000644338 [-0.00099515, -0.000289266]
  historical gap = 0.00243725 [0.00191952, 0.00296873]
  same-source gap = 0.00250533 [0.00191989, 0.00316719]
  confounding attenuation = -0.028

CONTENT
  CLIP text available = True
  CLIP text TOP/BOTTOM = 0.00058445 [-0.000605104, 0.00179019]
  CLIP image-full TOP/BOTTOM = 0.0232431 [0.019882, 0.0265391]
  PE content TOP/BOTTOM = 0.00894915 [0.00586515, 0.0119405]
  systematic BOTTOM content loss = True

CONTENT-BALANCED SUBSET
  pair count = 256
  definition = pre-registered lowest-50% |content asymmetry|; asymmetry = |CLIP text delta| when CLIP_TEXT available else |CLIP image delta| + |PE retained delta| (content fields only)
  uses causal results = NO
  TOP-BOTTOM causal gap = 0.00216326 [0.00128263, 0.00321546]
  CI = [0.00128263, 0.00321546]
  attenuation vs all = 0.137

CONTENT <-> CAUSAL
  Spearman = 0.13141971295247248
  quartile trend = {'q0': {'n': 128, 'mean_g': 0.0016953338345047086}, 'q1': {'n': 128, 'mean_g': 0.0026311911933589727}, 'q2': {'n': 128, 'mean_g': 0.0019249830802436918}, 'q3': {'n': 128, 'mean_g': 0.0037698114465456456}}
  interpretation = no pre-registered content-interaction threshold crossed

CORRECT-LOSS BASELINE
  PRE BB-TT = -0.0012917 [-0.00510967, 0.00241804]
  MID = -0.00132927 [-0.00513982, 0.00237037]
  POST = -0.00147198 [-0.00527929, 0.0022599]
  interpretation = L_BB - L_TT per ckpt: whether the BOTTOM crop itself is systematically harder under its own correct coordinates (not a final quality metric)

STRATA
  anchor START result = 0.00273076 [0.00182915, 0.00380564] (n=259)
  anchor END result = 0.00227456 [0.0015661, 0.00300143] (n=253)
  mild = 0.000684171 [0.000245959, 0.00111553] (n=350)
  medium = 0.00571216 [0.00424948, 0.0075245] (n=141)
  strong = 0.0113264 SMALL_N (n=21)
  latent <2 = 0.000286756 [-0.000143409, 0.000730861] (n=200)
  latent 2-4 = 0.00285134 [0.00199536, 0.00384116] (n=275)
  latent >=4 = 0.0119259 [0.00938192, 0.0147221] (n=37)

CLASSIFICATION
  MIXED_CONTENT_AND_DIRECTIONAL_ASYMMETRY

RECOMMENDATION
  REVIEW_CROP_AND_VERTICAL_SUPERVISION

VALIDATION
  old audit tests = <gate runner>
  V1 tests = <gate runner>
  V2 tests = <gate runner>
  new tests = <gate runner>
  total = <gate runner>
  skips = 0
  xfails = 0
  replay = REPLAY = PASS (9 point estimates + 3 arms x (3 checkpoint means+ci95 + 3 delta point+ci95) exactly equal to committed report)
  ruff = All checks passed (ruff 0.16.x, worktree context: new tooling 15 files + new test)
  py_compile = <gate runner>
  src/config diff = <gate runner>

IMMUTABILITY
  historical causal = <gate runner>
  V1 posthoc = <gate runner>
  V2 posthoc = <gate runner>
  offset balance V2 = <gate runner>
  prior tooling = <gate runner>
  /tmp evidence retained = <gate runner>

SECURITY
  secret hits = <gate runner>
  checkpoints staged = NO
  PNG staged = NO
  dataset staged = NO
  binary cache staged = NO

AUTHORIZATION
  longer P25 started = NO
  P50 started = NO
  production changed = NO

EXTERNAL REVIEW TARGET
  branch = camera-v2-vertical-bottom-supervision-review
  exact SHA = None

NEXT
  HARD STOP
  SEND SHA TO EXTERNAL REVIEWER
  EXTERNAL REVIEW REQUIRED BEFORE ANY TRAINING GO

== END ==
