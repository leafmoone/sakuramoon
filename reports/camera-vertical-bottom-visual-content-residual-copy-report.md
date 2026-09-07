== SakuraMoon Camera · Visual Content Residual Audit ==

BASE
  repository = leafmoone/sakuramoon (github)
  reviewed base = f3d183a35b3123e0b9c2a301fbc68ff5a276d345
  branch = camera-v2-visual-content-residual-review
  tooling commit = <VISUAL_CONTENT_TOOLING_HEAD>
  evidence commit = <VISUAL_CONTENT_EVIDENCE_HEAD - the commit containing this file>
  remote SHA = <ls-remote read-back; == evidence HEAD>
  pushed = YES
  force push = NO

INPUT INTEGRITY
  pair manifest sha = 7f3a81b248473d0b0591377c1bcd9f7df353066c9716836dabe90d9861f56297
  pairs = 512
  clip-image rows = 512
  pe rows = 512
  clip-text available = True
  causal rows = 512
  duplicate/missing = 0
  replay = <gate runner>

PHASE SEPARATION
  subset frozen before causal load = YES
  freeze script reads causal data = NO
  subset manifest sha = 15506002a0480e29e0b87c147bb018432349b99fe8d1af85b24be196b88db0ca

VISUAL SCORE
  definition =
    0.5*(rank(|CLIP_IMAGE_DELTA|)+rank(|PE_RETAINED_DELTA|))
  clip text used in primary score = NO
  tie handling = average ranks for ties; percentile rank in [0,1]; subset tie break by pair index

SUBSETS
  ALL = 512
  VISUAL_GEO_BALANCED_50 = 256
  VISUAL_GEO_BALANCED_25 = 128
  global50 = 256
  global25 = 128
  image-only50 = 256
  pe-only50 = 256
  joint-low intersection = 145

GEOMETRY PRESERVATION
  ALL START/END = 259/253
  ALL zoom = 350/141/21
  ALL latent = 200/275/37
  V50 START/END = 129/127
  V50 zoom = 175/71/10
  V50 latent = 100/138/18
  V25 START/END = 65/63
  V25 zoom = 88/36/4
  V25 latent = 50/69/9

VISUAL CONTENT
  ALL CLIP-image TOP-BOTTOM = 0.0232431
  CI = [0.019882, 0.0265391]
  ALL PE retained TOP-BOTTOM = 0.00894915
  CI = [0.00586515, 0.0119405]
  V50 CLIP-image = 0.0141699 [0.0112551, 0.0172162]
  V50 PE = 0.00231353 [-4.49402e-05, 0.00474361]
  V25 CLIP-image = 0.00963654 [0.00643882, 0.0136448]
  V25 PE = 0.00140684 [-0.000736482, 0.00363509]
  visual balance successful = A_visual mean ALL 0.5 -> V50 0.329726 -> V25 0.23329

CAUSAL GAP PRE->POST
  ALL = 0.00250533
  CI = [0.00191989, 0.00316719]
  V50 = 0.00200864
  CI = [0.00134108, 0.00271416]
  V25 = 0.00227716
  CI = [0.00132526, 0.00333803]
  RETENTION_50 = 0.801748
  RETENTION_25 = 0.908928
  ATTENUATION_50 = 0.198252
  ATTENUATION_25 = 0.0910721

CAUSAL GAP PRE->MID
  ALL = 0.0020667 [0.00155563, 0.00260406]
  V50 = 0.0017454 [0.00109455, 0.00244141]
  V25 = 0.00204399 [0.00114788, 0.00307982]

CAUSAL GAP MID->POST
  ALL = 0.00043863 [0.000184761, 0.000705578]
  V50 = 0.000263242 [5.05912e-05, 0.000488717]
  V25 = 0.000233175 [-5.04156e-05, 0.00050861]

CORRELATIONS
  rho signed CLIP-image vs G = 0.158126
  rho signed PE vs G = 0.131312
  rho abs CLIP-image vs abs G = 0.231713
  rho abs PE vs abs G = 0.0914757
  rho A_visual vs abs G = 0.202585
  rho CLIP-text vs G (secondary) = -0.0489796

QUARTILES
  Q0 = 0.00133608 [0.000605977, 0.00210295] (n=128)
  Q1 = 0.00148051 [0.000596172, 0.00238372] (n=128)
  Q2 = 0.00289857 [0.00181662, 0.0040636] (n=128)
  Q3 = 0.00430616 [0.00267015, 0.00633162] (n=128)
  monotonic content-gap relation = <assessed in audit.md section 7>

SENSITIVITY
  global visual 50 = 0.0014083 [0.000829487, 0.00199553]
  global visual 25 = 0.00133608 [0.000594442, 0.00209593]
  image-only 50 = 0.00233519 [0.00166366, 0.00303565]
  image-only 25 = 0.00262889 [0.00167325, 0.00369502]
  PE-only 50 = 0.00199405 [0.00131066, 0.00270219]
  PE-only 25 = 0.00208636 [0.00109378, 0.00317313]
  joint-low intersection = 0.00146127 [0.0006656295497869623, 0.002306399845100682]
  result = <assessed in audit.md section 10>

DIRECTIONAL RESIDUAL
  CONFIRMED

VISUAL CONTENT CONTRIBUTION
  STRONG_SUPPORTED

OVERALL CLASS
  MIXED_VISUAL_CONTENT_AND_DIRECTIONAL

RECOMMENDATION
  DESIGN_VERTICAL_MIRROR_BALANCED_SUPERVISION_WITH_CONTENT_GUARDS

LEGACY TOOL NOTE
  prior pooled_sign_mean imported = NO
  old reports modified = NO

VALIDATION
  audit tests = <gate runner>
  V1 tests = <gate runner>
  V2 tests = <gate runner>
  VBS tests = <gate runner>
  visual residual tests = <gate runner>
  total = <gate runner>
  skips = 0
  xfails = 0
  replay = <gate runner>
  ruff = <gate runner>
  py_compile = <gate runner>
  git diff check = <gate runner>

IMMUTABILITY
  original causal = <gate runner>
  V1 = <gate runner>
  V2 = <gate runner>
  offset balance = <gate runner>
  VBS tooling = <gate runner>
  VBS reports = <gate runner>
  src diff = EMPTY
  config diff = EMPTY
  /tmp evidence retained = YES

SECURITY
  secret hits = <gate runner>
  checkpoint staged = NO
  image staged = NO
  latent staged = NO
  dataset staged = NO

AUTHORIZATION
  longer P25 started = NO
  p50 started = NO
  production changed = NO

EXTERNAL REVIEW TARGET
  branch = camera-v2-visual-content-residual-review
  exact SHA = <recorded in session FINAL COPY after push>

NEXT
  HARD STOP
  SEND EXACT SHA TO EXTERNAL REVIEWER
  EXTERNAL REVIEW REQUIRED BEFORE ANY TRAINING/DESIGN IMPLEMENTATION GO

== END ==
