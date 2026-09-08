# Camera MBS Corrected 100U Treatment — RERUN #2 (post Phase-C) — EVIDENCE

**Status: PASS** — exactly 100 successful updates U118101..U118200, HARD STOP AT U118200,
no U118201, no auto-restart. Supervisor result v1: PASS
("terminal 118200 reached; ckpt COMPLETE; no U118201; records=100").

- Host: come7 (2x Hygon BW DCU, DTK, torch 2.9.0), run 2026-09-08 18:22:51 -> 18:55:10 CST
- GO: user 2026-09-08 17:42 (CORRECTED MBS 100U TREATMENT — RERUN #2, post Phase-C data-readiness/queue fix)
- This is the third start of the authorized U118100->U118200 window after two 0U FAILs and one
  setup-abort (see §Attempt ledger). The 17:42 GO was not consumed by any prior start: no
  checkpoint was ever written and no queue state advanced under it before this run.

## 1. Attempt ledger (distinguished, all frozen in /sakuramoon-runtime/failed-attempts/)

| Attempt | Time (CST) | Code | Outcome | Archive |
|---|---|---|---|---|
| 0U FAIL #1 (corrected 100U treatment) | 09-08 12:48 | 73d419c6 | 0 updates: dynamo recompile-limit compile incompatibility | corrected-mbs-0u-e6d7b103/ |
| 0U FAIL #2 (rerun #1) | 09-08 16:45 | b78ba02 | 0 updates: data-service cold-start race (socket-exists readiness check vs 16-shard warmup barrier > 30s client timeout) + queue cycle-0 order not cross-process reproducible (frozenset iteration) | corrected-mbs-0u-fail2-20260908-1645/ |
| rerun2 setup-abort | 09-08 17:52-18:09 | ef05e93 | Healthy run stopped at operator command: stack start env lacked RUN_ROOT export, train.pid landed in default /run/sakuramoon while tracked supervisor watched /run/sakuramoon-mbs-corrected-treatment-rerun2 (would false-positive FAIL at 18:12:24). 34 metric records (118101-118134) archived as progress evidence only; NO checkpoint written, NO queue state consumed, source ckpt unchanged | camera-mbs-rerun2-setup-abort-20260908-1752/ |
| **rerun2 (this run)** | **09-08 18:22-18:55** | **ef05e93** | **PASS, 100/100 updates** | — |

Fix chain (all pushed, all included in this run): 73d419c6 (worker-clone) -> b78ba02 (dynamo
recompile-limit fix + compile saturation smoke + supervisor stop-path hardening) -> 663d2e9
(fix: canonicalize data queue order and gate socket on warmup) -> ef05e93 (docs).

## 2. Code & environment identity

- Branch: camera-v2-mbs-corrected-100u-rerun2-evidence (worktree /sakuramoon-runtime/sakuramoon-camera-mbs-corrected-100u-rerun2)
- Base (externally reviewed): b78ba023f781535a9d8fd98b0fc543925e5f04a6; H1=663d2e9bf14b68c3c060acfcaf5660c96a343653; push head=ef05e937aefa890b38e7cb7e0808ee26f9f768b3; rev-list(b78ba02..ef05e93)=2; worktree clean pre-launch
- Tracked supervisor sha256: e86a209d7fa4b8e31aa3d54d5eb7be65d642cb201a80b68bc9437a8ba4066e51 (scripts/mbs_corrected_treatment_supervisor.py @ ef05e93)
- Config: train_g1_camera_v2_p25_mirror_v2_canary.toml, 23/23 field checks PASS (run_id g1_camera_v2_p25_mirror_v2, world 2, local 20, accum 20, global 800, planned 168000, canary_stop 118200, no auto-transition; camera hdm_shifted_square_v2 p=0.25 zoom 1.10-1.50; MBS vertical_mirror_pair_v1 min_shift 2.0 p=1.0 w=1.0; hybrid_cmuon_canonical_ns4_fp32_rescue; inductor dynamic max-autotune-no-cudagraphs recompile_limit=64 fail_on=true suppress=false; NO config edits this run)
- Env snapshot .env.training-stack.nul sha256: 4a79e8914dd6c1575a60449c30ae0086e288e3b28222d8838ba674671989597a (600, not symlink)
- DTK env: torch 2.9.0, 2 devices, no installs, no reboot; OMP/MKL=32

## 3. Pre-launch gates (all PASS)

- SOURCE ckpt_118100 manifest sha256: 821a8c12546a094865972fef0c23152e9754e01f82eed8353f7248b92076b892 (pre and post run, EXACT)
- CONTROL ckpt_118200 (g1_camera_v2_p25_mirror, v1 canary terminal) manifest sha256: f014827fb441aca8780a5bb710eeb2631a40aac112ed3aa6b07b6e3b65d657f7 (pre and post run, EXACT)
- Queue state pre-run: ABSENT (fresh cycle-0; reset performed 18:11 after setup-abort with backup data-service-mainset.json.pre-rerun2b-f43eefc2.backup, sha-verified)
- Canonical cycle-0 offline probe (spec §6, reviewed code, production manifest+selection, store.new(0) WITHOUT save): total rows 2956, full order sha256 d9d7081a782cf12bc9cfd2b28a5f6e791a66716932eec5e930b114e03203b8b5, BIT-IDENTICAL under PYTHONHASHSEED=7 and 12345 (Phase-C D2 fix verified live; production does not depend on fixed seed)
- v2 treatment roots (output_model/artifacts/runs) EMPTY pre-run; no other live workload; port 29501 bind-verified free; supervisor self-test-stop-env exit 0; training_stack.sh validate PASS
- RUN_ROOT fix (root cause of setup-abort): RUN_ROOT=/run/sakuramoon-mbs-corrected-treatment-rerun2 exported in the stack start env; train.pid landed in the tracked RUN_ROOT (supervisor saw it); no train.pid in default /run/sakuramoon

## 4. Supervised timeline (CST)

| t | event |
|---|---|
| 18:22:51 | supervisor 89502 started (nohup setsid), WAITING_FOR_START; startup_deadline=1200s wall=14400s stall=1200s first=118101 terminal=118200 |
| 18:23:04 | START invoked (explicit GO) with RUN_ROOT exported |
| 18:23:0x | data service 89669 started; 16/16 barrier shards cache-hit (warm from setup-abort attempt) -> socket /run/sakuramoon/data-service.sock created seconds after data start (Phase-C D1: socket only after wait_until_ready(); stack `[[ -S ]]` check correct again); 26 partials/19803 MiB retained, lookahead prefetch at ~110-124 MiB/s/stream |
| 18:23:1x | publisher 89775 + train 89899 started (trainer appears ONLY after socket); ranks 2/2 ready ~40s |
| 18:23:1x | supervisor -> RUNNING (train.pid in tracked RUN_ROOT) |
| 18:27:39 | first update 118101 (eligible=30 applied=30) |
| 18:28:39 | EARLY_EXPOSURE_GATE = PASS @ 118103 (cum severe=85 elig=85 sel=85 app=85) |
| 18:54:17 | terminal metric 118200 |
| 18:54:42 | trainer exited; terminal ckpt COMPLETE (update=118200, reason_ok=True); 20s settle |
| 18:55:02 | supervisor RESULT PASS written |
| 18:55:10 | stack stop rc=0 (data/publisher/train stopped, socket removed) |

Data-service client errors during the window: 0 (no broken pipe, no timeout, no traceback).
Teardown verified post-run: 0 matching processes, socket absent.

## 5. Training statistics (100 records, schema 12; full stats in -metrics-summary.json)

- Updates 118101..118200 contiguous, 0 missing, 0 dup; nonfinite_total = 0
- total_loss: mean 0.5854 (min 0.5513 / max 0.6267); high_noise_loss identical (low_noise_count=0)
- post_clip_grad_norm: mean 0.0439 (0.0212-0.1008); clip_fraction 0 throughout
- learning_rate constant 1.5625e-4 (5e-5 x 800/256); effective_batch constant 400
- throughput: samples/s mean 50.5 (median 52.2; first record 6.8 under compile); mean update interval 16.13s (14.37-30.30); wall span 118101->118200 = 1597.0s
- gpu_memory_allocated ~13.63 GiB flat; reserved peak 65.9 GiB
- compile gates CLEAN: no FailOnRecompileLimitHit / limit reached / accumulated failure / eager fallback / compile traceback; dynamo recompiles observed up to index 23 (< limit 64), TORCH_LOGS=recompiles captured
- phase seconds /100U: backward 736.5, dit_forward 406.5, qwen 347.7, vae 69.2, optimizer 61.4

## 6. MBS exposure (rank0 telemetry, supervisor cumulative)

- eligible = selected = applied = **1740** of 40000 logical samples (4.35%); per-record conservation severe==eligible==selected==applied holds on all 100 records
- extra_views = 1740; physical_views = 41740 > logical 40000; vertical_applied = 4505
- severity pool: lt2=2765, 2to4=1499, ge4=241 (sums to applied)
- degraded: 0 (data-service log: no degradation/fallback-to-redownload events)
- The corrected worker path is live: in the v1 canary (73d419c6 predecessor) eligible was 0 on the same dataset under the same policy — this run is the first with real vertical-mirror exposure (min_latent_shift=2.0, p=1.0, w=1.0)

## 7. Queue evidence

- Pre-run: fresh cycle-0 (state absent after 18:11 reset)
- Post-run state sha256: 7ad0ca0afb90cba95548026977109f4bb94e14f75e6f90b0e7f06036ba43415c; cycle=0; 2956 rows; first16/first32 EXACTLY equal to the offline canonical probe order (data-service prefetch order in data-service.log matches canonical head)
- Historical backups intact: .bak-v1-pre-v2-20260908-131634, .bak-v2-0u-fail-pre-rerun-20260908-164120, .pre-rerun2-797e8164.backup, .pre-rerun2b-f43eefc2.backup

## 8. Terminal checkpoint & hashes

- ckpt: /sakuramoon-runtime/output_model/g1_camera_v2_p25_mirror_v2/ckpt_118200_raw-118200-update-cadence — COMPLETE; identity update=118200, kind=raw, schema_version 4; trainer_state: attempted=successful=118200, cadence 100; stage_budget start_successful_update=58000 terminal_successful_update=168000 (UNCHANGED; no stage-finalize; reason update-cadence); growth G1 256px world2 alpha1.0
- Terminal ckpt full-tree sha256-of-sha256s: d2673914eba014ca389178f9bff191df4db31431b5de373ecabbfdfed4206c74
- metrics.jsonl (100 records) sha256: 085840c082600b8a23d7a08291d2ba68336389335a5c4e85a0607d49e7e4ade4
- supervisor-result.json sha256: 3f660e7d6b079d7c9363037e3fa206682d139ce44bba9ca84e1acb3a8875e6bf
- U118201: 0 occurrences in metrics.jsonl and train.log
- source/control re-hash post-run EXACT (see §3)

## 9. Scientific wording (binding)

- This treatment consumed the CANONICAL cycle-0 sequence (cross-process bit-identical after Phase-C D2).
- The control (v1 canary terminal U118200, g1_camera_v2_p25_mirror) is a SAME-DISTRIBUTION INDEPENDENT-SEQUENCE camera-only continuation control: it consumed a different (hash-randomized) cycle-0 order and no mirror pairs at all. It is NOT "exact matched-data".
- No claim is made that vertical asymmetry is "fixed"; the treatment merely applies vertical mirror-pair exposure (1740/40000) under identical base conditions. Effectiveness requires the downstream 3-way audit (not run this round; NO P50/PRODUCTION/FID/3-WAY-AUDIT this round).
- Conditional plan stands: a canonical camera-only control (sharing this exact deterministic cycle-0) costs 100U and is only spent if the 3-way audit lands borderline.

## 10. HARD STOP

Push of this evidence branch is the terminal action of the authorized window. No U118201, no
auto-restart, no eval, no further training. Machine left idle (stack stopped, 0 processes,
socket absent); come7 holds source ckpt_118100, control ckpt_118200 (mirror), and the new
treatment terminal ckpt_118200 (mirror_v2).
