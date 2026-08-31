# CMuon FP32 Rescue — D2 轮审计（POST-NS DETECTION + FP32 RESCUE）

日期：2026-08-31
宿主：crdnotebook-…-salt6（172.31.33.71，2×HCU，NFS /root/private_data）
代码：cmoun-guarded @ 7ec1c53+（候选实现 9ab667f；R1 toml a7d7e66）
Fork：ckpt_97100_raw-97100-update-cadence（12/12 manifest verified，AdamW S0 谱系）

## 0. 背景与路线裁决

- S1/S1-b（v1 幅度 guard）双 FAIL 结案：danger 类需要 skip 64–70% 才能覆盖（>50% 过紧线）。
- D1（pre-NS 结构判别器）离线 FAIL 结案（危险类 80.3% 的 safe 行高于最佳 FN=0 阈值；判别器无可行工作点）。
- 本路线（D2）：**保留 BF16 NS4 为第一路径，任何 BF16 NS 结果触碰 ceiling/非有限时，由 owner rank 用纯 FP32 重算同一 NS4 输入；FP32 结果再过同一条 post-NS 安全检查（有限 + Moonlight 合理带 [0.05×, 10×](0.2·lr) + ceiling）；通过则单次舍入回 BF16 走不变的 canonical broadcast/fingerprint/两阶段原子 commit；FP32 也失败才 fail-closed（CMuonSafetyError，零提交）。**
- KILL GATE（binding）：离线 replay 中 FP32 无法 100% rescue 已知 dangerous cases ⇒ 立即停止。

## 1. 离线验证（KILL GATE — PASS）

数据：16-obs shadow（ckpt_97100 fork，2657 tensor 全量 full-sample dump，SAFE 2650 / DANGEROUS 6 标签），artifacts-fp32-rescue/（92G）。

### 1.1 Replay（2657 tensor）
- BF16 catastrophic = 1（obs16 slot_07 q_proj c0：bf16 ns_max=756，delta_rms 7.78e-4 = 2.49× ceiling）
- 该输入 FP32 rescue：ns_max=0.31，delta_rms 1.498e-5 = 0.048× ceiling，moonlight_ok
- fp32_failed = 0，fp32_nonfinite = 0 → rescue_rate = 1/1 = 100%
- 注：chaos 是 run-dependent 的——6 个 D1 标签 dangerous 中仅 1 个在 replay 中复现 BF16 灾难（§1.2 的 repeats 测试刻画了分支概率）。

### 1.2 可重复性（×100 hard-sync repeats，7 个输入 = 6 D1 dangerous + 1 NFS）
- BF16 灾难率（每输入）：0.15–0.51（obs08 s02=0.25 / obs10 s02=0.15 / obs10 s04 k=0.0 / obs13 s07=0.51 / obs15 s07=0.18 / obs16 s07=0.38 / NFS k=0.0）；最坏 rms 9.77e-06 .. 6.26e+02（~5 个数量级摆动）
- FP32：**0/700 catastrophic，0 nonfinite，output spread = 0.0**（100 次 bit-deterministic）
- KILL GATE 判据（fp32_failed=0 且 fp32 repeat catastrophic=0）：**PASS**

### 1.3 SAFE 对齐（1958 分层样本，257 桶，≥1000 ✓）
- delta_ratio fp32/bf16：p50=0.9829 / p90=1.0108 / p99=1.0197（min 0.33）
- update cosine：p50=0.99663 / p90=1.0004 / p99=1.0016
- relative Fro error：p50=0.0903 / p90=0.4101 / p99=0.9164（角度主导，与 cosine/ratio 一致；同一 Muon 几何，非 bit-identical）

### 1.4 成本（5 生产形状，iters=50）
- 2560²=0.915× / 640×2560=0.918× / 6912×2560=0.939× / 2560×6912=1.000× / 2560×1024=0.960×（fp32/bf16 wall）
- **FP32 NS4 在该 HCU 上不慢于 BF16**；期望 rescue 频率 ~1.08/obs × ~0.6ms ≈ 0.65ms/obs ≈ 15.8s update 的 0.004% → 实际开销 ≈ 0

## 2. 实现（§7–§11）

- `src/sakuramoon/optim/fp32_rescue.py`：`HybridCMuonCanonicalNS4FP32Rescue(HybridCMuonGuardedCanonical)` + `build_fp32_rescue`（与 build_guarded_canonical 同签名）
- 机制要点：
  - 无 low-signal skip（retired 结构 pre-NS guard 不使用；所有梯度 chunk 进入 NS）
  - BF16 NS4 完全同生产路径（canonical owner-rank、单次批量 ceiling sync）
  - rescue pass 只在 owner rank、只在 all_reduce 之前、只对 owner-local 失败输入执行；成功 rescue 清 flag，单次舍入回 BF16 后走不变的 broadcast/fingerprint/commit
  - 仅 FP32 也失败才 raise（两阶段原子，零提交）
  - 计数器（bf16_attempts / bf16_safety_failures / fp32_attempts / fp32_rescues / fp32_rescue_failures / rescue_by_role）写入 guard state（"guard" 块内，checkpoint 顶层键集不变）
  - 注册：schema Literal + 校验器（新 name 要求 guard 段）；production 分派（同 forensic/telemetry/NS-map 校验）；checkpoint save/load 不变（isinstance 覆盖子类）
- 单元测试 A–H（tests/unit/optim/test_fp32_rescue.py）HCU 8/8 通过（88.95s）：
  A 安全步不触发 FP32 / B 坏BF16→FP32 rescue 成功且 delta 在带内 / C 坏BF16+坏FP32→CMuonSafetyError 零提交 / D 2-rank owner-only（各 rank 只 rescue 自己的输入）/ E rescue 后 broadcast 一致（spread=0、rank 参数 diff=0）/ F ckpt roundtrip 计数器 state-exact + 父类 ckpt 兼容 / G AdamW 路径与 core hybrid bit-identical / H retired 机制不可选（无 pre-NS skip、base skip gate 不被调用、schema 要求 guard 段）
- 回归：tests/unit/optim/ 全量 108/108 通过（216.73s）

## 3. R1 安全门（§12）

配置：config/train_g1_fp32_rescue_r1.toml（2 rank，ckpt_97100 fork，structural 647 mainset/预热缓存，W&B 关，外部看门狗 97300 停）
Gates：0 unresolved safety failure / 0 rank drift / 0 nonfinite / 0 catastrophic loss jump / 全部 BF16 事件被 rescue / 原子 ckpt resume PASS（97200 cadence ckpt）

[运行中 — 结果待填]

## 4. 裁决（§13/§14）

[待 R1 结果]
