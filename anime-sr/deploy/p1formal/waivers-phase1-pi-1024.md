# Phase I-P-1024 — 正式定义与 Waiver 记录（sakrua10，2026-08-28）

## 正式定义
- anime-sr "AnimeSR-Mage-UFlow" 4× anime blind SR（frozen Mage-VAE latent，LQ-centered residual flow matching）
- **Phase I-P-1024**：300,000 exposures / 18,750 steps / HR 1024 / LQ 256
- U-Flow + Pixel Condition Encoder（pixel features on）
- z_hr on-the-fly Mage encode（冻结 VAE）
- 双卡 DDP（torchrun nproc_per_node=2）
- 这是正式 **Pixel-Grounded 1024 pilot**；不宣称本轮完成 512/768/1024 mixed-resolution 训练
- 不修改模型架构 / loss / optimizer / 既定超参；不进入完整 6M–16M Phase I；不进入 Stage II
- Provenance：git commit c229f93（tag p1formal-launch-20260828，/root/anime-sr-p1formal）+ provenance/manifest.sha256

## DATA-WAIT-WAIVER
当前数据布局（staging webp /var/tmp/anime-sr-local/webp + NFS index + zhr onfly）下 producer 为 I/O 受限，
OMP=2 + 32 workers/rank + depth 4 为实测最优，data_wait ≈ 11.6%（canary 树 4.0–4.2% 不可迁移——不同数据布局/store 模式）。
本 pilot 判定阈值：
- **≤ 15%：PASS**
- **15–20%：WARN**（记录观察，不干预）
- **持续 > 20%：调查**（producer 遥测 + 系统侧）
- **持续 > 30% 或 HCU 明显饥饿：STOP**（报告并暂停）

## RESOLUTION-WAIVER
本次为 **1024-only** Phase I-P（V2 单 run 单桶，实际 bucket=1024，10,057 crops），不宣称验证 mixed-resolution。
若完整 M4 前仍决定使用 512/768/1024 curriculum，另行实现 mixed sampler（数据层改动，另行拍板）。

## 正式判门
- **#3 硬门 = ratio_4_1 ≤ 1.05 且 trajectory finite/stable**（稳定门）
- "l1_4 ≤ l1_1" 仅作为未来 quality-mode 发布判据，**不**因此中止健康的 Phase I-P
- probe cadence：2500 / 5000 / 7500 / 10000 / 12500 / 15000 / 17500 + final 18750
- 每探针记录：l1_anchor / l1_1 / improvement vs anchor / toward_1 / cos_v / l1_4 / ratio_4_1 /
  endpoint t0-t25-t50-t75 / trajectory deviation / data telemetry
- final 18750 额外：held-out validation + seam probe + gradient coverage +
  pixel-path gradient/activation evidence + #3 stability verdict

## Worker 死亡规则（进程池监控）
- 任何 "pool worker(s) died" 日志行必须立即记录并报告
- 单次恢复按现有 exactly-once recovery 继续
- **前 500 step 发生 ≥ 2 次 worker death → 主动停止调查**
- 现有累计 4 次死亡 hard abort 保持不变
- 不允许静默吞掉 worker crash

## Pixel Path Activation Probe
- step 100–500 范围确认（首个可用 ckpt = step 1000，save_every=1000）：
  trunk.proj_p64.weight / trunk.proj_p32.weight / trunk.proj_p16.weight /
  trunk.conditioner.gap_proj.weight 非零 grad；PixelConditionEncoder 主体出现非零梯度
- zero-init 为迁移设计要求；正式 pixel-feature 训练开始后这些权重必须开始学习
- **若到 step 500（以首个 ckpt 代证）Pixel Encoder 路径仍基本没有梯度 → 立即 STOP 并报告**，
  不允许把 latent-only 训练误认为 Phase I-P

## CODE-DELTA (08-29, end-of-run cleanup fix)
- latent_flow.py md5 88739d68d2e819d845982e0a99b1510c -> 139d10a1de89b12e5395260947a87ec5 (1 hunk @ L1306).
- Cause: end-of-run crash — DTK/HIP python's multiprocessing.pool.Pool lacks shutdown() (platform API surface: close/join/terminate only); after all 18,750 loop steps completed, crash at L1306 (both ranks, 08-29 02:50:32) prevented latest.pt + the run-end held-out probe from being written.
- Fix: compat branch — close()+join() when .shutdown absent (equivalent to shutdown(wait=True)). No training semantics changed (architecture/loss/optimizer/hyperparams untouched).
- Steps 18001-18750 run on the patched tree; steps 1-18000 on the manifest tree (88739d68...).
- Note: the pixel_features resume branch is designed fresh-optimizer (trunk-transition semantics); the optimizer state present in step-0018000.pt was not loaded. lr/schedule restored from step 18000 (lr 1.55e-05 at 18050, consistent with the 1.5e-05 decay tail); negligible effect over the final 750-step tail.
- Patch file: deploy/p1formal/patch_pool_shutdown_compat_20260829.patch (19 行 unified diff, git diff vs c229f93 树)
