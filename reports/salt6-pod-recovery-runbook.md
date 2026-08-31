# salt6 pod 重建恢复 runbook（D2 轮，08-31）

适用：平台回收 salt6 pod 后（跳板端口 ECONNREFUSED、内网 sshd 关闭、邻居 pod 正常）。
NFS `/root/private_data` 幸存；本地盘 `/root`、`/sakuramoon-runtime` 全灭。

## 0. 等待 pod 起来

- `ssh salt6` 跳板 42.228.13.144:10444 恢复连接即 pod 就绪（平台自动重建；
  若 30 分钟未恢复，需用户在平台控制台确认/重建 pod）。
- 起来后先 `hostname` + `uptime` 确认是新 pod（up < 1h），确认新内网 IP
  （`hostname -i`），并把新 IP 加入 no_proxy（见 4）。

## 1. 环境重建

```bash
source /etc/profile.d/zz-scnet-proxy.sh 2>/dev/null   # 若 /etc 也重建了：先重注入
bash /root/private_data/proxy-inject/reinject-proxy-salt.sh
source /opt/dtk/env.sh
export LD_LIBRARY_PATH=/opt/dtk/lib:$LD_LIBRARY_PATH
```

- DTK `/opt/dtk` 是平台镜像自带的（pod 重建后仍在）；若缺失走平台工单。
- 代码（NFS 上）：
  ```bash
  cd /root/private_data/sakuramoon-g1-forensic
  git fetch origin cmoun-guarded && git checkout origin/cmoun-guarded
  ```
- venv 重建（本地盘全灭）：
  ```bash
  cd /root/private_data/sakuramoon-g1-forensic
  uv venv /sakuramoon-runtime/sakuramoon-dtk-venv --python 3.11
  # 依赖装 sourcefind 镜像（仅可经代理）：
  source /etc/profile.d/zz-scnet-proxy.sh
  VIRTUAL_ENV=/sakuramoon-runtime/sakuramoon-dtk-venv \
    uv pip install --python /sakuramoon-runtime/sakuramoon-dtk-venv/bin/python \
    -i http://download.sourcefind.cn:65024/4/main/ --trusted-host download.sourcefind.cn \
    torch==2.9.0 torchaudio transformers accelerate "torchao==0.16.0" \
    numpy einops safetensors wandb "torchdata" pydantic loguru pillow webdataset
  ```
  （以 pyproject.toml 实际依赖为准；torch 走镜像直装，勿 pip index 探测。）
- venv torch 需要 `source /opt/dtk/env.sh` + `LD_LIBRARY_PATH=/opt/dtk/lib`
  （libgalaxyhip.so.5）。

## 2. 数据缓存重建（只补 16-obs 所需的最少分片）

- 16 obs = 12,800 samples = 5 个 2560-sample 分片：mainset rows[647:653]（648-653）。
- relay 源在 salt3（live G1 缓存），TSV 在 NFS：`/root/s6-relay-shards.tsv`
  （136 片，path-only 行）。只取前 6 行重跑 relay（或全量 136，22 分钟，
  缓存目录 /sakuramoon-runtime/cache，data-service LRU 需要 mtime 排队列序）。
  最少方案：
  ```bash
  head -6 /root/s6-relay-shards.tsv > /root/relay-16obs.tsv
  # 用 relay-s6.py 变体指向 /root/relay-16obs.tsv + 独立 state 文件
  ```
- toml 必须显式 `[data.cache] low_watermark_gib=192 high_watermark_gib=500`
  （P3 ns4_core 链不覆盖，base 默认 high=256 会启动时 LRU 驱逐）。
- structural mainset 状态在 NFS：`config/data-service-mainset-structural.json`
  （应为 647；若上次 16-obs 跑到一半被 pod 死亡打断，先重置回 647 再跑）。

## 3. 重跑 16-obs shadow（全量 dump）

```bash
# /root/structural-workload-env 是 NUL 分隔，本地盘灭了要重建：
python3 - <<'EOF'
vals = [
    "SAKURAMOON_STRUCTURAL_CALIBRATION_STEPS=16",
    "SAKURAMOON_STRUCTURAL_NS_REPEAT=5",
    "SAKURAMOON_STRUCTURAL_PI_ITERS=20",
    "SAKURAMOON_STRUCTURAL_SIGMA_METHOD=pi",
    "SAKURAMOON_STRUCTURAL_REFS_JSON=/root/private_data/sakuramoon/config/guard-refs-structural.json",
    "SAKURAMOON_STRUCTURAL_FULL_SAMPLE_OBS=16",
    "OMP_NUM_THREADS=8",
]
open("/root/structural-workload-env","wb").write(b"\x00".join(v.encode() for v in vals) + b"\x00")
EOF
bash /root/start-structural-shadow.sh start   # 脚本在 NFS? 若本地盘灭则从 repo 重建
```
- 日志 /root/sakuramoon-logs-structural/（本地，重建后重新 tee）。
- 期望 ~15 min，`[calibration] 完成: 16 次观察` 干净停止。
- **注意**：新 pod 的 forward/bwd 低位比特与旧 pod 不同 → 危险事件集合
  是新的混沌样本（与旧两次跑都不完全重合属正常；统计包络应一致：
  ~1 危险/obs、slot_02/07/08/21、attention k/q/gate/v）。

## 4. proxy no_proxy 补新 pod IP

`reinject-proxy-salt.sh` 会自动探测新 pod IP 写入 no_proxy（脚本已内置）。
若手动：`/root/private_data/proxy-inject/proxy.env` 的 no_proxy 追加
`<new_internal_ip>`，重跑脚本。

## 5. 重跑离线验证四件套

```bash
cd /root/private_data/sakuramoon-g1-forensic
PY=/sakuramoon-runtime/sakuramoon-dtk-venv/bin/python
S=/sakuramoon-runtime/artifacts/g1/structural-calibration/full-samples
# (a) 全量 replay（21 obs 若 100-obs 样本也重跑，否则只 16 obs = 2656 张量）
$PY -u scripts/fp32_rescue_replay.py replay --tensors "$S/obs-*/chunk-*.pt" \
  --device cuda:0 --out /sakuramoon-runtime/artifacts/g1/fp32-replay-all.json
# (b) 100-repeat 混沌专测（对 replay 中 bf16 catastrophic 的张量，选 hazard 最高的 2-3 个）
$PY -u scripts/fp32_rescue_replay.py repeats --tensors <危险.pt...> \
  --n 100 --device cuda:0 --out /sakuramoon-runtime/artifacts/g1/fp32-repeats.json
# (c) SAFE 分层对齐（≥1000）
$PY -u scripts/fp32_rescue_replay.py align \
  --tensors "$S/obs-*/chunk-*.pt" \
  --jsonl /sakuramoon-runtime/artifacts/g1/structural-calibration-rank0.jsonl \
           /sakuramoon-runtime/artifacts/g1/structural-calibration-rank1.jsonl \
  --min-samples 1000 --device cuda:0 --out /sakuramoon-runtime/artifacts/g1/fp32-align.json
# (d) 成本 benchmark
$PY -u scripts/fp32_rescue_replay.py benchmark --iters 50 --device cuda:0 \
  --out /sakuramoon-runtime/artifacts/g1/fp32-benchmark.json
```
- 全部结果落 NFS：`/root/private_data/sakuramoon-g1-forensic/artifacts-fp32-rescue/`
- **kill 判据**：replay 中 fp32_failed（BF16 catastrophic 且 FP32 也
  catastrophic/degenerate）== 0，且 repeats 中 FP32 catastrophic fraction
  == 0（或显著消失）→ 才进入实现阶段。

## 6. 数据幸存清单（NFS，已验证 08-31）

- `artifacts-fail-record/`：100-obs 分析 + 原始 rank jsonl×2 + counterexamples + 2 tensor
- `artifacts-fail-record/tensors/`：obs3 slot_07 k_proj（S1-b 崩溃张量）+ obs1 slot_00 out_proj
- `config/data-service-mainset-structural.json`、`config/guard-refs-structural.json`
- `s6-relay-shards.tsv`（relay 源清单，path-only）
- `proxy-inject/`（代理重注入源 + 脚本）
- repo git tree（cmoun-guarded @ f71215f）
