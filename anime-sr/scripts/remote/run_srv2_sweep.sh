#!/bin/bash
# SR_v2 throughput sweep (2026-08-31): arms run SEQUENTIALLY on the 2 DCU
# (one run at a time). Each arm = fresh init, 50,000 exposures (srv2_throughput
# overlay), wandb run per arm. Throughput is read from the it/s console lines
# and <out>/train-meta.json (consumer_img_s_per_rank).
set -u
# all outbound (wandb) needs the SCNet proxy: no proxy -> wandb.init times out
. /etc/profile.d/zz-scnet-proxy.sh
cd /root/private_data/anime-sr/m4canary-src/anime-sr || exit 1
export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/hyhal/lib:/opt/dtk-26.04/dcc/gcvm/lib
export PYTHONPATH=src
export OMP_NUM_THREADS=4
KEY=$(grep '^WANDB_API_KEY=' /root/private_data/anime-sr/.ms-wandb-creds | cut -d= -f2-)
if [ -z "$KEY" ]; then echo "[sweep] ERROR: no WANDB_API_KEY"; exit 1; fi
export WANDB_API_KEY="$KEY"
RUN=/root/private_data/anime-sr/srv2-run
CFG="config/base.toml config/data.toml config/m4_1024.toml config/srv2_throughput.toml"
# usage: run_sweep.sh [arms] [config_prefix] [out_base]
#   default: "bs8 bs16 bs24 bs32" srv2_tp srv2-tp
#   optimized round: run_sweep.sh "bs8 bs16 bs24" srv2_tp2 srv2-tp2
ARMS="${1:-bs8 bs16 bs24 bs32}"
PREFIX="${2:-srv2_tp}"
OUTBASE="${3:-srv2-tp}"
for arm in $ARMS; do
  LOG=$RUN/logs/sweep/${PREFIX}_$arm.log
  mkdir -p "$(dirname "$LOG")"
  echo "[sweep] arm ${PREFIX}/$arm start $(date)"
  /usr/local/bin/torchrun --nproc_per_node=2 -m anime_sr.cli.train_latent_flow \
    --config $CFG "config/${PREFIX}_$arm.toml" \
    --index-dir "$RUN/data/index" \
    --webp-dir "$RUN/data/webp" \
    --bucket-hr 1024 \
    --vae /root/private_data/anime-sr/model/vae/mage-vae.safetensors \
    --out-dir "$RUN/output_model/${OUTBASE}/$arm" \
    > "$LOG" 2>&1
  echo "[sweep] arm ${PREFIX}/$arm EXIT=$? $(date)"
done
echo "[sweep] ALL DONE $(date)"
