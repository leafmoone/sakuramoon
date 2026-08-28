#!/usr/bin/env bash
# dl_danbooru_v2.sh — resumable ranged batch downloader for danbooru-v2 shards.
#
# Downloads leafmoone/webdataset_danbooru_v2 shards (ms-hub blob API) into
# /root/private_data/anime-sr/danbooru-v2/data/1_2024/ (rw SCNet mount;
# /root/group_data is read-only — never write there).
#
# Resumable: partial shards keep a .part file, curl -C - appends from its
# current size; complete shards are skipped. Safe to re-run / nohup-restart.
#
# Completion is judged against the SERVER's true byte size (1-byte Range
# probe -> Content-Range total), NOT a hardcoded value: shard tar sizes
# vary slightly (probe: 2148065280..2148669440 across shards 0-4).
#
# Usage (on sakrua2):
#   nohup env N_SHARDS=21 bash /root/anime-sr-tools/dl_danbooru_v2.sh >> /root/danbooru_dl.log 2>&1 &
#   N_SHARDS=21  -> 200k images (~44 GB)
#   N_SHARDS=51  -> 500k images (~107 GB)
#
# Requires: curl + /root/anime-sr-env (MODELSCOPE_API_TOKEN, proxy vars).

set -u

HOST="modelscope.cn"
REPO="leafmoone/webdataset_danbooru_v2"
REV="master"
DATA_DIR="${DATA_DIR:-/root/private_data/anime-sr/danbooru-v2}"
N_SHARDS="${N_SHARDS:-21}"
PAR="${PAR:-4}"

[ -f /root/anime-sr-env ] && . /root/anime-sr-env
: "${MODELSCOPE_API_TOKEN:?need MODELSCOPE_API_TOKEN (source /root/anime-sr-env)}"

# Some distro curl builds ignore the http_proxy env var for HTTPS CONNECT;
# pass the proxy explicitly (empty when unset -> direct connection).
PROXY_X=""
[ -n "${http_proxy:-}" ] && PROXY_X="-x ${http_proxy}"

LOG="$DATA_DIR/download.log"
mkdir -p "$DATA_DIR/data/1_2024"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

blob_url() {
  # $1 = zero-padded 6-digit shard index
  echo "https://$HOST/api/v1/datasets/$REPO/repo?Revision=$REV&FilePath=data/1_2024/shard-$1.tar"
}

# True total byte size of a shard, straight from the server.
# The blob API 302-redirects to a signed CDN URL, so the probe MUST follow
# redirects (-L): the CDN answers a 1-byte Range with "206 + Content-Range:
# bytes 0-0/TOTAL" (lowercase header on HTTP/2). Fallback: a server that
# ignores Range answers 200 + Content-Length = total.
query_total() {
  local url=$1 hdr cr cl
  hdr=$(curl $PROXY_X -sS -L --max-time 90 -r 0-0 -o /dev/null -D - \
    -H "Authorization: Bearer $MODELSCOPE_API_TOKEN" \
    -H "Cookie: m_session_id=$MODELSCOPE_API_TOKEN" \
    "$url" 2>>"$LOG" | tr -d '\r') || return 1
  cr=$(printf '%s\n' "$hdr" | grep -i '^content-range:' | sed 's/.*\///')
  cl=$(printf '%s\n' "$hdr" | grep -i '^content-length:' | awk '{print $2}')
  printf '%s' "${cr:-$cl}"
}

dl_one() {
  local i=$1
  local z f part url total attempt rc
  z=$(printf "%06d" "$i")
  f="$DATA_DIR/data/1_2024/shard-$z.tar"
  part="$f.part"
  url=$(blob_url "$z")

  total=$(query_total "$url")
  if [ -z "$total" ]; then
    log "FAIL  $z (server size query failed)"
    return 1
  fi
  if [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$total" ]; then
    log "skip $z (complete, $total B)"
    return 0
  fi
  # incomplete final file from an older run -> fold into .part and resume
  if [ -f "$f" ] && [ ! -f "$part" ] && [ "$(stat -c %s "$f")" != "$total" ]; then
    mv -f "$f" "$part"
  fi
  log "start $z (total $total B)"
  for attempt in 1 2 3; do
    rc=0
    curl $PROXY_X -fsSL --max-time 900 \
      -H "Authorization: Bearer $MODELSCOPE_API_TOKEN" \
      -H "Cookie: m_session_id=$MODELSCOPE_API_TOKEN" \
      -C - -o "$part" "$url" >>"$LOG" 2>&1 || rc=$?
    # Success is judged by size: a 416 on a complete .part (we already have
    # every byte) still reports failure under -f, so size is authoritative.
    if [ "$(stat -c %s "$part" 2>/dev/null || echo 0)" = "$total" ]; then
      mv -f "$part" "$f"
      log "done  $z ($total B)"
      return 0
    fi
    log "retry $z attempt $attempt (rc=$rc, part=$(stat -c %s "$part" 2>/dev/null || echo 0)B want $total B)"
    sleep $((attempt * 10))
  done
  log "FAIL  $z after 3 attempts"
  return 1
}

# Concurrency-bounded fan-out: subshells inherit functions/vars/env, so no
# export -f needed. At most PAR downloads run at once.
log "=== dl_danbooru_v2 start: $N_SHARDS shards (0..$((N_SHARDS - 1))), PAR=$PAR ==="
for ((i = 0; i < N_SHARDS; i++)); do
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
  ( dl_one "$i" ) &
done
wait
# summary: recount from disk (subshells can't update the parent's variables)
done_n=0
for ((i = 0; i < N_SHARDS; i++)); do
  z=$(printf "%06d" "$i")
  [ -f "$DATA_DIR/data/1_2024/shard-$z.tar" ] && done_n=$((done_n + 1))
done
log "=== batch end: $done_n/$N_SHARDS shards complete ==="
[ "$done_n" -eq "$N_SHARDS" ]
