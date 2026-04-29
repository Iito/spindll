#!/usr/bin/env bash
# scripts/autoloop.sh — Karpathy autoresearch sweep on a perf metric.
# Usage: scripts/autoloop.sh <metric> <grid.jsonl>
# Metric: prompt_eval_tps | decode_tps | peak_rss_mb | p50_ms | p95_ms
# Grid: JSONL, one trial per line: {"label": "...", "patch_cmd": "...", "env": {"K":"V"}}
# Runs ONLY on ubuntu (mac is reserved for non-agent work).
set -euo pipefail

cd "$(dirname "$0")/.."

METRIC="${1:?metric required (decode_tps | prompt_eval_tps | peak_rss_mb | p50_ms | p95_ms)}"
GRID="${2:?grid jsonl path required}"

if [[ "$(uname)" == "Darwin" ]]; then
  echo "ERROR: autoloop is ubuntu-only. Run on the home server." >&2
  exit 2
fi

BACKEND="${BACKEND:-cuda}"
THRESHOLD="${THRESHOLD:-0.02}"
RUNS="${RUNS:-3}"
WALL_CAP_SEC="${WALL_CAP_SEC:-7200}"
MODEL="${MODEL:-llama3.2:1b}"

DATE=$(date +%Y%m%d)
LOG_DIR=".refs/autoloop"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/log-$DATE.jsonl"
BRANCH="autoloop/$DATE"

HIB_RE='^(prompt_eval_tps|decode_tps)$'
better() {
  local new="$1" old="$2"
  if [[ "$METRIC" =~ $HIB_RE ]]; then
    awk -v n="$new" -v o="$old" -v t="$THRESHOLD" 'BEGIN{exit !(n > o*(1+t))}'
  else
    awk -v n="$new" -v o="$old" -v t="$THRESHOLD" 'BEGIN{exit !(n < o*(1-t))}'
  fi
}

bench_median() {
  local vals=()
  for _ in $(seq 1 "$RUNS"); do
    local v
    v=$(./target/release/spindll bench "$MODEL" --json 2>/dev/null | jq -r ".$METRIC" || echo "")
    [[ -n $v && $v != null ]] && vals+=("$v")
  done
  printf '%s\n' "${vals[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'
}

build() {
  local feats="cli,http"
  [[ -n "$BACKEND" && "$BACKEND" != "none" ]] && feats="$feats,$BACKEND"
  cargo build --release --features "$feats"
}

git switch -C "$BRANCH"
build

BASELINE=$(bench_median)
echo "{\"event\":\"baseline\",\"metric\":\"$METRIC\",\"value\":$BASELINE,\"ts\":\"$(date -Iseconds)\"}" >> "$LOG"

START=$(date +%s)

while IFS= read -r row; do
  [[ -z "$row" ]] && continue
  NOW=$(date +%s)
  if (( NOW - START > WALL_CAP_SEC )); then
    echo "{\"event\":\"wall_cap\",\"ts\":\"$(date -Iseconds)\"}" >> "$LOG"
    break
  fi

  LABEL=$(echo "$row" | jq -r '.label // "trial"')
  PATCH_CMD=$(echo "$row" | jq -r '.patch_cmd // ""')
  ENV_KV=$(echo "$row" | jq -r '.env // {} | to_entries | map("\(.key)=\(.value)") | join(" ")')

  echo "==> trial: $LABEL"
  if [[ -n "$PATCH_CMD" ]]; then
    eval "$PATCH_CMD" || { echo "patch_cmd failed" >&2; continue; }
  fi
  if ! build 2>&1; then
    echo "{\"event\":\"build_fail\",\"label\":\"$LABEL\",\"ts\":\"$(date -Iseconds)\"}" >> "$LOG"
    git checkout -- .
    continue
  fi
  TRIAL=$(env $ENV_KV bash -c 'true' >/dev/null; bench_median)

  if better "$TRIAL" "$BASELINE"; then
    echo "{\"event\":\"keep\",\"label\":\"$LABEL\",\"baseline\":$BASELINE,\"trial\":$TRIAL,\"ts\":\"$(date -Iseconds)\"}" >> "$LOG"
    git add -A && git commit -m "autoloop: keep $LABEL ($METRIC $BASELINE -> $TRIAL)"
    BASELINE="$TRIAL"
  else
    echo "{\"event\":\"revert\",\"label\":\"$LABEL\",\"baseline\":$BASELINE,\"trial\":$TRIAL,\"ts\":\"$(date -Iseconds)\"}" >> "$LOG"
    git checkout -- .
  fi
done < "$GRID"

echo "==> autoloop done. final $METRIC = $BASELINE on branch $BRANCH"
echo "==> log: $LOG"
echo "NOTE: $BRANCH is local only. User must approve any push (see AGENTS.md push policy)."
