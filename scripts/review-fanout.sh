#!/usr/bin/env bash
# scripts/review-fanout.sh — multi-model review of HEAD vs <base-ref>.
# Pareto: R1 (Claude Opus 4.6) + R2 (Codex GPT-5.4).
# Release: also R3 (Codex GPT-5.3-codex). Set RELEASE=1.
set -euo pipefail

cd "$(dirname "$0")/.."

BASE="${1:-$(git merge-base HEAD origin/next 2>/dev/null || echo origin/main)}"
SHA=$(git rev-parse --short HEAD)
OUT=".refs/review"
mkdir -p "$OUT"

DIFF_FILE="$OUT/diff-$SHA.patch"
git diff "$BASE"...HEAD > "$DIFF_FILE"

if [[ ! -s "$DIFF_FILE" ]]; then
  echo "no diff vs $BASE — nothing to review"
  exit 0
fi

PROMPT="You are reviewing a Rust diff for spindll (LLM inference engine, GGUF + MLX). Output markdown with sections: Critical, High, Medium, Low, Nit. Each finding: one line location (file:line), one line problem, one line suggested fix. Be terse."

run_lane() {
  local label="$1" cmd="$2" out="$OUT/${label}-${SHA}.md"
  echo "==> lane $label -> $out"
  if eval "$cmd" < "$DIFF_FILE" > "$out" 2>&1; then
    echo "    ok"
  else
    echo "    LANE FAILED -- see $out" >&2
  fi
}

PIDS=()

# R1: Claude Opus 4.6
run_lane "r1-claude-opus-4-6" "claude --model claude-opus-4-6 -p '$PROMPT'" &
PIDS+=($!)

# R2: Codex GPT-5.4
run_lane "r2-codex-5-4" "codex exec --model gpt-5.4-xhigh '$PROMPT'" &
PIDS+=($!)

# R3: optional release lane
if [[ "${RELEASE:-0}" == "1" ]]; then
  run_lane "r3-codex-5-3" "codex exec --model gpt-5.3-codex '$PROMPT'" &
  PIDS+=($!)
fi

for pid in "${PIDS[@]}"; do wait "$pid" || true; done

# Collate.
COLLATED="$OUT/COLLATED-$SHA.md"
{
  echo "# Review COLLATED -- $SHA vs $BASE"
  echo "Generated: $(date -Iseconds)"
  echo
  for f in "$OUT"/r*-"$SHA".md; do
    [[ -f $f ]] || continue
    echo "---"
    echo "## $(basename "$f" .md)"
    echo
    cat "$f"
    echo
  done
} > "$COLLATED"

echo "==> collated -> $COLLATED"
