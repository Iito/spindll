#!/usr/bin/env bash
# scripts/review-fanout.sh — multi-model review of HEAD vs <base-ref>.
#
# The review contract lives in REVIEW.md, not in this script. Reviewers get
# that file plus the punchlist item the change claims to implement, so they can
# judge the diff against its stated acceptance criteria rather than against
# their own taste.
#
# Pareto (default): R1 Claude Opus 4.6 + R2 Codex GPT-5.4.
# Release: adds R3 Codex GPT-5.3-codex. Set RELEASE=1.
set -euo pipefail

cd "$(dirname "$0")/.."

BASE="${1:-$(git merge-base HEAD origin/next 2>/dev/null || echo origin/main)}"
SHA=$(git rev-parse --short HEAD)
OUT=".refs/review"
mkdir -p "$OUT"

DIFF_FILE="$OUT/diff-$SHA.patch"
git diff "$BASE"...HEAD >"$DIFF_FILE"

if [[ ! -s "$DIFF_FILE" ]]; then
  echo "no diff vs $BASE — nothing to review"
  exit 0
fi

if [[ ! -f REVIEW.md ]]; then
  echo "REVIEW.md is missing — that file is the review contract, refusing to run blind" >&2
  exit 1
fi

# The item under review: the top unchecked punchlist entry, or whatever the
# caller pins via PUNCHLIST_ITEM. Absent on a fresh clone; the review still runs.
ITEM="${PUNCHLIST_ITEM:-$(grep -m1 '^- \[ \]' docs/PUNCHLIST.md 2>/dev/null || true)}"
[[ -z "$ITEM" ]] && ITEM="(no punchlist item pinned — judge the diff on its own terms)"

PROMPT_FILE="$OUT/prompt-$SHA.md"
{
  cat REVIEW.md
  echo
  echo "---"
  echo
  echo "## The item this change claims to implement"
  echo
  echo "$ITEM"
  echo
  echo "---"
  echo
  echo "The diff vs \`$BASE\` follows on stdin. Review it against the contract above."
} >"$PROMPT_FILE"

run_lane() {
  # One `local` per line: macOS bash 3.2 expands ${label} in a combined
  # declaration before the assignment lands (breaks under `set -u`).
  local label="$1"
  local cmd="$2"
  local out="$OUT/${label}-${SHA}.md"
  echo "==> lane $label -> $out"
  if eval "$cmd" <"$DIFF_FILE" >"$out" 2>&1; then
    echo "    ok"
  else
    echo "    LANE FAILED -- see $out" >&2
  fi
}

PROMPT="$(cat "$PROMPT_FILE")"
PIDS=()

run_lane "r1-claude-opus-4-6" "claude --model claude-opus-4-6 -p \"\$PROMPT\"" &
PIDS+=($!)

run_lane "r2-codex-5-4" "codex exec --model gpt-5.4-xhigh \"\$PROMPT\"" &
PIDS+=($!)

if [[ "${RELEASE:-0}" == "1" ]]; then
  run_lane "r3-codex-5-3" "codex exec --model gpt-5.3-codex \"\$PROMPT\"" &
  PIDS+=($!)
fi

for pid in "${PIDS[@]}"; do wait "$pid" || true; done

COLLATED="$OUT/COLLATED-$SHA.md"
{
  echo "# Review COLLATED -- $SHA vs $BASE"
  echo "Generated: $(date -Iseconds)"
  echo
  echo "Item: $ITEM"
  echo
  echo "Contract: REVIEW.md. A finding is dropped only with a \`# silenced: <why>\` line beside it."
  echo
  for f in "$OUT"/r*-"$SHA".md; do
    [[ -f $f ]] || continue
    echo "---"
    echo "## $(basename "$f" .md)"
    echo
    cat "$f"
    echo
  done
} >"$COLLATED"

echo "==> collated -> $COLLATED"
