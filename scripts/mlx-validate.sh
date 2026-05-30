#!/usr/bin/env bash
# scripts/mlx-validate.sh — mac-side MLX validator.
# Picks up branches whose worklog tagged "mlx-validate-required",
# runs MLX-only build + tests, appends "mlx-validated" to the worklog.
# Usage: scripts/mlx-validate.sh <branch>
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "$(uname)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: mlx-validate runs only on Apple Silicon mac." >&2
  exit 2
fi

BRANCH="${1:?branch name required}"

git fetch origin "$BRANCH"
git switch "$BRANCH" 2>/dev/null || git switch -c "$BRANCH" "origin/$BRANCH"
git pull --ff-only origin "$BRANCH"

echo "==> cargo build --release --features cli,http,mlx"
cargo build --release --features cli,http,mlx

echo "==> cargo test --features cli,http,mlx --lib"
cargo test --features cli,http,mlx --lib

TS=$(date "+%Y-%m-%d %H:%M")
{
  echo
  echo "## $TS  mlx-validate  $BRANCH  ratchet=green  review=skipped"
  echo "- mlx build + lib tests pass on $(uname -srm)"
  echo "- tag: mlx-validated"
} >> docs/WORKLOG.md

git add docs/WORKLOG.md
git commit -m "mlx-validate: $BRANCH green on Apple Silicon"

echo "==> done. Push when ready (per AGENTS.md push policy)."
