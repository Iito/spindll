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

echo "==> cargo build --release --features cli,http,mlx,vision"
cargo build --release --features cli,http,mlx,vision

echo "==> cargo test --features cli,http,mlx,vision --lib"
cargo test --features cli,http,mlx,vision --lib

TS=$(date "+%Y-%m-%d %H:%M")
{
  echo
  echo "## $TS  mlx-validate  $BRANCH  ratchet=green  review=skipped"
  echo "- mlx build + lib tests pass on $(uname -srm)"
  echo "- tag: mlx-validated"
} >> docs/WORKLOG.md
# WORKLOG.md is a local per-host file (untracked + excluded) — append only, never commit.

echo "==> done. Worklog updated locally; nothing to push for the log itself."
