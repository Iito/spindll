#!/usr/bin/env bash
# scripts/ratchet.sh — fast pre-flight green gate for /implement.
# Target: <60s on M-series mac, <90s on ubuntu. If it grows, trim the test filter.
set -euo pipefail

cd "$(dirname "$0")/.."

FEATS="${RATCHET_FEATS:-cli,http}"
START=$(date +%s)

echo "==> cargo check --features $FEATS"
cargo check --features "$FEATS"

# Clippy is currently warn-only until the pre-existing 13 lints on `main` are
# resolved (tracked in docs/PUNCHLIST.md as the "[meta] clippy clean baseline" item).
# Once that lands, change `|| true` -> `|| { echo "clippy red"; exit 1; }` and
# upgrade to `-D warnings`.
echo "==> cargo clippy --features $FEATS"
cargo clippy --features "$FEATS" || echo "WARN: clippy red — see docs/PUNCHLIST.md '[meta] clippy clean baseline'" >&2

# Fast unit subset: all lib tests (currently only `scheduler::budget` and
# `model_store::registry` have #[cfg(test)] blocks, so this stays cheap).
# --bins as well as --lib: the CLI's own #[cfg(test)] blocks live in the
# `spindll` bin target, which `--lib` alone silently skips.
echo "==> cargo test --features $FEATS --lib --bins"
cargo test --features "$FEATS" --lib --bins

ELAPSED=$(( $(date +%s) - START ))
echo "==> ratchet green in ${ELAPSED}s"

CAP="${RATCHET_CAP:-90}"
if (( ELAPSED > CAP )); then
  echo "WARN: ratchet exceeded ${CAP}s cap. Trim the --lib filter, do not raise the cap." >&2
fi
