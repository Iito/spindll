#!/usr/bin/env bash
# scripts/ratchet.sh — fast pre-flight green gate for /implement.
# Target: <60s on M-series mac, <90s on ubuntu. If it grows, trim the test
# filter. Never raise the cap: a slow gate is a gate people skip.
set -euo pipefail

cd "$(dirname "$0")/.."

FEATS="${RATCHET_FEATS:-cli,http}"
START=$(date +%s)

echo "==> cargo check --features $FEATS"
cargo check --features "$FEATS"

# Strict since 2026-08-29. This was warn-only for months against a "13
# pre-existing lints" baseline that had since been cleaned up, so every
# ratchet=green in the worklog before that date meant "clippy unknown".
#
# --all-targets so #[cfg(test)] code is linted too, not just the build; --locked
# so a stale Cargo.lock cannot pass the gate. Deliberately NOT --all-features:
# cuda/vulkan/metal are mutually exclusive GPU backends, and enabling cuda makes
# llama-cpp-sys-2's build script fail without a CUDA toolchain installed.
echo "==> cargo clippy --features $FEATS --all-targets --locked -- -D warnings"
cargo clippy --features "$FEATS" --all-targets --locked -- -D warnings

# --bins as well as --lib: the CLI's own #[cfg(test)] blocks live in the
# `spindll` bin target, which `--lib` alone silently skips.
echo "==> cargo test --features $FEATS --lib --bins"
cargo test --features "$FEATS" --lib --bins

# The hooks are the enforcement layer for the AGENTS.md hard rules. They are
# shell-speed to test, so there is no excuse for them rotting silently.
echo "==> hook tests"
bash scripts/hook-tests.sh

# And the docs have to keep describing the harness that actually exists. This
# catches the class of drift that let clippy sit warn-only for months.
echo "==> harness lint"
python3 scripts/harness-lint.py

ELAPSED=$(($(date +%s) - START))
echo "==> ratchet green in ${ELAPSED}s"

CAP="${RATCHET_CAP:-90}"
if ((ELAPSED > CAP)); then
  echo "WARN: ratchet exceeded ${CAP}s cap. Trim the test filter, do not raise the cap." >&2
fi
