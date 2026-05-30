---
description: Multi-model review fanout on the current diff vs origin/next. Pareto = 2 lanes, release = 3 lanes.
allowed-tools: Bash, Read, Agent
argument-hint: [base-ref]
---

Run `bash scripts/review-fanout.sh ${1:-$(git merge-base HEAD origin/next)}`.

The script spawns:
- **R1**: `claude --model claude-opus-4-6` reviewing the diff.
- **R2**: `codex exec --model gpt-5.4-xhigh` reviewing the diff.
- **R3** (only if `RELEASE=1` env): `codex exec --model gpt-5.3-codex`.

Each lane writes a markdown report to `.refs/review/r<n>-<sha>.md` with severity-tagged findings. The script collates into `.refs/review/COLLATED-<sha>.md`.

After the script finishes, read the collated file and surface:
- Count of `crit` / `high` / `med` / `low` / `nit` findings.
- The **disjoint sets** — issues only one reviewer caught (PDF slide 16: each lane catches different classes).
- The diff-vs-baseline if a previous review exists for the same branch.

Do not auto-remediate. The user (or a subsequent `/implement` invocation) decides what to fix.
