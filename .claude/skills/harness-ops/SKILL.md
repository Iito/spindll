---
name: harness-ops
description: How the spindll agent harness itself works — the punchlist and worklog per-host file convention, seeding them on a fresh clone, the review fanout and collation layout, the autoloop and its control bands, and the cross-OS host split. Use when running or repairing /implement, /review, /autoloop, or /status, or when bootstrapping the harness on a new machine.
---

# Harness operations

## The punchlist and worklog are per-host local files

`docs/PUNCHLIST.md` and `docs/WORKLOG.md` have been **untracked, per-host**
since 2026-08-21. `.git/info/exclude` hides them, so they never show in
`git status`.

- **Never `git add` or commit them.** `git add` refuses; `git add -f` is blocked
  by the `guard-git.py` hook. An `/implement` close commits code only — the
  checkbox flip and worklog append stay on disk.
- **History keeps them only up to `d0f6d7e`.** A fresh clone gets neither. Seed
  from another host, or `git show d0f6d7e:docs/PUNCHLIST.md`, then add both
  paths to that clone's `.git/info/exclude`.
- **Checking out a pre-removal commit silently overwrites your local copies** —
  git treats ignored files as expendable. Back them up before history
  archaeology.

Why: the seed stays visible in early history, but no cross-host churn commit
from every `/implement` run.

### Formats

Punchlist: an ordered checkbox list of shippable units, newest sprint at the
bottom, backend tagged in brackets. Each item carries its own acceptance
criteria — that is the spec `/implement` and the review lanes work against.

Worklog: append-only, one entry per `/implement` close. Never mutate a prior
entry.

```
## YYYY-MM-DD HH:MM  <agent>  <branch>  ratchet=green  review=2/2-clean
- <one-line summary>
- files: <list>
- tag: mlx-validate-required   # only when MLX paths were touched
```

## Review fanout

`scripts/review-fanout.sh <base>` runs the lanes defined in `REVIEW.md`, which
is also the prompt — there is no second copy in the script. Reviewers get the
diff plus the punchlist item the change claims to implement.

Output lands in `.refs/review/`: one file per lane, plus
`COLLATED-<sha>.md`. A finding is dropped only with a one-line
`# silenced: <why>` beside it in the collated file.

`.refs/` is a local sink and is gitignored. Never commit logs.

## Ratchet

`scripts/ratchet.sh` is the fast green gate: `cargo check`, `cargo clippy -D
warnings`, `cargo test --lib`, and the hook tests. Target ≤60s on Apple
Silicon, ≤90s on the Linux host. **If it grows, trim the test filter — never
raise the cap.** A slow gate is a gate people skip.

Clippy went strict on 2026-08-29. It had been warn-only against a
"13 pre-existing lints" baseline that was cleaned up without anyone updating the
script, so every `ratchet=green` before that date meant "clippy unknown".

## Autoloop and control bands

`scripts/autoloop.sh <metric> <param-grid.json>` sweeps a parameter grid,
keeps winners, logs every trial to `.refs/autoloop/log-<date>.jsonl`.

Metrics: `prompt_eval_tps`, `decode_tps`, `peak_rss_mb`, `p50_ms`, `p95_ms`.
Keep-threshold defaults to +2% over the baseline median of 3 runs.

`bands.yaml` defines the control bands for the same metrics and
`scripts/bands-check.py` evaluates a metrics log against them. Tiers:

- **1σ** — log only.
- **2σ** — diagnose read-only; write findings, change nothing.
- **3σ** — propose a fix through the normal review gates.

Detection stays deterministic. An agent is invoked only after a breach is
confirmed by the numbers, never to decide whether one happened.

## Host split

| Host | Role |
|---|---|
| Linux server | Primary 24/7 harness host. Runs `/implement`, `/autoloop`, and the review lanes. Skips MLX. |
| Apple Silicon | MLX validator plus light interactive work. Runs `scripts/mlx-validate.sh` on branches tagged `mlx-validate-required`. |
| Windows | CI matrix only. No local daemon. |

Never run a scheduled harness on the same branch from two hosts at once. Hosts
sync through `git origin` only — no rsync, no live coupling.

## Scheduling

There is no scheduler installed today. A committed `nightshift.yml` was
described in the harness docs for months but never existed in the repo; the
claim was removed on 2026-08-29 rather than left as documentation of a system
nobody was running. If you adopt one, it must keep `auto_create_pr: false` —
the agent stops at `git commit`.
