---
description: Post-sprint metrics scraped from docs/WORKLOG.md.
allowed-tools: Read, Bash
argument-hint: [days=7]
---

Read `docs/WORKLOG.md` and `git log` for the last `${1:-7}` days. Print this table. Nothing writes `session=` durations or `release-close` tags to the worklog yet, so print `n/a` for the rows built on them (session time, the wall/session ratio, concurrent agents and ReleaseClose runs) rather than estimating:

| Metric | Value |
|--------|-------|
| Wall-clock hours | … |
| Agent session time | … (sum of `session=` durations) |
| Wall/session ratio | … (lower = more concurrency) |
| Concurrent agents (avg) | … |
| Commits | … |
| Insertions | … (`git log --shortstat`) |
| Deletions | … |
| Net lines | … |
| Review green rate | … (entries with `review=N/N-clean` ÷ total) |
| ReleaseClose runs | … (count of `release-close` tags) |

Then highlight:
- The single noisiest reviewer lane (highest finding rate).
- The cheapest punchlist item (smallest insertion count, still green).
- Any worklog entry tagged `mlx-validate-required` that has no matching `mlx-validated` follow-up — that is a stuck branch. Worklogs are per host and `scripts/mlx-validate.sh` writes to the Apple Silicon host's, so on any other host say that this check can't see validations.
