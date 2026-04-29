---
description: Post-sprint metrics scraped from docs/WORKLOG.md.
allowed-tools: Read, Bash
argument-hint: [days=7]
---

Read `docs/WORKLOG.md` and `git log` for the last `${1:-7}` days. Print a table matching PDF slide 13:

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
- Any worklog entry tagged `mlx-validate-required` that has no matching `mlx-validated` follow-up — that is a stuck branch.
