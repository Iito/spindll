---
description: Interactive planning. Refines docs/PUNCHLIST.md with the next batch of work.
allowed-tools: Read, Write, Edit, Glob, Grep, Agent, AskUserQuestion
argument-hint: [scope]
---

Interactive planning session. Goal: leave `docs/PUNCHLIST.md` in a state where `/implement` can run unattended for hours.

Steps:

1. Read current `docs/PUNCHLIST.md`, `docs/WORKLOG.md`, `CHANGELOG.md`, recent `git log` (last 30 commits).
2. Identify gaps. Use `Agent(subagent_type=Explore)` to scout 1–2 areas if scope unclear.
3. Propose 5–10 new punchlist items, each:
   - Self-contained (one PR's worth).
   - Has explicit acceptance criteria the test-first step in `/implement` can target.
   - Tagged with backend if relevant: `[gguf]`, `[mlx]`, `[cuda]`, `[http]`, `[grpc]`, `[cli]`, `[bench]`, `[meta]`.
4. Use `AskUserQuestion` for any genuine ambiguity (don't churn — max 2 questions).
5. Append the agreed items to `docs/PUNCHLIST.md` under a new `## Sprint <date>` heading.
6. Stop. Do not start `/implement`.
