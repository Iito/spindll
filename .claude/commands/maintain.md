---
description: Maintain stage. Evaluates perf against bands.yaml and responds at the tier the numbers earned — nothing more.
allowed-tools: Read, Edit, Write, Bash, Glob, Grep
---

You are running the **`/maintain`** stage on spindll. Bands are defined in
`bands.yaml`; the detector is `scripts/bands-check.py`. Background:
`.claude/skills/harness-ops/SKILL.md`.

**The tier decides your autonomy. You do not.** Never talk yourself up a tier
because a problem looks interesting, and never act at a tier the numbers did not
reach.

## Steps

1. **Measure.** If given a model, `python3 scripts/bands-check.py bench <model>`
   records every metric and evaluates. Otherwise
   `python3 scripts/bands-check.py check --json` judges the samples already on
   disk.

2. **Read the verdict.** `status: baselining` or `no-samples` means there is no
   baseline yet. Say so and stop — do not invent one.

3. **Act at the tier, and only at the tier.**

   - **tier 0** — report "inside bands" in one line. Stop.

   - **tier 1 (log)** — one line naming the metric, the move, and the sigma.
     Change nothing. One sample outside one sigma is weather.

   - **tier 2 (diagnose, read-only)** — investigate and write findings to
     `.refs/bands/findings-<YYYY-MM-DD>.md`: what moved, by how much, the
     commits in the window, and your best hypothesis with the evidence for it.
     **Edit no source file.** If the cause is obvious and the fix is one line,
     say so in the findings and still change nothing.

   - **tier 3 (propose)** — do the tier-2 diagnosis, then add a new item at the
     **top** of `docs/PUNCHLIST.md` with concrete acceptance criteria, tagged
     `[perf]`, naming the metric and the number to recover. Then stop. The fix
     ships through `/implement` like any other item: ratchet, review fanout, and
     the user's approval to publish. Never commit a fix here.

4. **Report** in one line: metric, tier, what you did, where the artefact is.

## Hard rules

- Never edit `bands.yaml` to make a breach go away. If a band is genuinely wrong
  — too tight for a noisy metric, wrong direction — say so and propose the
  change as its own punchlist item, with the sample data behind it.
- Never publish anything. The `guard-git.py` hook will stop you, but do not make
  it work for a living.
- A tier-3 proposal is a punchlist item, never a branch and never a commit.
