---
description: Spec-driven loop. Picks next docs/PUNCHLIST.md item, ships it through ratchet + multi-model review.
allowed-tools: Read, Edit, Write, Bash, Agent, TaskCreate, TaskUpdate, TaskList, Glob, Grep
---

You are running the **`/implement`** loop on spindll. The operating manual is
`AGENTS.md`; the review contract is `REVIEW.md`; harness conventions are in
`.claude/skills/harness-ops/SKILL.md`.

## Steps

1. **Pick the item.** Read `docs/PUNCHLIST.md`, take the first `- [ ]`. If there
   is none, stop and say so. **The item's acceptance criteria are the spec** —
   you are done when they are met, not when the code looks finished.

2. **Write the failing test first.** Add a `#[cfg(test)]` block next to the
   module you are about to change, capturing the acceptance criteria.

3. **Ratchet, expecting red.** `bash scripts/ratchet.sh`. Confirm your new test
   is the thing failing. **If anything unrelated is red, stop and report** — do
   not build on a red baseline.

4. **Implement.** Stay narrow: no drive-by refactors, no new dependencies
   without a >7-day age check.

5. **Ratchet, expecting green.** `bash scripts/ratchet.sh`. Block until green.
   Clippy is strict (`-D warnings`) and the hook tests run here too. If the
   ratchet exceeds 60s on Apple Silicon or 90s on the Linux host, trim the
   `--lib` filter — never raise the cap.

6. **MLX flag.** If you touched `mlx_bridge/`, `src/backend/mlx*.rs`, or any
   `feature = "mlx"` path, the worklog entry gets `tag: mlx-validate-required`.
   See `.claude/skills/mlx-validation/SKILL.md`.

7. **Review fanout.**
   `bash scripts/review-fanout.sh "$(git merge-base HEAD origin/next)"`.
   The script feeds `REVIEW.md` plus the punchlist item to each lane. Read
   `.refs/review/COLLATED-*.md`.

8. **Remediate.** Fix every `crit` and `high`. Re-run the ratchet, then re-run
   **only the lanes that flagged** — not all of them. Loop until no `crit` or
   `high` is open. Dropping a finding needs a `# silenced: <why>` line beside it
   in the collated file; never delete one silently.

9. **Close.** Flip the item to `- [x]` and append to `docs/WORKLOG.md`:

   ```
   ## YYYY-MM-DD HH:MM  claude  <branch>  ratchet=green  review=2/2-clean
   - <one-line summary>
   - files: <list>
   - tag: mlx-validate-required   # only when step 6 applied
   ```

   On Apple Silicon the R2 lane has no credentials by design — close honestly as
   `review=1/2-clean` with a note. Do not debug its auth.

10. **Commit code only. Do not publish.** `docs/PUNCHLIST.md` and
    `docs/WORKLOG.md` are per-host local files; the flip and the worklog entry
    stay on disk and never enter the commit. Then **stop** — the user decides
    when anything gets pushed.

11. **Feed the loop back.** If a review finding is one you have now seen twice,
    it is a missing rule, not a review problem. Mechanical → add it to
    `.claude/hooks/guard-git.py` with a test in `scripts/hook-tests.sh`.
    Judgment → add a line to `AGENTS.md`. Say which you did.

## Hard rules

The `guard-git.py` hook enforces the publish, `--no-verify`, amend, force-push
and local-file rules — it will deny the call and tell you why. Beyond those:

- One punchlist item per run. Do not chain.
- Stay on a feature branch. If you are on `main` or `next`, branch first. Base a
  quickfix on `main` and feature work on `next`; if the right base is ambiguous,
  ask "main or next?" before branching.
