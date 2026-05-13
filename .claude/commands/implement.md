---
description: Spec-driven loop. Picks next docs/PUNCHLIST.md item, ships it through ratchet + multi-model review.
allowed-tools: Read, Edit, Write, Bash, Agent, TaskCreate, TaskUpdate, TaskList, Glob, Grep
---

You are running the **`/implement`** loop on spindll. Source: `AGENTS.md` "spec-driven loop". Mirrors PDF slide 15.

## Steps

1. **Pick item.** Read `docs/PUNCHLIST.md`. Find first `- [ ]`. If none, stop and report. Note the item text — that is the spec.
2. **Plan tests.** Identify the module(s) the change touches. Existing test modules: `src/scheduler/budget.rs`, `src/model_store/registry.rs`. Add a failing `#[cfg(test)]` block that captures the spec's intent.
3. **Run ratchet (expect red).** `bash scripts/ratchet.sh`. Confirm the new test fails and nothing else regressed. If unrelated tests fail, stop and report — do not proceed on a red baseline.
4. **Implement.** Edit code until the new test passes. Stay narrow — no drive-by refactors, no new dependencies (PDF: harness evolves with project, not plugins).
5. **Run ratchet (expect green).** `bash scripts/ratchet.sh`. Block until green. If the run takes >60 s on mac / >90 s on ubuntu, trim the `--lib` filter in the script — do not raise the time cap.
6. **MLX flag.** If any edited path matches `mlx_bridge/`, `src/backend/mlx*.rs`, or any code gated on `feature = "mlx"`, set `MLX_REQUIRED=1` for the worklog entry.
7. **Review fanout.** `bash scripts/review-fanout.sh $(git merge-base HEAD origin/next)`. Wait for both lanes. Read `.refs/review/COLLATED-*.md`.
8. **Remediate.** For each `crit` / `high` finding, fix and re-run ratchet. Re-run review fanout *only on the lanes that flagged*, not all. Loop until no `crit`/`high` open.
9. **Close.** Flip the punchlist item to `- [x]`. Append a `docs/WORKLOG.md` entry:

   ```
   ## YYYY-MM-DD HH:MM  claude  <branch>  ratchet=green  review=2/2-clean
   - <one-line summary>
   - files: <list>
   - tag: mlx-validate-required   # only if MLX_REQUIRED=1
   ```

10. **Commit, do not push.** `git add` + `git commit`. Stop. The user pushes manually (memory rule + nightshift `auto_create_pr: false`).

## Hard rules

- No `--no-verify`, no amending pushed commits, no force-push.
- No new dependency without a >7-day age check — if uncertain, ask the user.
- Stay on the active feature branch. If on `main` or `next`, create a feature branch first (`git checkout -b feat/<short-slug>` from `next`).
- One punchlist item per run. Do not chain.
