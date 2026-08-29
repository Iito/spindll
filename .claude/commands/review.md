---
description: Multi-model review fanout on the current diff vs origin/next. Pareto = 2 lanes, release = 3 lanes.
allowed-tools: Bash, Read, Agent
argument-hint: [base-ref]
---

Run `bash scripts/review-fanout.sh ${1:-$(git merge-base HEAD origin/next)}`.

The contract every lane reviews against is **`REVIEW.md`** — severity ladder,
output shape, the five-nit cap, what does and does not earn a finding. The
script feeds that file plus the punchlist item the change claims to implement.
If a review comes back badly shaped, fix `REVIEW.md`, not the script.

Lanes:
- **R1** — `claude --model claude-opus-4-6`
- **R2** — `codex exec --model gpt-5.4-xhigh`
- **R3** — `codex exec --model gpt-5.3-codex`, only when `RELEASE=1`

Each lane writes `.refs/review/r<n>-<sha>.md`; the script collates into
`.refs/review/COLLATED-<sha>.md`.

On Apple Silicon the R2 lane has no OpenAI credentials by design. Report
`1/2-clean` honestly; do not retry it or suggest logging in. Run R2 on the Linux
harness host before a release.

After the script finishes, read the collated file and surface:

- Counts by severity: `crit` / `high` / `med` / `low` / `nit`.
- The **disjoint sets** — findings only one lane caught. That overlap gap is the
  whole reason for running more than one lane; it is the most informative part
  of the output.
- Any finding that also appeared in an earlier review of this branch. A repeat
  is a missing rule: mechanical ones belong in `.claude/hooks/guard-git.py` with
  a test in `scripts/hook-tests.sh`, judgment ones in `AGENTS.md`.

Do not auto-remediate. The user, or a later `/implement` run, decides what to
fix. Dropping a finding requires a `# silenced: <why>` line beside it in the
collated file — never a silent deletion.
