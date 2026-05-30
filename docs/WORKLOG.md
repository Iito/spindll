# Worklog

Append-only. One entry per `/implement` close. Format:

```
## YYYY-MM-DD HH:MM  <agent>  <branch>  ratchet=green|red  review=<lanes-clean>/<lanes-total>-clean
- <one-line summary>
- files: <list>
- tag: mlx-validate-required   # optional, only when MLX paths touched
- tag: mlx-validated            # appended by scripts/mlx-validate.sh on the mac
```

`/status` scrapes this file. Do not mutate prior entries — append only.

---

## 2026-04-30  bootstrap  feat/agent-harness  ratchet=skipped  review=skipped
- Harness scaffold installed: AGENTS.md (+ CLAUDE.md, .codex/AGENTS.md symlinks), .claude/{settings.json, commands/*.md}, docs/{PUNCHLIST,WORKLOG}.md, scripts/{ratchet,review-fanout,autoloop,mlx-validate}.sh, nightshift.yml.
- Branch base: origin/main.
- Next: smoke-test `/implement` against the seeded punchlist.
