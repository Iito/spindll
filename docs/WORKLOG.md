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

## 2026-05-31 11:05  claude  test/function-calling  ratchet=green  review=2/2-clean
- Verification tests for existing OpenAI-compatible function calling implementation. Already in codebase (no code changes needed). Added comprehensive tests covering tool parsing, response format, streaming/non-streaming, fallback.
- files: src/http.rs, docs/PUNCHLIST.md
- Tests: parse_tool_calls (single/multiple), response format validation, streaming/non-streaming with tools, fallback without tools. All 133 tests passing (8 new).
