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

## 2026-08-20 21:30  claude  feat/bump-mlx-swift-lm  ratchet=green  review=skipped
- mlx-validate-required
- Bumped mlx_bridge's mlx-swift-lm pin from main@0767814 (2026-06-17) to main@d242429 (2026-07-15) — the newest revision buildable with installed toolchains: #369 onward calls mlx-swift 0.31.5+ APIs (swift-tools 6.3; local ceiling is Swift 6.2.4 via Xcode 26.3), and lm main @ 2026-08-11+ additionally declares tools 6.2.
- Delta keeps: tool round-trip ordering fix (#409), ChatSession cancellation fixes (#389/#413/#423), safetensors-index fix (#408), Qwen3.5 windowed prefill + M-RoPE drift fix (#399), kvScheme KV compression hook (#230), seed in GenerateParameters (#377).
- No bridge source changes needed — FFI-relevant API surface unchanged.
- Validated: Swift release build clean; cargo build --features cli,http,mlx,vision green (metallib compiled); clippy -D warnings clean; 193 lib tests pass.
- Follow-up: installing Xcode 26.6 (Swift 6.3.3) locally + pinning an Xcode in ci-macos.yml unlocks latest upstream main — #465 guided-generation fix, #475 prompt-cache persistence, #521 thinking-budget enforcement, #442/#467/#468 Qwen3.5 decode perf (all relevant to issue #75).
