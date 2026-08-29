# Spindll — Agent Operating Manual

Single source of truth for **Claude Code** and **Codex CLI**. `CLAUDE.md` and
`.codex/AGENTS.md` are symlinks to this file.

Kept to one page on purpose. Detail lives in `.claude/skills/*/SKILL.md` —
plain markdown, readable by any agent, loaded on demand:

| Need | Read |
|---|---|
| Cutting a version, tagging, release notes, main/next sync | `.claude/skills/release-flow/SKILL.md` |
| MLX gating, when validation is required, Apple-Silicon build quirks | `.claude/skills/mlx-validation/SKILL.md` |
| Punchlist/worklog conventions, review fanout, autoloop, host split | `.claude/skills/harness-ops/SKILL.md` |
| What a review finding must look like | `REVIEW.md` |

## What this is

Rust-native LLM inference engine. GGUF via llama.cpp, MLX via a Swift FFI bridge
on Apple Silicon. Single binary; pulls models from Ollama and HuggingFace; serves
streaming inference over gRPC, HTTP/SSE, OpenAI-compatible `/v1`, Anthropic
Messages (`/v1/messages`), and a stateless subset of OpenAI Responses
(`/v1/responses`). Edition 2024.

## Build and test

```bash
cargo build --features cli,http                      # dev
cargo build --release --features cli,http,mlx        # Apple Silicon
cargo build --release --features cli,http,cuda       # Linux + CUDA
bash scripts/ratchet.sh                              # the green gate, <60s
```

Feature flags: `cli`, `http`, `cuda`, `metal`, `vulkan`, `mlx`, `vision`, `rpc`.
MLX is Apple-Silicon only — gate it
`#[cfg(all(target_arch = "aarch64", target_os = "macos", feature = "mlx"))]`.

Add `#[cfg(test)]` blocks alongside any module you change.

## Hard rules

These are enforced by `.claude/hooks/guard-git.py`, not by your memory. The hook
denies the call and tells you why; `scripts/hook-tests.sh` proves it still works.

- **Never push, open a PR, or cut a release without the user's approval, each
  time.** Approval is a lockfile the user touches by hand
  (`~/.local/state/spindll-harness/push.allowed`); one touch authorises one
  publish and the hook consumes it.
- **Never `--no-verify`.** The pre-commit hook enforces this repo's commit
  identity.
- **Never amend a pushed commit. Never force-push main or next.**
- **Never commit `docs/PUNCHLIST.md`, `docs/WORKLOG.md`, or `.refs/`** — they
  are per-host local files.
- **No new dependency without a >7-day age check.** If unsure, ask.

## Branch flow

- **Quickfix or hotfix → branch off `main`, PR to `main`.** It reaches `next`
  later via the user's main→next carry merge. Do not open a second PR for it.
- **Feature work → branch off `next`, PR to `next`.** `next` merges to `main`
  for releases.
- **Release docs follow the release**: changelog and version commits for a tag
  cut from `main` are main-line work, done in a detached sibling worktree.
- **If the right base is ambiguous, ask "main or next?"** before branching.

## The loop (`/implement`)

```
PUNCHLIST → tests first → code → RATCHET → REVIEW FANOUT → remediate → close
```

1. Take the top unchecked `docs/PUNCHLIST.md` item. Its acceptance criteria are
   the spec.
2. Write the failing test first.
3. Implement. Stay narrow — no drive-by refactors.
4. `bash scripts/ratchet.sh`. Block until green.
5. `bash scripts/review-fanout.sh <base>`. Contract is `REVIEW.md`.
6. Fix every `crit` and `high`. Re-run the ratchet, then re-run **only the lanes
   that flagged**.
7. Flip the checkbox, append a worklog entry, commit code only.
8. **Stop.** The user decides when to publish.

One item per run. Do not chain.

## Slash commands

| Command | Does |
|---|---|
| `/plan` | Interactive planning. Updates `docs/PUNCHLIST.md`. |
| `/implement` | The loop above, on the next punchlist item. |
| `/review` | Review fanout on the current diff. |
| `/autoloop` | Perf sweep across a parameter grid. |
| `/maintain` | Checks perf against `bands.yaml`; responds at the tier the numbers earned. |
| `/status` | Metrics scraped from the worklog. |

## Security baseline

`.refs/` is a local sink — never commit logs. Never put provider API keys in the
repo; they live in `~/.config/` or the environment. GitHub Action versions should
move to commit-SHA pins (open TODO).

## Lineage

Harness engineering: <https://openai.com/index/harness-engineering/> ·
AI-native SDLC: <https://claude.com/blog/the-ai-native-sdlc-playbook> ·
shisad: <https://github.com/shisa-ai/shisad>
