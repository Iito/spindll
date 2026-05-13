# Spindll — Agent Operating Manual

Single source of truth for both **Claude Code** and **Codex CLI** running against this repo. `CLAUDE.md` and `.codex/AGENTS.md` are symlinks to this file. Edit here, both tools see it.

## What this project is

Spindll = Rust-native LLM inference engine. GGUF (via llama.cpp) + MLX (via Swift FFI on Apple Silicon). Single binary that pulls models from Ollama / HuggingFace and serves streaming inference over gRPC + HTTP/SSE + OpenAI-compatible `/v1`.

Edition 2024. Active branch: usually `next` for in-flight work, `main` is stable. PRs land via review.

## Build / test / run

```bash
# Default dev build
cargo build --features cli,http
# Apple Silicon w/ MLX
cargo build --release --features cli,http,mlx
# Linux w/ CUDA
cargo build --release --features cli,http,cuda
# Linux w/ Vulkan
cargo build --release --features cli,http,vulkan
# Test (fast subset)
cargo test --features cli,http --lib
# Bench (real model required — see scripts/autoloop.sh for harness)
spindll bench <model-a> <model-b>
```

Feature flags: `cli`, `http`, `cuda`, `metal`, `vulkan`, `mlx`. `mlx` is **Apple-Silicon only** — gate with `#[cfg(all(target_arch="aarch64", target_os="macos", feature="mlx"))]`.

Existing test modules: `src/scheduler/budget.rs`, `src/model_store/registry.rs`. Add `#[cfg(test)]` blocks alongside any new module.

## Push / PR policy (HARD RULES)

- **Never `git push` or `gh pr create` without explicit user approval each time.** The user's auto-memory pins this rule.
- The ubuntu autoloop daemon honors this via a `~/.local/state/spindll-harness/push.allowed` lockfile that the user must touch (`touch ~/.local/state/spindll-harness/push.allowed`) before any push. The lockfile self-deletes after one consumed push.
- **Never `--no-verify`**, never amend pushed commits, never force-push to `main` or `next`.
- New work goes on a feature branch off `next`. Merge via PR.

## Punchlist + worklog (the core control point)

Two committed files drive `/implement`:

- `docs/PUNCHLIST.md` — ordered checkbox list of shippable units. `/implement` consumes top-most `[ ]`.
- `docs/WORKLOG.md` — append-only run log. One entry per `/implement` close: `## YYYY-MM-DD HH:MM  <agent>  <branch>  <ratchet?>  <review?>` + bullets.

Tag `mlx-validate-required` in a worklog entry whenever the change touches `mlx_bridge/`, `src/backend/mlx*.rs`, or any `feature = "mlx"`-gated path. The mac-side `scripts/mlx-validate.sh` picks these up.

## The spec-driven loop (`/implement`)

Mirrors PDF slide 15 (Lopopolo / shisad-dev `$implement v0.6.4`):

```
PUNCHLIST  →  CODER  →  TARGETED VALIDATION  →  REVIEW FANOUT  →  COLLATE-FIX-RESUBMIT  →  RELEASE CLOSE
```

Steps the slash command executes:
1. Read top unchecked punchlist item.
2. Write tests first (`#[cfg(test)]` next to the module under change).
3. Implement code.
4. **Ratchet gate** (`scripts/ratchet.sh`, target <60 s): `cargo check` + `cargo clippy -- -D warnings` + targeted `cargo test --lib`. Block until green.
5. **Review fanout** (`scripts/review-fanout.sh <base>`): R1 Claude Opus 4.6 + R2 Codex GPT-5.4 (Pareto). Promote to R1+R2+R3 (GPT-5.3-codex) for release-tagged PRs.
6. Remediate findings, re-run ratchet, re-run review fanout *only on non-green lanes*.
7. Flip punchlist item `[x]`, append worklog entry.
8. Stop. Do not push. User decides when to release.

## Karpathy autoloop (`/autoloop`)

Run on ubuntu via `systemd --user` timer. Form:

```
Define metric → Modify → Verify → Improved? Keep : Revert → Log → Repeat
```

Driver: `scripts/autoloop.sh <metric> <param-grid.json>`. Logs to `.refs/autoloop/log-<date>.jsonl`. Metrics: `prompt_eval_tps`, `decode_tps`, `peak_rss_mb`, `p50_ms`, `p95_ms`. Keep-threshold default = +2 % over baseline median of 3 runs.

## Ratchet definition

`scripts/ratchet.sh` runs:
- `cargo check --features cli,http`
- `cargo clippy --features cli,http -- -D warnings`
- `cargo test --features cli,http --lib budget registry` (the two existing test modules — fast)

Target ≤ 60 s on M-series mac, ≤ 90 s on the ubuntu box. If it grows, trim the test filter, do not raise the cap. Full `cargo test --release --features cli,http,mlx` runs only on mac at the MLX validator step.

## Reviewer fanout policy

- Pareto: 2 lanes (R1 Claude Opus 4.6, R2 Codex GPT-5.4).
- Release-tagged: 3 lanes (R1+R2+R3 Codex GPT-5.3-codex).
- Each reviewer reads the full diff vs base + a synopsis of the punchlist item. Output is markdown with severity (`crit`, `high`, `med`, `low`, `nit`).
- Findings collated into `.refs/review/COLLATED-<sha>.md`.
- A finding is **silenced** only with a one-line `# silenced: <why>` comment in the collated file. No silent drops.

## Cross-OS split

| Host | Role |
|------|------|
| **Ubuntu home server** | Primary 24/7 harness host. nightshift runs `/implement`, `/autoloop`, `/review` lanes. Skips MLX (Apple-only feature). |
| **Mac (Apple Silicon)** | MLX validator + light interactive work. Runs `scripts/mlx-validate.sh` on branches tagged `mlx-validate-required`. Never run a nightshift schedule concurrently with the ubuntu one on the same branch. |
| **Windows** | CI matrix only (`.github/workflows/ci.yml`). No local daemon. |

Sync between hosts: `git origin` only. No rsync, no live coupling.

### Scheduler: nightshift (not systemd)

We use [nightshift](https://github.com/marcus/nightshift) instead of hand-rolled systemd timers. Why: budget guard, multi-provider (Claude + Codex + Copilot), commit-only safety mode, single config file (`nightshift.yml`) committed to the repo so each new host bootstraps the same way.

Bootstrap on a fresh host (mac or ubuntu):
```
git clone https://github.com/Iito/spindll.git
cd spindll
brew install marcus/tap/nightshift     # or: go install github.com/marcus/nightshift/cmd/nightshift@latest
nightshift daemon start
```

`nightshift.yml` (in repo root) sets:
- `auto_create_pr: false` — agent stops at `git commit`, never pushes, never opens a PR. User does `gh pr create` manually.
- Schedules: `/implement` every 30 min on ubuntu, `/autoloop` nightly 02:00.
- Budget cap: 75 % of daily token allotment.

## Security baseline (lightweight)

- `.refs/` is a local sink (gitignored). Never commit logs.
- Don't add a dependency without checking it's >7 days old (PDF slide 24). Future: `cargo-deny` config.
- Don't bypass `cargo clippy -- -D warnings` in the ratchet.
- All GitHub Action versions in `.github/workflows/*.yml` should move to commit-SHA pins (TODO, separate PR).
- Never store provider API keys in repo. Keep them in `~/.config/` or env.

## Slash command index

| Command | What it does | Driver |
|---------|--------------|--------|
| `/plan` | Interactive Opus 4.7 planning. Updates `docs/PUNCHLIST.md`. | Claude Code |
| `/implement` | Spec-driven loop end-to-end on the next punchlist item. | `.claude/commands/implement.md` |
| `/review` | Multi-model review fanout on current diff. | `.claude/commands/review.md` → `scripts/review-fanout.sh` |
| `/autoloop` | Karpathy perf sweep. | `.claude/commands/autoloop.md` → `scripts/autoloop.sh` |
| `/status` | Post-sprint metrics from worklog. | `.claude/commands/status.md` |

Codex CLI mirrors at `.codex/prompts/` (when added).

## Pointers (PDF lineage)

- Lopopolo "harness engineering": <https://openai.com/index/harness-engineering/>
- shisad: <https://github.com/shisa-ai/shisad>
- StrongDM dark factory: <https://factory.strongdm.ai/>
- AgentsView: <https://github.com/wesm/agentsview>
- Workflow writeup: <https://agenticcodingweekly.com> (ACW #14)
