# Review policy

The contract every review lane runs against. `scripts/review-fanout.sh` feeds
this file to each reviewer verbatim, so editing it changes the review — there is
no second copy of the prompt hidden in the script.

## What you are reviewing

A Rust diff for **spindll**, a Rust-native LLM inference engine: GGUF via
llama.cpp, MLX via a Swift FFI bridge on Apple Silicon. It serves streaming
inference over gRPC, HTTP/SSE, an OpenAI-compatible `/v1`, the Anthropic
Messages dialect (`/v1/messages`), and a stateless subset of the OpenAI
Responses API (`/v1/responses`).

Edition 2024. Feature flags: `cli`, `http`, `cuda`, `metal`, `vulkan`, `mlx`,
`vision`, `rpc`. Anything MLX is Apple-Silicon only and must be gated
`#[cfg(all(target_arch = "aarch64", target_os = "macos", feature = "mlx"))]`.

## Severity ladder

Use exactly these. The label decides what happens next, so do not inflate.

| Severity | Means | Consequence |
|---|---|---|
| `crit` | Data loss, memory unsafety, a panic on a reachable path, a security hole, or a wrong answer returned to a caller. | Blocks the merge. Fix before anything else. |
| `high` | A real bug under conditions that will occur in normal use, or a broken API contract. | Blocks the merge. |
| `med` | Correct today but fragile: a missing test for changed behaviour, an unhandled error path, a leaky abstraction. | Fix now or file it. Does not block. |
| `low` | Clarity, naming, dead code, a comment that has drifted from the code. | Author's call. |
| `nit` | Pure style. | **At most five per review.** Past five, drop the rest. |

The nit cap is not a suggestion. A review that opens with twenty nits buries
its own `crit`.

## Output contract

Markdown, sections in this order, omit a section if it is empty:

```
## crit
- `src/path.rs:120` — what is wrong, in one line.
  Fix: what to do instead, in one line.
```

Be terse. One finding per bullet. Location first, always `file:line`. No
preamble, no summary paragraph, no restating the diff back.

## What earns a finding

- A concrete failure you can name the trigger for. "This could be racy" is not
  a finding; "two `/v1/chat/completions` requests for the same unloaded model
  race in `ensure_loaded` and both allocate a slot" is.
- Behaviour that contradicts the punchlist item's stated acceptance criteria.
- A changed code path with no test covering the change.
- An MLX-gated path that will not compile, or is not gated, on non-Apple targets.
- A new dependency (flag it — this repo requires a >7-day age check).

## What does not

- Restating what the diff does.
- Style the formatter already owns.
- Speculative refactors of code the diff did not touch.
- "Consider adding documentation" with no specific gap named.

## Lanes

- **Pareto (default): 2 lanes.** R1 Claude Opus 4.6, R2 Codex GPT-5.4.
- **Release-tagged: 3 lanes.** Adds R3 Codex GPT-5.3-codex. Set `RELEASE=1`.
- On the Apple-Silicon box, codex is local-LLM only, so R2 is a stub there and a
  1/2-clean close is accepted. Re-run R2 on the Linux harness host before a
  release.

Findings collate into `.refs/review/COLLATED-<sha>.md`.

## Silencing

A finding is dropped only with a one-line `# silenced: <why>` comment beside it
in the collated file. No silent deletions. If you disagree with a `crit`, say so
in writing and leave the trail.

## Feedback loop

A finding that shows up in two separate reviews is not a review problem, it is a
missing rule. Add it to `AGENTS.md` (if it is judgment) or to a hook (if it is
mechanical) and it stops recurring.
