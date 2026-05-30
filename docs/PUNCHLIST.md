# Punchlist

`/implement` consumes the top-most unchecked item. Add new items at the bottom under a sprint heading. Tag backend in brackets where relevant.

## Sprint 2026-04-30 (seed)

- [ ] [meta] **Clippy clean baseline.** Resolve the 13 pre-existing clippy errors on `main` (sample: `result_large_err` on `tonic::Status`, `needless_range_loop` at `src/http.rs:530`, `collapsible_if` at `src/http.rs:606`). Then in `scripts/ratchet.sh` swap `cargo clippy ... || echo WARN ...` back to `cargo clippy ... -- -D warnings` and remove the warn-only comment. Acceptance: `bash scripts/ratchet.sh` is green with strict clippy.
- [ ] [meta] Smoke-test `/implement` end-to-end. Acceptance: `/implement` writes a worklog entry, runs ratchet green, runs review fanout, and flips this checkbox to `[x]` without manual edits.
- [ ] [cli] Add `spindll --version-short` flag that prints just the semver (e.g. `0.5.0`) with no trailing newline. Acceptance: a `#[test]` in the CLI module asserts the exact byte sequence; ratchet stays green.
- [ ] [bench] Extend `spindll bench` to emit `--json` mode with fields `prompt_eval_tps`, `decode_tps`, `peak_rss_mb`, `p50_ms`, `p95_ms`. Required by `scripts/autoloop.sh`. Acceptance: a unit test parses the JSON and round-trips.
- [ ] [meta] Add `cargo-deny` config (`deny.toml`) with a `min-release-age = 7d` advisory. Acceptance: a CI step runs `cargo deny check advisories` and passes on current `Cargo.lock`.

## Conventions

- One acceptance line per item, written so the test-first step can target it.
- Tags: `[gguf]`, `[mlx]`, `[cuda]`, `[vulkan]`, `[http]`, `[grpc]`, `[cli]`, `[bench]`, `[meta]`.
- Cross-cutting items go under `[meta]`.
- Items that touch `mlx_bridge/`, `src/backend/mlx*.rs`, or `feature = "mlx"` paths get `[mlx]` AND will be auto-tagged `mlx-validate-required` in the worklog by `/implement`.
