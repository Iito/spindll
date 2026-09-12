---
name: release-flow
description: How to cut a spindll release — version bump and changelog commits on main via a detached worktree, lightweight tag, the release.yml gates, hand-curated notes, and the main/next fast-forward lockstep. Use when tagging a version, preparing release commits, writing release notes, or syncing main and next after a release.
---

# Cutting a spindll release

Verified against v0.9.0 (2026-08-21) and every v0.9.x since.

## 1. Two commits on `main`, in a detached sibling worktree

```bash
git worktree add --detach ../spindll-release origin/main
```

Never do release work in the shared `main` checkout. The user works there and
its branch moves underfoot — v0.9.1's first tag landed on `feat/rpc-sharding`
that way and had to be killed.

Two commits, in this order:

1. `chore(release): bump version to X.Y.Z` — `Cargo.toml` plus the spindll
   stanza in `Cargo.lock`.
2. `docs(changelog): set X.Y.Z release date` — `## [Unreleased]` becomes
   `## [X.Y.Z] - <date>`.

Validate with `cargo check --features cli,http` before either lands.

**Worktree gotcha.** Cargo walks parent directories for `.cargo/config.toml`, so
a worktree nested under the main checkout inherits its config — including
`GGML_RPC = "ON"` while main sits on an RPC branch, which fails CMake on any
branch pinning crates.io `llama-cpp-sys-2`. Put worktrees as **siblings**
(`../spindll-release`), or drop an uncommitted worktree-local
`.cargo/config.toml` with `[env] GGML_RPC = "OFF"`. After a failure, clear the
half-configured CMake dir: `rm -rf target/debug/build/llama-cpp-sys-2-*`.

## 2. Tag

Lightweight `vX.Y.Z` on the **changelog-date commit**. Pushing it triggers
`.github/workflows/release.yml`.

Never tag before the bump lands: `check-version` hard-fails unless
`Cargo.toml`'s version equals the tag. A post-publish artifact defect gets a
patch release, never a moved tag.

## 3. What release.yml does

- `check-version` — hard gate on version/tag equality.
- `-rc.N` tags are skipped when a newer RC of the same version exists.
- `test` — `cargo test --features cli,http`.
- `create-release` — fires once tests pass; build jobs attach macOS ARM, Linux
  Vulkan and Windows Vulkan artifacts as they finish.

## 4. Notes are hand-curated, never auto-generated

The workflow carries the body from the newest RC pre-release of the same version
(retiring that RC), else a placeholder. Curate with
`gh release edit vX.Y.Z --notes-file`.

Format mirrors v0.8.0: `## vX.Y.Z`, a one-line summary with PR refs, "What's
Included" (Added / Fixed / Changed / Infrastructure, from the changelog),
"Verification", and a compare link.

## 5. Post-publish smoke test

Download the macOS artifact and run an MLX generation on the **oldest supported
macOS (15)**. v0.9.0's metallib targeted Metal language 4.0 (the macos-26
runner's SDK default) and macOS 15's loader rejected it; `build.rs` now pins
`-mmacosx-version-min=15.0` on the metal compile. This check exists because that
shipped.

## 6. Branch sync

`main` and `next` advance in lockstep with identical SHAs. Releases land on main
by **fast-forward** — the GitHub merge buttons don't fit (merge commits are
disabled, rebase refuses branches containing merges). Push
`origin/next:refs/heads/main` directly; admin bypasses the review rule and the
open next→main PR auto-flips to MERGED. After cutting, fast-forward `next` to
`main` and push so both tips match.

## Approval

Every push and every `gh pr create` in this flow needs the user's explicit
approval, each time. The `guard-git.py` hook enforces it via a lockfile the user
touches by hand; it is consumed by one publish.
