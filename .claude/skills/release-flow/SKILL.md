---
name: release-flow
description: How to cut a spindll release. Version bump and changelog commits through a PR into next and then the next→main PR, a lightweight tag on main, the release.yml gates, hand-curated notes, and resetting next to main after the merge. Use when tagging a version, preparing release commits, writing release notes, or syncing main and next after a release.
---

# Cutting a spindll release

Verified against v0.9.0 (2026-08-21) and every v0.9.x since. The PR path in §1
and §6 replaced direct pushes to `main` on 2026-11-27, and no release has gone
through it yet.

## 1. Two commits on a branch off `next`, in a sibling worktree

```bash
git fetch origin
git worktree add -b release/X.Y.Z ../spindll-release origin/next
```

Never do release work in the shared `main` checkout. The user works there and
its branch moves underfoot — v0.9.1's first tag landed on `feat/rpc-sharding`
that way and had to be killed.

Two commits, in this order:

1. `chore(release): bump version to X.Y.Z` — `Cargo.toml` plus the spindll
   stanza in `Cargo.lock`.
2. `docs(changelog): set X.Y.Z release date` — `## [Unreleased]` becomes
   `## [X.Y.Z] - <date>`.

Validate with `cargo check --features cli,http`, then open a PR into `next`.
Never push these commits to `main`: they reach it like any other change, through
`next`'s PR, its CI and its merge, then the next→main PR (§6).

**Worktree gotcha.** Cargo walks parent directories for `.cargo/config.toml`, so
a worktree nested under the main checkout inherits its config — including
`GGML_RPC = "ON"` while main sits on an RPC branch, which fails CMake on any
branch pinning crates.io `llama-cpp-sys-2`. Put worktrees as **siblings**
(`../spindll-release`), or drop an uncommitted worktree-local
`.cargo/config.toml` with `[env] GGML_RPC = "OFF"`. After a failure, clear the
half-configured CMake dir: `rm -rf target/debug/build/llama-cpp-sys-2-*`.

## 2. Tag

Lightweight `vX.Y.Z` on **`main`'s changelog-date commit**, once the next→main
PR has merged. The merge rewrites commits, so tag the copy on `main`, never the
one `next` had. Pushing the tag triggers `.github/workflows/release.yml`.

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

`next` reaches `main` only through the next→main PR, merged once its CI passes.
**Never push to `main`**, not even a fast-forward.

The repo allows only squash and rebase merges, and both rewrite commits. So the
merge leaves `main` with copies of `next`'s commits: the same tree, different
SHAs. Until `next` is reset to `main`, any later next→main PR lists those commits
again and its rebase conflicts on `CHANGELOG.md`. Reset `next` right after the
merge, before anything else lands on it:

```bash
git fetch origin
git diff --quiet origin/main origin/next && echo "same tree"
git push --force-with-lease=refs/heads/next:$(git rev-parse origin/next) \
  origin origin/main:refs/heads/next
```

If the trees differ, something landed on `next` after the merge, and the reset
would drop it: stop and ask the user. `--force-with-lease` refuses if `next`
moved after the fetch. Agents never force-push `next` (an AGENTS.md hard rule),
so the user runs this one.

## Approval

Every push (the tag's included), every `gh pr create` and every `gh pr merge` in
this flow needs the user's explicit approval, each time. The `guard-git.py` hook
enforces it via a lockfile the user touches by hand; it is consumed by one
publish.
