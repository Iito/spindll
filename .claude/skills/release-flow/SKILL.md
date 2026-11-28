---
name: release-flow
description: How to cut a spindll release. Version bump, changelog and release-fix commits through a PR straight into main, a lightweight tag on main, the release.yml gates, hand-curated notes, carrying release PRs into next, and resetting next to main. Use when tagging a version or release candidate, preparing release commits, writing release notes, or syncing main and next after a release.
---

# Cutting a spindll release

Verified against v0.9.0 (2026-08-21) and every v0.9.x since. Release work moved
from direct pushes to `main` to PRs on 2026-11-27, and on 2026-11-28 from PRs
into `next` to PRs straight into `main` (§1, §6). v0.10.0-rc.2 is the first
release cut this way.

## 1. Release commits on a branch off `main`, in a sibling worktree

Release work is what a tag needs and nothing more: the version bump, the
changelog, and fixes for the release itself (an RC whose build broke, the bug a
patch release ships for). It branches off `main`, so a release never waits on,
or picks up, unfinished work on `next`. Features keep going into `next`.

```bash
git fetch origin
git worktree add --no-track -b release/X.Y.Z ../spindll-release origin/main
```

Never do release work in the shared `main` checkout. The user works there and
its branch moves underfoot — v0.9.1's first tag landed on `feat/rpc-sharding`
that way and had to be killed.

For a final release, two commits, in this order (skip the bump if an RC already
made it):

1. `chore(release): bump version to X.Y.Z` — `Cargo.toml` plus the spindll
   stanza in `Cargo.lock`.
2. `docs(changelog): set X.Y.Z release date` — `## [Unreleased]` becomes
   `## [X.Y.Z] - <date>`.

An `-rc.N` tag has to match `Cargo.toml` too, since `check-version` strips the
suffix before comparing, so the bump lands with the first RC. The changelog
heading stays `[Unreleased]` until the final release. Later RCs need only what
the tag still lacks, such as missing changelog entries or a release.yml fix.

Validate with `cargo check --features cli,http`, then open a PR into `main`. CI
runs on it as on any PR. **Never push these commits to `main`**: they land
through the PR's merge, and then get carried into `next` (§6).

**Worktree gotcha.** Cargo walks parent directories for `.cargo/config.toml`, so
a worktree nested under the main checkout inherits its config — including
`GGML_RPC = "ON"` while main sits on an RPC branch, which fails CMake on any
branch pinning crates.io `llama-cpp-sys-2`. Put worktrees as **siblings**
(`../spindll-release`), or drop an uncommitted worktree-local
`.cargo/config.toml` with `[env] GGML_RPC = "OFF"`. After a failure, clear the
half-configured CMake dir: `rm -rf target/debug/build/llama-cpp-sys-2-*`.

## 2. Tag

Lightweight `vX.Y.Z` (or `vX.Y.Z-rc.N`) on **the commit the release PR's merge
left on `main`**, the newest one if a rebase merge left several. The merge
rewrites commits, so tag the copy on `main`, never the branch's. Pushing the tag
triggers `.github/workflows/release.yml`. It doesn't wait for the carry into
`next` (§6).

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

`main` moves only through two kinds of PR: release PRs (§1), and the next→main
PR, which brings features over once they're ready to ship. **Never push to
`main`**, not even a fast-forward.

The repo allows only squash and rebase merges, and both rewrite commits, so a
merge never leaves `main` and `next` sharing SHAs. Three habits keep them in
step.

**After a release PR merges, carry it into `next`.** If nothing has landed on
`next` since its last reset, `next` is still an ancestor of `main`, and the user
resets it (below) instead. Otherwise open a carry PR into `next`: a branch off
`next` with the release commits cherry-picked. Resolve any `CHANGELOG.md`
conflict there, since features add entries on `next` while releases move
headings on `main`.

```bash
git fetch origin
git merge-base --is-ancestor origin/next origin/main && echo "reset, no carry PR"
```

**Before a next→main PR, check that every release PR has been carried.** A
trial merge of `main` into `next` must leave `next`'s tree unchanged (needs Git
2.38 or later):

```bash
git fetch origin
[ "$(git merge-tree --write-tree origin/next origin/main)" = \
  "$(git rev-parse 'origin/next^{tree}')" ] && echo "next has everything on main"
```

If it prints nothing, open the missing carry PR first. Otherwise the next→main
PR can conflict on `CHANGELOG.md`, and after it merges the trees won't match for
the reset.

**After the next→main PR merges, reset `next` to `main`**, before anything else
lands on `next`. The merge leaves `main` with copies of `next`'s commits, the
same tree under different SHAs. Until the reset, any later next→main PR lists
them again and its rebase conflicts on `CHANGELOG.md`.

```bash
git fetch origin
git diff --quiet origin/main origin/next && echo "same tree"
git push --force-with-lease=refs/heads/next:$(git rev-parse origin/next) \
  origin origin/main:refs/heads/next
```

The reset is safe when the trees match (after a next→main merge) or when `next`
is an ancestor of `main` (after a release PR, with nothing new on `next`). If
neither holds, the reset would drop work: stop and ask the user.
`--force-with-lease` refuses if `next` moved after the fetch. Agents never
force-push `next` (an AGENTS.md hard rule), so the user runs this one.

## Approval

Every push (the tag's included), every `gh pr create` and every `gh pr merge` in
this flow needs the user's explicit approval, each time. The `guard-git.py` hook
enforces it via a lockfile the user touches by hand; it is consumed by one
publish.
