---
name: mlx-validation
description: When and how to validate spindll's MLX path — which changes require it, the mlx-validate-required worklog tag, what CI already covers on macOS versus what only the local release-level validator catches, and the Apple-Silicon build gotchas. Use when touching mlx_bridge/, src/backend/mlx*.rs, any feature = "mlx" code, or the Swift toolchain pin.
---

# Validating the MLX path

MLX is Apple-Silicon only. Gate every MLX code path:

```rust
#[cfg(all(target_arch = "aarch64", target_os = "macos", feature = "mlx"))]
```

## When validation is required

Tag the worklog entry `mlx-validate-required` when the change touches any of:

- `mlx_bridge/` (the Swift FFI bridge)
- `src/backend/mlx*.rs`
- any path gated on `feature = "mlx"`

Then run `scripts/mlx-validate.sh <branch>` on Apple Silicon. It doesn't read
the tag: it switches the checkout it runs in to that branch, builds and tests it
with `--features cli,http,mlx,vision`, and appends an entry tagged
`mlx-validated` to that host's worklog when it passes. Run it in a spare
worktree, never in the user's checkout.

## What CI already covers, and what it does not

| | Runs on | Covers |
|---|---|---|
| `.github/workflows/ci.yml` | Linux + Windows matrix | Build/test, clippy (plus the `ffi` crate's clippy and tests), vision and vulkan feature checks. Path-scoped to `src/**`, `tests/**`, `proto/**`, `ffi/**`, `.cargo/**`, `Cargo.*`, `build.rs`. **Does not list `mlx_bridge/**`.** |
| `.github/workflows/ci-macos.yml` | `macos-26`, Xcode 26.6 | **Debug** build + lib tests, `--features cli,http,mlx,vision`, and an `ios-check` job that lints the library for iPhone and the Simulator. Scoped to MLX-relevant paths *including* `mlx_bridge/**`. |
| `scripts/mlx-validate.sh` | Local Apple Silicon | **Release** build + full lib tests. The heavy validator. |

Net effect: a Swift-bridge-only change runs **only** the macOS job. Shared-code
changes run all three platforms. Neither CI job is a release-level build, so
`mlx-validate-required` still means a human runs the local validator before the
change ships.

## Release-level build gotchas on Apple Silicon

- **Non-interactive shells have no Homebrew on PATH.** `protoc` and `cmake` live
  in `/opt/homebrew/bin`, so a build driven over SSH fails in `build.rs` with
  "Could not find protoc" unless you prefix
  `export PATH="/opt/homebrew/bin:$PATH"`.
- **MLX needs the on-demand Metal toolchain** (macOS/Xcode 26):
  `xcodebuild -downloadComponent MetalToolchain` (~688 MB, no sudo required).
- **Metal language version floor.** `build.rs` pins
  `-mmacosx-version-min=15.0` on the metal compile. Without it the metallib
  targets Metal 4.0 (the macos-26 SDK default) and macOS 15's loader rejects it
  at runtime — this shipped once, in v0.9.0.
- **Swift toolchain and SDK**: mlx-swift-lm is pinned to a release tag
  (`exact:` in `mlx_bridge/Package.swift`), with its default
  FoundationModelsIntegration trait off. The bridge needs Swift 6.3+ and the
  macOS 26 SDK: a swift.org toolchain on an older Xcode fails, and `build.rs`
  stops early with the fix. The test hardware runs Swift natively with no
  toolchain overlay.

## Review lanes on Apple Silicon

The R2 review lane (`codex exec`) has no OpenAI credentials on the Apple-Silicon
machines by design — that CLI is used there only for local models. Close
`/implement` runs honestly as `review=1/2-clean` with a lane note. Do not retry
or debug R2 auth there. Run R2 on the Linux harness host before a release.
