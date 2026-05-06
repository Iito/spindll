# MlxBridge

Swift Package that wraps [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) behind a C ABI for Rust FFI. Built into a static library by `build.rs` at the repo root when spindll is compiled with `--features mlx` on Apple Silicon.

See [`docs/mlx-bridge.md`](../docs/mlx-bridge.md) for the C ABI reference, build pipeline, prompt KV cache details, and Rust integration notes.
