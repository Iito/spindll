# MLX Swift Bridge

`mlx_bridge/` is a Swift Package Manager project that wraps [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) behind a small C ABI so Rust can drive MLX inference through FFI. It is built into a static library (`libMlxBridge.a`) by `build.rs` when spindll is compiled with `--features mlx` on Apple Silicon (`aarch64-apple-darwin`).

## Layout

```
mlx_bridge/
  Package.swift                       # SPM manifest, pins mlx-swift-lm + swift-transformers
  Package.resolved                    # Locked dependency versions
  Sources/MlxBridge/
    MlxBridge.swift                   # @_cdecl FFI entry points + PromptCache
    include/
      mlx_bridge.h                    # C header consumed by Rust
```

The library is `.static` and sets a minimum deployment target of macOS 15 (`platforms: [.macOS(.v15)]`) — it runs on macOS 15 and later. It targets Swift language mode 5 to allow the `DispatchSemaphore` + `Box` pattern that bridges async results across Task boundaries.

## C ABI

The headers in `include/mlx_bridge.h` define the exported surface. All entry points are blocking — internally each one drives an `await` on `ModelContainer` and waits on a semaphore so the caller does not need to host an async runtime.

| Symbol | Purpose |
|---|---|
| `mlx_model_load(path)` | Load an MLX-format directory (safetensors + config.json). Returns an opaque `MlxModelHandle*` (NULL on failure). |
| `mlx_model_free(handle)` | Release the handle and all associated MLX resources. |
| `mlx_generate(handle, prompt, max_tokens, temp, top_p, seed, cb, ctx)` | Tokenize `prompt` directly (no chat template) and stream detokenized chunks via `cb`. Returns generated token count, or `-1` on error. |
| `mlx_apply_chat_template(handle, messages_json)` | Render the tokenizer's Jinja chat template over a JSON `[{role, content}, ...]` array and return a UTF-8 string. Caller frees with `mlx_free_string`. |
| `mlx_free_string(s)` | Release a string returned by `mlx_apply_chat_template`. NULL-safe. |

### Header drift

`mlx_bridge.h` does not currently declare `mlx_chat_generate`, which is implemented in Swift and used by the Rust side via its own `extern "C"` declaration in `src/backend/mlx_swift.rs`. `mlx_chat_generate` is the fast path: it applies the chat template, looks up the prompt KV cache, and streams generation in a single `ModelContainer.perform` entry — no decode→encode round-trip. Add the declaration to the header if you want C consumers other than Rust to use it.

## Prompt KV cache

`mlx_chat_generate` keeps an in-memory LRU of the last 4 prompt token sequences and their post-prefill `KVCache` snapshots. On a hit, the cache is restored, trimmed by one token, and `TokenIterator` is seeded with the final prompt token only — TTFT collapses to a single decode step instead of full prefill. Access is serialised by `ModelContainer`'s actor, so no extra locking is needed.

The cache lives on the `ModelState` retained by `mlx_model_load` and is released by `mlx_model_free`.

## Token streaming

Both generate paths use `NaiveStreamingDetokenizer` and:

1. Stop on EOS, UNK, any `eosTokenIds` from the model config, or `extraEOSTokens` resolved through `convertTokenToId` (this is what catches Gemma3's `<end_of_turn>`, Phi's `<|end|>`, SmolLM's `<turn|>`, etc.).
2. Flush a final detokenizer chunk after the loop so partial-UTF-8 bytes are not dropped at `maxTokens` mid-codepoint exits.
3. Call `Stream().synchronize()` before tearing down `perform` so in-flight async evals complete cleanly. Mirrors mlx-swift-lm's own `runSynchronousGenerationLoop`.

The token callback returns `1` to continue or `0` to cancel; cancellation breaks the loop without flushing the final chunk.

## Build pipeline

`build.rs` runs the bridge build only when both conditions hold:

```rust
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
if std::env::var("CARGO_FEATURE_MLX").is_ok() { build_mlx_bridge()?; }
```

The build steps are:

1. `swift build --package-path mlx_bridge --configuration release --arch arm64`
   produces `mlx_bridge/.build/release/libMlxBridge.a`.
2. `compile_mlx_metallib` walks `mlx_bridge/.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal/`, compiles every `.metal` file with `xcrun -sdk macosx metal`, and links the resulting `.air` files into `mlx.metallib` next to the spindll binary. MLX's `device.cpp` discovers the metallib via `load_colocated_library("mlx")`, so it must sit next to the binary.
3. Cargo links the static archive plus `Foundation`, `Metal`, `Accelerate`, `MetalPerformanceShaders`, and adds rpaths for the Xcode Swift toolchain and `/usr/lib/swift` so `libswift_Concurrency.dylib` resolves at runtime.

If `xcrun metal` reports "missing Metal Toolchain", the build aborts with the install hint:

```
xcodebuild -downloadComponent MetalToolchain
```

Recompilation is gated on `mlx_bridge/Sources` and `mlx_bridge/Package.swift` (`rerun-if-changed`), so editing Swift code or bumping dependencies in `Package.swift` triggers a rebuild on the next `cargo build`.

## Updating dependencies

The Swift dependencies are pinned in `Package.resolved`. To upgrade:

```bash
swift package --package-path mlx_bridge update
```

After updating, rerun `cargo build --features cli,mlx` so `build.rs` rebuilds the static archive and refreshes `mlx.metallib`.

## Rust integration

The Rust side lives in `src/backend/mlx_swift.rs` and implements `InferenceBackend` for an `MlxModel` newtype around the opaque handle. Loads happen on a dedicated thread because Swift expects async work to run off the calling thread, and the FFI callback marshals UTF-8 chunks back into the `ModelManager` streaming pipeline.
