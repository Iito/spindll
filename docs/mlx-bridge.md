# MLX Swift Bridge

`mlx_bridge/` is a Swift Package Manager project that wraps [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) behind a small C ABI so Rust can drive MLX inference through FFI. It is built into a static library (`libMlxBridge.a`) by `build.rs` when spindll is compiled with `--features mlx` on Apple Silicon (`aarch64-apple-darwin`).

## Layout

```
mlx_bridge/
  Package.swift                       # SPM manifest, pins mlx-swift-lm + swift-transformers
  Package.resolved                    # Locked dependency versions
  Sources/MlxBridge/
    MlxBridge.swift                   # @_cdecl FFI entry points + prompt cache (RAM and disk tiers)
    include/
      mlx_bridge.h                    # C header for C callers (Rust declares the functions itself)
```

The library is `.static` and sets a minimum deployment target of macOS 15 (`platforms: [.macOS(.v15)]`) — it runs on macOS 15 and later. It targets Swift language mode 5 to allow the `DispatchSemaphore` + `Box` pattern that bridges async results across Task boundaries.

## C ABI

The bridge exports 12 `@_cdecl` functions, and `include/mlx_bridge.h` declares 9 of them (see [Header drift](#header-drift)). Loading, generation, template rendering and embedding block: each one runs its async work in a Swift `Task` and waits on a semaphore, so the caller does not need to host an async runtime.

| Symbol | Purpose |
|---|---|
| `mlx_model_load(path)` | Load an MLX-format directory (safetensors + config.json), as a VLM if mlx-swift-lm can, else as an LLM. Returns an opaque `MlxModelHandle*` (NULL on failure). |
| `mlx_model_free(handle)` | Release the handle and all associated MLX resources. |
| `mlx_model_is_vlm(handle)` | `1` if the model loaded as a VLM (takes images), else `0`. |
| `mlx_generate(handle, prompt, max_tokens, temp, top_p, seed, cb, ctx)` | Tokenize `prompt` directly (no chat template) and stream detokenized chunks via `cb`. Returns generated token count, or `-1` on error. |
| `mlx_chat_generate(handle, messages_json, tools_json, max_tokens, temp, top_p, seed, cb, ctx)` | Apply the chat template and generate in one call, with the prompt KV cache. Returns like `mlx_generate`. |
| `mlx_chat_generate_vision(handle, messages_json, image_path, max_tokens, temp, top_p, seed, cb, ctx)` | VLM chat: attach the image at `image_path` (NULL for none) to the last user message. No prompt cache. Fails for a model without a chat template. |
| `mlx_apply_chat_template(handle, messages_json, tools_json)` | Render the tokenizer's Jinja chat template over a JSON `[{role, content}, ...]` array and return a UTF-8 string. `tools_json` is a JSON array of OpenAI tool specs bound to the template's `tools` variable, or NULL. Caller frees with `mlx_free_string`. |
| `mlx_free_string(s)` | Release a string returned by `mlx_apply_chat_template`. NULL-safe. |
| `mlx_embed(handle, text, out_data, out_len)` | Mean-pool the input embedding rows of `text`'s tokens into a unit vector, a heuristic rather than a trained embedding. Returns the prompt token count, or `-1`. Caller frees `*out_data` with `mlx_free_floats`. |
| `mlx_free_floats(data)` | Release a vector returned by `mlx_embed`. NULL-safe. |
| `mlx_set_cache_budgets(ram_bytes, disk_bytes)` | Set the prompt cache's byte budgets; `0` turns a tier off. See [Prompt KV cache](#prompt-kv-cache). |
| `mlx_set_kv_scheme(scheme)` | Set the KV cache scheme for later loads, or clear it with NULL. Returns `0`, or `-1` with the setting unchanged for a name mlx-swift-lm doesn't know. |

The three generate calls take `seed` but ignore it.

### Header drift

`mlx_bridge.h` does not declare `mlx_chat_generate`, `mlx_set_cache_budgets` or `mlx_set_kv_scheme`. Rust doesn't read the header: `src/backend/mlx_swift.rs` declares all 12 functions in its own `extern "C"` block. `mlx_chat_generate` is the fast path: it applies the chat template, looks up the prompt KV cache, and streams generation in a single `ModelContainer.perform` entry — no decode→encode round-trip. Add the missing declarations to the header if C consumers other than Rust need them.

## Prompt KV cache

`mlx_chat_generate` reuses KV state across requests that share a token prefix. A lookup takes the stored entry that shares the longest prefix with the new prompt, if that prefix is at least 16 tokens. On a hit the bridge restores the shared prefix and prefills only the tokens past it. When the whole prompt is cached it holds back the last token, since `TokenIterator` needs one to produce the logits it samples from. After generation the live cache, which covers the prompt plus the generated tokens, is stored for the next turn's prompt to extend. Caches that can't be trimmed, such as Mamba/SSM hybrids or a sliding-window cache past its window, are not stored, and a restore falls back to a full prefill when a layer can't rewind far enough. `mlx_generate` and `mlx_chat_generate_vision` don't use the cache.

Entries live in two tiers:

- **RAM**: one per loaded model, with its own byte budget (default 512 MB, `--kv-ram-cache`). It drops least-recently-used entries when over budget. `--no-kv-ram-cache` turns it off, and new entries then go straight to disk. It lives on the `ModelState` retained by `mlx_model_load` and is released by `mlx_model_free`. Access is serialised by `ModelContainer`'s actor, so no extra locking is needed.
- **Disk**: one tier shared by all models, in `~/.spindll/cache/mlx/` (default 2 GB, `--kv-cache`, the flag that also sizes the GGUF disk cache). It keeps prefixes across restarts and drops least-recently-used entries when over budget. It only takes sequences of 512 tokens or more. When a new entry is saved, each older RAM entry it doesn't extend is copied to disk, once. A prompt of 512 tokens or more checks the disk tier, and loads an entry from it only if that entry covers at least 64 more tokens than the RAM hit. Files are named `<digest>-<scheme or f16>_<token digest>.safetensors`, where `<digest>` is the SHA-256 of the model's `config.json`, so entries are kept apart per model config and per scheme. `SPINDLL_MLX_DISK_CACHE=0` turns the tier off, and `--no-kv-cache` turns off both.

`mlx_set_cache_budgets` (`set_cache_budgets` in Rust) sets both budgets. Call it before the first load: each model's RAM tier reads its budget when the model loads, and the disk tier reads its budget on first use. Without the call, the defaults above apply. `spindll serve` passes its `--kv-ram-cache` and `--kv-cache` values.

### KV scheme

Entries keep the KV state as computed, in f16, so greedy output with the cache matches output without it. `--kv-scheme` (`set_kv_scheme` in Rust, called before the first MLX load) trades that for memory. Generation then uses one of mlx-swift-lm's compressed layouts, such as `turbo8v3` (TurboQuant: 8-bit keys, 3-bit values), `turbo0v4`, `turbo4` or `affine8`, and cache entries stay in that layout. `f16`, the default, means no scheme. The scheme applies to all three generate calls and is part of each model's disk namespace, so switching schemes starts a fresh disk cache. Gemma 2, GPT-OSS, Bitnet, Apertus and Baichuan-M1 keep the f16 cache under any scheme, with a warning at load: their attention writes the KV cache directly, and a compressed cache would crash them on the first prompt.

### Prefill step

`mlx_generate` and `mlx_chat_generate` prefill a prompt of more than 512 tokens (on a cache hit, the uncached rest) in steps of `min(2048, length - 1)` tokens. Only the last token reaches the forward that samples the first output token, so only that token is projected onto the vocabulary. A prompt of 512 tokens or fewer, mlx-swift-lm's default step, goes through that forward whole, which costs less there than another pass over the weights. Larger steps hold activations for up to 2048 tokens at a time while a long prompt fills. A model with a sliding window still caps the step at its window, and the vision path keeps mlx-swift-lm's default step.

## VLM image budget

Input images are pre-scaled before they reach the processor when they exceed
**~1M pixels** (`kDefaultMaxPixels`, ≈1024×1024). Vision prefill cost scales
linearly with patch count while decode speed is unaffected, and model
preprocessor configs ship permissive budgets (`qwen3.5-9b-4bit` allows 16.8M px
— about two minutes of prefill for a 12 MP photo); the cap turns that into ~5 s
with no decode cost. The resize preserves aspect ratio, never upscales, and
logs `[mlx-bridge] resized image from W×H to W'×H'` when it fires.

`SPINDLL_VLM_MAX_PIXELS` overrides the budget per process; `0` (or negative)
disables the cap entirely and defers to the model's own preprocessor config —
use this when caption fidelity on fine detail matters more than TTFT.

## Token streaming

All three generate paths use `NaiveStreamingDetokenizer` and:

1. Stop on EOS, UNK, any `eosTokenIds` from the model config, or `extraEOSTokens` resolved through `convertTokenToId` (this is what catches Gemma3's `<end_of_turn>`, Phi's `<|end|>`, SmolLM's `<turn|>`, etc.).
2. Call the detokenizer's `finish()` after the loop. `next()` holds back text that ends in an incomplete UTF-8 sequence, and calling it again keeps holding it, so a reply cut mid-codepoint (at `maxTokens`, say) would lose its tail. `finish()` emits that text, with U+FFFD for the incomplete bytes.
3. Call `Stream.defaultStream.synchronize()` before tearing down `perform` so in-flight async evals complete cleanly. Mirrors mlx-swift-lm's own `runSynchronousGenerationLoop`. Not `Stream()`: since mlx-swift 0.33 that creates a new stream, and waiting on it returns at once.

The token callback returns `1` to continue or `0` to cancel; cancellation breaks the loop without flushing the final chunk.

## Build pipeline

`build.rs` runs the bridge build only when all three conditions hold: an Apple Silicon macOS build host (the `cfg`), the `mlx` feature, and a macOS target. A build for any other target, such as iOS, with `--features mlx` gets a warning and no bridge:

```rust
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
if std::env::var("CARGO_FEATURE_MLX").is_ok() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        build_mlx_bridge()?;
    } else {
        println!("cargo:warning=mlx: the MLX bridge builds for macOS only; skipping it for this target");
    }
}
```

The build steps are:

1. `swift build --package-path mlx_bridge --configuration release --arch arm64`
   produces `mlx_bridge/.build/release/libMlxBridge.a`.
   SwiftPM must be Swift 6.3+ (mlx-swift 0.31.5+ declares `swift-tools-version: 6.3`).
   `build.rs` picks the toolchain in this order: an explicit `$TOOLCHAINS`; the
   default `swift` when it is 6.3+; the newest swift.org toolchain found under
   `~/Library/Developer/Toolchains` or `/Library/Developer/Toolchains`
   (selected via `TOOLCHAINS` for the SwiftPM child process only). If none
   qualifies it aborts with install hints. Xcode 26.4+ satisfies this natively;
   Xcode 26.0–26.3 needs a [swift.org toolchain](https://www.swift.org/install/macos/)
   alongside — the metallib step below stays on the `xcode-select`'d Xcode either way.
   The SDK must be macOS 26 or newer, whatever the toolchain: mlx-swift-lm relies on
   Sendable annotations that SDK added, so a swift.org toolchain on Xcode 16 fails
   deep in the build. `build.rs` checks `xcrun --show-sdk-version` first and stops
   with the fix. To build on a Mac whose selected Xcode is older, point the build at
   an Xcode 26 install with `DEVELOPER_DIR=/Applications/<Xcode 26>.app/Contents/Developer`;
   that Xcode needs its own Metal Toolchain.
2. `compile_mlx_metallib` walks `mlx_bridge/.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal/`, compiles every `.metal` file with `xcrun -sdk macosx metal`, and links the resulting `.air` files into `mlx.metallib` next to the spindll binary. MLX's `device.cpp` discovers the metallib via `load_colocated_library("mlx")`, so it must sit next to the binary. The binary also embeds a copy and writes it out at startup if the file next to it is missing or a different size. If that write fails, `ModelManager::new` leaves the MLX backend out, with a warning. A built metallib is reused only while `mlx.metallib.stamp` beside it matches a fingerprint of the shader tree and the compile flags, so an mlx-swift bump recompiles it rather than shipping the old kernels.
3. Cargo links the static archive plus `Foundation`, `Metal`, `Accelerate`, `MetalPerformanceShaders`, and adds rpaths for the Xcode Swift toolchain and `/usr/lib/swift` so `libswift_Concurrency.dylib` resolves at runtime.

If `xcrun metal` reports "missing Metal Toolchain", the build aborts with the install hint:

```
xcodebuild -downloadComponent MetalToolchain
```

Recompilation is gated on `mlx_bridge/Sources`, `mlx_bridge/Package.swift` and `mlx_bridge/Package.resolved` (`rerun-if-changed`), so editing Swift code or bumping dependencies, in the manifest or the lockfile alone, triggers a rebuild on the next `cargo build`.

## Updating dependencies

`Package.swift` pins mlx-swift-lm to one release (`exact: "3.33.0"`), and `Package.resolved` locks every package. `swift package update` can't move an exact pin, so to upgrade mlx-swift-lm, change that version in `Package.swift` first, then resolve:

```bash
swift package --package-path mlx_bridge update
```

Without a manifest change, the update only moves packages within their declared ranges (swift-transformers within 1.3.x, for one). It needs Swift 6.3+, like the build: set `TOOLCHAINS=<bundle id>` if the default `swift` is older.

After updating, rerun `cargo build --features cli,mlx` so `build.rs` rebuilds the static archive and refreshes `mlx.metallib`.

## Rust integration

The Rust side lives in `src/backend/mlx_swift.rs`. `MlxBackend` implements `InferenceBackend`, and its loads return an `MlxSwiftEngine`, which owns the opaque handle and implements `BackendModel`. A load runs on the caller's thread, which blocks until Swift finishes. Generation and embedding each spawn a thread, so the bridge's semaphore wait never blocks a Tokio worker, and the FFI callback sends UTF-8 chunks over a bounded channel back into the `ModelManager` streaming pipeline. `set_cache_budgets` and `set_kv_scheme` wrap the two setters.
