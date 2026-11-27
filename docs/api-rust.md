# Rust Library API

Add spindll to your project:

```toml
[dependencies]
spindll = { git = "https://github.com/Iito/spindll.git", features = ["http"] }
```

`http` adds the HTTP server and router used below; the gRPC server is always built. On Apple Silicon, add `mlx` for the MLX backend. The server samples also need `tokio` in your own dependencies, and the router sample needs `axum` 0.8.

## ModelManager

The primary entry point for multi-model inference. Routes loads to the matching `InferenceBackend` by `ModelFormat` (GGUF → llama.cpp, MLX → mlx-swift-lm on Apple Silicon builds with the `mlx` feature).

```rust
use spindll::engine::{ModelManager, GenerateParams};
use spindll::model_store::ModelStore;

// Create a manager with 4096 context, auto GPU layers, default memory cap.
// memory_budget = 0 caps loaded models at total physical RAM, read once here.
// Free RAM is measured before each load, but only to auto-size a GGUF
// model's n_ctx when n_ctx here is 0. Pass an explicit number (e.g.
// 8_000_000_000) for a hard cap, or u64::MAX for "no eviction".
let mut manager = ModelManager::new(4096, None, 0)?;

// Enable KV cache (2GB). Use enable_kv_cache_with_dir() for a custom path.
manager.enable_kv_cache(2_000_000_000);

// Its in-memory tier (512MB) is separate and stays off until enabled.
manager.enable_kv_ram_cache(512 * 1_048_576);

// Enable continuous batching (8 concurrent sequences per GGUF model;
// MLX models are gated out via supports_batching()).
manager.set_batch_slots(8);

// Enable RAM cache for recently-evicted model weights (4GB budget). After an
// unload or eviction the manager reads the GGUF file once to pull it into the
// OS page cache, so a reload skips the disk. It keeps no bytes itself.
// `spindll serve` skips it on macOS; the library doesn't.
manager.enable_ram_cache(4_000_000_000);

// Load a model — the manager picks the backend automatically based on
// the format detected from the path (file → GGUF, directory → MLX).
// Load it under its canonical registry key: the servers resolve request
// names to that key, so a model loaded under an alias can be loaded twice.
let store = ModelStore::new(None);
let m = store.resolve("llama3.1:8b")?;
manager.load_model_with_digest(&m.key, &m.path, None, m.digest.clone())?;

// Generate with streaming callback
manager.generate(&m.key, "Hello!", &GenerateParams::default(), None, |token| {
    print!("{token}");
    true // return false to stop early
})?;
```

### MLX prompt cache

MLX models don't use the manager's KV cache. The Swift bridge keeps its own prompt cache, on by default: a RAM tier per model (512 MB) and a disk tier shared by all models (2 GB). See [mlx-bridge.md](mlx-bridge.md#prompt-kv-cache). Only `generate_chat` reaches it. To change the budgets or compress the cache, call these before the first MLX load. They exist only on Apple Silicon macOS builds with the `mlx` feature:

```rust
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
{
    use spindll::backend::mlx_swift;
    mlx_swift::set_cache_budgets(512 * 1_048_576, 2_000_000_000); // RAM, disk; 0 turns a tier off
    mlx_swift::set_kv_scheme(Some("turbo8v3"))?; // None keeps the f16 cache
}
```

`spindll serve` sets both from `--kv-ram-cache`, `--kv-cache` and `--kv-scheme`.

### LoadOptions and eviction priority

For fine-grained control over model lifecycle, use `load_model_with_options` instead of `load_model_with_digest`:

```rust
use spindll::engine::{LoadOptions, EvictionPriority};
use std::time::Duration;

let opts = LoadOptions {
    gpu_layers: None,                           // auto-detect
    digest: m.digest.clone(),
    priority: EvictionPriority::High,           // evict last
    idle_reload: Some(Duration::from_secs(60)), // reload 60s after eviction
    ..Default::default()                        // mmproj_path, with `vision`
};
manager.load_model_with_options(&m.key, &m.path, opts)?;
```

`EvictionPriority` has three tiers — `Low`, `Normal` (default), `High`. Low-priority models are evicted first regardless of recency; within the same tier, LRU order applies. `idle_reload` reloads a model evicted under memory pressure once the manager has been idle that long with no request in flight, if the model then fits the budget with a 10% margin; otherwise it stays unloaded. It needs a manager wrapped with `into_arc()` (see [Starting servers](#starting-servers)) and a running tokio runtime: `Arc::new(manager)` turns it off silently.

### Backend traits (advanced)

`ModelManager` has no public way to add a backend: `new` registers llama.cpp, plus MLX on Apple Silicon builds with the `mlx` feature. For direct backend access, call `load_model` on a backend yourself (such as `spindll::backend::llamacpp::LlamaCppBackend::new()?`) and use the `BackendModel` it returns, or implement `InferenceBackend` for your own. The trait is:

```rust
use spindll::backend::{InferenceBackend, BackendModel, BackendLoadParams};

pub trait InferenceBackend: Send + Sync {
    fn load_model(&self, path: &Path, params: BackendLoadParams)
        -> anyhow::Result<Box<dyn BackendModel>>;
    fn name(&self) -> &str;
}
```

`BackendLoadParams` carries:
- `n_ctx: u32` — requested context size; `0` means auto-resolve to the largest n_ctx that fits weights + KV + compute buffers within `memory_budget`.
- `n_gpu_layers: Option<u32>` — `None` to auto-detect.
- `memory_budget: u64` — live availability snapshotted before the load; `0` means unlimited. Backends that auto-size n_ctx use this as the budget ceiling.
- `mmproj_path: Option<PathBuf>` (with the `vision` feature): the multimodal projector GGUF. `Some` sets up the vision pipeline.

## ModelStore (pulling)

```rust
use spindll::model_store::{ModelStore, FormatPreference};

let store = ModelStore::new(None);

// Auto: MLX-first on Apple Silicon builds with the `mlx` feature, GGUF
// fallback. q4_k_m default for HuggingFace GGUF repos; Ollama pulls ignore
// the quant, since the tag decides.
let path = store.pull("llama3.1:8b", None, FormatPreference::Auto)?;

// Force a specific format / quant
let path = store.pull("Qwen/Qwen2.5-3B-Instruct-GGUF", Some("q5_k_m"), FormatPreference::Gguf)?;

// FormatPreference::Mlx errors if no MLX equivalent is found
```

## Starting servers

```rust
use std::sync::Arc;

// into_arc() sets up internal weak references needed by the idle-reload watcher.
let manager = manager.into_arc();
let store = Arc::new(ModelStore::new(None));

// HTTP server (spindll's "http" feature). Spawn it first: each call keeps
// serving, so awaiting one would keep the other from starting.
tokio::spawn(spindll::http::start_http_server(8080, manager.clone(), store.clone()));

// gRPC server (always available). Returns on SIGINT or SIGTERM.
spindll::grpc::start_server(50051, manager.clone(), store.clone()).await?;
```

## Embedding the HTTP router

To mount spindll's API into your own axum server alongside other routes:

```rust
use std::sync::Arc;

let manager = manager.into_arc(); // not Arc::new(), which disables idle reload
let store = Arc::new(ModelStore::new(None));

// Get the router without binding to a port
let spindll_router = spindll::http::router(manager, store);

// Nest it under a prefix, or merge with your own routes
let app = axum::Router::new()
    .nest("/spindll", spindll_router);

let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
axum::serve(listener, app).await?;
```

## Engine (single-model)

For simpler use cases that don't need multi-model management (GGUF only):

```rust
use spindll::engine::{Engine, GenerateParams};

let engine = Engine::load(&path, None, 2048)?; // &Path, gpu layers, n_ctx
engine.generate("prompt", &GenerateParams::default(), |token| {
    print!("{token}");
    true
})?;
```
