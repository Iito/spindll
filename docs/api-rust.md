# Rust Library API

Add spindll to your project:

```toml
[dependencies]
spindll = { git = "https://github.com/Iito/spindll.git" }
```

## ModelManager

The primary entry point for multi-model inference. Routes loads to the matching `InferenceBackend` by `ModelFormat` (GGUF → llama.cpp, MLX → mlx-swift-lm on Apple Silicon).

```rust
use spindll::engine::{ModelManager, GenerateParams};
use spindll::model_store::ModelStore;

// Create a manager with 4096 context, auto GPU, dynamic memory tracking.
// memory_budget = 0 → live-tracking auto-mode: every load and eviction
// re-snapshots free RAM, so spindll never exceeds what the system can give.
// Pass an explicit number (e.g. 8_000_000_000) for a hard cap, or u64::MAX
// for "no eviction".
let mut manager = ModelManager::new(4096, None, 0)?;

// Enable KV cache (2GB)
manager.enable_kv_cache(2_000_000_000);

// Enable continuous batching (8 concurrent sequences per GGUF model;
// MLX models are gated out via supports_batching()).
manager.set_batch_slots(8);

// Load a model — the manager picks the backend automatically based on
// the format detected from the path (file → GGUF, directory → MLX).
let store = ModelStore::new(None);
let path = store.resolve_model_path("llama3.1:8b")?;
let digest = store.resolve_model_digest("llama3.1:8b").unwrap_or_default();
manager.load_model_with_digest("llama3.1:8b", &path, None, digest)?;

// Generate with streaming callback
manager.generate("llama3.1:8b", "Hello!", &GenerateParams::default(), None, |token| {
    print!("{token}");
    true // return false to stop early
})?;
```

### Backend traits (advanced)

For direct backend access, implement `InferenceBackend` and add it to a custom `ModelManager`. The trait is:

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

## ModelStore (pulling)

```rust
use spindll::model_store::{ModelStore, FormatPreference};

let store = ModelStore::new(None);

// Auto: MLX-first on Apple Silicon, GGUF fallback. q4_k_m default for GGUF.
let path = store.pull("llama3.1:8b", None, FormatPreference::Auto)?;

// Force a specific format / quant
let path = store.pull("Qwen/Qwen2.5-3B-Instruct-GGUF", Some("q5_k_m"), FormatPreference::Gguf)?;

// FormatPreference::Mlx errors if no MLX equivalent is found
```

## Starting servers

```rust
use std::sync::Arc;

let manager = Arc::new(manager);
let store = Arc::new(ModelStore::new(None));

// gRPC server (always available)
spindll::grpc::start_server(50051, manager.clone(), store.clone()).await?;

// HTTP server (requires "http" feature)
#[cfg(feature = "http")]
spindll::http::start_http_server(8080, manager.clone(), store.clone()).await?;
```

## Embedding the HTTP router

To mount spindll's API into your own axum server alongside other routes:

```rust
use std::sync::Arc;

let manager = Arc::new(manager);
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

For simpler use cases that don't need multi-model management:

```rust
use spindll::engine::{Engine, GenerateParams};

let engine = Engine::load(path, None, 2048)?;
engine.generate("prompt", &GenerateParams::default(), |token| {
    print!("{token}");
    true
})?;
```
