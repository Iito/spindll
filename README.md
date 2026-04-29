# Spindll

**Spindle + LL(ama).** A Rust-native GGUF and MLX inference engine with model management.

A single binary that pulls models from Ollama's registry or HuggingFace, manages local storage, and serves streaming inference over gRPC and HTTP. Multi-model, memory-aware, GPU-accelerated, with an OpenAI-compatible API. On Apple Silicon, runs MLX models natively via a Swift bridge in addition to GGUF via llama.cpp.

## Quick Start

```bash
# Build with HTTP support
cargo build --release --features cli,http

# Pull a model
spindll pull llama3.1:8b

# Or import from an existing Ollama installation
spindll import --from-ollama

# Start the server (gRPC on 50051, HTTP on 8080)
spindll serve
```

Models are loaded automatically on first request, or explicitly via the `Load` RPC / `POST /load` endpoint.

## Features

- **Pull from Ollama or HuggingFace** -- auto-detects source from model name format; on Apple Silicon, resolves to MLX-format models when available, falls back to GGUF
- **Pluggable backends** -- `InferenceBackend` trait dispatches to `llama.cpp` for GGUF and `mlx-swift-lm` for MLX, with an extension point for new engines
- **Smart quant default** -- `pull` without `--quant` picks q4_k_m by priority list (q4_k_m > q5_k_m > q4_0 > … > fp16) instead of grabbing the first GGUF in the repo
- **Streaming inference** -- token-by-token output over gRPC, HTTP/SSE, or OpenAI-compatible API
- **OpenAI-compatible API** -- `/v1/chat/completions`, `/v1/completions`, and tool/function calling for AnythingLLM, Open WebUI, and any OpenAI client
- **Multi-model** -- multiple models loaded concurrently, LRU eviction when budget exceeded
- **Continuous batching** -- concurrent requests to the same model share a single context via sequence IDs
- **KV cache** -- disk-backed prefix caching with optional ChaCha20-Poly1305 encryption at rest
- **Chat template fallback** -- reads template from GGUF metadata, falls back to ChatML for models without one
- **GGUF metadata** -- extracts model name, description, and architecture from file headers
- **Memory-aware** -- configurable budget; budget-aware n_ctx auto-resolution at load time prevents silent OOMs
- **GPU acceleration** -- Metal (macOS) auto-detected, CUDA / Vulkan (Linux) supported
- **Embeddable** -- use as a Rust crate in your own project, no subprocess needed

## CLI

```
spindll pull <model> [flags]          # pull from Ollama registry or HuggingFace
spindll list                          # show local models with metadata
spindll rm <model>                    # delete a local model
spindll run [options] <model> "prompt" # one-shot inference (no server)
spindll bench <model> [other]         # benchmark one or two models (any format)
spindll serve [options]               # start gRPC + HTTP server
spindll import --from-ollama          # migrate existing Ollama models
spindll status                        # query a running server
```

Model names follow Ollama conventions (`llama3.1:8b`, `qwen2:0.5b`) or HuggingFace repos (`TheBloke/Llama-3-8B-GGUF`, `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`).

### Pull options

```
--quant <STR>              Pick a specific quant (e.g. q4_k_m, q5_k_m, fp16). Without
                           this flag, the picker prefers q4_k_m by default.
--gguf                     Force GGUF, skip MLX resolution on Apple Silicon
--mlx                      Force MLX, error if no MLX equivalent is found
```

### Serve options

```
--port <PORT>              gRPC port [default: 50051]
--http-port <PORT>         HTTP/SSE port [default: 8080]
--ctx-size <N>             Context window size [default: 2048]
--gpu-layers <N>           GPU layers (omit to auto-detect)
--budget <SIZE>            Memory budget, e.g. "8G". Default: full live availability
                           (free + inactive + purgeable + speculative on macOS;
                           sysinfo's available_memory elsewhere). Pass "0" to use
                           total RAM (trust unified-memory paging).
--kv-cache [<SIZE>]        Enable KV cache for prompt prefixes [default: 2G]
--batch-slots <N>          Concurrent sequence slots per model [default: 0 = disabled]
--ram-cache [<SIZE>]       Keep recently-evicted models warm in RAM [default: 4G; no-op on macOS]
```

### Run options

```
--ctx-size <N>             Context window [default: 0 = budget-safe auto]
--budget <SIZE>            Memory budget, e.g. "8G"; omit=live RAM, "0"=total RAM
--kv-cache [<SIZE>]        Enable KV cache for prompt prefixes [default: 2G]
```

While `serve` is running it writes a JSON lockfile (`pid`, `grpc_port`, `http_port`) to the system temp dir. `spindll status` reads this file to auto-detect the port, so `--port` is optional when a local server is running. Stale lockfiles are cleaned automatically when the referenced PID is no longer alive.

## API

Full API reference covering HTTP, OpenAI-compatible `/v1`, gRPC, and Rust library usage: **[docs/API.md](docs/API.md)**

Quick summary of available interfaces:

| Interface | Port | Feature flag | Use case |
|-----------|------|-------------|----------|
| gRPC | 50051 | none (always on) | Programmatic access, mesh integrations |
| HTTP/SSE | 8080 | `http` | Web frontends, custom integrations |
| OpenAI `/v1` | 8080 | `http` | AnythingLLM, Open WebUI, any OpenAI client (chat, completions, tool calling) |

## Using as a Rust library

```toml
[dependencies]
spindll = { git = "https://github.com/Iito/spindll.git" }
```

```rust
use spindll::engine::{ModelManager, GenerateParams};
use spindll::model_store::ModelStore;

let store = ModelStore::new(None);
let path = store.resolve_model_path("llama3.1:8b")?;
let digest = store.resolve_model_digest("llama3.1:8b").unwrap_or_default();

let manager = ModelManager::new(2048, None, 0)?;
manager.load_model_with_digest("llama3.1:8b", &path, None, digest)?;

manager.generate("llama3.1:8b", "Hello!", &GenerateParams::default(), None, |token| {
    print!("{token}");
    true
})?;
```

See [docs/API.md](docs/API.md) for the full library API including batch scheduling, KV cache, and server startup.

## Architecture

```
CLI / gRPC / HTTP+SSE / OpenAI /v1
              |
    Model Manager (multi-model slots, LRU eviction, memory budget)
         |                |
  Batch Scheduler     Per-request context
  (continuous batching,   (KV cache, encryption)
   sequence pooling)
         |                |
    Inference Backends (`InferenceBackend` trait)
       │                                  │
   llama.cpp via llama-cpp-2     mlx-swift-lm via Swift FFI
   (GGUF, all platforms,         (MLX, Apple Silicon only,
    GPU offload)                  --features mlx)
              |
    Model Store (Ollama registry, HuggingFace, GGUF metadata, local registry)
```

## Storage

Models are stored in `~/.spindll/models/<repo>/<file>`. A JSON registry at `~/.spindll/registry.json` tracks downloaded models with GGUF metadata. KV cache files are stored in `~/.spindll/cache/`.

## Feature flags

| Flag | Description |
|------|-------------|
| `cli` | Standalone binary (clap argument parsing, pretty logging) |
| `http` | HTTP/SSE server with OpenAI-compatible `/v1` API (axum) |
| `mlx` | MLX Swift backend on `aarch64-apple-darwin` (links against the bundled `mlx_bridge` Swift package) |
| `cuda` | CUDA GPU support in llama.cpp |
| `metal` | Metal GPU support in llama.cpp |
| `vulkan` | Vulkan GPU support in llama.cpp |

The gRPC server and core engine are always compiled -- no feature flag needed for library consumers or for embedding spindll in another binary.

## Prerequisites

- [Rust toolchain](https://rustup.rs/) (stable, edition 2024)
- CMake (for llama.cpp compilation)
- Swift toolchain ≥ 5.9 and Xcode command-line tools (only when building with `--features mlx` on Apple Silicon)

## License

Apache 2.0
