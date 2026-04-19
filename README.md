# Spindll

**Spindle + LL(ama).** A Rust-native GGUF inference engine with model management.

A single binary that pulls models from Ollama's registry or HuggingFace, manages local storage, and serves streaming inference over gRPC. Multi-model, memory-aware, GPU-accelerated.

## Quick Start

```bash
cargo build --release

# Pull a model
spindll pull llama3.1:8b

# Or import from an existing Ollama installation
spindll import --from-ollama

# Start the server
spindll serve --port 50051 --budget 8G
```

Models are loaded and unloaded dynamically via the `Load` / `Unload` gRPC RPCs.

## Features

- **Pull from Ollama or HuggingFace** — auto-detects source from model name format
- **Streaming inference** — token-by-token output over gRPC via llama.cpp
- **Multi-model** — multiple models loaded concurrently, LRU eviction when budget exceeded
- **Model-native chat templates** — reads the template from GGUF metadata, no hardcoded formats
- **Memory-aware** — configurable budget, refuses to load models that won't fit
- **GPU acceleration** — Metal (macOS) auto-detected, CUDA (Linux) supported

## CLI

```
spindll pull <model>                  # pull from Ollama registry or HuggingFace
spindll list                          # show local models
spindll rm <model>                    # delete a local model
spindll serve [--port] [--budget]     # start gRPC server
spindll import --from-ollama          # migrate existing Ollama models
```

Model names follow Ollama conventions (`llama3.1:8b`, `qwen2:0.5b`) or HuggingFace repos (`TheBloke/Llama-3-8B-GGUF`).

## gRPC API

```protobuf
service Spindll {
  rpc Generate (GenerateRequest) returns (stream GenerateResponse);
  rpc Chat (ChatRequest) returns (stream ChatResponse);
  rpc Load (LoadRequest) returns (LoadResponse);
  rpc Unload (UnloadRequest) returns (UnloadResponse);
  rpc List (ListRequest) returns (ListResponse);
  rpc Status (StatusRequest) returns (StatusResponse);
}
```

## Architecture

```
CLI / gRPC API
      |
  Model Manager (multi-model slots, LRU eviction, memory budget)
      |
  Inference Engine (llama.cpp via llama-cpp-2, streaming, GPU offload)
      |
  Model Store (Ollama registry, HuggingFace, local registry)
```

## Storage

Models are stored in `~/.spindll/models/<repo>/<file>` — flat and inspectable. A JSON registry at `~/.spindll/registry.json` tracks downloaded models.

## Prerequisites

- [Rust toolchain](https://rustup.rs/) (stable)
- CMake

## License

Apache 2.0
