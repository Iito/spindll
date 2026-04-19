# Spindll

**Spindle + LL(ama).** A Rust-native GGUF inference engine with model management.

Spindll aims to be a single binary that pulls models from HuggingFace, manages local storage, and serves inference over gRPC. Work in progress.

## Status

Early development. The project scaffolding is in place with module stubs for:

- **Model Store** — download and manage GGUF models from HuggingFace
- **Inference Engine** — llama.cpp bindings (not yet implemented)
- **Scheduler** — memory-aware model loading (not yet implemented)
- **gRPC API** — streaming inference server (not yet implemented)
- **CLI** — `pull`, `list`, `rm`, `serve`, `run`, `import`, `status`

## Build

```bash
git clone https://github.com/Iito/spindll.git
cd spindll
cargo build --release
```

### Prerequisites

- [Rust toolchain](https://rustup.rs/) (stable)
- CMake (for llama.cpp compilation, when engine is implemented)

## Storage

Models are stored in `~/.spindll/models/<repo>/<file>` — flat and inspectable.

## License

Apache 2.0
