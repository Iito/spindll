# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.5.0] - 2026-04-28

### Added

- **Multi-backend trait system** — pluggable `InferenceBackend` and `BackendModel` traits replace
  hardcoded llama.cpp calls, enabling new backends without touching the manager or CLI.
- **MLX Swift backend** — native Apple Silicon inference via MLX Swift, auto-selected on
  `aarch64-apple-darwin` when the `mlx` feature is enabled.
- **Platform-aware model pulling** — `spindll pull` resolves MLX-format models on Apple Silicon
  via `mlx-community` repos, falling back to GGUF elsewhere. Explicit `--gguf` / `--mlx` flags
  override auto-detection.
- **MLX repo resolver** — maps Ollama names and HuggingFace GGUF repos to their
  `mlx-community` equivalents using a hardcoded table + HF API search fallback.
- **Registry versioning** — `registry.json` carries a `version` field with automatic forward
  migration on load and a read-only guard for files written by newer spindll versions.
- **`base_model` field** — canonical model identity in the registry, enabling cross-format name
  resolution (e.g. find the MLX entry when the user types an Ollama name).
- **Format column in `spindll list`** — shows `gguf` or `mlx` next to each model.
- **`resolve_model_format` API** — callers can query a model's on-disk format before loading.
- **Format-aware bench command** — `spindll bench` dispatches through the backend trait,
  supporting both GGUF and MLX models.
- **`download_hf_auto`** — single HuggingFace download entry point that auto-detects GGUF vs MLX
  from repo contents.
- **gRPC `ModelInfo` fields** — `format`, `base_model`, and `display_name` per model;
  `prefer_format` (per-host hint) at the `ListResponse` level. Clients should prefer
  `display_name` over `name` for picker UIs.
- **Quant priority list** — q4_k_m > q5_k_m > q4_0 > … > fp16 ranking when no `--quant` is
  specified, picked from any HuggingFace repo with multiple GGUF variants.
- **Display-name disambiguation** — registry entries from the same repo with different quants
  surface as `Repo (q4_k_m)` vs `Repo (fp16)` instead of duplicate labels.
- **Dynamic column widths in `spindll list`** — MODEL and ARCH columns size to their longest
  entry so `mlx-community/...` paths don't wrap or truncate.

### Changed

- `ModelManager` stores `Box<dyn BackendModel>` instead of raw `LlamaModel` + `LlamaContext`.
- `ModelStore::pull` signature gains a `FormatPreference` argument
  (`Auto` / `Gguf` / `Mlx`); existing gRPC and HTTP handlers pass `Auto`.
- CLI `run` and `bench` commands dispatch through `backend_for_format()` instead of
  separate per-format code paths.
- KV cache byte estimation in `total_loaded_bytes` downcasts to `LlamaCppModel` — non-GGUF
  backends report weight size only.
- Context window sizing moved from the manager into each backend's `load_model`, threaded
  through `BackendLoadParams.memory_budget`.
- `BackendLoadParams` gains a `memory_budget: u64` field; pass `0` for no backend cap.
- Default memory budget no longer applies a 20% reserve — `available_memory_platform` already
  excludes wired/active pages so the reserve was double-counting OS overhead.
- `pull` default GGUF picker prefers q4_k_m (was: first file in repo, often fp16).

### Fixed

- `BackendAlreadyInitialized` error on chat requests after engine startup —
  `LlamaBackend::init()` is now a `OnceLock` singleton in `backend::llamacpp`.
- Context window silently exceeding budget — `resolve_n_ctx` clamps auto context
  sizing to the backend load budget and floors at 512 tokens.
- `n_batch == n_ctx` now set in every context-creation site (Engine, BatchScheduler,
  manager's KV-cached path, `LlamaCppModel::generate`) — prevents GGML_ASSERT crashes on
  prompts longer than 512 tokens.
- `context_length` backfill now re-reads GGUF headers when the stored value is 0.
- MLX directory size reported as 0 due to `symlink_metadata` not following HF hub symlinks.
- macOS available-memory calculation now includes `speculative_count` (file-cache prefetch
  pages the kernel reclaims first under pressure) — recovers 1–2 GB that was understated.

### MLX bridge correctness

- Switch generation loop from `AsyncStream<Generation>` to synchronous `TokenIterator`,
  matching mlx-swift-lm's `runSynchronousGenerationLoop`.
- `extraEOSTokens` now resolved through `convertTokenToId` so Gemma3 `<end_of_turn>`,
  Phi `<|end|>`, and SmolLM `<turn|>` tokens actually stop generation.
- Final detokenizer flush after the loop so partial-UTF-8 bytes aren't dropped on
  maxTokens mid-codepoint exits.
- `Stream().synchronize()` before `perform` teardown to drain in-flight async evals.
- `Memory.cacheLimit = 64MB` moved into `mlx_model_load` to amortise across runs.

## [0.4.0] - 2026-04-28

### Added

- CUDA, Metal, and Vulkan GPU backend feature flags (`--features cuda`, etc.).
- Windows support with hard-link + copy fallback for model store.
- Cross-platform CI (Linux, macOS, Windows).

## [0.3.0] - 2026-04-27

### Added

- Multi-model manager with LRU eviction and memory budgeting.
- Continuous batching scheduler for concurrent request multiplexing.
- Encrypted KV cache with model-digest keying.
- RAM cache for fast model reload after eviction.
- HTTP/SSE server with OpenAI-compatible `/v1/chat/completions` API.
- Ollama registry pull (native blob protocol).
- Ollama model import via symlink discovery.
- GGUF metadata reading (`general.name`, `general.architecture`, context length).
- Lockfile-based port auto-detection between CLI and server.

### Fixed

- Prompt token count not tracked in usage stats.
- Registry not updated on model removal.

## [0.2.0] - 2026-04-26

### Added

- gRPC server with generate, chat, list, status, load, and unload RPCs.
- Auto-detect pull source from model name format (Ollama vs HuggingFace).
- Model chat template support (replaces hardcoded prompt format).
- Memory budget enforcement with system memory detection.

## [0.1.0] - 2026-04-25

### Added

- Initial release.
- Model store with HuggingFace GGUF download and local registry.
- GPU detection and layer offloading.
- Streaming token generation via llama.cpp.
- `pull`, `list`, `rm` CLI commands.
