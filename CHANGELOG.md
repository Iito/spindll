# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.5.1] - 2026-05-30

### Fixed

- **Memory budget live re-evaluation** — stale memory budget snapshot no longer prevents model loads after external processes free RAM. The budget is now re-evaluated on each load attempt, respecting explicit `--budget` flags without clamping to stale available memory.

## [0.5.0] - 2026-05-23

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
- **Per-model eviction priority + idle-reload watcher** — models can be pinned or
  deprioritised for eviction; idle-reload watches previously-loaded models and
  brings them back when memory permits.
- **MLX prompt KV cache** — prefix caching for MLX models with fused chat generate,
  matching the GGUF backend's disk-backed cache.
- **MLX chat template support** — reads Jinja chat templates via the Swift bridge,
  falling back to ChatML when the model ships without one.
- **`spindll search`** — search for models across HuggingFace and Ollama registries,
  ranked by host hardware compatibility (preferred format first, models that fit
  in available RAM before those that don't, then by download count).
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
- **Standalone binary** — embedded `mlx.metallib` in binary for standalone installs.
- `docs/mlx-bridge.md` documenting the `mlx_bridge/` Swift package: C ABI,
  prompt KV cache, build pipeline, and Rust FFI integration.

### Changed

- `ModelManager` stores `Box<dyn BackendModel>` instead of raw `LlamaModel` + `LlamaContext`.
- `ModelStore::pull` signature gains a `FormatPreference` argument
  (`Auto` / `Gguf` / `Mlx`); existing gRPC and HTTP handlers pass `Auto`.
- CLI `run` and `bench` commands dispatch through `backend_for_format()` instead of
  separate per-format code paths.
- `run` command routes through `ModelManager` instead of dispatching to backends
  directly; gains `--ctx-size` and `--budget` flags.
- Context window sizing moved from the manager into each backend's `load_model`, threaded
  through `BackendLoadParams.memory_budget`.
- `BackendLoadParams` gains a `memory_budget: u64` field; pass `0` for live-tracking auto-mode.
- Default memory budget no longer applies a 20% reserve — `available_memory_platform` already
  excludes wired/active pages so the reserve was double-counting OS overhead.
- `pull` default GGUF picker prefers q4_k_m (was: first file in repo, often fp16).
- `bench` command gated from release builds.
- Bench throughput reporting separates decode tok/s from total tok/s.

### Fixed

- `BackendAlreadyInitialized` error on chat requests after engine startup —
  `LlamaBackend::init()` is now a `OnceLock` singleton in `backend::llamacpp`.
- Context window silently exceeding available memory — `resolve_n_ctx` clamps to
  `min(budget, available_ram)` and floors at 512 tokens.
- `n_batch == n_ctx` now set in every context-creation site — prevents GGML_ASSERT crashes on
  prompts longer than 512 tokens.
- `context_length` backfill now re-reads GGUF headers when the stored value is 0.
- **MLX KV cache corruption** — quantize cache snapshots before storing to prevent
  stale float buffers on cache hits; deep-copy `MambaCache` state to prevent
  shared-buffer corruption across generations.
- **MLX ChatML fallback** — models without a chat template no longer panic.
- MLX pull/run/rm bugs: import path resolution, model removal, incorrect format detection.
- MLX backend skipped gracefully when metallib not found next to binary.
- MLX directory size reported as 0 due to `symlink_metadata` not following HF hub symlinks.
- `platform_prefers_mlx` gated on the `mlx` feature flag.
- Reject MLX pull on unsupported platforms instead of downloading unusable weights.
- Split GGUF models: download all shards instead of only the first file.
- Suppress llama.cpp C-level log messages from leaking into terminal output.
- Xcode toolchain rpath for Swift concurrency dylib on macOS.
- Honor `--budget 0` flag and guard registry save against empty model stores.
- **Linux budget-aware loading** — batch scheduler weight in memory budget calculations,
  `clamp_budget_to_live` for over-allocation, `checked_div` in `resolve_n_ctx`.
- macOS available-memory now includes `speculative_count`, recovering 1–2 GB.

### MLX bridge correctness

- Synchronous `TokenIterator` replacing `AsyncStream<Generation>`.
- `extraEOSTokens` resolved through `convertTokenToId` so Gemma3, Phi, and SmolLM
  stop tokens work correctly.
- Final detokenizer flush for partial-UTF-8 bytes on maxTokens exits.
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
