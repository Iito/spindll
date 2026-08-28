# Changelog

All notable changes to this project will be documented in this file.

## [0.10.0] - 2026-08-28

### Added

- **`spindll serve <model>`, `spindll load`, `spindll unload`** — `serve` takes an optional model and preloads it before the ports open, so the server only accepts traffic once the model is ready and a failed load exits non-zero instead of leaving a warm-looking server with nothing in it. `load` / `unload` drive a running server over gRPC (`--port`, else the server's lockfile), which works on `--features cli` builds and servers started with `--http-port 0`.

### Fixed

- **A model is resident under exactly one name** — loading `llama3.1:8b` and requesting `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (the id `/v1/models` advertises) put the same weights in memory twice, each counted separately against the eviction budget. Every load path now resolves to the canonical registry key, and `unload` accepts either form.
- **`/load` and `/unload` no longer block an async worker** — both ran their synchronous work inline, stalling in-flight SSE streams on that thread; unload additionally re-reads the whole model to warm the RAM cache.
- **The server shuts down on SIGINT/SIGTERM** — nothing handled either signal, so the lockfile survived every ordinary Ctrl-C and the next client command read a dead server's ports out of it. The lockfile is now also only removed by the process that wrote it, so a second `serve` that fails to bind can't erase a healthy server's record.
- **`spindll serve --ram-cache <model>` no longer swallows the model name** — the optional flag value consumed the positional argument, silently falling back to the default cache size and preloading nothing.
- **`cargo build --features cli` compiles again** — the Anthropic and Responses dialect modules were not gated behind the `http` feature they depend on.

## [0.9.3] - 2026-08-22

### Fixed

- **`spindll pull` mirrors `chat_template.jinja`** — newer HuggingFace repos ship the chat template as a standalone Jinja file instead of a `tokenizer_config.json` field; the MLX download filter skipped it, so affected models silently lost their template. Models pulled before this fix need a re-pull to pick the file up.
- **Missing chat templates are loud, and vision fails fast** — the MLX bridge now probes the tokenizer's template once at load. A template-less model prints a prominent warning with the re-pull remedy; text requests keep the generic ChatML fallback (now with a hint); image requests are rejected with a clear error instead of silently generating image-blind — the template is what inserts the image tokens vision conditioning needs.

## [0.9.2] - 2026-08-21

### Fixed

- **`spindll rm` accepts the names `spindll list` prints** — multi-variant GGUF repos list as `<repo> (<quant>)` (e.g. `cjpais/llava-1.6-mistral-7b-gguf (q8_0)`), but deletion looked the raw string up as a registry key and failed with "not found". Name resolution now understands the display form, disambiguating variants by quant tag; the same resolution applies to `DELETE /models/{id}` over HTTP.
- **`spindll rm` fully removes downloaded artifacts** — deletion previously left the store-materialized `mmproj` projector behind (vision models like LLaVA ship one), let emptied repo/org directories accumulate, and leaked dangling symlinks. All three are cleaned up; directory pruning stops at the first level still holding another variant. Externally-imported models keep their confirmation prompt and never lose source files.

## [0.9.1] - 2026-08-21

### Fixed

- **macOS release artifact: the MLX Metal library loads on macOS 15 again** — `build.rs` now pins `-mmacosx-version-min=15.0` (the `mlx_bridge` platform floor) when compiling `mlx.metallib`. Built on the macos-26 runner it previously defaulted to Metal language 4.0, which macOS 15's loader rejects, so every MLX operation failed with "This library is using language version 4.0 which is not supported on this OS". GGUF was unaffected. v0.9.0's macOS artifact carries the defect; local builds on macOS 15 never did.

## [0.9.0] - 2026-08-21

### Added

- **Anthropic Messages API** — `POST /v1/messages` implements the Anthropic dialect on the shared engine chat path: string/block content, `system`, tools via `input_schema` with `tool_use`/`tool_result` round-trip, `tool_choice` (`auto`/`any`/`tool`/`none`), `stop_sequences` (streamed output holds back partial matches so a stop string split across tokens never leaks), image blocks behind the `vision` feature, and the Messages SSE grammar (named events, no `[DONE]`). Point Claude Code at spindll with `ANTHROPIC_BASE_URL=http://localhost:8080`.
- **OpenAI Responses API** — `POST /v1/responses` serves the stateless subset agent clients use with `store: false`: `input` as string or item array (`message`, `function_call`/`function_call_output`, `developer`→`system`), `instructions`, flat function tools, and the item-based SSE grammar with monotonic `sequence_number`s, terminating in `response.completed` / `response.incomplete` / `response.failed` per spec. Codex CLI works with its default `wire_api = "responses"`. `previous_response_id` is rejected with a clear 400; `input_image` parts are not mapped yet.
- **`reasoning_content` on `/v1/chat/completions`** (#75) — thinking models' `<think>`-delimited reasoning no longer floods `message.content`: it is split into `message.reasoning_content` (non-streaming) / `delta.reasoning_content` (streaming), the mlx-vlm / DeepSeek response shape. Handles both explicit tags and templates that force the think block open (probed once per model at load). Tool calls are parsed from the visible answer only. When a block was split, `usage.completion_tokens_details.reasoning_tokens` reports the reasoning share.

### Fixed

- **`finish_reason` reports `"length"`** on `/v1/chat/completions` when generation exhausted `max_tokens` — previously always `"stop"`, which hid truncation (e.g. a reasoning pass consuming the whole budget) from clients. (#75)
- **`spindll import <dir>` accepts sharded MLX models** — `config.json` plus any `*.safetensors` (e.g. `model-00001-of-00002.safetensors` + index) now qualifies; previously only a single-file `model.safetensors` did, even though sharded models served fine. (#75)
- **`no backend available for mlx format` names the remedy** — the error now says to rebuild with `--features mlx` (or that the platform is unsupported) instead of leaving the fix to be guessed. (#75)
- **Missing Metal Toolchain fails the MLX build fast** — `build.rs` probes `xcrun metal` before the multi-minute Swift build when `mlx.metallib` needs compiling, so the `xcodebuild -downloadComponent MetalToolchain` hint is the first thing the build says, not the last. (#75)

### Changed

- **VLM input images are pre-scaled to ~1M pixels by default** — vision prefill scales linearly with patch count while decode speed is unaffected, and models ship permissive budgets (`qwen3.5-9b-4bit` allows 16.8M px — ≈2 minutes of prefill for an uncapped 12 MP photo, vs ~5 s capped). `SPINDLL_VLM_MAX_PIXELS` overrides the budget; `0` disables the cap. Same-box control vs mlx-vlm 0.6.14 on identical weights: decode parity, faster prefill at every image size. (#75, #79)
- **mlx-swift-lm bumped to main@7871b09 (2026-08-17)** — picks up upstream's Qwen3.5 prefill rework, recovering VLM decode throughput (issue #75's repro: 8 → ~39 tok/s). The MLX bridge now requires a **Swift 6.3+ toolchain**: `build.rs` selects one automatically (honors `$TOOLCHAINS`, falls back to an installed swift.org toolchain) while metallib compilation stays on the xcode-select'd toolchain. (#76)

### Infrastructure

- **macOS CI runs on macos-26 / Xcode 26.6** — the MLX job moved runners to satisfy the Swift 6.3 floor; actionlint taught the new runner label. (#76)

## [0.8.0] - 2026-08-11

### Added

- **Faithful Jinja chat templating (GGUF)** — the llama.cpp backend renders the model's embedded `tokenizer.chat_template` with minijinja (Python-compatible string methods via pycompat; Hugging Face `trim_blocks`/`lstrip_blocks` whitespace) instead of llama.cpp's legacy substring formatter, which only recognized a fixed set of formats and errored on anything else. Falls back to the legacy formatter when a model ships no embedded template or the template can't be rendered. Brings GGUF to parity with the MLX backend, which already rendered real Jinja via swift-transformers.
- **Native tool templating** — `tools` / `tool_choice` now flow through the model's own chat template, so tool specs are rendered in the format each model was trained on rather than a generic injected preamble (kept only as a fallback for templates that don't declare `tools`). Works on both backends — the MLX Swift bridge passes tools to swift-transformers' `applyChatTemplate(messages:tools:)`. Extraction (`parse_tool_calls`) is unchanged.
- **Grammar-constrained tool decoding** — new opt-in `grammar` Cargo feature (enables `llama-cpp-2/common`). When `tool_choice` is `required` or names a function, a lazy GBNF grammar derived from the tool schemas (`json_schema_to_grammar`) forces the model to emit a syntactically valid tool call, triggered only after a tool-call opener (`<tool_call>` / `[TOOL_CALLS]` / `<|python_tag|>`) so plain text stays unconstrained. Off by default (no build cost).

### Changed

- **`BackendModel` trait (breaking)** — `apply_chat_template` and `generate_chat` now take `tools: &[ToolSpec]` and `tool_choice: &ToolChoice`; external implementers and callers of the trait must update.

## [0.7.4] - 2026-08-09

### Fixed

- **No more `HOME not set` panics** — home-directory resolution now falls back to `USERPROFILE` when `HOME` is unset (the norm in Windows shells), instead of panicking. Covers `import::ollama_dir()`, `import::hf_cache_dir()`, `ModelStore::new(None)`, and `KvCache::new()`; the Ollama and HuggingFace importers return a clear error when no home directory can be determined at all.

### Changed

- **`import::ollama_dir()` / `import::hf_cache_dir()` (breaking)** — now return `Option<PathBuf>` (`None` when no home directory can be determined) instead of a `PathBuf` that could only be produced by panicking.

## [0.7.3] - 2026-08-05

### Changed

- **llama-cpp-2 0.1.151 → 0.1.154** — binding-level upstream updates: bindings for the three missing KV-cache functions, sampler offloading to the backend, an accessor for tensor buffer-type overrides, `GGML_*` environment variables forwarded to the CMake build (e.g. `GGML_CPU_REPACK=OFF` for mmap-friendly weight layouts), Vulkan support for Android cross-builds, and an MSVC debug CRT link fix. The bundled llama.cpp advances from a June 7 to a July 30 upstream snapshot. (#74)

## [0.7.2] - 2026-08-02

### Fixed

- **MLX import into a fresh model store** — `import_from_hf` linked `models/<owner>/<repo>` without creating `models/<owner>` first, so importing an MLX model from the HuggingFace cache failed with ENOENT on a fresh store and aborted the whole import — including GGUF models staged earlier in the same run, since the registry is only written at the end. `link_or_copy_path` now creates the destination's parent directory, covering all call sites. (#72)

## [0.7.1] - 2026-07-12

### Added

- **Apache-2.0 licensing** — `LICENSE` file, SPDX headers on all source files, and a copyright notice in the README.

### Changed

- **llama-cpp-2 0.1.150 → 0.1.151** — binding-level upstream updates: MTP speculative-decoding support, a model-load progress callback on `LlamaModelParams`, an optional Intel MKL build feature, and a Vulkan build fix for `x86_64-pc-windows-gnu`. The bundled llama.cpp version is unchanged.

### Infrastructure

- **Automated release publishing** — `release.yml` opens the GitHub release (or RC pre-release) as soon as tests pass; each platform build uploads its artifact as it finishes. Release notes carry over from the newest RC of the same version, and superseded RC pre-releases are retired when the final tag lands.
- **Dependency auto-update PRs target `main`** — the daily llama-cpp-2 / mlx-swift upstream checkers now branch off and PR against `main`, so staged patch bumps release directly from `main` instead of riding `next`.

## [0.7.0] - 2026-06-14

### Added

- **Vision / multimodal inference** — chat requests can include images over gRPC (`Message.parts` / `ContentPart`) and the OpenAI HTTP API (`image_url` base64 `data:` URIs). GGUF vision via llama.cpp's `mtmd` API; MLX VLM decode path (Qwen2.5-VL) on Apple Silicon. Gated behind a new `vision` Cargo feature; auto-downloads the `mmproj` projector sibling and enforces a 32 MB per-image decode cap.
- **AnythingLLM native provider support** — enhanced OpenAI API with per-model metadata:
  - `GET /v1/models` now includes architecture, context_length, format, size_bytes, capabilities, created timestamp
  - `GET /v1/models/{id}` endpoint for per-model config queries
  - `GET /v1/status` endpoint for server status with model inventory
- **Run command chat template and system prompt** — `spindll run` now uses the model's chat template (via `generate_chat`) and injects a default system prompt ("You are a helpful assistant."). Add `--system` flag to override, `--max-tokens` to control output length.
- **Tool / function calling** — `tools` and `tool_choice` on the OpenAI `/v1/chat/completions` API and the gRPC `Chat` RPC. Prompt-injection based: tool specs are rendered into the system prompt and the model's output (`<tool_call>` / Hermes / Llama-3.1 / Mistral wrappers) is parsed back into OpenAI-shaped calls. `tool_choice` is honored (`none` disables tools; `required` / named instruct the model); streaming emits incremental `tool_calls` deltas.
- **CLI `ls` / `remove` aliases** — `spindll ls` aliases `list`; `spindll remove` aliases `rm`.
- **Sidecar chat-template override** — the llama.cpp backend now loads a `<model-file>.jinja` sidecar next to the model and uses it in place of the GGUF's embedded chat template. Contents may be a raw Jinja template or a built-in name (e.g. `gemma`, `chatml`). Mirrors llama.cpp's `--chat-template-file` and lets a model shipping a broken or unusable template be corrected without re-quantizing.

### Refactored

- **Budget calculation clarity** — extracted `MemoryBudget::load_budget_with_scheduler()` method to clarify the interaction between configured budgets, available RAM, and scheduler overhead. Added regression tests for default-mode clamping behavior (PR #47 follow-up).
- **Max-tokens default handling** — `--max-tokens` in run command is optional (no clap default) to avoid duplicating `GenerateParams::default().max_tokens` (512). Falls back to library default when not provided (PR #46 follow-up).

### Fixed

- **Gemma chat template** — fold the `system` role into the first user turn when the model's template rejects a standalone system role, fixing `failed to apply chat template: ffi error -1` on Gemma. Also unblocks tool calling on Gemma (which injects a system preamble).
- **Double BOS on chat prompts** — chat templates emit `bos_token` themselves, so tokenizing the rendered prompt with `AddBos::Always` prepended a second BOS. The duplicate is now collapsed (raw, non-templated prompts still get a BOS). A double BOS degrades output and, on models whose BOS/EOS differ from the Llama defaults (e.g. Gemma), can make the model emit end-of-turn immediately.

## [0.6.0] - 2026-05-30

### Added

- **Model source tracking** — registry tracks how each model entered Spindll (downloaded from Ollama/HuggingFace, imported from local Ollama/HF cache, or manually imported). Five source types enable multi-engine compatibility.
- **Extended import command** — `spindll import --from-hf` discovers models in local HuggingFace cache; `spindll import "/path/to/model"` validates and symlinks arbitrary GGUF/MLX files.
- **Smart model cleanup** — `spindll rm` auto-deletes models Spindll owns, prompts for externally-managed models. `--purge` flag skips confirmation; users can respond "no" to keep models registered.
- **Registry v2 migration** — auto-detects and infers source types for existing models on first load, ensuring backward compatibility with 0.5.1 registries.
- **Embeddings support** — `POST /v1/embeddings` OpenAI-compatible endpoint; MLX and GGUF embedding extraction with input validation and rate limiting.
- **MLX prompt cache disk tier** — extends prompt cache beyond RAM with adaptive quantization; longest-prefix-match reuse (not exact match); freshest cache kept near-lossless.
- **Search command** — `spindll search <query>` across HuggingFace + Ollama with hardware-aware ranking, quant-aware sizing, FITS column, and `--format`/`--sort` flags.
- **Device/GPU selection** — `--device` flag for serve/run commands; device-aware backend selection and per-model GPU pinning.
- **Improved benchmarking** — before-after merge gate mode, separate decode tok/s from total tok/s, auto-detect platform features.

### Fixed

- **KV cache hardening** — fixed cross-tenant RAM leak, sampling crashes (hit and miss paths), hardened restore paths.
- **Embeddings refinements** — array-len cap, right-size n_batch, OpenAI compat fixes, separate error counters.
- **Search ranking** — rank by total system RAM on dedicated GPU, current available on shared; backfill HF model sizes from API and safetensors metadata.

## [0.5.1] - 2026-05-30

### Fixed

- **Memory budget live re-evaluation** — stale memory budget snapshot no longer prevents model loads after external processes free RAM. The budget is now re-evaluated on each load attempt, respecting explicit `--budget` flags without clamping to stale available memory.

### Documentation

- **Linux build and runtime dependencies** — documented complete dev build requirements for bare Ubuntu (libssl-dev, clang, libclang-dev, etc.) and end-user runtime dependencies (libssl3, libgomp1) to address CI/bare-Ubuntu parity gaps.

## [0.5.0] - 2026-05-10

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
- Bench memory footprint measurement switched from raw `phys_footprint_mb` FFI to the `memory-stats` crate.

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
- README links to `docs/API.md` (removed in the v0.5.0 docs split) now point to `docs/README.md` and `docs/api-rust.md`.

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
