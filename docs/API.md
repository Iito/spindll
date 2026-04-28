# Spindll API Reference

Spindll exposes three interfaces: an HTTP/SSE API (with OpenAI compatibility), a gRPC API, and a Rust library API. All three share the same underlying engine.

## HTTP API

The HTTP server runs on port 8080 by default (requires the `http` feature flag). All endpoints return JSON. CORS is enabled for all origins.

### GET /health

Health check.

```bash
curl http://localhost:8080/health
```

```json
{"status": "ok"}
```

### GET /models

List all models in the store with format, metadata, and loaded state.

```bash
curl http://localhost:8080/models
```

```json
[
  {
    "name": "ollama/llama3.1/8b.gguf",
    "size_bytes": 4920000000,
    "quantization": "",
    "digest": "sha256:abc123...",
    "loaded": true,
    "model_name": "Llama 3.1 8B Instruct",
    "description": "",
    "architecture": "llama",
    "context_length": 8192,
    "format": "gguf",
    "base_model": "llama-3.1-8b-instruct",
    "display_name": "llama3.1:8b"
  },
  {
    "name": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "size_bytes": 4526682492,
    "loaded": false,
    "architecture": "llama",
    "context_length": 0,
    "format": "mlx",
    "base_model": "llama-3.1-8b-instruct",
    "display_name": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
  }
]
```

| Field | Source |
|-------|--------|
| `model_name` | GGUF `general.name` metadata (empty for MLX) |
| `description` | GGUF `general.description` metadata |
| `architecture` | `general.architecture` (GGUF) or `model_type` from `config.json` (MLX) |
| `context_length` | Trained context size from GGUF metadata (effective size capped by `--ctx-size` when loaded; 0 for MLX) |
| `loaded` | Whether the model is currently in memory |
| `format` | `"gguf"` or `"mlx"` — pick the matching backend automatically |
| `base_model` | Stable cross-format identifier — same value for the GGUF and MLX variants of one logical model |
| `display_name` | Human-readable label for picker UIs; disambiguates same-repo quants (`Repo (q4_k_m)` vs `Repo (fp16)`) |

### POST /chat (SSE)

Chat inference with Server-Sent Events streaming. Auto-loads the model if not already in memory.

```bash
curl -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "hello"}],
    "params": {"max_tokens": 256, "temperature": 0.7}
  }'
```

Response is `text/event-stream`:

```
data: {"type":"token","content":"Hi"}

data: {"type":"token","content":" there"}

data: {"type":"done"}
```

Error events:

```
data: {"type":"error","error":"model not found"}
```

**Request body:**

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `model` | string | yes | |
| `messages` | array of `{role, content}` | yes | |
| `params.max_tokens` | integer | no | 512 |
| `params.temperature` | float | no | 0.8 |
| `params.top_p` | float | no | 0.95 |
| `params.top_k` | integer | no | 40 |
| `params.seed` | integer | no | 42 |

### POST /load

Load a model into memory.

```bash
curl -X POST http://localhost:8080/load \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b"}'
```

```json
{"already_loaded": false}
```

### POST /pull

Download a model from Ollama registry or HuggingFace. On Apple Silicon, attempts to resolve an MLX-format model first (e.g. `llama3.1:8b` → `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`), falling back to GGUF.

```bash
curl -X POST http://localhost:8080/pull \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b"}'
```

```json
{"status": "ok"}
```

**Request body:**

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `model` | string | yes | |
| `quantization` | string | no | Picker chooses by priority: `q4_k_m > q5_k_m > q4_0 > … > fp16` |

For HuggingFace repos with multiple GGUF variants, omitting `quantization` picks the lowest-ranked match (q4_k_m by default). Pass `"fp16"` (or any specific quant string) to override.

The HTTP `/pull` endpoint always uses `FormatPreference::Auto` — to force GGUF or MLX explicitly, use the CLI (`spindll pull --gguf` / `--mlx`) or the gRPC `Pull` RPC with library access.

### DELETE /models/{id}

Remove a model from disk (and unload from memory if loaded).

```bash
curl -X DELETE http://localhost:8080/models/ollama%2Fllama3.1%2F8b.gguf
```

```json
{"status": "ok"}
```

### POST /models/{id}/unload

Unload a model from memory without deleting it from disk.

```bash
curl -X POST http://localhost:8080/models/ollama%2Fllama3.1%2F8b.gguf/unload
```

```json
{"status": "ok"}
```

---

## OpenAI-compatible API (`/v1`)

Drop-in compatible with any OpenAI client. These endpoints follow the [OpenAI API specification](https://platform.openai.com/docs/api-reference).

### Client configuration

| Client | Setting |
|--------|---------|
| AnythingLLM | Provider: Custom (OpenAI Compatible), Base URL: `http://localhost:8080/v1` |
| Open WebUI | OpenAI API URL: `http://localhost:8080/v1` |
| Python openai SDK | `base_url="http://localhost:8080/v1"`, `api_key="any"` |
| curl | Base URL: `http://localhost:8080/v1` |

No API key is required. Clients that mandate one can use any non-empty string.

### GET /v1/models

```bash
curl http://localhost:8080/v1/models
```

```json
{
  "object": "list",
  "data": [
    {"id": "ollama/llama3.1/8b.gguf", "object": "model", "owned_by": "spindll"}
  ]
}
```

### POST /v1/chat/completions

#### Streaming (default)

```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": true
  }'
```

```
data: {"object":"chat.completion.chunk","model":"llama3.1:8b","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}

data: {"object":"chat.completion.chunk","model":"llama3.1:8b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

#### Non-streaming

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": false
  }'
```

```json
{
  "object": "chat.completion",
  "model": "llama3.1:8b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hi there!"},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 4,
    "total_tokens": 16
  }
}
```

**Request body:**

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `model` | string | yes | |
| `messages` | array of `{role, content}` | yes | |
| `stream` | boolean | no | true |
| `max_tokens` | integer | no | 512 |
| `temperature` | float | no | 0.8 |
| `top_p` | float | no | 0.95 |
| `seed` | integer | no | 42 |
| `tools` | array of tool objects | no | |
| `tool_choice` | string or object | no | (accepted, not yet enforced) |

#### Tool / function calling

Pass `tools` to enable function calling. Tool definitions are injected into the prompt and the model's output is parsed for tool call JSON.

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          },
          "required": ["city"]
        }
      }
    }],
    "stream": false
  }'
```

When the model calls a tool, the response uses `finish_reason: "tool_calls"`:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"Tokyo\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

Send tool results back as a `tool` role message:

```json
{
  "messages": [
    {"role": "user", "content": "What is the weather in Tokyo?"},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_abc123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}]},
    {"role": "tool", "tool_call_id": "call_abc123", "content": "{\"temp\": 22, \"condition\": \"sunny\"}"},
  ]
}
```

**Notes:**
- Tool calling works best with models fine-tuned for it (Llama 3.1+, Qwen 2.5+, Mistral v0.3+)
- `tool_choice` is accepted for compatibility but not yet used to constrain selection
- When streaming with tools, output is buffered to parse tool calls before sending

### POST /v1/completions

Raw text completion (no chat template applied). Use this for code completion, text continuation, and other non-chat tasks.

#### Streaming (default)

```bash
curl -N -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "prompt": "The capital of France is",
    "stream": true
  }'
```

```
data: {"object":"text_completion","model":"llama3.1:8b","choices":[{"index":0,"text":" Paris","finish_reason":null}]}

data: {"object":"text_completion","model":"llama3.1:8b","choices":[{"index":0,"text":"","finish_reason":"stop"}]}

data: [DONE]
```

#### Non-streaming

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "prompt": "The capital of France is",
    "stream": false
  }'
```

```json
{
  "object": "text_completion",
  "model": "llama3.1:8b",
  "choices": [
    {
      "index": 0,
      "text": " Paris, the city of light.",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 8,
    "total_tokens": 14
  }
}
```

**Request body:**

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `model` | string | yes | |
| `prompt` | string | yes | |
| `stream` | boolean | no | true |
| `max_tokens` | integer | no | 512 |
| `temperature` | float | no | 0.8 |
| `top_p` | float | no | 0.95 |
| `seed` | integer | no | 42 |

**Error format:**

```json
{
  "error": {
    "message": "description of what went wrong",
    "type": "server_error"
  }
}
```

---

## gRPC API

The gRPC server runs on port 50051 by default and is always available (no feature flag required).

```protobuf
service Spindll {
  rpc Generate (GenerateRequest) returns (stream GenerateResponse);
  rpc Chat (ChatRequest) returns (stream ChatResponse);
  rpc Prefill (PrefillRequest) returns (PrefillResponse);
  rpc Load (LoadRequest) returns (LoadResponse);
  rpc Unload (UnloadRequest) returns (UnloadResponse);
  rpc List (ListRequest) returns (ListResponse);
  rpc Pull (PullRequest) returns (stream PullProgress);
  rpc Delete (DeleteRequest) returns (DeleteResponse);
  rpc Status (StatusRequest) returns (StatusResponse);
}
```

See [`proto/spindll.proto`](../proto/spindll.proto) for full message definitions including all request/response fields.

### Key RPCs

**Generate** -- streaming text completion from a raw prompt.

**Chat** -- streaming chat completion from a message history. Applies the model's built-in chat template (falls back to ChatML). Supports an optional `encryption_key` for encrypted KV cache isolation.

**Prefill** -- encode a prompt into the KV cache without generating tokens. Used by orchestrators to pre-warm the cache before the user's request arrives.

**Load / Unload** -- explicit model lifecycle control. Load returns `already_loaded: true` for idempotent preloading.

**List** -- returns local models with their `format`, `base_model`, and `display_name`, plus a per-host `prefer_format` hint at the response level (`"mlx"` on Apple Silicon, `"gguf"` elsewhere). Clients should prefer `display_name` over `name` for picker UIs and use `base_model` to group format variants of the same logical model.

**Pull** -- download a model. `PullRequest.quantization` is honored when set; empty triggers the q4_k_m-first priority picker. The server handler always uses `FormatPreference::Auto` (MLX-first on Apple Silicon, GGUF fallback).

**Status** -- returns loaded models, memory info (RAM/VRAM), device list, and engine metrics (cache hit rate, tokens/second, request counts).

---

## Rust Library API

Add spindll to your project:

```toml
[dependencies]
spindll = { git = "https://github.com/Iito/spindll.git" }
```

### ModelManager

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

#### Backend traits (advanced)

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

### ModelStore (pulling)

```rust
use spindll::model_store::{ModelStore, FormatPreference};

let store = ModelStore::new(None);

// Auto: MLX-first on Apple Silicon, GGUF fallback. q4_k_m default for GGUF.
let path = store.pull("llama3.1:8b", None, FormatPreference::Auto)?;

// Force a specific format / quant
let path = store.pull("Qwen/Qwen2.5-3B-Instruct-GGUF", Some("q5_k_m"), FormatPreference::Gguf)?;

// FormatPreference::Mlx errors if no MLX equivalent is found
```

### Starting servers

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

### Embedding the HTTP router

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

### Engine (single-model)

For simpler use cases that don't need multi-model management:

```rust
use spindll::engine::{Engine, GenerateParams};

let engine = Engine::load(path, None, 2048)?;
engine.generate("prompt", &GenerateParams::default(), |token| {
    print!("{token}");
    true
})?;
```
