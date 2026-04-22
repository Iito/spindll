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

List all models in the store with GGUF metadata and loaded state.

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
    "architecture": "llama"
  }
]
```

| Field | Source |
|-------|--------|
| `model_name` | GGUF `general.name` metadata |
| `description` | GGUF `general.description` metadata |
| `architecture` | GGUF `general.architecture` metadata |
| `loaded` | Whether the model is currently in memory |

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

Download a model from Ollama registry or HuggingFace.

```bash
curl -X POST http://localhost:8080/pull \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b"}'
```

```json
{"status": "ok"}
```

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

**Prefill** -- encode a prompt into the KV cache without generating tokens. Used by orchestrators (e.g. Parley) to pre-warm the cache before the user's request arrives.

**Load / Unload** -- explicit model lifecycle control. Load returns `already_loaded: true` for idempotent preloading.

**Status** -- returns loaded models, memory info (RAM/VRAM), device list, and engine metrics (cache hit rate, tokens/second, request counts).

---

## Rust Library API

Add spindll to your project:

```toml
[dependencies]
spindll = { git = "https://github.com/Iito/spindll.git" }
```

### ModelManager

The primary entry point for multi-model inference:

```rust
use spindll::engine::{ModelManager, GenerateParams};
use spindll::model_store::ModelStore;

// Create a manager with 4096 context, auto GPU, 8GB budget
let mut manager = ModelManager::new(4096, None, 8_000_000_000)?;

// Enable KV cache (2GB)
manager.enable_kv_cache(2_000_000_000);

// Enable continuous batching (8 concurrent sequences per model)
manager.set_batch_slots(8);

// Load a model
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
