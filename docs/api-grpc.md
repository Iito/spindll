# gRPC API

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

## Key RPCs

**Generate** -- streaming text completion from a raw prompt. It never loads a model: the model must already be in memory (from Load, or an earlier Chat or Prefill), or the stream fails with `INTERNAL` (model not loaded).

**Chat** -- streaming chat completion from a message history. Applies the model's built-in chat template (falls back to ChatML). Loads the model if it isn't in memory; if that fails (an unknown name or a failed load), the stream ends with `NOT_FOUND`. Supports tool calling (`tools` / `tool_choice` in, `tool_calls` out; see [Tool calling](#tool-calling)) and an optional `encryption_key` that isolates this caller's KV cache entries and encrypts the ones on disk (GGUF only). The key is used only when it is exactly 32 bytes; any other length is ignored without an error. With the `vision` feature, `Message.parts` can carry images (`type: "image"`, raw bytes in `image_data`, up to 32 MiB each). Without it, image parts are dropped silently.

**Prefill** -- encode a prompt into the KV cache without generating tokens. Used by orchestrators to pre-warm the cache before the user's request arrives. Loads the model like Chat does, and takes the same `encryption_key`. On MLX models it does nothing and returns `tokens_cached: 0`.

`GenerateParams.stop` is ignored: Generate and Chat have no stop sequences.

**Load / Unload** -- explicit model lifecycle control. Load resolves the name to its canonical registry key and loads the model under that key. `LoadResponse.message` names the key (`loaded <key>`, or `<key> already loaded` with `already_loaded: true`, for idempotent preloading). A name that doesn't resolve fails with `NOT_FOUND`, and a failed load with `INTERNAL`. Unload takes an alias or the key, and fails with `NOT_FOUND` if the model isn't loaded. `LoadRequest` accepts optional fields for fine-grained lifecycle control:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | required | Model name or registry key |
| `gpu_layers` | int32 | 0 (CPU only) | GPU layers to offload, GGUF only. Unset or 0 keeps every layer on the CPU. A negative value uses the server default: `spindll serve --gpu-layers`, or every layer when the build has GPU support. `spindll load` sends -1 |
| `priority` | EvictionPriority | NORMAL | Eviction tier (see below) |
| `idle_reload_secs` | uint32 | 0 (disabled) | Seconds after eviction to attempt automatic reload |

**EvictionPriority** -- controls which models are evicted first when budget is exceeded. Low evicts first, LRU tiebreak within tier.

```protobuf
enum EvictionPriority {
  PRIORITY_NORMAL = 0;
  PRIORITY_LOW = 1;
  PRIORITY_HIGH = 2;
}
```

`PRIORITY_HIGH` models are evicted last; `PRIORITY_LOW` models are evicted before any normal-priority model regardless of recency. Combine with `idle_reload_secs` to keep a model warm: if it gets evicted under pressure, the manager will reload it once memory frees up.

**List** -- returns local models with their `format`, `base_model`, and `display_name`, plus a per-host `prefer_format` hint at the response level (`"mlx"` on Apple Silicon builds with the `mlx` feature, `"gguf"` elsewhere). Clients should prefer `display_name` over `name` for picker UIs and use `base_model` to group format variants of the same logical model. `quantization` and `last_used` are always empty.

**Pull** -- download a model. `PullRequest.quantization` is honored when set; empty triggers the q4_k_m-first priority picker. The server handler always uses `FormatPreference::Auto` (MLX first on Apple Silicon builds with the `mlx` feature, GGUF fallback). The stream sends one frame when the pull starts and one with `done: true` when it finishes, with no byte progress in between.

**Delete** -- not implemented: returns `UNIMPLEMENTED`. Use HTTP `DELETE /models/{id}` or `spindll rm` instead.

**Status** -- returns loaded models, RAM figures (the VRAM fields of `MemoryInfo` are always 0), the device list (taken from the build's features, not detected), and engine metrics (cache hit rate, tokens/second, request counts).

## Tool calling

`ChatRequest` carries the same tool calling as the OpenAI `/v1` surface. The model's chat template renders the tools when it supports them; injecting them into the system prompt is the fallback (see [api-openai.md](api-openai.md#tool--function-calling)):

| Field | Type | Description |
|-------|------|-------------|
| `tools` | repeated `Tool` | Tool definitions (`name`, `description`, `parameters_json` — a JSON Schema string) |
| `tool_choice` | string | `"none"` / `"auto"` / `"required"` / a function name |

Parsed calls arrive on the final `ChatResponse` (the `done: true` frame):

| Field | Type | Description |
|-------|------|-------------|
| `tool_calls` | repeated `ToolCall` | Parsed calls (`id`, `name`, `arguments` JSON string) |
| `finish_reason` | string | `"tool_calls"` when a call fired, else `"stop"` |

```protobuf
message Tool {
  string name = 1;
  string description = 2;      // optional
  string parameters_json = 3;  // JSON Schema for the arguments (optional)
}

message ToolCall {
  string id = 1;
  string name = 2;
  string arguments = 3;  // JSON string (OpenAI shape)
}
```

Without tools, tokens stream as usual. With tools active, nothing streams during generation: the server buffers the output and parses the calls. Any leftover prose then arrives as one `done: false` frame, followed by the final `done: true` frame, which carries the calls and an empty `token`.
