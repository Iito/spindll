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

**Generate** -- streaming text completion from a raw prompt.

**Chat** -- streaming chat completion from a message history. Applies the model's built-in chat template (falls back to ChatML). Supports an optional `encryption_key` for encrypted KV cache isolation.

**Prefill** -- encode a prompt into the KV cache without generating tokens. Used by orchestrators to pre-warm the cache before the user's request arrives.

**Load / Unload** -- explicit model lifecycle control. Load returns `already_loaded: true` for idempotent preloading.

**List** -- returns local models with their `format`, `base_model`, and `display_name`, plus a per-host `prefer_format` hint at the response level (`"mlx"` on Apple Silicon, `"gguf"` elsewhere). Clients should prefer `display_name` over `name` for picker UIs and use `base_model` to group format variants of the same logical model.

**Pull** -- download a model. `PullRequest.quantization` is honored when set; empty triggers the q4_k_m-first priority picker. The server handler always uses `FormatPreference::Auto` (MLX-first on Apple Silicon, GGUF fallback).

**Status** -- returns loaded models, memory info (RAM/VRAM), device list, and engine metrics (cache hit rate, tokens/second, request counts).
