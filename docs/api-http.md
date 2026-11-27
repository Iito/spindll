# HTTP API

The HTTP server runs on port 8080 by default (requires the `http` feature flag). All endpoints return JSON. CORS is enabled for all origins.

## GET /health

Health check.

```bash
curl http://localhost:8080/health
```

```json
{"status": "ok"}
```

## GET /models

List all models in the store with metadata and loaded state.

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
    "context_length": 8192
  },
  {
    "name": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "size_bytes": 4526682492,
    "quantization": "",
    "digest": "sha256:def456...",
    "loaded": false,
    "model_name": "",
    "description": "",
    "architecture": "llama",
    "context_length": 0
  }
]
```

| Field | Source |
|-------|--------|
| `quantization` | Always an empty string |
| `model_name` | GGUF `general.name` metadata. For MLX, `_name_or_path` from `config.json` (empty when absent), or the repo id for a model imported from the HuggingFace cache |
| `description` | GGUF `general.description` metadata |
| `architecture` | `general.architecture` (GGUF) or `model_type` from `config.json` (MLX) |
| `context_length` | Trained context size from GGUF metadata, 0 for MLX. A loaded model reports its effective size instead, capped by `--ctx-size` |
| `loaded` | Whether the model is currently in memory |

This endpoint has no `format`, `base_model` or `display_name`. The gRPC `List` RPC returns all three, and `GET /v1/models` returns `format`.

## POST /chat (SSE)

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

## POST /load

Load a model into memory.

```bash
curl -X POST http://localhost:8080/load \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b"}'
```

```json
{"already_loaded": false}
```

**Request body:**

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `model` | string | yes | |
| `gpu_layers` | integer | no | server default |

`model` takes any name the store resolves (`llama3.1:8b`, a registry key, a repo name). The model loads under its canonical registry key, and `already_loaded` is checked against that key. `gpu_layers` applies to GGUF models only. Leaving it out, or passing a negative value, uses the server default: `spindll serve --gpu-layers`, or every layer when the build has GPU support.

A name that doesn't resolve returns 404, and a failed load returns 500, both as `{"error": "..."}`.

## POST /pull

Download a model from Ollama registry or HuggingFace. On Apple Silicon builds with the `mlx` feature, attempts to resolve an MLX-format model first (e.g. `llama3.1:8b` → `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`), falling back to GGUF. On those builds `quantization` also selects the MLX variant (default `4bit`).

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

The HTTP `/pull` endpoint always uses `FormatPreference::Auto`, and so does the gRPC `Pull` RPC. To force GGUF or MLX, use the CLI (`spindll pull --gguf` / `--mlx`).

## DELETE /models/{id}

Remove a model from disk and from the registry.

```bash
curl -X DELETE http://localhost:8080/models/ollama%2Fllama3.1%2F8b.gguf
```

```json
{"status": "ok"}
```

`{id}` takes any name form, but a name that matches several models is refused until it names one exactly. An unknown or ambiguous name returns 404 with `{"error": "..."}`.

The model is unloaded first only when `{id}` is its exact registry key. Under an alias such as `llama3.1:8b`, the files are deleted and the model stays loaded.

A model added with `spindll import` (linked from Ollama's store, the HuggingFace cache or a local path) is not deleted outright. The server prints `delete symlink? (y/N)` on its own terminal and the request waits for an answer typed there. Only `y` or `yes` removes the store's link (a copy on Windows); the original files stay. Any other answer keeps the model, and so does a stdin at end of file (such as `/dev/null`). The reply is `{"status": "ok"}` either way.

## POST /models/{id}/unload

Unload a model from memory without deleting it from disk.

```bash
curl -X POST http://localhost:8080/models/ollama%2Fllama3.1%2F8b.gguf/unload
```

```json
{"status": "ok"}
```

`{id}` can be an alias or the registry key. A model that isn't loaded returns 404 with `{"error": "..."}`.
