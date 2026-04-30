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

## POST /pull

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

## DELETE /models/{id}

Remove a model from disk (and unload from memory if loaded).

```bash
curl -X DELETE http://localhost:8080/models/ollama%2Fllama3.1%2F8b.gguf
```

```json
{"status": "ok"}
```

## POST /models/{id}/unload

Unload a model from memory without deleting it from disk.

```bash
curl -X POST http://localhost:8080/models/ollama%2Fllama3.1%2F8b.gguf/unload
```

```json
{"status": "ok"}
```
