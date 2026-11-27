# OpenAI-compatible API (`/v1`)

Drop-in compatible with any OpenAI client. These endpoints follow the [OpenAI API specification](https://platform.openai.com/docs/api-reference).

## Client configuration

| Client | Setting |
|--------|---------|
| AnythingLLM | Provider: Custom (OpenAI Compatible), Base URL: `http://localhost:8080/v1` |
| Open WebUI | OpenAI API URL: `http://localhost:8080/v1` |
| Python openai SDK | `base_url="http://localhost:8080/v1"`, `api_key="any"` |
| curl | Base URL: `http://localhost:8080/v1` |

No API key is required. Clients that mandate one can use any non-empty string.

## GET /v1/models

```bash
curl http://localhost:8080/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "ollama/llama3.1/8b.gguf",
      "object": "model",
      "owned_by": "spindll",
      "created": 1767225600,
      "architecture": "llama",
      "context_length": 131072,
      "format": "gguf",
      "size_bytes": 4920000000,
      "capabilities": {"chat": true, "completions": true, "embeddings": true}
    }
  ]
}
```

Each item has the standard `id`, `object`, `created` (when the model was downloaded or imported, in Unix seconds) and `owned_by`, plus:

| Field | Notes |
|-------|-------|
| `architecture` | Omitted when unknown |
| `context_length` | Trained context size from the registry, not the loaded size. Omitted when 0, which is always the case for MLX |
| `format` | `"gguf"` or `"mlx"` |
| `size_bytes` | Size on disk |
| `capabilities` | `chat`, `completions` and `embeddings` are `true` for every model; nothing checks what the model supports |

## GET /v1/models/{id}

One model, in the same shape as a `/v1/models` item. `{id}` must be the exact registry key from `/v1/models`, URL-encoded. Aliases such as `llama3.1:8b` are not resolved here.

```bash
curl http://localhost:8080/v1/models/ollama%2Fllama3.1%2F8b.gguf
```

An unknown id returns 404:

```json
{"error": {"message": "model not found", "type": "invalid_request_error"}}
```

## GET /v1/status

```bash
curl http://localhost:8080/v1/status
```

```json
{
  "status": "ok",
  "version": "0.10.0",
  "models": {
    "total": 2,
    "loaded": 1,
    "loaded_models": ["ollama/llama3.1/8b.gguf"]
  }
}
```

`version` is the server's version. `total` counts the models in the store, and `loaded_models` lists the keys of the models in memory.

## POST /v1/chat/completions

### Streaming

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
data: {"object":"chat.completion.chunk","model":"llama3.1:8b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"object":"chat.completion.chunk","model":"llama3.1:8b","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}

data: {"object":"chat.completion.chunk","model":"llama3.1:8b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

The first chunk carries only `delta.role`. A streamed response has no `usage` unless the request sets `"stream_options": {"include_usage": true}`. Then one more chunk, with `"choices": []` and the `usage` object, arrives just before `[DONE]`.

### Non-streaming (default)

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
| `stream` | boolean | no | false |
| `stream_options.include_usage` | boolean | no | false |
| `max_tokens` | integer | no | 512 |
| `temperature` | float | no | 0.8 |
| `top_p` | float | no | 0.95 |
| `seed` | integer | no | 42 |
| `tools` | array of tool objects | no | |
| `tool_choice` | string or object | no | `"auto"` when tools present |

Other fields are ignored. That includes `stop`: stop sequences are not supported on this endpoint.

### Images (`vision` feature)

A message's `content` can also be an array of `text` and `image_url` parts. Image parts need the `vision` build feature:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "What is in this picture?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}}
  ]
}
```

- `url` must be a base64 `data:` URI. spindll never fetches `http(s)` URLs.
- Allowed media types: `image/png`, `image/jpeg`, `image/webp`, `image/gif`, `image/bmp`.
- Each image can be up to 32 MiB once decoded.
- The model must be a vision model: a GGUF loaded with its multimodal projector (mmproj), or an MLX VLM.
- A bad image (not a base64 `data:` URI, too large, or another media type) fails the request with a 500, or with an error event when streaming.
- Without the `vision` feature, any `image_url` part gets a 400.

### Tool / function calling

Pass `tools` to enable function calling. The model's chat template renders the tools when it supports them; injecting them into the system prompt is the fallback (see the notes below). The model's output is parsed back into tool calls, recognizing the common wrapper formats (Hermes `<tool_call>`, Llama-3.1 `<|python_tag|>`, Mistral `[TOOL_CALLS]`) as well as bare JSON.

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
    {"role": "tool", "tool_call_id": "call_abc123", "content": "{\"temp\": 22, \"condition\": \"sunny\"}"}
  ]
}
```

**Notes:**
- Tool calling works best with models fine-tuned for it (Llama 3.1+, Qwen 2.5+, Mistral v0.3+).
- Tool rendering: on GGUF models, a chat template that references `tools` renders them in the model's trained format; otherwise the tool list is injected into the system prompt. MLX models always pass the tools to their template, with no injection fallback. Requests with images always use injection.
- `tool_choice` is honored: `"none"` disables tools entirely (the model never sees them, and nothing is parsed); `"auto"` (the default when `tools` is present) lets the model decide; `"required"` and `{"type": "function", "function": {"name": …}}` ask for a call. That instruction is part of the injected preamble only; a template that renders the tools itself receives the tool list without it. Neither path forces a call.
- With the `grammar` build feature, GGUF models also get a lazy GBNF grammar when `tool_choice` is `"required"` or names a function. It starts once the model opens a tool call (`<tool_call>`, `[TOOL_CALLS]` or `<|python_tag|>`) and then holds the output to a valid call: a listed tool name and an `arguments` object (matched against the tool's schema when one function is named). Text before the opener is not constrained. MLX models, requests with images, and servers running with `--batch-slots` get no grammar.
- Streaming with tools: output is buffered to parse the calls, then emitted as OpenAI-style **incremental** `tool_calls` deltas (per-call `index`, then `id`/`name`, then `arguments`).

### Reasoning models (`reasoning_content`)

Thinking models (Qwen3 / Qwen3.5, DeepSeek-R1, …) emit a reasoning pass delimited by `<think>` / `</think>` before the visible answer. spindll splits it out automatically — no request option needed — following the response shape mlx-vlm and the DeepSeek API use:

- **Non-streaming:** the reasoning arrives in `message.reasoning_content`; `message.content` holds only the answer.
- **Streaming:** reasoning arrives as `delta.reasoning_content` chunks, the answer as `delta.content` chunks.
- **Tools:** tool calls are parsed from the answer only, so a call the model merely *plans* inside its think block is not mistaken for one it made.

```json
{
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "reasoning_content": "The user wants a title. The sign reads…",
      "content": "{\"title\": \"Winter Evening Sake Shop\"}"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 137,
    "completion_tokens": 214,
    "total_tokens": 351,
    "completion_tokens_details": {"reasoning_tokens": 96}
  }
}
```

Both delimiter conventions are handled: models that emit `<think>` themselves, and chat templates that force the block open (the generation prompt ends with `<think>`, so the stream starts mid-reasoning — detected once per model at load). A single leading think block is recognized; delimiters appearing after visible content stream through as ordinary text.

**Truncation is detectable.** `finish_reason` is `"length"` when generation exhausted `max_tokens` (previously always `"stop"`). If the reasoning consumed the whole budget, `content` is empty (or absent from deltas), the partial reasoning is still returned in `reasoning_content`, and `usage.completion_tokens_details.reasoning_tokens` shows where the budget went (in a stream, only when `stream_options.include_usage` is set). `reasoning_tokens` is a stream-piece approximation of the token count.

## POST /v1/completions

Raw text completion (no chat template applied). Use this for code completion, text continuation, and other non-chat tasks.

### Streaming

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

### Non-streaming (default)

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
| `stream` | boolean | no | false |
| `max_tokens` | integer | no | 512 |
| `temperature` | float | no | 0.8 |
| `top_p` | float | no | 0.95 |
| `seed` | integer | no | 42 |

Other fields are ignored, including `stop` and `stream_options`: there are no stop sequences, and a stream carries no `usage`. `finish_reason` is always `"stop"` here, even when `max_tokens` was reached.

## POST /v1/embeddings

```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "input": ["first text", "second text"]
  }'
```

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.0123, -0.0456, 0.0789]},
    {"object": "embedding", "index": 1, "embedding": [0.0234, -0.0567, 0.0891]}
  ],
  "model": "llama3.1:8b",
  "usage": {"prompt_tokens": 6, "total_tokens": 6}
}
```

(Vectors shortened.)

**Request body:**

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `model` | string | yes | |
| `input` | string or array of strings | yes | |
| `encoding_format` | string | no | `"float"` |

- `input` is a string or an array of up to 2048 strings. Each one must be non-empty and at most 32,768 bytes. Token-id arrays are rejected.
- `encoding_format` is `"float"` or `"base64"` (the little-endian float32 bytes, base64-encoded). Any other value is rejected.
- `dimensions` is not supported: a request that sets it is rejected.
- Each rejection above is a 400. Most failures after that return 500, an unknown model name included.
- Each input is embedded on its own, and `usage.prompt_tokens` sums their token counts.
- Vectors are L2-normalized. GGUF models run a forward pass and use the model's pooling, or the last token's embedding when it has none; input past the context size is cut off. MLX models return the mean of the input token embeddings with no forward pass, so their vectors are only a rough approximation.

## POST /v1/responses

The [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) — the stateless subset agent clients use with `store: false`. Codex CLI works with its default `wire_api = "responses"`; no provider override needed:

```toml
# ~/.codex/config.toml
model = "llama3.1:8b"                 # a spindll model id (see /v1/models)
model_provider = "spindll"

[model_providers.spindll]
name = "spindll"
base_url = "http://localhost:8080/v1"
```

### Non-streaming

```bash
curl -X POST http://localhost:8080/v1/responses   -H "Content-Type: application/json"   -d '{
    "model": "llama3.1:8b",
    "instructions": "Be brief.",
    "input": "hello",
    "max_output_tokens": 256
  }'
```

```json
{
  "id": "resp_0123456789abcdef",
  "object": "response",
  "status": "completed",
  "model": "llama3.1:8b",
  "output": [
    {
      "type": "message",
      "id": "msg_...",
      "status": "completed",
      "role": "assistant",
      "content": [{"type": "output_text", "text": "Hi! How can I help?", "annotations": []}]
    }
  ],
  "usage": {
    "input_tokens": 12,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 8,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 20
  }
}
```

`cached_tokens` and `reasoning_tokens` are always 0. This endpoint does not split out think blocks: a `<think>` block the model writes stays in the output text.

### Streaming

Set `"stream": true`. Item-based events, each carrying a monotonic `sequence_number`, no `[DONE]` sentinel:

```
event: response.created
event: response.in_progress
event: response.output_item.added
event: response.content_part.added
event: response.output_text.delta        (one per token)
event: response.output_text.done
event: response.content_part.done
event: response.output_item.done
event: response.completed
```

The terminal event is `response.completed`, or `response.incomplete` when `max_output_tokens` was hit, or `response.failed` (with `response.error.code` / `message`) on error.

With tools in the request, the output is buffered and replayed through the same events once generation ends, so the text arrives as a single delta.

### Scope

| Accepted | Notes |
|----------|-------|
| `input` | String, or item array: `message` (typed or bare `{role, content}`), `function_call`, `function_call_output`; `developer` role maps to `system` |
| `instructions` | Prepended as the system turn |
| `tools` | Flat `{"type": "function", "name", ...}` definitions; hosted tool types (`web_search`, `local_shell`, …) are skipped |
| `tool_choice` | `auto` / `none` / `required` / `{"type": "function", "name"}` |
| `max_output_tokens`, `temperature`, `top_p`, `stream` | Passed through |
| `store`, `reasoning`, `include`, `prompt_cache_key`, `text`, and other stateful/hosted fields | Accepted and ignored (nothing is persisted); replayed `reasoning` items are dropped |

Function calls come back as `function_call` output items (arguments as one fragment via `response.function_call_arguments.delta`); send results back as `function_call_output` items keyed by `call_id`.

**Rejected with a 400:** `previous_response_id` (spindll is stateless; Codex's `store: false` path never sends it), `input_image` parts (not mapped yet), and a request with neither `input` nor `instructions`.

**Error format:**

`/v1/models`, `/v1/chat/completions`, `/v1/completions` and `/v1/embeddings` return errors in this shape. `type` is `"server_error"` whether the status is 400 or 500:

```json
{
  "error": {
    "message": "description of what went wrong",
    "type": "server_error"
  }
}
```

The 404 from `GET /v1/models/{id}` uses `"invalid_request_error"`. `/v1/responses` uses the same shape with `"invalid_request_error"` for a 400 and `"api_error"` for a 500.

When a chat or text completion stream fails, including a model that fails to load, the server sends one `data: {"error": {...}}` event in this shape and closes the stream without `[DONE]`. The HTTP status is still 200. A failing `/v1/responses` stream ends with `response.failed` instead.

A request body that doesn't parse (for example, one without `model`) is rejected before these handlers run, with a plain-text 400, 415 or 422.
