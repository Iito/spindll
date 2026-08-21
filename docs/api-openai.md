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
    {"id": "ollama/llama3.1/8b.gguf", "object": "model", "owned_by": "spindll"}
  ]
}
```

## POST /v1/chat/completions

### Streaming (default)

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

### Non-streaming

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
| `tool_choice` | string or object | no | `"auto"` when tools present |

### Tool / function calling

Pass `tools` to enable function calling. Tool definitions are injected into the system prompt (Spindll uses prompt injection, not a model-native grammar — see the note below) and the model's output is parsed back into tool calls, recognizing the common wrapper formats (Hermes `<tool_call>`, Llama-3.1 `<|python_tag|>`, Mistral `[TOOL_CALLS]`) as well as bare JSON.

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
- Tool calling works best with models fine-tuned for it (Llama 3.1+, Qwen 2.5+, Mistral v0.3+).
- `tool_choice` is honored: `"none"` disables tools entirely (no injection, no parsing); `"auto"` (the default when `tools` is present) lets the model decide; `"required"` and `{"type": "function", "function": {"name": …}}` instruct the model to call a tool. Without a grammar these are prompt-level guidance, not a hard constraint.
- Emission is **prompt-injection only** — llama.cpp's OpenAI-compatible chat-template + GBNF-grammar helper was removed in llama-cpp-2 0.1.150, so there is no model-native grammar constraint today.
- Streaming with tools: output is buffered to parse the calls, then emitted as OpenAI-style **incremental** `tool_calls` deltas (per-call `index`, then `id`/`name`, then `arguments`).

## POST /v1/completions

Raw text completion (no chat template applied). Use this for code completion, text continuation, and other non-chat tasks.

### Streaming (default)

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

### Non-streaming

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
  "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}
}
```

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

**Rejected with a 400:** `previous_response_id` (spindll is stateless — Codex's `store: false` path never sends it) and `input_image` parts (not mapped yet).

**Error format:**

```json
{
  "error": {
    "message": "description of what went wrong",
    "type": "server_error"
  }
}
```
