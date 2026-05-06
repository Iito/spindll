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
| `tool_choice` | string or object | no | (accepted, not yet enforced) |

### Tool / function calling

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

**Error format:**

```json
{
  "error": {
    "message": "description of what went wrong",
    "type": "server_error"
  }
}
```
