# Anthropic-compatible API (`/v1/messages`)

Implements the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) dialect so Anthropic SDK clients — Claude Code included — can use spindll as a local backend.

## Client configuration

| Client | Setting |
|--------|---------|
| Claude Code | `ANTHROPIC_BASE_URL=http://localhost:8080`, `ANTHROPIC_AUTH_TOKEN=any`, model set to a spindll model id |
| Python anthropic SDK | `Anthropic(base_url="http://localhost:8080", api_key="any")` |
| curl | `POST http://localhost:8080/v1/messages` |

No API key is required; auth headers are ignored. Model ids are spindll ids (see `GET /v1/models`), not Anthropic model names.

## POST /v1/messages

### Non-streaming

```bash
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "max_tokens": 256,
    "system": "Be brief.",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

```json
{
  "id": "msg_0123456789abcdef",
  "type": "message",
  "role": "assistant",
  "model": "llama3.1:8b",
  "content": [{"type": "text", "text": "Hi! How can I help?"}],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {"input_tokens": 12, "output_tokens": 8}
}
```

### Streaming

Set `"stream": true`. Events follow the Messages SSE grammar — named `event:` lines, no `[DONE]` sentinel:

```
event: message_start
data: {"type":"message_start","message":{...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{...}}

event: message_stop
data: {"type":"message_stop"}
```

### Tool use

Tools are declared with `input_schema`; results come back as `tool_use` content blocks, and tool results are sent as `tool_result` blocks in a `user` turn:

```bash
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "max_tokens": 256,
    "tools": [{
      "name": "get_weather",
      "description": "Get current weather",
      "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }],
    "messages": [{"role": "user", "content": "Weather in Paris?"}]
  }'
```

A tool call answers with `stop_reason: "tool_use"` and a `{"type": "tool_use", "id", "name", "input"}` block. Streaming tool calls arrive as a `tool_use` `content_block_start` followed by one `input_json_delta` (arguments are parsed from the completed output, not token-by-token).

## Supported request fields

| Field | Notes |
|-------|-------|
| `model`, `max_tokens`, `messages` | Required (`max_tokens` missing → Anthropic-shaped 400) |
| `system` | String or text-block array |
| `messages[].content` | String or block array: `text`, `image`, `tool_use`, `tool_result` |
| `tools`, `tool_choice` | `{type: auto \| any \| tool \| none}`; rendered through the model's native tool template |
| `stop_sequences` | Honored in both modes; streamed output holds back a partial match so a stop string split across tokens never leaks |
| `temperature`, `top_p`, `top_k` | Passed through |
| `stream` | Messages SSE grammar |
| `image` blocks | `base64` source, or `url` source with a `data:` URI only — spindll never fetches remote URLs. Requires the `vision` build feature |

Unknown content block types (`thinking`, `redacted_thinking`, `document`, …) are accepted and dropped, so clients that replay conversation history work unchanged. Unknown top-level fields are ignored.

## Stop reasons

`end_turn`, `max_tokens`, `stop_sequence` (with `stop_sequence` set to the matched string), and `tool_use`.

## Not implemented

- `/v1/messages/count_tokens`, batches, files
- prompt caching fields (accepted, ignored; `usage` never reports cache reads)
- extended thinking (no `thinking` config; thinking blocks in history are dropped)
- server-side tools (web search etc.) — only client-executed function tools
