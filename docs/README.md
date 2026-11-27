# Spindll API Reference

Spindll exposes three interfaces: an HTTP/SSE API (with OpenAI compatibility), a gRPC API, and a Rust library API. All three share the same underlying engine.

- [HTTP API](api-http.md) — native endpoints for model management and SSE chat
- [OpenAI-compatible API](api-openai.md) — drop-in `/v1` endpoints for any OpenAI client, including the Responses API (`/v1/responses`) Codex CLI speaks
- [Anthropic-compatible API](api-anthropic.md) — the Messages API (`/v1/messages`) for Anthropic SDK clients, Claude Code included
- [gRPC API](api-grpc.md) — streaming RPCs with prefill, batching, and status
- [Rust Library API](api-rust.md) — embed spindll as a crate with `ModelManager` and backend traits
- [MLX Swift Bridge](mlx-bridge.md) — C ABI, build pipeline, and prompt KV cache for the `mlx_bridge/` Swift package

For apps that embed spindll, such as iOS keyboards:

- [C API](../ffi/include/spindll.h): the `spindll-ffi` crate's header over `Scorer`, which scores candidate text with a small model
- [SpindllKit](../ios/SpindllKit/README.md): a Swift package that wraps the C API, and how to build its xcframework
- [iOS](IOS.md): building for iPhone and the Simulator, the memory budget, and the Metal-free build a keyboard extension needs
