# Spindll API Reference

Spindll exposes three interfaces: an HTTP/SSE API (with OpenAI compatibility), a gRPC API, and a Rust library API. All three share the same underlying engine.

- [HTTP API](api-http.md) — native endpoints for model management and SSE chat
- [OpenAI-compatible API](api-openai.md) — drop-in `/v1` endpoints for any OpenAI client
- [gRPC API](api-grpc.md) — streaming RPCs with prefill, batching, and status
- [Rust Library API](api-rust.md) — embed spindll as a crate with `ModelManager` and backend traits
