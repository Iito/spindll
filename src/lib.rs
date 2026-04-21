//! Spindll — a Rust-native GGUF inference engine with model management.
//!
//! The default library includes the gRPC server so embedders (e.g. Parley)
//! can start it in-process. The `cli` feature adds the standalone binary extras
//! (argument parsing, pretty logging) and is not needed by library consumers.
//!
//! # Modules
//!
//! - [`engine`] — Model loading, inference, KV caching, and operational metrics.
//! - [`model_store`] — Download, import, and resolve GGUF models from HuggingFace or Ollama.
//! - [`scheduler`] — Memory budget detection and enforcement for model loading.
//! - [`grpc`] — Tonic-based gRPC server exposing inference and model management RPCs.
//! - [`proto`] — Auto-generated protobuf types for the spindll wire protocol.

pub mod engine;
pub mod model_store;
pub mod scheduler;
pub mod grpc;

/// Auto-generated protobuf types for the spindll gRPC protocol.
pub mod proto {
    tonic::include_proto!("spindll");
}
