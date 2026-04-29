//! Spindll — a Rust-native GGUF inference engine with model management.
//!
//! Pull models from Ollama or HuggingFace, load them with GPU acceleration,
//! and serve streaming inference over gRPC, HTTP/SSE, or an OpenAI-compatible API.
//! Supports multi-model concurrency with LRU eviction, continuous batching,
//! encrypted KV caching, and configurable memory budgets.
//!
//! The default library includes the gRPC server so embedders
//! can start it in-process. The `cli` feature adds the standalone binary extras
//! (argument parsing, pretty logging). The `http` feature adds an HTTP/SSE server
//! with an OpenAI-compatible `/v1` API for use with AnythingLLM, Open WebUI, etc.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use spindll::engine::{ModelManager, GenerateParams};
//! use spindll::model_store::ModelStore;
//!
//! let store = ModelStore::new(None);
//! let path = store.resolve_model_path("llama3.1:8b").unwrap();
//! let digest = store.resolve_model_digest("llama3.1:8b").unwrap_or_default();
//!
//! let manager = ModelManager::new(2048, None, 0).unwrap();
//! manager.load_model_with_digest("llama3.1:8b", &path, None, digest).unwrap();
//!
//! manager.generate("llama3.1:8b", "Hello!", &GenerateParams::default(), None, |token| {
//!     print!("{token}");
//!     true
//! }).unwrap();
//! ```
//!
//! # Modules
//!
//! - [`engine`] — Model loading, inference, continuous batching, KV caching, and metrics.
//! - [`engine::batch`] — Batch scheduler for multiplexing concurrent requests.
//! - [`model_store`] — Download, import, and resolve GGUF models from HuggingFace or Ollama.
//! - [`scheduler`] — Memory budget detection and enforcement for model loading.
//! - [`grpc`] — Tonic-based gRPC server exposing inference and model management RPCs.
//! - [`http`] — HTTP/SSE server with OpenAI-compatible `/v1` API (requires `http` feature).
//! - [`proto`] — Auto-generated protobuf types for the spindll wire protocol.
//!
//! # Feature flags
//!
//! | Flag | Description |
//! |------|-------------|
//! | `cli` | Standalone binary with argument parsing and logging |
//! | `http` | HTTP/SSE server with OpenAI-compatible `/v1` API |

pub mod engine;
pub mod lockfile;
pub mod model_store;
pub mod scheduler;
pub mod grpc;
pub mod backend;
#[cfg(feature = "http")]
pub mod http;

/// Auto-generated protobuf types for the spindll gRPC protocol.
pub mod proto {
    tonic::include_proto!("spindll");
}
