//! Spindll — a Rust-native GGUF inference engine with model management.
//!
//! The default library includes the gRPC server so embedders (e.g. parley)
//! can start it in-process. The `cli` feature adds the standalone binary extras
//! (argument parsing, pretty logging) and is not needed by library consumers.

pub mod engine;
pub mod model_store;
pub mod scheduler;
pub mod grpc;

pub mod proto {
    tonic::include_proto!("spindll");
}
