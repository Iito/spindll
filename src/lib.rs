//! Spindll — a Rust-native GGUF inference engine with model management.
//!
//! This is the library root. Other Rust projects can depend on spindll
//! as a crate and use these modules directly, without running the server.

pub mod model_store;
pub mod engine;
pub mod scheduler;

#[cfg(feature = "cli")]
pub mod grpc;

// Re-export the generated protobuf types so consumers can use them.
#[cfg(feature = "cli")]
pub mod proto {
    tonic::include_proto!("spindll");
}
