//! Inference Engine — loads GGUF models and runs inference.
//!
//! Wraps llama.cpp via Rust bindings. Handles token generation,
//! streaming output, and GPU offloading.

pub mod streaming;
