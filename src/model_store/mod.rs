//! Model Store — manages model files on disk.
//!
//! Handles downloading from HuggingFace, tracking local models,
//! and import from Ollama.

pub mod download;
pub mod registry;
pub mod import;
