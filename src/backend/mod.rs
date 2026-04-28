mod traits;
pub mod llamacpp;
pub use traits::{BackendLoadParams, BackendModel, InferenceBackend};

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub mod mlx_swift;
