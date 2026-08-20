// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

mod traits;
pub mod llamacpp;
#[cfg(feature = "rpc")]
pub mod rpc_ffi;
pub use traits::{BackendLoadParams, BackendModel, EmbedResult, InferenceBackend};

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub mod mlx_swift;
