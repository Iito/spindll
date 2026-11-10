// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

mod traits;
pub mod llamacpp;

/// Layers llama.cpp offloads when the caller leaves it unset: all of them
/// wherever a GPU backend is built in, none otherwise. On iOS that means CPU
/// unless the `metal` feature is on, and small models are faster there anyway
/// (83 vs 37 tok/s for Gemma 3 270M on an iPhone 16 Pro Max). Never on the
/// Simulator: its Metal returns garbage for llama.cpp's kernels, which is why
/// llama.cpp's own iOS example runs it on the CPU too.
pub(crate) const fn default_gpu_layers() -> u32 {
    let gpu_built = cfg!(target_os = "macos")
        || cfg!(feature = "cuda")
        || cfg!(feature = "metal")
        || cfg!(feature = "vulkan");
    if gpu_built && !cfg!(all(target_os = "ios", target_abi = "sim")) {
        999
    } else {
        0
    }
}
pub use traits::{BackendLoadParams, BackendModel, EmbedResult, InferenceBackend};

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
pub mod mlx_swift;
