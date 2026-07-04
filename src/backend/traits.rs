// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

use std::path::Path;
#[cfg(feature = "vision")]
use std::path::PathBuf;

use crate::engine::streaming::{GenerateParams, GenerateResult};

pub struct BackendLoadParams {
    /// Requested context size. 0 = auto-resolve to the largest n_ctx that
    /// fits weights + KV + compute buffers within `memory_budget`.
    pub n_ctx: u32,
    pub n_gpu_layers: Option<u32>,
    /// Live memory available for this load (bytes), snapshotted before the
    /// model's weights are mmap'd. 0 = unlimited. Backends that auto-size
    /// n_ctx use this as the budget ceiling.
    pub memory_budget: u64,
    /// Path to the multimodal projector GGUF file. When `Some`, the backend
    /// should initialise its vision pipeline (e.g. `MtmdContext`).
    #[cfg(feature = "vision")]
    pub mmproj_path: Option<PathBuf>,
}

pub trait InferenceBackend: Send + Sync {
    fn load_model(
        &self,
        path: &Path,
        params: BackendLoadParams,
    ) -> anyhow::Result<Box<dyn BackendModel>>;

    fn name(&self) -> &str;
}

pub trait BackendModel: Send + Sync {
    fn generate(
        &self,
        prompt: &str,
        params: &GenerateParams,
        on_token: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult>;

    fn apply_chat_template(
        &self,
        messages: &[(String, String)],
    ) -> anyhow::Result<String>;

    /// Apply the chat template and generate in one call.
    ///
    /// The default implementation calls `apply_chat_template` then `generate`.
    /// Backends that can fuse the two operations (e.g. MLX, which avoids a
    /// decode → encode round-trip across the FFI boundary) should override this.
    fn generate_chat(
        &self,
        messages: &[(String, String)],
        params: &GenerateParams,
        on_token: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult> {
        let prompt = self.apply_chat_template(messages)?;
        self.generate(&prompt, params, on_token)
    }

    fn n_ctx(&self) -> u32;

    fn n_ctx_train(&self) -> u32 {
        0
    }

    fn size_bytes(&self) -> u64;

    fn supports_batching(&self) -> bool {
        false
    }

    /// Per-token KV bytes for eviction sizing. Required so a new backend
    /// cannot silently underflow `total_loaded_bytes` by forgetting it.
    fn kv_bytes_per_token(&self) -> u64;

    fn as_any(&self) -> &dyn std::any::Any;

    fn embed(&self, _text: &str) -> anyhow::Result<EmbedResult> {
        anyhow::bail!("embeddings not supported by this backend")
    }

    /// Returns `true` if the model was loaded with a multimodal projector
    /// and can accept image inputs.
    #[cfg(feature = "vision")]
    fn supports_vision(&self) -> bool {
        false
    }

    /// Generate a response from a multimodal conversation (text + images).
    ///
    /// The default implementation returns an error — backends must override
    /// this when they support vision.
    #[cfg(feature = "vision")]
    fn generate_multimodal(
        &self,
        _messages: &[crate::engine::multimodal::MultimodalMessage],
        _params: &GenerateParams,
        _on_token: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult> {
        anyhow::bail!("this backend does not support multimodal / vision input")
    }
}

pub struct EmbedResult {
    pub embedding: Vec<f32>,
    pub prompt_tokens: u32,
}
