use std::path::Path;

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
}
