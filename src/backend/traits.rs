use std::path::Path;

use crate::engine::streaming::{GenerateParams, GenerateResult};

pub struct BackendLoadParams {
    pub n_ctx: u32,
    pub n_gpu_layers: Option<u32>,
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

    fn as_any(&self) -> &dyn std::any::Any;
}
