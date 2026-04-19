pub mod streaming;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::num::NonZeroU32;
use std::path::Path;

pub struct Engine {
    backend: LlamaBackend,
    model: LlamaModel,
    n_ctx: u32,
}

impl Engine {
    pub fn load(path: &Path, n_gpu_layers: u32, n_ctx: u32) -> anyhow::Result<Self> {
        let backend = LlamaBackend::init()?;

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(&backend, path, &model_params)
            .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

        Ok(Self {
            backend,
            model,
            n_ctx,
        })
    }

    pub fn model(&self) -> &LlamaModel {
        &self.model
    }

    pub fn backend(&self) -> &LlamaBackend {
        &self.backend
    }

    pub fn create_context(
        &self,
    ) -> anyhow::Result<llama_cpp_2::context::LlamaContext<'_>> {
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx));
        self.model
            .new_context(&self.backend, params)
            .map_err(|e| anyhow::anyhow!("failed to create context: {e}"))
    }
}
