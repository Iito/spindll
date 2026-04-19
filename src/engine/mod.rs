pub mod streaming;

pub use streaming::GenerateParams;

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
    /// Load a model, auto-detecting GPU. Pass n_gpu_layers=None to offload all layers.
    pub fn load(path: &Path, n_gpu_layers: Option<u32>, n_ctx: u32) -> anyhow::Result<Self> {
        let backend = LlamaBackend::init()?;

        let gpu_layers = n_gpu_layers.unwrap_or_else(|| {
            if cfg!(target_os = "macos") {
                // Metal: offload everything
                999
            } else {
                // CPU fallback — user can override with explicit count
                0
            }
        });

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(gpu_layers);
        let model = LlamaModel::load_from_file(&backend, path, &model_params)
            .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

        let device = if gpu_layers > 0 && cfg!(target_os = "macos") {
            "metal"
        } else if gpu_layers > 0 {
            "cuda"
        } else {
            "cpu"
        };
        println!("loaded {} layers on {device}", model.n_layer());

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

    pub fn generate(
        &self,
        prompt: &str,
        params: &GenerateParams,
        on_token: impl FnMut(&str) -> bool,
    ) -> anyhow::Result<()> {
        let mut ctx = self.create_context()?;
        streaming::generate_streaming(&self.model, &mut ctx, prompt, params, on_token)
    }
}
