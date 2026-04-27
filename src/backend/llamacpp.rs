use std::num::NonZeroU32;
use std::path::Path;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;

use crate::engine::streaming::{GenerateParams, GenerateResult, generate_streaming};
use crate::engine::apply_chat_template_with_fallback;
use super::traits::{BackendLoadParams, BackendModel, InferenceBackend};

pub struct LlamaCppBackend {
    backend: LlamaBackend,
}

impl LlamaCppBackend {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            backend: LlamaBackend::init()?,
        })
    }
}

impl InferenceBackend for LlamaCppBackend {
    fn load_model(
        &self,
        path: &Path,
        params: BackendLoadParams,
    ) -> anyhow::Result<Box<dyn BackendModel>> {
        let gpu_layers = params.n_gpu_layers.unwrap_or_else(|| {
            if cfg!(target_os = "macos")
                || cfg!(feature = "cuda")
                || cfg!(feature = "metal")
                || cfg!(feature = "vulkan")
            {
                999
            } else {
                0
            }
        });

        let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
        let model = LlamaModel::load_from_file(&self.backend, path, &model_params)
            .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

        let n_ctx_train = model.n_ctx_train();
        let n_ctx = if params.n_ctx > 0 && n_ctx_train > 0 {
            std::cmp::min(n_ctx_train, params.n_ctx)
        } else if params.n_ctx > 0 {
            params.n_ctx
        } else {
            n_ctx_train
        };

        let size_bytes = model.size();

        let device = if gpu_layers == 0 {
            "cpu"
        } else if cfg!(target_os = "macos") || cfg!(feature = "metal") {
            "metal"
        } else if cfg!(feature = "cuda") {
            "cuda"
        } else if cfg!(feature = "vulkan") {
            "vulkan"
        } else {
            "cpu"
        };
        tracing::info!(
            layers = model.n_layer(),
            device,
            size_bytes,
            n_ctx,
            n_ctx_train,
            "model loaded"
        );

        Ok(Box::new(LlamaCppModel {
            backend: LlamaBackend::init()?,
            model,
            n_ctx,
            n_ctx_train,
            size_bytes,
            gpu_layers,
        }))
    }

    fn name(&self) -> &str {
        "llamacpp"
    }
}

pub struct LlamaCppModel {
    backend: LlamaBackend,
    model: LlamaModel,
    n_ctx: u32,
    n_ctx_train: u32,
    size_bytes: u64,
    gpu_layers: u32,
}

impl LlamaCppModel {
    pub fn llama_model(&self) -> &LlamaModel {
        &self.model
    }

    pub fn llama_backend(&self) -> &LlamaBackend {
        &self.backend
    }

    pub fn gpu_layers(&self) -> u32 {
        self.gpu_layers
    }
}

impl BackendModel for LlamaCppModel {
    fn generate(
        &self,
        prompt: &str,
        params: &GenerateParams,
        on_token: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult> {
        let ctx_params =
            LlamaContextParams::default().with_n_ctx(NonZeroU32::new(self.n_ctx));
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("failed to create context: {e}"))?;
        generate_streaming(&self.model, &mut ctx, prompt, params, on_token)
    }

    fn apply_chat_template(
        &self,
        messages: &[(String, String)],
    ) -> anyhow::Result<String> {
        apply_chat_template_with_fallback(&self.model, messages)
    }

    fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    fn n_ctx_train(&self) -> u32 {
        self.n_ctx_train
    }

    fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
