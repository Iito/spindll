//! Inference engine — model loading, text generation, KV caching, and metrics.

pub mod kv_cache;
pub mod manager;
pub mod metrics;
pub mod streaming;

pub use kv_cache::KvCache;
pub use manager::ModelManager;
pub use metrics::Metrics;
pub use streaming::{GenerateParams, GenerateResult};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::num::NonZeroU32;
use std::path::Path;

/// Single-model inference engine wrapping a loaded GGUF model.
///
/// For multi-model use cases (the common Parley path), prefer [`ModelManager`] instead.
/// `Engine` is useful when you need a lightweight wrapper around exactly one model.
pub struct Engine {
    backend: LlamaBackend,
    model: LlamaModel,
    n_ctx: u32,
    model_digest: String,
    kv_cache: Option<KvCache>,
}

impl Engine {
    /// Load a model, auto-detecting GPU. Pass n_gpu_layers=None to offload all layers.
    #[tracing::instrument(skip(path), fields(n_ctx, gpu_layers))]
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
        tracing::info!(layers = model.n_layer(), device, "model loaded");

        Ok(Self {
            backend,
            model,
            n_ctx,
            model_digest: String::new(),
            kv_cache: None,
        })
    }

    /// Returns a reference to the underlying llama.cpp model.
    pub fn model(&self) -> &LlamaModel {
        &self.model
    }

    /// Returns a reference to the llama.cpp backend instance.
    pub fn backend(&self) -> &LlamaBackend {
        &self.backend
    }

    /// Create a new inference context with the configured `n_ctx` window size.
    pub fn create_context(
        &self,
    ) -> anyhow::Result<llama_cpp_2::context::LlamaContext<'_>> {
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx));
        self.model
            .new_context(&self.backend, params)
            .map_err(|e| anyhow::anyhow!("failed to create context: {e}"))
    }

    /// Apply the model's built-in chat template to a list of (role, content) messages.
    pub fn apply_chat_template(
        &self,
        messages: &[(String, String)],
    ) -> anyhow::Result<String> {
        let tmpl = self.model.chat_template(None)
            .map_err(|e| anyhow::anyhow!("model has no chat template: {e}"))?;

        let chat_messages: Vec<llama_cpp_2::model::LlamaChatMessage> = messages
            .iter()
            .map(|(role, content)| {
                llama_cpp_2::model::LlamaChatMessage::new(role.clone(), content.clone())
                    .map_err(|e| anyhow::anyhow!("invalid chat message: {e}"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        self.model
            .apply_chat_template(&tmpl, &chat_messages, true)
            .map_err(|e| anyhow::anyhow!("failed to apply chat template: {e}"))
    }

    /// Enable the disk-backed KV cache with the given maximum size in bytes.
    pub fn enable_kv_cache(&mut self, max_bytes: u64) {
        self.kv_cache = Some(KvCache::new(max_bytes));
    }

    /// Set the model file digest for KV cache keying.
    pub fn set_model_digest(&mut self, digest: String) {
        self.model_digest = digest;
    }

    /// Generate text from a prompt, streaming tokens through the `on_token` callback.
    ///
    /// Returns `false` from `on_token` to stop generation early. Uses KV cache
    /// if enabled, falling back to uncached generation otherwise.
    #[tracing::instrument(skip(self, params, on_token), fields(prompt_len = prompt.len()))]
    pub fn generate(
        &self,
        prompt: &str,
        params: &GenerateParams,
        on_token: impl FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult> {
        let mut ctx = self.create_context()?;
        match &self.kv_cache {
            Some(cache) => streaming::generate_streaming_cached(
                &self.model, &mut ctx, prompt, params, "default", &self.model_digest, cache, None, on_token,
            ),
            None => streaming::generate_streaming(&self.model, &mut ctx, prompt, params, on_token),
        }
    }
}
