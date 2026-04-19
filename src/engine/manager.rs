use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::num::NonZeroU32;

use super::streaming::{GenerateParams, GenerateResult, generate_streaming};

pub struct LoadedModel {
    pub model: LlamaModel,
    pub n_ctx: u32,
    pub size_bytes: u64,
    pub last_used: RwLock<Instant>,
    pub gpu_layers: u32,
}

pub struct ModelManager {
    backend: LlamaBackend,
    models: RwLock<HashMap<String, LoadedModel>>,
    default_n_ctx: u32,
    default_gpu_layers: u32,
}

impl ModelManager {
    pub fn new(n_ctx: u32, gpu_layers: Option<u32>) -> anyhow::Result<Self> {
        let backend = LlamaBackend::init()?;

        let default_gpu_layers = gpu_layers.unwrap_or_else(|| {
            if cfg!(target_os = "macos") { 999 } else { 0 }
        });

        Ok(Self {
            backend,
            models: RwLock::new(HashMap::new()),
            default_n_ctx: n_ctx,
            default_gpu_layers,
        })
    }

    pub fn load_model(
        &self,
        name: &str,
        path: &Path,
        gpu_layers: Option<u32>,
    ) -> anyhow::Result<()> {
        let layers = gpu_layers.unwrap_or(self.default_gpu_layers);

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(layers);
        let model = LlamaModel::load_from_file(&self.backend, path, &model_params)
            .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

        let size_bytes = model.size();

        let device = if layers > 0 && cfg!(target_os = "macos") {
            "metal"
        } else if layers > 0 {
            "cuda"
        } else {
            "cpu"
        };
        println!("loaded {name} ({} layers on {device})", model.n_layer());

        let loaded = LoadedModel {
            model,
            n_ctx: self.default_n_ctx,
            size_bytes,
            last_used: RwLock::new(Instant::now()),
            gpu_layers: layers,
        };

        self.models.write().unwrap().insert(name.to_string(), loaded);
        Ok(())
    }

    pub fn unload_model(&self, name: &str) -> anyhow::Result<()> {
        self.models
            .write()
            .unwrap()
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("model '{name}' not loaded"))?;
        println!("unloaded {name}");
        Ok(())
    }

    pub fn is_loaded(&self, name: &str) -> bool {
        self.models.read().unwrap().contains_key(name)
    }

    pub fn loaded_models(&self) -> Vec<(String, u64, u32)> {
        self.models
            .read()
            .unwrap()
            .iter()
            .map(|(name, m)| (name.clone(), m.size_bytes, m.gpu_layers))
            .collect()
    }

    /// Run a closure with a reference to a loaded model.
    /// Updates last_used timestamp.
    pub fn with_model<F, R>(&self, name: &str, f: F) -> anyhow::Result<R>
    where
        F: FnOnce(&LlamaModel, &LlamaBackend, u32) -> anyhow::Result<R>,
    {
        let models = self.models.read().unwrap();
        let loaded = models
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("model '{name}' not loaded"))?;

        *loaded.last_used.write().unwrap() = Instant::now();
        f(&loaded.model, &self.backend, loaded.n_ctx)
    }

    pub fn generate(
        &self,
        model_name: &str,
        prompt: &str,
        params: &GenerateParams,
        on_token: impl FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult> {
        self.with_model(model_name, |model, backend, n_ctx| {
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(n_ctx));
            let mut ctx = model
                .new_context(backend, ctx_params)
                .map_err(|e| anyhow::anyhow!("failed to create context: {e}"))?;
            generate_streaming(model, &mut ctx, prompt, params, on_token)
        })
    }

    pub fn apply_chat_template(
        &self,
        model_name: &str,
        messages: &[(String, String)],
    ) -> anyhow::Result<String> {
        self.with_model(model_name, |model, _, _| {
            let tmpl = model
                .chat_template(None)
                .map_err(|e| anyhow::anyhow!("model has no chat template: {e}"))?;

            let chat_messages: Vec<llama_cpp_2::model::LlamaChatMessage> = messages
                .iter()
                .map(|(role, content)| {
                    llama_cpp_2::model::LlamaChatMessage::new(role.clone(), content.clone())
                        .map_err(|e| anyhow::anyhow!("invalid chat message: {e}"))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            model
                .apply_chat_template(&tmpl, &chat_messages, true)
                .map_err(|e| anyhow::anyhow!("failed to apply chat template: {e}"))
        })
    }
}
