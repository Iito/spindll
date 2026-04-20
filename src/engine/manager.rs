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
    memory_budget: u64, // max bytes for loaded models, 0 = unlimited
}

impl ModelManager {
    pub fn new(n_ctx: u32, gpu_layers: Option<u32>, memory_budget: u64) -> anyhow::Result<Self> {
        let backend = LlamaBackend::init()?;

        let default_gpu_layers = gpu_layers.unwrap_or_else(|| {
            if cfg!(target_os = "macos") { 999 } else { 0 }
        });

        Ok(Self {
            backend,
            models: RwLock::new(HashMap::new()),
            default_n_ctx: n_ctx,
            default_gpu_layers,
            memory_budget,
        })
    }

    fn total_loaded_bytes(&self) -> u64 {
        self.models
            .read()
            .unwrap()
            .values()
            .map(|m| m.size_bytes)
            .sum()
    }

    /// Evict least-recently-used models until `needed` bytes fit within budget.
    fn evict_for(&self, needed: u64) -> anyhow::Result<()> {
        if self.memory_budget == 0 {
            return Ok(()); // unlimited
        }

        loop {
            let used = self.total_loaded_bytes();
            if used + needed <= self.memory_budget {
                return Ok(());
            }

            // Find LRU model
            let models = self.models.read().unwrap();
            if models.is_empty() {
                anyhow::bail!(
                    "model needs {:.1} GB but budget is {:.1} GB",
                    needed as f64 / 1_073_741_824.0,
                    self.memory_budget as f64 / 1_073_741_824.0
                );
            }

            let lru_name = models
                .iter()
                .min_by_key(|(_, m)| *m.last_used.read().unwrap())
                .map(|(name, _)| name.clone())
                .unwrap();
            drop(models);

            eprintln!("evicting {lru_name} (lru) to make room");
            self.models.write().unwrap().remove(&lru_name);
        }
    }

    pub fn load_model(
        &self,
        name: &str,
        path: &Path,
        gpu_layers: Option<u32>,
    ) -> anyhow::Result<()> {
        // Estimate size from file for budget check before loading
        let file_size = std::fs::metadata(path)?.len();
        self.evict_for(file_size)?;

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
        eprintln!("loaded {name} ({} layers on {device})", model.n_layer());

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
        eprintln!("unloaded {name}");
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
