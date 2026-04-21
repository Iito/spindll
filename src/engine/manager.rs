use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::num::NonZeroU32;

use super::kv_cache::KvCache;
use super::metrics::Metrics;
use super::streaming::{GenerateParams, GenerateResult, generate_streaming, generate_streaming_cached};

/// A model that has been loaded into memory and is ready for inference.
pub struct LoadedModel {
    /// The underlying llama.cpp model handle.
    pub model: LlamaModel,
    /// Context window size (number of tokens).
    pub n_ctx: u32,
    /// Approximate memory footprint in bytes.
    pub size_bytes: u64,
    /// Timestamp of the last inference request (used for LRU eviction).
    pub last_used: RwLock<Instant>,
    /// Number of layers offloaded to GPU.
    pub gpu_layers: u32,
    /// SHA-256 digest of the model file, used for KV cache keying.
    pub digest: String,
}

/// Multi-model manager with LRU eviction and memory budgeting.
///
/// This is the primary entry point for Parley: load models by name, run inference,
/// and let the manager handle eviction when memory is tight.
pub struct ModelManager {
    backend: LlamaBackend,
    models: RwLock<HashMap<String, LoadedModel>>,
    default_n_ctx: u32,
    default_gpu_layers: u32,
    memory_budget: u64, // max bytes for loaded models, 0 = unlimited
    kv_cache: Option<KvCache>,
    metrics: Arc<Metrics>,
}

impl ModelManager {
    /// Create a new manager. Pass `gpu_layers = None` to auto-detect (all layers on macOS Metal,
    /// CPU-only elsewhere). Set `memory_budget` to 0 for unlimited.
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
            kv_cache: None,
            metrics: Arc::new(Metrics::new()),
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

            tracing::warn!(model = %lru_name, "evicting LRU model");
            self.models.write().unwrap().remove(&lru_name);
        }
    }

    /// Load a model from a GGUF file. Evicts LRU models if the memory budget would be exceeded.
    pub fn load_model(
        &self,
        name: &str,
        path: &Path,
        gpu_layers: Option<u32>,
    ) -> anyhow::Result<()> {
        self.load_model_with_digest(name, path, gpu_layers, String::new())
    }

    /// Load a model with an explicit file digest for KV cache keying.
    ///
    /// Prefer this over [`load_model`](Self::load_model) when the digest is already known
    /// (e.g. from the model store registry) to avoid recomputing it.
    #[tracing::instrument(skip(self, path, digest), fields(file_size))]
    pub fn load_model_with_digest(
        &self,
        name: &str,
        path: &Path,
        gpu_layers: Option<u32>,
        digest: String,
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
        tracing::info!(name, layers = model.n_layer(), device, size_bytes, "model loaded");

        let loaded = LoadedModel {
            model,
            n_ctx: self.default_n_ctx,
            size_bytes,
            last_used: RwLock::new(Instant::now()),
            gpu_layers: layers,
            digest,
        };

        self.models.write().unwrap().insert(name.to_string(), loaded);
        Ok(())
    }

    #[tracing::instrument(skip(self))]
    pub fn unload_model(&self, name: &str) -> anyhow::Result<()> {
        self.models
            .write()
            .unwrap()
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("model '{name}' not loaded"))?;
        tracing::info!(name, "model unloaded");
        Ok(())
    }

    /// Returns `true` if a model with the given name is currently loaded.
    pub fn is_loaded(&self, name: &str) -> bool {
        self.models.read().unwrap().contains_key(name)
    }

    /// List all loaded models as `(name, size_bytes, gpu_layers, digest)` tuples.
    pub fn loaded_models(&self) -> Vec<(String, u64, u32, String)> {
        self.models
            .read()
            .unwrap()
            .iter()
            .map(|(name, m)| (name.clone(), m.size_bytes, m.gpu_layers, m.digest.clone()))
            .collect()
    }

    /// Run a closure with a reference to a loaded model.
    /// Updates last_used timestamp.
    pub fn with_model<F, R>(&self, name: &str, f: F) -> anyhow::Result<R>
    where
        F: FnOnce(&LlamaModel, &LlamaBackend, u32, &str) -> anyhow::Result<R>,
    {
        let models = self.models.read().unwrap();
        let loaded = models
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("model '{name}' not loaded"))?;

        *loaded.last_used.write().unwrap() = Instant::now();
        f(&loaded.model, &self.backend, loaded.n_ctx, &loaded.digest)
    }

    /// Enable the disk-backed KV cache with the given max size in bytes.
    pub fn enable_kv_cache(&mut self, max_bytes: u64) {
        self.kv_cache = Some(KvCache::new(max_bytes));
    }

    /// Enable the KV cache with a custom directory.
    pub fn enable_kv_cache_with_dir(&mut self, dir: std::path::PathBuf, max_bytes: u64) {
        self.kv_cache = Some(KvCache::with_dir(dir, max_bytes));
    }

    /// Returns a reference to the KV cache, if enabled.
    pub fn kv_cache(&self) -> Option<&KvCache> {
        self.kv_cache.as_ref()
    }

    /// Returns a reference to the engine's metrics counters.
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// Run inference on a loaded model, streaming tokens through `on_token`.
    ///
    /// Automatically uses KV cache if enabled. Records prompt/completion metrics.
    /// Pass `encryption_key` to encrypt cached KV state at rest.
    #[tracing::instrument(skip(self, prompt, params, on_token, encryption_key), fields(model = model_name))]
    pub fn generate(
        &self,
        model_name: &str,
        prompt: &str,
        params: &GenerateParams,
        encryption_key: Option<&[u8; 32]>,
        on_token: impl FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult> {
        let start = Instant::now();
        let result = self.with_model(model_name, |model, backend, n_ctx, digest| {
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(n_ctx));
            let mut ctx = model
                .new_context(backend, ctx_params)
                .map_err(|e| anyhow::anyhow!("failed to create context: {e}"))?;

            match &self.kv_cache {
                Some(cache) => generate_streaming_cached(
                    model, &mut ctx, prompt, params, model_name, digest, cache, encryption_key, on_token,
                ),
                None => generate_streaming(model, &mut ctx, prompt, params, on_token),
            }
        });

        let elapsed_us = start.elapsed().as_micros() as u64;
        match &result {
            Ok(stats) => {
                if params.prefill_only {
                    self.metrics.record_prefill(
                        stats.prompt_tokens as u64, elapsed_us, stats.cache_hit,
                    );
                } else {
                    self.metrics.record_generate(
                        stats.prompt_tokens as u64,
                        stats.completion_tokens as u64,
                        elapsed_us,
                        stats.cache_hit,
                    );
                }
            }
            Err(_) => self.metrics.record_error(),
        }
        result
    }

    /// Apply the model's built-in chat template to a list of `(role, content)` messages.
    ///
    /// Returns the formatted prompt string ready for generation.
    pub fn apply_chat_template(
        &self,
        model_name: &str,
        messages: &[(String, String)],
    ) -> anyhow::Result<String> {
        self.with_model(model_name, |model, _, _, _| {
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
