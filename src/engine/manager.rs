use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::num::NonZeroU32;

use crate::backend::llamacpp::LlamaCppModel;
use crate::backend::{BackendLoadParams, BackendModel, InferenceBackend};
use crate::model_store::registry::ModelFormat;

use super::batch::{BatchEvent, BatchRequest, BatchScheduler};
use super::kv_cache::KvCache;
use super::metrics::Metrics;
use super::ram_cache::RamCache;
use super::streaming::{GenerateParams, GenerateResult, generate_streaming_cached};

/// A model that has been loaded into memory and is ready for inference.
pub struct LoadedModel {
    /// The backend model, dispatching to GGUF or MLX as appropriate.
    pub model: Box<dyn BackendModel>,
    /// Path to the model file or directory on disk.
    pub file_path: PathBuf,
    /// Context window size (number of tokens).
    pub n_ctx: u32,
    /// Trained context length from metadata (0 if unknown).
    pub n_ctx_train: u32,
    /// Approximate memory footprint in bytes.
    pub size_bytes: u64,
    /// Timestamp of the last inference request (used for LRU eviction).
    pub last_used: RwLock<Instant>,
    /// Number of layers offloaded to GPU.
    pub gpu_layers: u32,
    /// SHA-256 digest of the model file, used for KV cache keying.
    pub digest: String,
    /// On-disk format of this model.
    pub format: ModelFormat,
    /// Channel to submit requests to this model's batch scheduler (if running).
    pub batch_tx: Option<std::sync::mpsc::Sender<BatchRequest>>,
}

/// Multi-model manager with LRU eviction and memory budgeting.
///
/// This is the primary entry point for Parley: load models by name, run inference,
/// and let the manager handle eviction when memory is tight.
pub struct ModelManager {
    backends: Vec<Box<dyn InferenceBackend>>,
    models: RwLock<HashMap<String, LoadedModel>>,
    default_n_ctx: u32,
    default_gpu_layers: u32,
    memory_budget: u64,
    kv_cache: Option<KvCache>,
    ram_cache: Option<RamCache>,
    metrics: Arc<Metrics>,
    batch_slots: usize,
}

impl ModelManager {
    /// Create a new manager. Pass `gpu_layers = None` to auto-detect (all layers on macOS Metal,
    /// CPU-only elsewhere). Set `memory_budget` to 0 for unlimited.
    pub fn new(n_ctx: u32, gpu_layers: Option<u32>, memory_budget: u64) -> anyhow::Result<Self> {
        #[allow(unused_mut)]
        let mut backends: Vec<Box<dyn InferenceBackend>> = vec![
            Box::new(crate::backend::llamacpp::LlamaCppBackend::new()?),
        ];

        #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
        backends.push(Box::new(crate::backend::mlx_swift::MlxBackend));

        let default_gpu_layers = gpu_layers.unwrap_or_else(|| {
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

        Ok(Self {
            backends,
            models: RwLock::new(HashMap::new()),
            default_n_ctx: n_ctx,
            default_gpu_layers,
            memory_budget,
            kv_cache: None,
            ram_cache: None,
            metrics: Arc::new(Metrics::new()),
            batch_slots: 0,
        })
    }

    /// Set the number of concurrent sequence slots for batch scheduling.
    /// When > 0, each loaded model gets a dedicated batch scheduler thread
    /// that multiplexes concurrent requests through a single context.
    pub fn set_batch_slots(&mut self, slots: usize) {
        self.batch_slots = slots;
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
            return Ok(());
        }

        loop {
            let used = self.total_loaded_bytes();
            if used + needed <= self.memory_budget {
                return Ok(());
            }

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
            let evicted = self.models.write().unwrap().remove(&lru_name);
            if let (Some(cache), Some(model)) = (&self.ram_cache, &evicted) {
                cache.warm(&lru_name, &model.file_path);
            }
        }
    }

    fn backend_for_format(&self, format: &ModelFormat) -> anyhow::Result<&dyn InferenceBackend> {
        let target = match format {
            ModelFormat::Gguf => "llamacpp",
            ModelFormat::Mlx => "mlx",
        };
        self.backends
            .iter()
            .find(|b| b.name() == target)
            .map(|b| b.as_ref())
            .ok_or_else(|| anyhow::anyhow!("no backend available for {target} format"))
    }

    fn infer_format(path: &Path) -> ModelFormat {
        if path.is_dir() {
            ModelFormat::Mlx
        } else {
            ModelFormat::Gguf
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
        let format = Self::infer_format(path);
        let backend = self.backend_for_format(&format)?;

        let from_ram_cache = self
            .ram_cache
            .as_ref()
            .and_then(|c| c.get(name))
            .is_some();

        let file_size = if path.is_dir() {
            std::fs::read_dir(path)?
                .filter_map(|e| e.ok())
                .filter_map(|e| std::fs::metadata(e.path()).ok())
                .map(|m| m.len())
                .sum()
        } else {
            std::fs::metadata(path)?.len()
        };
        self.evict_for(file_size)?;

        let layers = gpu_layers.unwrap_or(self.default_gpu_layers);

        let load_params = BackendLoadParams {
            n_ctx: self.default_n_ctx,
            n_gpu_layers: Some(layers),
        };

        let model = backend.load_model(path, load_params)?;

        if from_ram_cache {
            if let Some(cache) = &self.ram_cache {
                cache.remove(name);
            }
        }

        let n_ctx = model.n_ctx();
        let n_ctx_train = model.n_ctx_train();
        let size_bytes = model.size_bytes();

        // Batch scheduling: GGUF-only, gated on supports_batching().
        let batch_tx = if self.batch_slots > 0 && model.supports_batching() {
            let (tx, rx) = std::sync::mpsc::channel::<BatchRequest>();
            let max_seq = self.batch_slots;
            let model_name = name.to_string();

            let sched_backend = crate::backend::llamacpp::shared_backend();
            let sched_params = LlamaModelParams::default().with_n_gpu_layers(layers);
            let sched_model = LlamaModel::load_from_file(sched_backend, path, &sched_params)
                .map_err(|e| anyhow::anyhow!("failed to load scheduler model: {e}"))?;

            std::thread::Builder::new()
                .name(format!("batch-{model_name}"))
                .spawn(move || {
                    if let Err(e) =
                        BatchScheduler::run(&sched_model, sched_backend, n_ctx, max_seq, rx)
                    {
                        tracing::error!(model = model_name, "batch scheduler exited: {e}");
                    }
                })?;

            tracing::info!(name, slots = max_seq, "batch scheduler started");
            Some(tx)
        } else {
            None
        };

        let loaded = LoadedModel {
            model,
            file_path: path.to_path_buf(),
            n_ctx,
            n_ctx_train,
            size_bytes,
            last_used: RwLock::new(Instant::now()),
            gpu_layers: layers,
            digest,
            format,
            batch_tx,
        };

        self.models.write().unwrap().insert(name.to_string(), loaded);
        Ok(())
    }

    #[tracing::instrument(skip(self))]
    pub fn unload_model(&self, name: &str) -> anyhow::Result<()> {
        let removed = self
            .models
            .write()
            .unwrap()
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("model '{name}' not loaded"))?;
        if let Some(cache) = &self.ram_cache {
            cache.warm(name, &removed.file_path);
        }
        tracing::info!(name, "model unloaded");
        Ok(())
    }

    /// Returns `true` if a model with the given name is currently loaded.
    pub fn is_loaded(&self, name: &str) -> bool {
        self.models.read().unwrap().contains_key(name)
    }

    /// List all loaded models as `(name, size_bytes, gpu_layers, digest, n_ctx, n_ctx_train)` tuples.
    pub fn loaded_models(&self) -> Vec<(String, u64, u32, String, u32, u32)> {
        self.models
            .read()
            .unwrap()
            .iter()
            .map(|(name, m)| {
                (
                    name.clone(),
                    m.size_bytes,
                    m.gpu_layers,
                    m.digest.clone(),
                    m.n_ctx,
                    m.n_ctx_train,
                )
            })
            .collect()
    }

    /// Enable the RAM cache for recently-evicted models.
    pub fn enable_ram_cache(&mut self, max_bytes: u64) {
        self.ram_cache = Some(RamCache::new(max_bytes));
    }

    /// Returns a reference to the RAM cache, if enabled.
    pub fn ram_cache(&self) -> Option<&RamCache> {
        self.ram_cache.as_ref()
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
    /// When the model has a batch scheduler running, the request is submitted
    /// to the shared decode loop. Otherwise falls back to a per-request context
    /// (using the disk-backed KV cache if enabled for GGUF models).
    ///
    /// Pass `encryption_key` to encrypt cached KV state at rest (non-batched GGUF path only).
    #[tracing::instrument(
        skip(self, prompt, params, on_token, encryption_key),
        fields(model = model_name)
    )]
    pub fn generate(
        &self,
        model_name: &str,
        prompt: &str,
        params: &GenerateParams,
        encryption_key: Option<&[u8; 32]>,
        mut on_token: impl FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult> {
        let batch_tx = {
            let models = self.models.read().unwrap();
            models.get(model_name).and_then(|m| {
                *m.last_used.write().unwrap() = Instant::now();
                m.batch_tx.clone()
            })
        };

        let start = Instant::now();
        let result = if let Some(tx) = batch_tx {
            Self::generate_via_batch(tx, prompt, params, &mut on_token)
        } else {
            self.generate_direct(model_name, prompt, params, encryption_key, &mut on_token)
        };

        let elapsed_us = start.elapsed().as_micros() as u64;
        match &result {
            Ok(stats) => {
                if params.prefill_only {
                    self.metrics
                        .record_prefill(stats.prompt_tokens as u64, elapsed_us, stats.cache_hit);
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

    /// Submit a request to the model's batch scheduler and bridge events to the callback.
    fn generate_via_batch(
        tx: std::sync::mpsc::Sender<BatchRequest>,
        prompt: &str,
        params: &GenerateParams,
        on_token: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult> {
        let (resp_tx, mut resp_rx) = tokio::sync::mpsc::channel::<BatchEvent>(32);

        let req = BatchRequest {
            prompt: prompt.to_string(),
            params: GenerateParams {
                max_tokens: params.max_tokens,
                temperature: params.temperature,
                top_p: params.top_p,
                top_k: params.top_k,
                seed: params.seed,
                prefill_only: params.prefill_only,
            },
            response_tx: resp_tx,
        };

        tx.send(req)
            .map_err(|_| anyhow::anyhow!("batch scheduler is not running"))?;

        loop {
            match resp_rx.blocking_recv() {
                Some(BatchEvent::Token(piece)) => {
                    if !on_token(&piece) {
                        break;
                    }
                }
                Some(BatchEvent::Done {
                    prompt_tokens,
                    completion_tokens,
                }) => {
                    return Ok(GenerateResult {
                        prompt_tokens,
                        completion_tokens,
                        cache_hit: false,
                    });
                }
                Some(BatchEvent::Error(msg)) => {
                    anyhow::bail!(msg);
                }
                None => {
                    anyhow::bail!("batch scheduler dropped the response channel");
                }
            }
        }

        Ok(GenerateResult {
            prompt_tokens: 0,
            completion_tokens: 0,
            cache_hit: false,
        })
    }

    /// Per-request context path. Uses KV cache for GGUF models when available,
    /// otherwise delegates to the backend trait.
    fn generate_direct(
        &self,
        model_name: &str,
        prompt: &str,
        params: &GenerateParams,
        encryption_key: Option<&[u8; 32]>,
        on_token: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult> {
        let models = self.models.read().unwrap();
        let loaded = models
            .get(model_name)
            .ok_or_else(|| anyhow::anyhow!("model '{}' not loaded", model_name))?;
        *loaded.last_used.write().unwrap() = Instant::now();

        // KV cache path: GGUF models with cache enabled get the cached generate path.
        if let Some(cache) = &self.kv_cache {
            if let Some(llama) = loaded.model.as_any().downcast_ref::<LlamaCppModel>() {
                let ctx_params =
                    LlamaContextParams::default().with_n_ctx(NonZeroU32::new(loaded.n_ctx));
                let mut ctx = llama
                    .llama_model()
                    .new_context(llama.llama_backend(), ctx_params)
                    .map_err(|e| anyhow::anyhow!("failed to create context: {e}"))?;
                return generate_streaming_cached(
                    llama.llama_model(),
                    &mut ctx,
                    prompt,
                    params,
                    model_name,
                    &loaded.digest,
                    cache,
                    encryption_key,
                    on_token,
                );
            }
        }

        loaded.model.generate(prompt, params, on_token)
    }

    /// Apply the model's built-in chat template to a list of `(role, content)` messages.
    /// Falls back to ChatML if the model has no embedded template.
    ///
    /// Returns the formatted prompt string ready for generation.
    pub fn apply_chat_template(
        &self,
        model_name: &str,
        messages: &[(String, String)],
    ) -> anyhow::Result<String> {
        let models = self.models.read().unwrap();
        let loaded = models
            .get(model_name)
            .ok_or_else(|| anyhow::anyhow!("model '{}' not loaded", model_name))?;
        *loaded.last_used.write().unwrap() = Instant::now();
        loaded.model.apply_chat_template(messages)
    }
}
