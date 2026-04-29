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
use super::streaming::{
    CachedGenerateRequest, GenerateParams, GenerateResult, generate_streaming_cached,
};

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
    /// Batch scheduler GGUF weight copy.
    pub scheduler_size_bytes: u64,
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

/// Multi-model manager with LRU memory budgeting.
///
/// Load by name, run inference, evict when tight.
pub struct ModelManager {
    backends: Vec<Box<dyn InferenceBackend>>,
    models: RwLock<HashMap<String, LoadedModel>>,
    default_n_ctx: u32,
    default_gpu_layers: u32,
    memory_budget: u64,
    clamp_budget_to_live: bool,
    kv_cache: Option<KvCache>,
    ram_cache: Option<RamCache>,
    metrics: Arc<Metrics>,
    batch_slots: usize,
}

impl ModelManager {
    /// Create a manager. `memory_budget = 0` means total RAM cap.
    pub fn new(n_ctx: u32, gpu_layers: Option<u32>, memory_budget: u64) -> anyhow::Result<Self> {
        #[allow(unused_mut)]
        let mut backends: Vec<Box<dyn InferenceBackend>> =
            vec![Box::new(crate::backend::llamacpp::LlamaCppBackend::new()?)];

        #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
        backends.push(Box::new(crate::backend::mlx_swift::MlxBackend));

        let default_gpu_layers = gpu_layers.unwrap_or({
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

        let (memory_budget, clamp_budget_to_live) = if memory_budget == 0 {
            (sysinfo::System::new_all().total_memory(), false)
        } else {
            (memory_budget, true)
        };

        Ok(Self {
            backends,
            models: RwLock::new(HashMap::new()),
            default_n_ctx: n_ctx,
            default_gpu_layers,
            memory_budget,
            clamp_budget_to_live,
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
            .map(loaded_model_memory_bytes)
            .sum()
    }

    /// Budget for next load. Live budgets clamp to current RAM.
    fn effective_budget(&self) -> u64 {
        if !self.clamp_budget_to_live {
            return self.memory_budget;
        }
        let live = crate::scheduler::budget::MemoryBudget::detect(None).available_ram;
        std::cmp::min(self.memory_budget, live)
    }

    /// Evict least-recently-used models until `needed` bytes fit within budget.
    fn evict_for(&self, needed: u64) -> anyhow::Result<()> {
        let budget = self.effective_budget();
        if budget == 0 {
            return Ok(());
        }

        loop {
            let used = self.total_loaded_bytes();
            if used + needed <= budget {
                return Ok(());
            }

            let models = self.models.read().unwrap();
            if models.is_empty() {
                anyhow::bail!(
                    "model needs {:.1} GB but budget is {:.1} GB",
                    needed as f64 / 1_073_741_824.0,
                    budget as f64 / 1_073_741_824.0
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

        let from_ram_cache = self.ram_cache.as_ref().and_then(|c| c.get(name)).is_some();

        let file_size = if path.is_dir() {
            std::fs::read_dir(path)?
                .filter_map(|e| e.ok())
                .filter_map(|e| std::fs::metadata(e.path()).ok())
                .map(|m| m.len())
                .sum()
        } else {
            std::fs::metadata(path)?.len()
        };
        let planned_scheduler_bytes = scheduler_weight_bytes(&format, file_size, self.batch_slots);
        self.evict_for(planned_load_weight_bytes(
            file_size,
            planned_scheduler_bytes,
        ))?;

        // Snapshot before weights consume live RAM.
        let available_before = crate::scheduler::budget::MemoryBudget::detect(None).available_ram;
        let load_budget = if self.clamp_budget_to_live {
            std::cmp::min(self.memory_budget, available_before)
        } else {
            self.memory_budget
        };
        let load_budget = load_budget.saturating_sub(planned_scheduler_bytes);

        let layers = gpu_layers.unwrap_or(self.default_gpu_layers);

        let load_params = BackendLoadParams {
            n_ctx: self.default_n_ctx,
            n_gpu_layers: Some(layers),
            memory_budget: load_budget,
        };

        let model = backend.load_model(path, load_params)?;

        if let (true, Some(cache)) = (from_ram_cache, &self.ram_cache) {
            cache.remove(name);
        }

        let n_ctx = model.n_ctx();
        let n_ctx_train = model.n_ctx_train();
        let size_bytes = model.size_bytes();

        // Scheduler needs its own GGUF model.
        let (batch_tx, scheduler_size_bytes) = if self.batch_slots > 0 && model.supports_batching()
        {
            let (tx, rx) = std::sync::mpsc::channel::<BatchRequest>();
            let max_seq = self.batch_slots;
            let model_name = name.to_string();

            let sched_backend = crate::backend::llamacpp::shared_backend();
            let sched_params = LlamaModelParams::default().with_n_gpu_layers(layers);
            let sched_model = LlamaModel::load_from_file(sched_backend, path, &sched_params)
                .map_err(|e| anyhow::anyhow!("failed to load scheduler model: {e}"))?;
            let scheduler_size_bytes = sched_model.size();

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
            (Some(tx), scheduler_size_bytes)
        } else {
            (None, 0)
        };

        let loaded = LoadedModel {
            model,
            file_path: path.to_path_buf(),
            n_ctx,
            n_ctx_train,
            size_bytes,
            scheduler_size_bytes,
            last_used: RwLock::new(Instant::now()),
            gpu_layers: layers,
            digest,
            format,
            batch_tx,
        };

        self.models
            .write()
            .unwrap()
            .insert(name.to_string(), loaded);
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

    /// List all loaded models as `(name, memory_bytes, gpu_layers, digest, n_ctx, n_ctx_train)` tuples.
    pub fn loaded_models(&self) -> Vec<(String, u64, u32, String, u32, u32)> {
        self.models
            .read()
            .unwrap()
            .iter()
            .map(|(name, m)| {
                (
                    name.clone(),
                    loaded_model_memory_bytes(m),
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
        tracing::info!(
            model = model_name,
            prompt_chars = prompt.len(),
            max_tokens = params.max_tokens,
            "generate request received"
        );

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
                tracing::info!(
                    prompt_tokens = stats.prompt_tokens,
                    completion_tokens = stats.completion_tokens,
                    elapsed_ms = elapsed_us / 1000,
                    "generation complete"
                );
                if params.prefill_only {
                    self.metrics.record_prefill(
                        stats.prompt_tokens as u64,
                        elapsed_us,
                        stats.cache_hit,
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
            Err(e) => {
                tracing::error!(error = %e, elapsed_ms = elapsed_us / 1000, "generation failed");
                self.metrics.record_error();
            }
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
        if let (Some(cache), Some(llama)) = (
            &self.kv_cache,
            loaded.model.as_any().downcast_ref::<LlamaCppModel>(),
        ) {
            // Avoid llama.cpp's 512-token default batch cap.
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(loaded.n_ctx))
                .with_n_batch(loaded.n_ctx);
            let mut ctx = llama
                .llama_model()
                .new_context(llama.llama_backend(), ctx_params)
                .map_err(|e| anyhow::anyhow!("failed to create context: {e}"))?;
            return generate_streaming_cached(
                CachedGenerateRequest {
                    model: llama.llama_model(),
                    ctx: &mut ctx,
                    prompt,
                    params,
                    model_name,
                    model_digest: &loaded.digest,
                    cache,
                    encryption_key,
                },
                on_token,
            );
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

fn kv_bytes_for(model: &LlamaModel, n_ctx: u32) -> u64 {
    crate::backend::llamacpp::kv_bytes_per_token(model) * n_ctx as u64
}

fn loaded_model_memory_bytes(model: &LoadedModel) -> u64 {
    let kv = model
        .model
        .as_any()
        .downcast_ref::<LlamaCppModel>()
        .map(|lm| kv_bytes_for(lm.llama_model(), model.n_ctx))
        .unwrap_or(0);

    model
        .size_bytes
        .saturating_add(model.scheduler_size_bytes)
        .saturating_add(kv)
}

fn scheduler_weight_bytes(
    format: &ModelFormat,
    model_weight_bytes: u64,
    batch_slots: usize,
) -> u64 {
    if batch_slots > 0 && *format == ModelFormat::Gguf {
        model_weight_bytes
    } else {
        0
    }
}

fn planned_load_weight_bytes(model_weight_bytes: u64, scheduler_weight_bytes: u64) -> u64 {
    model_weight_bytes.saturating_add(scheduler_weight_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;

    struct DummyModel;

    impl BackendModel for DummyModel {
        fn generate(
            &self,
            _prompt: &str,
            _params: &GenerateParams,
            _on_token: &mut dyn FnMut(&str) -> bool,
        ) -> anyhow::Result<GenerateResult> {
            Ok(GenerateResult {
                prompt_tokens: 0,
                completion_tokens: 0,
                cache_hit: false,
            })
        }

        fn apply_chat_template(&self, _messages: &[(String, String)]) -> anyhow::Result<String> {
            Ok(String::new())
        }

        fn n_ctx(&self) -> u32 {
            0
        }

        fn size_bytes(&self) -> u64 {
            0
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn dummy_loaded(size_bytes: u64, scheduler_size_bytes: u64) -> LoadedModel {
        LoadedModel {
            model: Box::new(DummyModel),
            file_path: PathBuf::new(),
            n_ctx: 0,
            n_ctx_train: 0,
            size_bytes,
            scheduler_size_bytes,
            last_used: RwLock::new(Instant::now()),
            gpu_layers: 0,
            digest: String::new(),
            format: ModelFormat::Mlx,
            batch_tx: None,
        }
    }

    fn dummy_manager(memory_budget: u64, clamp_budget_to_live: bool) -> ModelManager {
        ModelManager {
            backends: Vec::new(),
            models: RwLock::new(HashMap::new()),
            default_n_ctx: 0,
            default_gpu_layers: 0,
            memory_budget,
            clamp_budget_to_live,
            kv_cache: None,
            ram_cache: None,
            metrics: Arc::new(Metrics::new()),
            batch_slots: 0,
        }
    }

    mod scheduler_weight_bytes {
        use super::*;

        #[test]
        fn returns_model_size_when_batching_gguf() {
            let extra = super::super::scheduler_weight_bytes(&ModelFormat::Gguf, 128, 1);

            assert_eq!(extra, 128);
        }

        #[test]
        fn returns_zero_when_batching_disabled() {
            let extra = super::super::scheduler_weight_bytes(&ModelFormat::Gguf, 128, 0);

            assert_eq!(extra, 0);
        }

        #[test]
        fn returns_zero_for_mlx_models() {
            let extra = super::super::scheduler_weight_bytes(&ModelFormat::Mlx, 128, 1);

            assert_eq!(extra, 0);
        }
    }

    mod planned_load_weight_bytes {
        #[test]
        fn includes_scheduler_weight_copy() {
            let needed = super::super::planned_load_weight_bytes(128, 128);

            assert_eq!(needed, 256);
        }

        #[test]
        fn saturates_on_overflow() {
            let needed = super::super::planned_load_weight_bytes(u64::MAX, 1);

            assert_eq!(needed, u64::MAX);
        }
    }

    mod loaded_model_memory_bytes {
        use super::*;

        #[test]
        fn includes_scheduler_weight_copy() {
            let loaded = dummy_loaded(128, 256);

            assert_eq!(super::super::loaded_model_memory_bytes(&loaded), 384);
        }
    }

    mod model_manager {
        use super::*;

        #[test]
        fn effective_budget_keeps_total_ram_mode() {
            let manager = dummy_manager(123, false);

            assert_eq!(manager.effective_budget(), 123);
        }

        #[test]
        fn loaded_models_reports_scheduler_weight_copy() {
            let manager = dummy_manager(1_000, false);
            manager
                .models
                .write()
                .unwrap()
                .insert("model".to_string(), dummy_loaded(128, 256));

            assert_eq!(manager.loaded_models()[0].1, 384);
        }

        #[test]
        fn evict_for_counts_scheduler_weight_copy() {
            let manager = dummy_manager(500, false);
            manager
                .models
                .write()
                .unwrap()
                .insert("model".to_string(), dummy_loaded(128, 256));

            manager.evict_for(128).unwrap();

            assert!(!manager.is_loaded("model"));
        }
    }
}
