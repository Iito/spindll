use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};
use std::time::{Duration, Instant};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::num::NonZeroU32;
use tokio::task::JoinHandle;

use crate::backend::llamacpp::LlamaCppModel;
use crate::backend::{BackendLoadParams, BackendModel, InferenceBackend};
use crate::model_store::registry::ModelFormat;

use super::batch::{BatchEvent, BatchRequest, BatchScheduler};
use super::kv_cache::KvCache;
use super::kv_ram_cache::KvRamCache;
use super::metrics::Metrics;
use super::ram_cache::RamCache;
use super::streaming::{GenerateParams, GenerateResult, generate_streaming_cached};

/// Eviction tier. Low evicts first, LRU tiebreak within tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvictionPriority {
    Low,
    #[default]
    Normal,
    High,
}

impl EvictionPriority {
    fn evict_order(self) -> u8 {
        match self {
            Self::Low => 0,
            Self::Normal => 1,
            Self::High => 2,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// `None` = auto.
    pub gpu_layers: Option<u32>,
    pub digest: String,
    pub priority: EvictionPriority,
    /// Reload this long after eviction under VRAM pressure. `None` disables.
    pub idle_reload: Option<Duration>,
}

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
    /// Memory used by the batch scheduler's GGUF model copy (0 if no scheduler).
    pub scheduler_size_bytes: u64,
    /// Timestamp of the last inference request (used for LRU eviction).
    pub last_used: RwLock<Instant>,
    /// Number of layers offloaded to GPU.
    pub gpu_layers: u32,
    /// Original gpu_layers request, reused on idle-reload.
    pub requested_gpu_layers: Option<u32>,
    /// SHA-256 digest of the model file, used for KV cache keying.
    pub digest: String,
    /// On-disk format of this model.
    pub format: ModelFormat,
    /// Channel to submit requests to this model's batch scheduler (if running).
    pub batch_tx: Option<std::sync::mpsc::Sender<BatchRequest>>,
    pub priority: EvictionPriority,
    pub idle_reload: Option<Duration>,
}

#[derive(Debug, Clone)]
struct PendingReload {
    name: String,
    path: PathBuf,
    digest: String,
    gpu_layers: Option<u32>,
    priority: EvictionPriority,
    idle_reload: Duration,
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
    kv_cache: Option<KvCache>,
    kv_ram_cache: Option<KvRamCache>,
    ram_cache: Option<RamCache>,
    metrics: Arc<Metrics>,
    batch_slots: usize,
    /// Idle-reload gating: timestamp of last load/generate.
    last_activity: RwLock<Instant>,
    /// Idle-reload waits for this to hit zero.
    active_requests: AtomicUsize,
    /// Set by [`ModelManager::into_arc`]; required for idle-reload watchers.
    weak_self: OnceLock<Weak<ModelManager>>,
    watchers: Mutex<HashMap<String, JoinHandle<()>>>,
}

fn scheduler_weight_bytes(format: &ModelFormat, model_weight: u64, batch_slots: usize) -> u64 {
    if batch_slots > 0 && *format == ModelFormat::Gguf {
        model_weight
    } else {
        0
    }
}

fn planned_load_weight_bytes(model_weight: u64, scheduler_weight: u64) -> u64 {
    model_weight + scheduler_weight
}

fn loaded_model_memory_bytes(m: &LoadedModel) -> u64 {
    m.size_bytes
        + m.scheduler_size_bytes
        + m.model.kv_bytes_per_token() * m.n_ctx as u64
}

/// Bumps `active_requests` for the lifetime of an inference call.
struct ActiveGuard<'a>(&'a AtomicUsize);
impl<'a> ActiveGuard<'a> {
    fn new(c: &'a AtomicUsize) -> Self {
        c.fetch_add(1, Ordering::SeqCst);
        Self(c)
    }
}
impl<'a> Drop for ActiveGuard<'a> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl ModelManager {
    /// Create a new manager. Pass `gpu_layers = None` to auto-detect (all layers on macOS Metal,
    /// CPU-only elsewhere). Pass `memory_budget = 0` to use total physical RAM as the cap
    /// (recommended on Apple Silicon's unified memory).
    pub fn new(n_ctx: u32, gpu_layers: Option<u32>, memory_budget: u64) -> anyhow::Result<Self> {
        #[allow(unused_mut)]
        let mut backends: Vec<Box<dyn InferenceBackend>> = vec![
            Box::new(crate::backend::llamacpp::LlamaCppBackend::new()?),
        ];

        #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
        {
            if ensure_mlx_metallib() {
                backends.push(Box::new(crate::backend::mlx_swift::MlxBackend));
            } else {
                tracing::warn!("mlx.metallib not available; MLX backend disabled");
            }
        }

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

        let memory_budget = if memory_budget == 0 {
            sysinfo::System::new_all().total_memory()
        } else {
            memory_budget
        };

        Ok(Self {
            backends,
            models: RwLock::new(HashMap::new()),
            default_n_ctx: n_ctx,
            default_gpu_layers,
            memory_budget,
            kv_cache: None,
            kv_ram_cache: None,
            ram_cache: None,
            metrics: Arc::new(Metrics::new()),
            batch_slots: 0,
            last_activity: RwLock::new(Instant::now()),
            active_requests: AtomicUsize::new(0),
            weak_self: OnceLock::new(),
            watchers: Mutex::new(HashMap::new()),
        })
    }

    #[cfg(test)]
    pub fn with_backends(backends: Vec<Box<dyn InferenceBackend>>, memory_budget: u64) -> Self {
        Self {
            backends,
            models: RwLock::new(HashMap::new()),
            default_n_ctx: 2048,
            default_gpu_layers: 0,
            memory_budget,
            kv_cache: None,
            kv_ram_cache: None,
            ram_cache: None,
            metrics: Arc::new(Metrics::new()),
            batch_slots: 0,
            last_activity: RwLock::new(Instant::now()),
            active_requests: AtomicUsize::new(0),
            weak_self: OnceLock::new(),
            watchers: Mutex::new(HashMap::new()),
        }
    }

    /// Wrap into `Arc<Self>` with a `Weak` self-ref for idle-reload watchers.
    /// Required if any load uses `idle_reload`; `Arc::new(mgr)` disables it silently.
    pub fn into_arc(self) -> Arc<Self> {
        let arc = Arc::new(self);
        let _ = arc.weak_self.set(Arc::downgrade(&arc));
        arc
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

    fn effective_budget(&self) -> u64 {
        self.memory_budget
    }

    /// Evict by (priority, LRU) until `needed` bytes fit. Returns reload specs
    /// for any evicted model with `idle_reload` set; caller arms watchers.
    fn evict_for(&self, needed: u64) -> anyhow::Result<Vec<PendingReload>> {
        let mut pending = Vec::new();
        let budget = self.effective_budget();
        if budget == 0 {
            return Ok(pending);
        }

        loop {
            let used = self.total_loaded_bytes();
            if used + needed <= budget {
                return Ok(pending);
            }

            let victim = {
                let models = self.models.read().unwrap();
                if models.is_empty() {
                    anyhow::bail!(
                        "model needs {:.1} GB but budget is {:.1} GB",
                        needed as f64 / 1_073_741_824.0,
                        budget as f64 / 1_073_741_824.0
                    );
                }
                pick_evict_victim(models.iter().map(|(name, m)| {
                    (name.as_str(), m.priority, *m.last_used.read().unwrap())
                }))
                .map(|s| s.to_string())
                .unwrap()
            };

            tracing::warn!(model = %victim, "evicting model under VRAM pressure");
            let evicted = self.models.write().unwrap().remove(&victim);
            if let Some(model) = evicted {
                if let Some(cache) = &self.ram_cache {
                    cache.warm(&victim, &model.file_path);
                }
                if let Some(d) = model.idle_reload {
                    pending.push(PendingReload {
                        name: victim.clone(),
                        path: model.file_path.clone(),
                        digest: model.digest.clone(),
                        gpu_layers: model.requested_gpu_layers,
                        priority: model.priority,
                        idle_reload: d,
                    });
                }
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
        self.load_model_with_options(
            name,
            path,
            LoadOptions { gpu_layers, ..LoadOptions::default() },
        )
    }

    /// Load a model with an explicit file digest for KV cache keying.
    pub fn load_model_with_digest(
        &self,
        name: &str,
        path: &Path,
        gpu_layers: Option<u32>,
        digest: String,
    ) -> anyhow::Result<()> {
        self.load_model_with_options(
            name,
            path,
            LoadOptions { gpu_layers, digest, ..LoadOptions::default() },
        )
    }

    #[tracing::instrument(skip(self, path, opts), fields(file_size))]
    pub fn load_model_with_options(
        &self,
        name: &str,
        path: &Path,
        opts: LoadOptions,
    ) -> anyhow::Result<()> {
        self.cancel_watcher(name);
        let LoadOptions { gpu_layers, digest, priority, idle_reload } = opts;
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
        let planned_scheduler_bytes = scheduler_weight_bytes(&format, file_size, self.batch_slots);
        let pending_reloads = self.evict_for(
            planned_load_weight_bytes(file_size, planned_scheduler_bytes),
        )?;

        for spec in pending_reloads {
            self.arm_reload_watcher(spec);
        }

        // Snapshot live availability BEFORE the load so the backend can
        // budget-resolve n_ctx against memory not yet consumed by weights.
        // Once weights are mmap'd / uploaded to Metal, available memory drops,
        // so the snapshot has to happen here, not inside the backend.
        let mem = crate::scheduler::budget::MemoryBudget::detect(None);
        let load_budget = std::cmp::min(self.memory_budget, mem.available_ram);
        let load_budget = load_budget.saturating_sub(planned_scheduler_bytes);

        let layers = gpu_layers.unwrap_or(self.default_gpu_layers);

        let load_params = BackendLoadParams {
            n_ctx: self.default_n_ctx,
            n_gpu_layers: Some(layers),
            memory_budget: load_budget,
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

        // Scheduler needs its own GGUF model copy.
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

        let scheduler_size_bytes = if batch_tx.is_some() { size_bytes } else { 0 };
        let loaded = LoadedModel {
            model,
            file_path: path.to_path_buf(),
            n_ctx,
            n_ctx_train,
            size_bytes,
            scheduler_size_bytes,
            last_used: RwLock::new(Instant::now()),
            gpu_layers: layers,
            requested_gpu_layers: gpu_layers,
            digest,
            format,
            batch_tx,
            priority,
            idle_reload,
        };

        self.models.write().unwrap().insert(name.to_string(), loaded);
        self.record_activity();
        Ok(())
    }

    fn record_activity(&self) {
        *self.last_activity.write().unwrap() = Instant::now();
    }

    fn cancel_watcher(&self, name: &str) {
        if let Some(handle) = self.watchers.lock().unwrap().remove(name) {
            handle.abort();
        }
    }

    /// No-ops without `into_arc` or a tokio runtime.
    fn arm_reload_watcher(&self, spec: PendingReload) {
        let Some(weak) = self.weak_self.get().cloned() else {
            tracing::debug!(model = %spec.name, "idle-reload disabled: manager not in Arc");
            return;
        };
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::debug!(model = %spec.name, "idle-reload disabled: no tokio runtime");
            return;
        };
        let name = spec.name.clone();
        let task = handle.spawn(reload_watcher(weak, spec));
        if let Some(prev) = self.watchers.lock().unwrap().insert(name, task) {
            prev.abort();
        }
    }

    #[tracing::instrument(skip(self))]
    pub fn unload_model(&self, name: &str) -> anyhow::Result<()> {
        self.cancel_watcher(name);
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
    /// The in-memory RAM tier is independent — callers must enable it
    /// explicitly via [`Self::enable_kv_ram_cache`] so `--no-kv-ram-cache`
    /// can fully disable it.
    pub fn enable_kv_cache(&mut self, max_bytes: u64) {
        self.kv_cache = Some(KvCache::new(max_bytes));
    }

    /// Enable the in-memory KV state cache with the given max size in bytes.
    pub fn enable_kv_ram_cache(&mut self, max_bytes: u64) {
        self.kv_ram_cache = Some(KvRamCache::new(max_bytes));
    }

    /// Disable the in-memory KV state cache.
    pub fn disable_kv_ram_cache(&mut self) {
        self.kv_ram_cache = None;
    }

    /// Returns a reference to the KV RAM cache, if enabled.
    pub fn kv_ram_cache(&self) -> Option<&KvRamCache> {
        self.kv_ram_cache.as_ref()
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
        let _active = ActiveGuard::new(&self.active_requests);
        let batch_tx = {
            let models = self.models.read().unwrap();
            models.get(model_name).and_then(|m| {
                *m.last_used.write().unwrap() = Instant::now();
                m.batch_tx.clone()
            })
        };
        *self.last_activity.write().unwrap() = Instant::now();

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
        if let Some(cache) = &self.kv_cache {
            if let Some(llama) = loaded.model.as_any().downcast_ref::<LlamaCppModel>() {
                // n_batch == n_ctx so prefill batches always fit. Default n_batch=512
                // hits GGML_ASSERT and crashes on prompts longer than 512 tokens.
                let ctx_params = LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(loaded.n_ctx))
                    .with_n_batch(loaded.n_ctx);
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
                    self.kv_ram_cache.as_ref(),
                    encryption_key,
                    on_token,
                );
            }
        }

        loaded.model.generate(prompt, params, on_token)
    }

    /// Apply the model's chat template and generate in one step.
    ///
    /// Backends that support fused template + generation (e.g. MLX) use a
    /// single FFI call, eliminating the decode → encode round-trip. All other
    /// backends fall back to `apply_chat_template` + `generate`.
    #[tracing::instrument(
        skip(self, messages, params, on_token, encryption_key),
        fields(model = model_name)
    )]
    pub fn generate_chat(
        &self,
        model_name: &str,
        messages: &[(String, String)],
        params: &GenerateParams,
        encryption_key: Option<&[u8; 32]>,
        mut on_token: impl FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult> {
        let _active = ActiveGuard::new(&self.active_requests);
        let batch_tx = {
            let models = self.models.read().unwrap();
            models.get(model_name).and_then(|m| {
                *m.last_used.write().unwrap() = Instant::now();
                m.batch_tx.clone()
            })
        };
        *self.last_activity.write().unwrap() = Instant::now();

        let start = Instant::now();
        let result = if let Some(tx) = batch_tx {
            // Batch scheduler path is llama.cpp-only and requires a prompt string.
            let prompt = self.apply_chat_template(model_name, messages)?;
            Self::generate_via_batch(tx, &prompt, params, &mut on_token)
        } else {
            self.generate_chat_direct(model_name, messages, params, encryption_key, &mut on_token)
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
            Err(e) => {
                tracing::error!(error = %e, elapsed_ms = elapsed_us / 1000, "generation failed");
                self.metrics.record_error();
            }
        }
        result
    }

    /// Per-request context path for `generate_chat`.
    fn generate_chat_direct(
        &self,
        model_name: &str,
        messages: &[(String, String)],
        params: &GenerateParams,
        encryption_key: Option<&[u8; 32]>,
        on_token: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult> {
        let models = self.models.read().unwrap();
        let loaded = models
            .get(model_name)
            .ok_or_else(|| anyhow::anyhow!("model '{}' not loaded", model_name))?;
        *loaded.last_used.write().unwrap() = Instant::now();

        // KV cache path: GGUF models need the prompt as a string.
        if let Some(cache) = &self.kv_cache {
            if let Some(llama) = loaded.model.as_any().downcast_ref::<LlamaCppModel>() {
                let prompt = loaded.model.apply_chat_template(messages)?;
                let ctx_params = LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(loaded.n_ctx))
                    .with_n_batch(loaded.n_ctx);
                let mut ctx = llama
                    .llama_model()
                    .new_context(llama.llama_backend(), ctx_params)
                    .map_err(|e| anyhow::anyhow!("failed to create context: {e}"))?;
                return generate_streaming_cached(
                    llama.llama_model(),
                    &mut ctx,
                    &prompt,
                    params,
                    model_name,
                    &loaded.digest,
                    cache,
                    self.kv_ram_cache.as_ref(),
                    encryption_key,
                    on_token,
                );
            }
        }

        // All other backends (MLX): fused template + generation.
        loaded.model.generate_chat(messages, params, on_token)
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

    /// Compute an embedding vector for a text string.
    #[tracing::instrument(skip(self, text), fields(model = model_name))]
    pub fn embed(
        &self,
        model_name: &str,
        text: &str,
    ) -> anyhow::Result<crate::backend::EmbedResult> {
        let _active = ActiveGuard::new(&self.active_requests);
        let models = self.models.read().unwrap();
        let loaded = models
            .get(model_name)
            .ok_or_else(|| anyhow::anyhow!("model '{}' not loaded", model_name))?;
        *loaded.last_used.write().unwrap() = Instant::now();
        self.record_activity();

        let start = Instant::now();
        let result = loaded.model.embed(text);
        let elapsed_us = start.elapsed().as_micros() as u64;

        match &result {
            Ok(r) => {
                tracing::info!(
                    prompt_tokens = r.prompt_tokens,
                    embedding_dim = r.embedding.len(),
                    elapsed_ms = elapsed_us / 1000,
                    "embedding complete"
                );
                self.metrics.record_embed(r.prompt_tokens as u64, elapsed_us);
            }
            Err(e) => {
                tracing::error!(error = %e, elapsed_ms = elapsed_us / 1000, "embedding failed");
                self.metrics.record_embed_error();
            }
        }
        result
    }
}

/// Lowest priority first, oldest `last_used` as tiebreak.
fn pick_evict_victim<'a, I>(candidates: I) -> Option<&'a str>
where
    I: IntoIterator<Item = (&'a str, EvictionPriority, Instant)>,
{
    candidates
        .into_iter()
        .min_by_key(|(_, p, t)| (p.evict_order(), *t))
        .map(|(name, _, _)| name)
}

/// Sleep idle_reload, recheck idle + active_requests + headroom, reload if safe.
async fn reload_watcher(weak: Weak<ModelManager>, spec: PendingReload) {
    loop {
        tokio::time::sleep(spec.idle_reload).await;
        let Some(mgr) = weak.upgrade() else { return };
        if mgr.is_loaded(&spec.name) {
            return;
        }
        let idle_for = mgr.last_activity.read().unwrap().elapsed();
        if idle_for < spec.idle_reload {
            continue;
        }
        if mgr.active_requests.load(Ordering::SeqCst) > 0 {
            continue;
        }
        if mgr.memory_budget != 0 {
            let needed = match std::fs::metadata(&spec.path) {
                Ok(m) => m.len(),
                Err(e) => {
                    tracing::warn!(model = %spec.name, error = %e, "idle-reload abort: stat failed");
                    return;
                }
            };
            let used = mgr.total_loaded_bytes();
            let margin = mgr.memory_budget / 10;
            if used.saturating_add(needed).saturating_add(margin) > mgr.memory_budget {
                tracing::debug!(model = %spec.name, "idle-reload skipped: insufficient headroom");
                return;
            }
        }
        let opts = LoadOptions {
            gpu_layers: spec.gpu_layers,
            digest: spec.digest.clone(),
            priority: spec.priority,
            idle_reload: Some(spec.idle_reload),
        };
        let mgr_blocking = mgr.clone();
        let path = spec.path.clone();
        let name = spec.name.clone();
        let res = tokio::task::spawn_blocking(move || {
            mgr_blocking.load_model_with_options(&name, &path, opts)
        })
        .await;
        match res {
            Ok(Ok(())) => tracing::info!(model = %spec.name, "idle-reloaded"),
            Ok(Err(e)) => tracing::warn!(model = %spec.name, error = %e, "idle-reload failed"),
            Err(e) => tracing::warn!(model = %spec.name, error = %e, "idle-reload task panicked"),
        }
        return;
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
static MLX_METALLIB_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mlx.metallib"));

/// Ensure mlx.metallib is available next to the binary where MLX's
/// `load_colocated_library("mlx")` expects it.
///
/// If a sidecar file already exists and matches the embedded size, keep it.
/// Otherwise extract the embedded copy directly next to the binary.
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
fn ensure_mlx_metallib() -> bool {
    let exe_dir = match std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
    {
        Some(d) => d,
        None => return false,
    };

    let sidecar = exe_dir.join("mlx.metallib");
    if sidecar.exists() {
        let up_to_date = sidecar
            .metadata()
            .map(|m| m.len() == MLX_METALLIB_BYTES.len() as u64)
            .unwrap_or(false);
        if up_to_date {
            return true;
        }
    }
    if exe_dir.join("Resources/mlx.metallib").exists() {
        return true;
    }

    // Extract embedded copy next to binary.
    match std::fs::write(&sidecar, MLX_METALLIB_BYTES) {
        Ok(_) => {
            tracing::info!("extracted mlx.metallib → {}", sidecar.display());
            true
        }
        Err(e) => {
            tracing::warn!(error = %e, "cannot write mlx.metallib next to binary");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::streaming::GenerateResult;

    struct FakeBackend;

    impl InferenceBackend for FakeBackend {
        fn load_model(
            &self,
            _path: &Path,
            _params: BackendLoadParams,
        ) -> anyhow::Result<Box<dyn BackendModel>> {
            Ok(Box::new(FakeModel))
        }
        fn name(&self) -> &str { "llamacpp" }
    }

    struct FakeModel;

    impl BackendModel for FakeModel {
        fn generate(
            &self,
            _prompt: &str,
            _params: &GenerateParams,
            _on_token: &mut dyn FnMut(&str) -> bool,
        ) -> anyhow::Result<GenerateResult> {
            Ok(GenerateResult::default())
        }
        fn apply_chat_template(
            &self,
            _messages: &[(String, String)],
        ) -> anyhow::Result<String> {
            Ok(String::new())
        }
        fn n_ctx(&self) -> u32 { 2048 }
        fn n_ctx_train(&self) -> u32 { 4096 }
        fn size_bytes(&self) -> u64 { 100 }
        fn kv_bytes_per_token(&self) -> u64 { 1 }
        fn as_any(&self) -> &dyn std::any::Any { self }
    }

    fn test_manager(budget: u64) -> ModelManager {
        ModelManager::with_backends(vec![Box::new(FakeBackend)], budget)
    }

    fn t(secs: u64) -> Instant {
        let base = Instant::now();
        base.checked_sub(Duration::from_secs(1_000)).unwrap_or(base)
            + Duration::from_secs(secs)
    }

    #[test]
    fn priority_evicts_low_first_even_when_recent() {
        let cands = vec![
            ("low_recent", EvictionPriority::Low, t(999)),
            ("normal_old", EvictionPriority::Normal, t(900)),
        ];
        assert_eq!(pick_evict_victim(cands), Some("low_recent"));
    }

    #[test]
    fn priority_lru_tiebreak_within_tier() {
        let cands = vec![
            ("a", EvictionPriority::Normal, t(950)),
            ("b", EvictionPriority::Normal, t(900)),
        ];
        assert_eq!(pick_evict_victim(cands), Some("b"));
    }

    #[test]
    fn high_priority_evicted_last() {
        let cands = vec![
            ("high_old", EvictionPriority::High, t(900)),
            ("normal_recent", EvictionPriority::Normal, t(999)),
        ];
        assert_eq!(pick_evict_victim(cands), Some("normal_recent"));
    }

    #[test]
    fn evict_order_total_ordering() {
        assert!(EvictionPriority::Low.evict_order() < EvictionPriority::Normal.evict_order());
        assert!(EvictionPriority::Normal.evict_order() < EvictionPriority::High.evict_order());
    }

    #[test]
    fn empty_candidates_returns_none() {
        let empty: Vec<(&str, EvictionPriority, Instant)> = vec![];
        assert_eq!(pick_evict_victim(empty), None);
    }

    fn fake_model_file(dir: &Path, name: &str, size: u64) -> PathBuf {
        let p = dir.join(name);
        let f = std::fs::File::create(&p).unwrap();
        f.set_len(size).unwrap();
        p
    }

    #[tokio::test]
    async fn idle_reload_fires_after_quiet_window() {
        let dir = tempfile::tempdir().unwrap();
        // Budget fits 2 models (2*2148=4296) but not 3 (4296+100 > 4300).
        let mgr = test_manager(4300);
        let mgr = mgr.into_arc();

        let p_a = fake_model_file(dir.path(), "a.gguf", 100);
        let p_b = fake_model_file(dir.path(), "b.gguf", 100);
        let p_c = fake_model_file(dir.path(), "c.gguf", 100);

        mgr.load_model_with_options("a", &p_a, LoadOptions {
            priority: EvictionPriority::Low,
            idle_reload: Some(Duration::from_millis(50)),
            ..Default::default()
        }).unwrap();
        mgr.load_model_with_options("b", &p_b, LoadOptions {
            priority: EvictionPriority::Normal,
            ..Default::default()
        }).unwrap();
        assert!(mgr.is_loaded("a"));

        // Loading "c" triggers eviction of "a" (Low priority), arming the watcher
        mgr.load_model_with_options("c", &p_c, LoadOptions::default()).unwrap();
        assert!(!mgr.is_loaded("a"), "a should have been evicted");

        // Unload "c" to free headroom for reload
        mgr.unload_model("c").unwrap();

        // Push last_activity into the past so the idle check passes
        *mgr.last_activity.write().unwrap() = Instant::now() - Duration::from_millis(200);

        // Poll until the idle watcher reloads "a" (generous timeout for slow CI)
        let deadline = Instant::now() + Duration::from_secs(5);
        while !mgr.is_loaded("a") && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(mgr.is_loaded("a"), "model should be reloaded by idle watcher");
    }

    #[tokio::test]
    async fn idle_reload_blocks_on_insufficient_headroom() {
        let dir = tempfile::tempdir().unwrap();
        // Budget fits 2 models (4296) but not 3 (4296+100 > 4300).
        // After evict+reload check: used(4296) + needed(100) + margin(430) > 4300.
        let mgr = test_manager(4300);
        let mgr = mgr.into_arc();

        let p_a = fake_model_file(dir.path(), "a.gguf", 100);
        let p_b = fake_model_file(dir.path(), "b.gguf", 100);
        let p_c = fake_model_file(dir.path(), "c.gguf", 100);

        mgr.load_model_with_options("a", &p_a, LoadOptions {
            priority: EvictionPriority::Low,
            idle_reload: Some(Duration::from_millis(50)),
            ..Default::default()
        }).unwrap();
        mgr.load_model_with_options("b", &p_b, LoadOptions {
            priority: EvictionPriority::Normal,
            ..Default::default()
        }).unwrap();

        // Evict "a" by loading "c"
        mgr.load_model_with_options("c", &p_c, LoadOptions::default()).unwrap();
        assert!(!mgr.is_loaded("a"));

        // Don't free "c" — budget stays full. Reload should skip due to headroom.
        *mgr.last_activity.write().unwrap() = Instant::now() - Duration::from_millis(200);
        tokio::time::sleep(Duration::from_millis(300)).await;

        assert!(!mgr.is_loaded("a"), "reload should not fire when budget is full");
    }

    #[tokio::test]
    async fn into_arc_arms_self_weak() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = test_manager(4300);
        // Plain Arc::new — no Weak, so idle-reload cannot fire
        let mgr = Arc::new(mgr);

        let p_a = fake_model_file(dir.path(), "a.gguf", 100);
        let p_b = fake_model_file(dir.path(), "b.gguf", 100);
        let p_c = fake_model_file(dir.path(), "c.gguf", 100);

        mgr.load_model_with_options("a", &p_a, LoadOptions {
            priority: EvictionPriority::Low,
            idle_reload: Some(Duration::from_millis(50)),
            ..Default::default()
        }).unwrap();
        mgr.load_model_with_options("b", &p_b, LoadOptions::default()).unwrap();

        mgr.load_model_with_options("c", &p_c, LoadOptions::default()).unwrap();
        assert!(!mgr.is_loaded("a"));

        mgr.unload_model("c").unwrap();
        *mgr.last_activity.write().unwrap() = Instant::now() - Duration::from_millis(200);
        tokio::time::sleep(Duration::from_millis(300)).await;

        assert!(!mgr.is_loaded("a"), "reload must not fire without into_arc()");
    }

    #[test]
    fn evict_for_skips_high_when_low_present() {
        let dir = tempfile::tempdir().unwrap();
        // Each model costs 100 (file) + 1*2048 (kv) = 2148 in total_loaded_bytes.
        // Budget fits 3 models (6444) but not 4 (6444 + 100 file > 6500).
        let mgr = test_manager(6500);

        let p_low = fake_model_file(dir.path(), "low.gguf", 100);
        let p_norm = fake_model_file(dir.path(), "norm.gguf", 100);
        let p_high = fake_model_file(dir.path(), "high.gguf", 100);

        mgr.load_model_with_options("low", &p_low, LoadOptions {
            priority: EvictionPriority::Low, ..Default::default()
        }).unwrap();
        mgr.load_model_with_options("norm", &p_norm, LoadOptions {
            priority: EvictionPriority::Normal, ..Default::default()
        }).unwrap();
        mgr.load_model_with_options("high", &p_high, LoadOptions {
            priority: EvictionPriority::High, ..Default::default()
        }).unwrap();

        let p_new = fake_model_file(dir.path(), "new.gguf", 100);
        mgr.load_model_with_options("new", &p_new, LoadOptions::default()).unwrap();

        let models = mgr.models.read().unwrap();
        assert!(!models.contains_key("low"), "Low-priority model should be evicted");
        assert!(models.contains_key("norm"));
        assert!(models.contains_key("high"));
        assert!(models.contains_key("new"));
    }

    fn real_gguf_path() -> Option<PathBuf> {
        let p = PathBuf::from(std::env::var("HOME").ok()?)
            .join(".spindll/models/Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf");
        p.exists().then_some(p)
    }

    #[test]
    #[ignore] // requires downloaded model: cargo test -- --ignored
    fn gguf_kv_bytes_per_token_is_positive() {
        let path = real_gguf_path().expect("GGUF model not found");
        let mgr = ModelManager::new(512, None, 0).unwrap();
        mgr.load_model("test", &path, None).unwrap();
        let models = mgr.models.read().unwrap();
        let loaded = &models["test"];
        assert!(loaded.model.kv_bytes_per_token() > 0, "kv_bytes_per_token must be positive");
    }

    #[test]
    #[ignore]
    fn gguf_prefill_only_returns_zero_completion_tokens() {
        let path = real_gguf_path().expect("GGUF model not found");
        let mgr = ModelManager::new(512, None, 0).unwrap();
        mgr.load_model("test", &path, None).unwrap();
        let params = crate::engine::streaming::GenerateParams {
            prefill_only: true,
            ..Default::default()
        };
        let mut token_count = 0u32;
        let result = mgr.generate("test", "Hello world", &params, None, |_| {
            token_count += 1;
            true
        }).unwrap();
        assert_eq!(result.completion_tokens, 0);
        assert_eq!(token_count, 0);
    }

    #[test]
    #[ignore]
    fn gguf_cancel_callback_stops_generation() {
        let path = real_gguf_path().expect("GGUF model not found");
        let mgr = ModelManager::new(512, None, 0).unwrap();
        mgr.load_model("test", &path, None).unwrap();
        let params = crate::engine::streaming::GenerateParams {
            max_tokens: 100,
            ..Default::default()
        };
        let mut token_count = 0u32;
        let _ = mgr.generate("test", "Count from 1 to 50", &params, None, |_| {
            token_count += 1;
            false // cancel immediately
        });
        assert!(token_count <= 1, "expected at most 1 token, got {token_count}");
    }

    #[test]
    #[ignore]
    fn gguf_embed_produces_normalized_vector() {
        let path = real_gguf_path().expect("GGUF model not found");
        let mgr = ModelManager::new(512, None, 0).unwrap();
        mgr.load_model("test", &path, None).unwrap();
        let result = mgr.embed("test", "Hello world").unwrap();
        assert!(!result.embedding.is_empty(), "embedding should not be empty");
        assert!(result.prompt_tokens > 0);
        let norm: f32 = result.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "embedding should be L2-normalized, got norm={norm}");
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    fn real_mlx_path() -> Option<PathBuf> {
        let p = PathBuf::from(std::env::var("HOME").ok()?)
            .join(".spindll/models/mlx-community/Qwen2.5-3B-Instruct-4bit");
        p.exists().then_some(p)
    }

    #[test]
    #[ignore]
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    fn mlx_kv_bytes_per_token_is_positive() {
        let path = real_mlx_path().expect("MLX model not found");
        let mgr = ModelManager::new(512, None, 0).unwrap();
        mgr.load_model("test-mlx", &path, None).unwrap();
        let models = mgr.models.read().unwrap();
        let loaded = &models["test-mlx"];
        assert!(loaded.model.kv_bytes_per_token() > 0);
    }

    #[test]
    #[ignore]
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    fn mlx_prefill_only_returns_zero_completion_tokens() {
        let path = real_mlx_path().expect("MLX model not found");
        let mgr = ModelManager::new(512, None, 0).unwrap();
        mgr.load_model("test-mlx", &path, None).unwrap();
        let params = crate::engine::streaming::GenerateParams {
            prefill_only: true,
            ..Default::default()
        };
        let mut token_count = 0u32;
        let result = mgr.generate_chat("test-mlx",
            &[("user".into(), "Hello".into())],
            &params, None, |_| { token_count += 1; true },
        ).unwrap();
        assert_eq!(result.completion_tokens, 0);
        assert_eq!(token_count, 0);
    }

    #[test]
    #[ignore]
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    fn mlx_cancel_callback_stops_generation() {
        let path = real_mlx_path().expect("MLX model not found");
        let mgr = ModelManager::new(512, None, 0).unwrap();
        mgr.load_model("test-mlx", &path, None).unwrap();
        let params = crate::engine::streaming::GenerateParams {
            max_tokens: 100,
            ..Default::default()
        };
        let mut token_count = 0u32;
        let _ = mgr.generate_chat("test-mlx",
            &[("user".into(), "Count from 1 to 50".into())],
            &params, None, |_| { token_count += 1; false },
        );
        assert!(token_count <= 1, "expected at most 1 token, got {token_count}");
    }

    #[test]
    fn scheduler_weight_bytes_gguf_with_batch() {
        assert_eq!(scheduler_weight_bytes(&ModelFormat::Gguf, 1000, 4), 1000);
    }

    #[test]
    fn scheduler_weight_bytes_mlx_is_zero() {
        assert_eq!(scheduler_weight_bytes(&ModelFormat::Mlx, 1000, 4), 0);
    }

    #[test]
    fn scheduler_weight_bytes_no_batch_is_zero() {
        assert_eq!(scheduler_weight_bytes(&ModelFormat::Gguf, 1000, 0), 0);
    }

    #[test]
    fn planned_load_includes_scheduler() {
        assert_eq!(planned_load_weight_bytes(500, 500), 1000);
    }

    #[test]
    fn effective_budget_returns_stored_value() {
        let mgr = test_manager(8_000_000_000);
        assert_eq!(mgr.effective_budget(), 8_000_000_000);
    }
}
