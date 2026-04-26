use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::num::NonZeroU32;

use super::batch::{BatchEvent, BatchRequest, BatchScheduler};
use super::kv_cache::KvCache;
use super::metrics::Metrics;
use super::ram_cache::RamCache;
use super::streaming::{GenerateParams, GenerateResult, generate_streaming, generate_streaming_cached};

/// A model that has been loaded into memory and is ready for inference.
pub struct LoadedModel {
    /// The underlying llama.cpp model handle.
    pub model: LlamaModel,
    /// Path to the GGUF file on disk.
    pub file_path: PathBuf,
    /// Context window size (number of tokens).
    pub n_ctx: u32,
    /// Trained context length from GGUF metadata (0 if unknown).
    pub n_ctx_train: u32,
    /// Approximate memory footprint in bytes.
    pub size_bytes: u64,
    /// Timestamp of the last inference request (used for LRU eviction).
    pub last_used: RwLock<Instant>,
    /// Number of layers offloaded to GPU.
    pub gpu_layers: u32,
    /// SHA-256 digest of the model file, used for KV cache keying.
    pub digest: String,
    /// Channel to submit requests to this model's batch scheduler (if running).
    pub batch_tx: Option<std::sync::mpsc::Sender<BatchRequest>>,
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
    ram_cache: Option<RamCache>,
    metrics: Arc<Metrics>,
    /// Maximum concurrent sequences per model for batch scheduling.
    /// 0 disables batching (each request gets its own context).
    batch_slots: usize,
}

impl ModelManager {
    /// Create a new manager. Pass `gpu_layers = None` to auto-detect (all layers on macOS Metal,
    /// CPU-only elsewhere). Pass `memory_budget = 0` to use total physical RAM as the cap
    /// (recommended on Apple Silicon's unified memory).
    pub fn new(n_ctx: u32, gpu_layers: Option<u32>, memory_budget: u64) -> anyhow::Result<Self> {
        super::suppress_llama_log();

        let backend = LlamaBackend::init()?;

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
            backend,
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
            .map(|m| m.size_bytes + kv_bytes_for(&m.model, m.n_ctx))
            .sum()
    }

    /// Pick the largest `n_ctx` that fits all three caps:
    /// 1. `default_n_ctx` (user-requested, 0 = unlimited)
    /// 2. `n_ctx_train` (model's trained max)
    /// 3. remaining memory budget after weights + already-loaded models
    ///
    /// Returns at least 512 even if budget is tight — refusing to load is the
    /// caller's job (via `evict_for`), this just sizes the context.
    fn resolve_n_ctx(
        &self,
        n_ctx_train: u32,
        weights: u64,
        kv_bpt: u64,
        available_before_load: u64,
    ) -> u32 {
        // Effective budget: the lesser of the configured cap and what the
        // system actually has free right now. `available_before_load` was
        // snapshotted before this model's weights were mmap'd, so subtracting
        // `weights` is correct — they have not yet shrunk the snapshot.
        let budget = if self.memory_budget > 0 {
            std::cmp::min(self.memory_budget, available_before_load)
        } else {
            available_before_load
        };

        // Hard caps from user and model. When the user explicitly sets
        // `--ctx-size N`, that's a deliberate ask — honor it as a hard target
        // and let llama.cpp / Metal surface OOM rather than silently shrinking
        // the window. Budget-aware sizing only applies on the `--ctx-size 0`
        // (auto) path.
        let user_explicit = self.default_n_ctx > 0;
        let user_cap = if user_explicit { self.default_n_ctx } else { u32::MAX };
        let train_cap = if n_ctx_train == 0 { u32::MAX } else { n_ctx_train };
        let mut n_ctx = std::cmp::min(user_cap, train_cap);

        // Budget cap (auto path only): solve
        // `weights + n_ctx * (kv_bpt + compute_bpt) <= budget`.
        // `compute_bpt` accounts for the Metal/CPU compute buffers ggml
        // allocates per context (roughly proportional to n_batch == n_ctx).
        // No additional OS reserve — `available_before_load` (free + inactive
        // + purgeable on macOS) already excludes what the OS holds, and
        // double-reserving makes auto silently undersize ctx vs. what an
        // explicit `--ctx-size` request would happily allocate.
        const COMPUTE_BPT: u64 = 8 * 1024; // ~8 KB/token across MTL0+CPU

        let bpt = kv_bpt + COMPUTE_BPT;
        if !user_explicit && budget > 0 && bpt > 0 {
            let remaining = budget.saturating_sub(weights);
            let budget_cap = (remaining / bpt).min(u32::MAX as u64) as u32;
            n_ctx = std::cmp::min(n_ctx, budget_cap);
        }

        // Floor at 512: refusing to load is `evict_for`'s job; this just sizes.
        let resolved = std::cmp::max(n_ctx, 512);
        tracing::debug!(
            user_cap, train_cap, weights, kv_bpt, available_before_load,
            budget, resolved, "resolved n_ctx"
        );
        resolved
    }

    /// Effective budget for the next load: the configured cap, lowered to
    /// what the system can actually provide right now. We use `available_ram`
    /// (free + inactive + purgeable on macOS) rather than the detect-default's
    /// 80% reserve — KV/n_ctx headroom is handled in `resolve_n_ctx`, and the
    /// `--budget 0` (total RAM) opt-in must not be silently downgraded.
    fn effective_budget(&self) -> u64 {
        if self.memory_budget == 0 {
            return 0; // unlimited
        }
        let live = crate::scheduler::budget::MemoryBudget::detect(None).available_ram;
        std::cmp::min(self.memory_budget, live)
    }

    /// Evict least-recently-used models until `needed` bytes fit within budget.
    fn evict_for(&self, needed: u64) -> anyhow::Result<()> {
        let budget = self.effective_budget();
        if budget == 0 {
            return Ok(()); // unlimited
        }

        loop {
            let used = self.total_loaded_bytes();
            if used + needed <= budget {
                return Ok(());
            }

            // Find LRU model
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
        let from_ram_cache = self
            .ram_cache
            .as_ref()
            .and_then(|c| c.get(name))
            .is_some();

        // Estimate size from file for budget check before loading
        let file_size = std::fs::metadata(path)?.len();
        self.evict_for(file_size)?;

        // Snapshot currently-available system memory BEFORE loading the model.
        // This becomes the budget ceiling for n_ctx — `min(configured_budget,
        // available_now)`. Snapshotting before load avoids double-counting:
        // once weights are mmap'd / uploaded to Metal, available memory drops.
        let available_before =
            crate::scheduler::budget::MemoryBudget::detect(None).available_ram;

        let layers = gpu_layers.unwrap_or(self.default_gpu_layers);

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(layers);
        let model = LlamaModel::load_from_file(&self.backend, path, &model_params)
            .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

        if from_ram_cache {
            if let Some(cache) = &self.ram_cache {
                cache.remove(name);
            }
        }

        let size_bytes = model.size();

        // Resolve n_ctx with three caps: user-requested (`default_n_ctx`),
        // model's trained length, and what fits in the remaining budget after
        // accounting for this model's weights + KV cache for already-loaded
        // models. KV cache scales linearly with n_ctx, so for large contexts
        // it can dwarf the weights.
        let n_ctx_train = model.n_ctx_train();
        let kv_bpt = kv_bytes_per_token(&model);
        let n_ctx = self.resolve_n_ctx(n_ctx_train, size_bytes, kv_bpt, available_before);

        let device = if layers == 0 {
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
            name = name,
            layers = model.n_layer(),
            device = device,
            size_bytes = size_bytes,
            n_ctx = n_ctx,
            n_ctx_train = n_ctx_train,
            "model loaded"
        );

        // Optionally start a batch scheduler for this model.
        let batch_tx = if self.batch_slots > 0 {
            let (tx, rx) = std::sync::mpsc::channel::<BatchRequest>();
            let n_ctx = n_ctx;
            let max_seq = self.batch_slots;
            let model_name = name.to_string();

            // The batch scheduler needs its own context, which requires a
            // reference to the model. Since LlamaModel isn't Send, we load
            // a second handle for the scheduler thread.
            let sched_params = LlamaModelParams::default().with_n_gpu_layers(layers);
            let sched_model = LlamaModel::load_from_file(&self.backend, path, &sched_params)
                .map_err(|e| anyhow::anyhow!("failed to load scheduler model: {e}"))?;
            let sched_backend = LlamaBackend::init()?;

            std::thread::Builder::new()
                .name(format!("batch-{model_name}"))
                .spawn(move || {
                    if let Err(e) = BatchScheduler::run(
                        &sched_model, &sched_backend, n_ctx, max_seq, rx,
                    ) {
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
            .map(|(name, m)| (name.clone(), m.size_bytes, m.gpu_layers, m.digest.clone(), m.n_ctx, m.n_ctx_train))
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
    /// (using the disk-backed KV cache if enabled).
    ///
    /// Pass `encryption_key` to encrypt cached KV state at rest (non-batched path only).
    #[tracing::instrument(skip(self, prompt, params, on_token, encryption_key), fields(model = model_name))]
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

        // Check if this model has a batch scheduler.
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
            self.generate_direct(model_name, prompt, params, encryption_key, on_token)
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

        tx.send(req).map_err(|_| anyhow::anyhow!("batch scheduler is not running"))?;

        // Drain events from the batch scheduler.
        loop {
            match resp_rx.blocking_recv() {
                Some(BatchEvent::Token(piece)) => {
                    if !on_token(&piece) {
                        // Client abort — drop the receiver so the scheduler
                        // detects the closed channel on its next send.
                        break;
                    }
                }
                Some(BatchEvent::Done { prompt_tokens, completion_tokens }) => {
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

        // If we broke out of the loop (client abort), wait for Done/Error.
        // The scheduler will notice the closed channel and finish the sequence.
        Ok(GenerateResult {
            prompt_tokens: 0,
            completion_tokens: 0,
            cache_hit: false,
        })
    }

    /// Per-request context path (original behavior).
    fn generate_direct(
        &self,
        model_name: &str,
        prompt: &str,
        params: &GenerateParams,
        encryption_key: Option<&[u8; 32]>,
        on_token: impl FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult> {
        self.with_model(model_name, |model, backend, n_ctx, digest| {
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(n_ctx))
                .with_n_batch(n_ctx);
            let mut ctx = model
                .new_context(backend, ctx_params)
                .map_err(|e| anyhow::anyhow!("failed to create context: {e}"))?;

            match &self.kv_cache {
                Some(cache) => generate_streaming_cached(
                    model, &mut ctx, prompt, params, model_name, digest, cache, encryption_key, on_token,
                ),
                None => generate_streaming(model, &mut ctx, prompt, params, on_token),
            }
        })
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
        self.with_model(model_name, |model, _, _, _| {
            super::apply_chat_template_with_fallback(model, messages)
        })
    }
}

/// KV cache bytes per token for a loaded model. The cache stores both K and V
/// tensors across all layers, sized by `n_head_kv * head_dim` per layer.
/// Assumes fp16 (2 bytes/element) — quantized KV would change this.
fn kv_bytes_per_token(model: &LlamaModel) -> u64 {
    let n_layer = model.n_layer() as u64;
    let n_head_kv = model.n_head_kv() as u64;
    let n_head = model.n_head().max(1) as u64;
    let n_embd = model.n_embd() as u64;
    let head_dim = n_embd / n_head;
    const KV: u64 = 2; // K and V
    const FP16: u64 = 2;
    KV * n_layer * n_head_kv * head_dim * FP16
}

fn kv_bytes_for(model: &LlamaModel, n_ctx: u32) -> u64 {
    kv_bytes_per_token(model) * n_ctx as u64
}

