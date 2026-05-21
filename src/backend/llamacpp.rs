use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

use crate::engine::streaming::{GenerateParams, GenerateResult, generate_streaming};
use crate::engine::apply_chat_template_with_fallback;
use super::traits::{BackendLoadParams, BackendModel, EmbedResult, InferenceBackend};

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

pub fn shared_backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| {
        // Install the log filter BEFORE init so the backend's own ggml /
        // Metal init banners go through it instead of straight to stderr.
        crate::engine::suppress_llama_log();
        let mut backend = LlamaBackend::init().expect("LlamaBackend::init failed");
        backend.void_logs();
        backend
    })
}

/// KV cache bytes per token for a loaded model — `2 * n_layer * n_head_kv *
/// head_dim * 2` (K+V, fp16). Used for budget-aware n_ctx sizing and for
/// estimating already-loaded KV usage during eviction.
pub fn kv_bytes_per_token(model: &LlamaModel) -> u64 {
    let n_layer = model.n_layer() as u64;
    let n_head_kv = model.n_head_kv() as u64;
    let n_head = model.n_head().max(1) as u64;
    let n_embd = model.n_embd() as u64;
    let head_dim = n_embd / n_head;
    const KV: u64 = 2;
    const FP16: u64 = 2;
    KV * n_layer * n_head_kv * head_dim * FP16
}

/// Pure n_ctx sizing logic, separated from model loading for testability.
///
/// `kv_bpt` is pre-computed from the model (KV cache bytes per token).
/// `requested == 0` triggers auto-sizing from budget; `requested > 0` is
/// user-explicit and bypasses the budget cap.
fn resolve_n_ctx_pure(
    requested: u32,
    n_ctx_train: u32,
    kv_bpt: u64,
    weights: u64,
    memory_budget: u64,
) -> u32 {
    let user_explicit = requested > 0;
    let user_cap = if user_explicit { requested } else { u32::MAX };
    let train_cap = if n_ctx_train == 0 { u32::MAX } else { n_ctx_train };
    let mut n_ctx = std::cmp::min(user_cap, train_cap);

    // ~8 KB/token across MTL0+CPU compute buffers (n_batch == n_ctx).
    const COMPUTE_BPT: u64 = 8 * 1024;

    if !user_explicit && memory_budget > 0 {
        let bpt = kv_bpt + COMPUTE_BPT;
        let remaining = memory_budget.saturating_sub(weights);
        if let Some(tokens) = remaining.checked_div(bpt) {
            let budget_cap = tokens.min(u32::MAX as u64) as u32;
            n_ctx = std::cmp::min(n_ctx, budget_cap);
        }
    }

    std::cmp::max(n_ctx, 512)
}

fn resolve_n_ctx(
    model: &LlamaModel,
    requested: u32,
    n_ctx_train: u32,
    weights: u64,
    memory_budget: u64,
) -> u32 {
    resolve_n_ctx_pure(requested, n_ctx_train, kv_bytes_per_token(model), weights, memory_budget)
}

pub struct LlamaCppBackend;

impl LlamaCppBackend {
    pub fn new() -> anyhow::Result<Self> {
        let _ = shared_backend();
        Ok(Self)
    }
}

impl InferenceBackend for LlamaCppBackend {
    fn load_model(
        &self,
        path: &Path,
        params: BackendLoadParams,
    ) -> anyhow::Result<Box<dyn BackendModel>> {
        let gpu_layers = params.n_gpu_layers.unwrap_or(if cfg!(target_os = "macos")
            || cfg!(feature = "cuda")
            || cfg!(feature = "metal")
            || cfg!(feature = "vulkan")
        {
            999
        } else {
            0
        });

        let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
        let model = LlamaModel::load_from_file(shared_backend(), path, &model_params)
            .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

        let n_ctx_train = model.n_ctx_train();
        let size_bytes = model.size();
        let n_ctx = resolve_n_ctx(&model, params.n_ctx, n_ctx_train, size_bytes, params.memory_budget);

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
            spec_ctx: Mutex::new(None),
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

/// SAFETY: llama.cpp contexts are safe to use from a single thread at a time;
/// Mutex ensures exclusive access.
struct SendCtx(LlamaContext<'static>);
unsafe impl Send for SendCtx {}

pub struct LlamaCppModel {
    // SAFETY: spec_ctx borrows from `model` via FFI handles; transmuted to
    // 'static. MUST be declared above `model` so it drops first.
    spec_ctx: Mutex<Option<SendCtx>>,
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
        shared_backend()
    }

    pub fn gpu_layers(&self) -> u32 {
        self.gpu_layers
    }

    /// Borrow a persistent context for speculative decoding. First call creates;
    /// subsequent calls reset KV and reuse, eliminating per-call ctx allocation.
    pub fn with_spec_ctx<R>(
        &self,
        _n_batch_min: u32,
        f: impl FnOnce(&LlamaModel, &mut LlamaContext) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        let mut guard = self.spec_ctx.lock().unwrap();
        if guard.is_none() {
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(self.n_ctx))
                .with_n_batch(self.n_ctx);
            let ctx = self
                .model
                .new_context(shared_backend(), ctx_params)
                .map_err(|e| anyhow::anyhow!("failed to create context: {e}"))?;
            // SAFETY: see field doc.
            let ctx_static: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
            *guard = Some(SendCtx(ctx_static));
        }
        let ctx = &mut guard.as_mut().unwrap().0;
        let _ = ctx.clear_kv_cache_seq(Some(0), None, None);
        // SAFETY: narrow lifetime back to &self.
        let ctx: &mut LlamaContext<'_> = unsafe { std::mem::transmute(ctx) };
        f(&self.model, ctx)
    }
}

impl BackendModel for LlamaCppModel {
    fn generate(
        &self,
        prompt: &str,
        params: &GenerateParams,
        on_token: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult> {
        // n_batch == n_ctx so prefill batches always fit. Default n_batch=512
        // hits GGML_ASSERT and crashes the engine on prompts longer than 512.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_n_batch(self.n_ctx);
        let mut ctx = self
            .model
            .new_context(shared_backend(), ctx_params)
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

    fn kv_bytes_per_token(&self) -> u64 {
        kv_bytes_per_token(&self.model)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn embed(&self, text: &str) -> anyhow::Result<EmbedResult> {
        let tokens = self
            .model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        if tokens.is_empty() {
            anyhow::bail!("empty input produces no tokens");
        }

        let n_tokens = tokens.len().min(self.n_ctx as usize);
        let n_batch: u32 = (n_tokens as u32).min(self.n_ctx);

        let ctx_params = LlamaContextParams::default()
            .with_embeddings(true)
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_n_batch(n_batch);
        let mut ctx = self
            .model
            .new_context(shared_backend(), ctx_params)
            .map_err(|e| anyhow::anyhow!("failed to create embedding context: {e}"))?;

        let mut batch = LlamaBatch::new(n_tokens.max(512), 1);
        let last_idx = n_tokens as i32 - 1;
        for (i, token) in (0_i32..).zip(tokens[..n_tokens].iter()) {
            batch.add(*token, i, &[0], i == last_idx)?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

        let raw = match ctx.embeddings_seq_ith(0) {
            Ok(emb) => emb.to_vec(),
            Err(llama_cpp_2::EmbeddingsError::NonePoolType) => {
                ctx.embeddings_ith(last_idx)
                    .map_err(|e| anyhow::anyhow!("embeddings_ith failed: {e}"))?
                    .to_vec()
            }
            Err(e) => anyhow::bail!("embeddings_seq_ith failed: {e}"),
        };

        let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        let embedding = if norm > 0.0 {
            raw.iter().map(|x| x / norm).collect()
        } else {
            raw
        };

        Ok(EmbedResult {
            embedding,
            prompt_tokens: n_tokens as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A representative KV-per-token value (Llama-3-8B: 32 layers, 8 kv_heads, 128 head_dim, fp16).
    const KV_BPT: u64 = 2 * 32 * 8 * 128 * 2; // 131_072 bytes

    #[test]
    fn explicit_n_ctx_ignores_budget() {
        // User asked for 4096 — budget can only fit ~800 tokens; honour the request.
        let budget = KV_BPT * 800 + 8 * 1024 * 800;
        let n_ctx = resolve_n_ctx_pure(4096, 131_072, KV_BPT, 0, budget);
        assert_eq!(n_ctx, 4096);
    }

    #[test]
    fn auto_n_ctx_capped_by_budget() {
        // weights = 0, budget fits exactly 2000 tokens worth of KV+compute.
        let bpt = KV_BPT + 8 * 1024;
        let budget = bpt * 2000;
        let n_ctx = resolve_n_ctx_pure(0, 131_072, KV_BPT, 0, budget);
        assert_eq!(n_ctx, 2000);
    }

    #[test]
    fn auto_n_ctx_budget_after_weights() {
        // weights consume half the budget.
        let bpt = KV_BPT + 8 * 1024;
        let budget = bpt * 4000;
        let weights = bpt * 2000;
        let n_ctx = resolve_n_ctx_pure(0, 131_072, KV_BPT, weights, budget);
        assert_eq!(n_ctx, 2000);
    }

    #[test]
    fn auto_n_ctx_capped_by_train_length() {
        // Budget fits 100k tokens but the model was only trained on 8k.
        let bpt = KV_BPT + 8 * 1024;
        let budget = bpt * 100_000;
        let n_ctx = resolve_n_ctx_pure(0, 8192, KV_BPT, 0, budget);
        assert_eq!(n_ctx, 8192);
    }

    #[test]
    fn auto_n_ctx_floors_at_512_when_budget_tiny() {
        // Budget is 1 byte — impossible to fit any tokens; floor kicks in.
        let n_ctx = resolve_n_ctx_pure(0, 131_072, KV_BPT, 0, 1);
        assert_eq!(n_ctx, 512);
    }

    #[test]
    fn explicit_n_ctx_floors_at_512() {
        // Absurdly small explicit request still gets the minimum floor.
        let n_ctx = resolve_n_ctx_pure(64, 131_072, KV_BPT, 0, 0);
        assert_eq!(n_ctx, 512);
    }

    #[test]
    fn unlimited_budget_zero_no_budget_cap() {
        // memory_budget == 0 means unlimited — auto n_ctx only capped by n_ctx_train.
        let n_ctx = resolve_n_ctx_pure(0, 4096, KV_BPT, 0, 0);
        assert_eq!(n_ctx, 4096);
    }
}
