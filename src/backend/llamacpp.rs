use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;

use crate::engine::streaming::{GenerateParams, GenerateResult, generate_streaming};
use crate::engine::apply_chat_template_with_fallback;
use super::traits::{BackendLoadParams, BackendModel, InferenceBackend};

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

pub fn shared_backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| {
        // Install the log filter BEFORE init so the backend's own ggml /
        // Metal init banners go through it instead of straight to stderr.
        crate::engine::suppress_llama_log();
        LlamaBackend::init().expect("LlamaBackend::init failed")
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

/// Pick the largest n_ctx that fits user request, model's trained max, and
/// remaining memory budget after weights. On `requested == 0` (auto), caps
/// n_ctx at `(memory_budget - weights) / (kv_bpt + compute_bpt)`. Explicit
/// `requested > 0` bypasses the budget cap — let llama.cpp/Metal surface
/// OOM rather than silently undersize what the user asked for. Floors at
/// 512; refusing to load is the caller's job.
fn resolve_n_ctx(
    model: &LlamaModel,
    requested: u32,
    n_ctx_train: u32,
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
        let kv_bpt = kv_bytes_per_token(model);
        let bpt = kv_bpt + COMPUTE_BPT;
        if bpt > 0 {
            let remaining = memory_budget.saturating_sub(weights);
            let budget_cap = (remaining / bpt).min(u32::MAX as u64) as u32;
            n_ctx = std::cmp::min(n_ctx, budget_cap);
        }
    }

    std::cmp::max(n_ctx, 512)
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
}
