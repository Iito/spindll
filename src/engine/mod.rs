//! Inference engine — model loading, text generation, KV caching, and metrics.

pub mod batch;
pub mod kv_cache;
pub mod kv_ram_cache;
pub mod manager;
pub mod metrics;
#[cfg(feature = "vision")]
pub mod multimodal;
pub mod ram_cache;
pub mod streaming;

pub use batch::{BatchEvent, BatchRequest, BatchScheduler};
pub use kv_cache::KvCache;
pub use kv_ram_cache::KvRamCache;
pub use manager::{EvictionPriority, LoadOptions, ModelManager};
pub use metrics::Metrics;
#[cfg(feature = "vision")]
pub use multimodal::{ContentPart, MultimodalMessage};
pub use ram_cache::RamCache;
pub use streaming::{GenerateParams, GenerateResult};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::num::NonZeroU32;
use std::path::Path;

/// Single-model inference engine wrapping a loaded GGUF model.
///
/// For multi-model use cases, prefer [`ModelManager`] instead.
/// `Engine` is useful when you need a lightweight wrapper around exactly one model.
pub struct Engine {
    model: LlamaModel,
    n_ctx: u32,
    model_digest: String,
    kv_cache: Option<KvCache>,
    kv_ram_cache: Option<KvRamCache>,
}

impl Engine {
    /// Load a model, auto-detecting GPU. Pass n_gpu_layers=None to offload all layers.
    #[tracing::instrument(skip(path), fields(n_ctx, gpu_layers))]
    pub fn load(path: &Path, n_gpu_layers: Option<u32>, n_ctx: u32) -> anyhow::Result<Self> {
        let backend = crate::backend::llamacpp::shared_backend();

        let gpu_layers = n_gpu_layers.unwrap_or({
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

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(gpu_layers);
        let model = LlamaModel::load_from_file(backend, path, &model_params)
            .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;

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
        tracing::info!(layers = model.n_layer(), device, "model loaded");

        Ok(Self {
            model,
            n_ctx,
            model_digest: String::new(),
            kv_cache: None,
            kv_ram_cache: None,
        })
    }

    /// Returns a reference to the underlying llama.cpp model.
    pub fn model(&self) -> &LlamaModel {
        &self.model
    }

    /// Returns a reference to the llama.cpp backend instance.
    pub fn backend(&self) -> &LlamaBackend {
        crate::backend::llamacpp::shared_backend()
    }

    /// Create a new inference context with the configured `n_ctx` window size.
    pub fn create_context(
        &self,
    ) -> anyhow::Result<llama_cpp_2::context::LlamaContext<'_>> {
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx));
        self.model
            .new_context(self.backend(), params)
            .map_err(|e| anyhow::anyhow!("failed to create context: {e}"))
    }

    /// Apply the model's built-in chat template to a list of (role, content) messages.
    /// Falls back to ChatML if the model has no embedded template.
    pub fn apply_chat_template(
        &self,
        messages: &[(String, String)],
    ) -> anyhow::Result<String> {
        apply_chat_template_with_fallback(&self.model, messages)
    }

    /// Enable the disk-backed KV cache with the given maximum size in bytes.
    /// Also enables the in-memory RAM tier (512 MB) unless already configured.
    pub fn enable_kv_cache(&mut self, max_bytes: u64) {
        self.kv_cache = Some(KvCache::new(max_bytes));
        if self.kv_ram_cache.is_none() {
            self.kv_ram_cache = Some(KvRamCache::new(512 * 1_048_576));
        }
    }

    /// Enable the in-memory KV state cache with the given maximum size in bytes.
    pub fn enable_kv_ram_cache(&mut self, max_bytes: u64) {
        self.kv_ram_cache = Some(KvRamCache::new(max_bytes));
    }

    /// Disable the in-memory KV state cache.
    pub fn disable_kv_ram_cache(&mut self) {
        self.kv_ram_cache = None;
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
                &self.model, &mut ctx, prompt, params, "default", &self.model_digest,
                cache, self.kv_ram_cache.as_ref(), None, on_token,
            ),
            None => streaming::generate_streaming(&self.model, &mut ctx, prompt, params, on_token),
        }
    }
}

/// Apply a model's chat template, falling back to ChatML if none is embedded.
///
/// Some templates — notably Gemma's — reject a standalone `system` role and make
/// llama.cpp's apply return an FFI error (`ffi error -1`). When that happens and
/// a system message is present, retry with the system content folded into the
/// first user turn (the shape those models expect) instead of failing the
/// request.
pub(crate) fn apply_chat_template_with_fallback(
    model: &LlamaModel,
    messages: &[(String, String)],
) -> anyhow::Result<String> {
    use llama_cpp_2::model::LlamaChatTemplate;

    let tmpl = match model.chat_template(None) {
        Ok(t) => t,
        Err(_) => {
            tracing::debug!("model has no chat template, using ChatML fallback");
            LlamaChatTemplate::new("chatml")
                .map_err(|e| anyhow::anyhow!("failed to create ChatML template: {e}"))?
        }
    };

    match render_chat(model, &tmpl, messages) {
        Ok(rendered) => Ok(rendered),
        Err(first_err) if has_system_message(messages) => {
            tracing::debug!(
                "chat template rejected the system role; folding it into the first user turn"
            );
            render_chat(model, &tmpl, &fold_system_into_user(messages)).map_err(|retry_err| {
                anyhow::anyhow!(
                    "failed to apply chat template: {first_err} \
                     (also failed after folding the system role into the user turn: {retry_err})"
                )
            })
        }
        Err(e) => Err(anyhow::anyhow!("failed to apply chat template: {e}")),
    }
}

/// Build `LlamaChatMessage`s and render them with `tmpl` (assistant generation
/// prompt appended). Returns the raw backend error so the caller can decide
/// whether to retry.
fn render_chat(
    model: &LlamaModel,
    tmpl: &llama_cpp_2::model::LlamaChatTemplate,
    messages: &[(String, String)],
) -> anyhow::Result<String> {
    let chat_messages: Vec<llama_cpp_2::model::LlamaChatMessage> = messages
        .iter()
        .map(|(role, content)| {
            llama_cpp_2::model::LlamaChatMessage::new(role.clone(), content.clone())
                .map_err(|e| anyhow::anyhow!("invalid chat message: {e}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(model.apply_chat_template(tmpl, &chat_messages, true)?)
}

/// True if any message carries the `system` role.
fn has_system_message(messages: &[(String, String)]) -> bool {
    messages.iter().any(|(role, _)| role == "system")
}

/// Fold every `system` message into the first following `user` turn — the shape
/// Gemma-style templates (which forbid a standalone system role) expect. System
/// text is prepended to the first user message; with no user message present it
/// becomes one.
fn fold_system_into_user(messages: &[(String, String)]) -> Vec<(String, String)> {
    let mut system_parts: Vec<&str> = Vec::new();
    let mut rest: Vec<(String, String)> = Vec::new();
    for (role, content) in messages {
        if role == "system" {
            if !content.is_empty() {
                system_parts.push(content);
            }
        } else {
            rest.push((role.clone(), content.clone()));
        }
    }
    if system_parts.is_empty() {
        return rest;
    }
    let system_text = system_parts.join("\n\n");
    match rest.iter().position(|(role, _)| role == "user") {
        Some(idx) => {
            let merged = format!("{system_text}\n\n{}", rest[idx].1);
            rest[idx].1 = merged;
        }
        None => rest.insert(0, ("user".to_string(), system_text)),
    }
    rest
}

/// Suppress llama.cpp's built-in stderr logging.
///
/// Installs a callback that drops INFO / DEBUG noise (model metadata
/// dumps, Metal device discovery, ggml init banners) while forwarding
/// WARN and ERROR through tracing so real failures aren't silenced.
/// Idempotent — `llama_log_set` overwrites the previous callback.
pub(crate) fn suppress_llama_log() {
    unsafe {
        llama_cpp_sys_2::llama_log_set(Some(noop_llama_log), std::ptr::null_mut());
    }
}

unsafe extern "C" fn noop_llama_log(
    level: llama_cpp_sys_2::ggml_log_level,
    text: *const std::ffi::c_char,
    _user_data: *mut std::ffi::c_void,
) {
    // Current ggml.h enum: 0=NONE, 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR, 5=CONT.
    // Forward WARN and ERROR through tracing; drop everything else.
    if text.is_null() {
        return;
    }
    let lvl = level;
    if lvl != llama_cpp_sys_2::GGML_LOG_LEVEL_WARN
        && lvl != llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR
    {
        return;
    }
    let msg = unsafe { std::ffi::CStr::from_ptr(text) }
        .to_string_lossy()
        .trim_end()
        .to_string();
    if msg.is_empty() {
        return;
    }
    if lvl == llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR {
        tracing::error!(target: "llama_cpp", "{msg}");
    } else {
        tracing::warn!(target: "llama_cpp", "{msg}");
    }
}

#[cfg(test)]
mod chat_template_tests {
    use super::{fold_system_into_user, has_system_message};

    fn m(role: &str, content: &str) -> (String, String) {
        (role.to_string(), content.to_string())
    }

    #[test]
    fn folds_system_into_first_user() {
        let folded = fold_system_into_user(&[
            m("system", "Be terse."),
            m("user", "Hi"),
            m("assistant", "Hello"),
            m("user", "Bye"),
        ]);
        assert_eq!(folded.len(), 3);
        assert_eq!(folded[0], m("user", "Be terse.\n\nHi"));
        assert_eq!(folded[1].0, "assistant");
        assert_eq!(folded[2], m("user", "Bye"));
        assert!(!has_system_message(&folded));
    }

    #[test]
    fn system_only_becomes_user() {
        assert_eq!(
            fold_system_into_user(&[m("system", "Rules.")]),
            vec![m("user", "Rules.")]
        );
    }

    #[test]
    fn system_without_user_inserts_user_turn() {
        let folded = fold_system_into_user(&[m("system", "Ctx"), m("assistant", "ok")]);
        assert_eq!(folded[0], m("user", "Ctx"));
        assert_eq!(folded[1], m("assistant", "ok"));
    }

    #[test]
    fn multiple_systems_joined() {
        assert_eq!(
            fold_system_into_user(&[m("system", "A"), m("system", "B"), m("user", "Q")]),
            vec![m("user", "A\n\nB\n\nQ")]
        );
    }

    #[test]
    fn no_system_left_unchanged() {
        let msgs = vec![m("user", "Hi"), m("assistant", "Hello")];
        assert_eq!(fold_system_into_user(&msgs), msgs);
        assert!(!has_system_message(&msgs));
    }
}
