//! Inference engine — model loading, text generation, KV caching, and metrics.

pub mod batch;
pub(crate) mod chat_template;
pub mod kv_cache;
pub mod kv_ram_cache;
pub mod manager;
pub mod metrics;
#[cfg(feature = "vision")]
pub mod multimodal;
pub mod ram_cache;
pub mod streaming;
pub mod tools;

pub use batch::{BatchEvent, BatchRequest, BatchScheduler};
pub use kv_cache::KvCache;
pub use kv_ram_cache::KvRamCache;
pub use manager::{EvictionPriority, LoadOptions, ModelManager};
pub use metrics::Metrics;
#[cfg(feature = "vision")]
pub use multimodal::{ContentPart, MultimodalMessage};
pub use ram_cache::RamCache;
pub use streaming::{GenerateParams, GenerateResult};
pub use tools::{ToolCall, ToolChoice, ToolSpec};

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
        apply_chat_template_with_fallback(&self.model, messages, &[], &ToolChoice::None, None)
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

/// Apply a model's chat template to `messages`, producing a prompt.
///
/// Prefers **faithful Jinja rendering** of the template baked into the GGUF
/// (`tokenizer.chat_template`) via [`chat_template::render`] — this handles
/// arbitrary/unknown templates, thinking tags, and native tool formatting that
/// llama.cpp's legacy substring formatter cannot. Falls back to the legacy
/// formatter (with a ChatML default) when the model ships no Jinja template or
/// minijinja can't render it.
///
/// Some templates — notably Gemma's — reject a standalone `system` role. When
/// that happens and a system message is present, we retry with the system
/// content folded into the first user turn (the shape those models expect)
/// before giving up.
///
/// When `tools` are active, they are rendered through the template's `tools`
/// variable if the template supports it (the model's trained tool format);
/// otherwise the tool preamble is injected into the messages. Exactly one of the
/// two paths runs, so native and injected tool descriptions never both appear.
///
/// `override_tmpl`, when set, takes precedence over the model's embedded
/// template. A raw Jinja override is rendered by minijinja; a built-in template
/// *name* (e.g. "gemma", "chatml") is resolved by the legacy formatter. This
/// lets a model that ships a broken template be corrected without re-quantizing
/// it (see the sidecar `.jinja` convention in the llama.cpp backend).
pub(crate) fn apply_chat_template_with_fallback(
    model: &LlamaModel,
    messages: &[(String, String)],
    tools: &[ToolSpec],
    tool_choice: &ToolChoice,
    override_tmpl: Option<&str>,
) -> anyhow::Result<String> {
    // Tools are "active" only when present and not explicitly disabled.
    let tools_active = !tools.is_empty() && !matches!(tool_choice, ToolChoice::None);

    if let Some(source) = jinja_template_source(model, override_tmpl) {
        // Render `tools` natively only when the template actually consumes them
        // (references `tools`); otherwise fall back to injecting the preamble.
        // Never both — a single tool-system avoids native/injected drift.
        let native_tools = (tools_active && source.contains("tools"))
            .then(|| tools_to_oai_value(tools))
            .flatten();
        let injected;
        let msgs = if tools_active && native_tools.is_none() {
            injected = inject_tool_preamble(messages, tools, tool_choice);
            injected.as_slice()
        } else {
            messages
        };

        match try_render_jinja(model, &source, msgs, native_tools.as_ref()) {
            Ok(rendered) => return Ok(rendered),
            Err(e) => tracing::debug!(
                "Jinja chat template render failed ({e}); falling back to legacy formatter"
            ),
        }
    }

    // The legacy substring formatter can't render tools natively — inject.
    if tools_active {
        let injected = inject_tool_preamble(messages, tools, tool_choice);
        apply_legacy_chat_template(model, &injected, override_tmpl)
    } else {
        apply_legacy_chat_template(model, messages, override_tmpl)
    }
}

/// The OpenAI `tools` array as a JSON value for the template's `tools` variable.
fn tools_to_oai_value(tools: &[ToolSpec]) -> Option<serde_json::Value> {
    let json = tools::tools_to_oai_json(tools)?;
    serde_json::from_str(&json).ok()
}

/// Fold the tool-injection preamble into `messages` (merged into the first system
/// turn, or prepended as one) — the fallback for templates/formatters that can't
/// render `tools` natively. Keeps a single tool-system so native and injected
/// tool descriptions never both appear.
fn inject_tool_preamble(
    messages: &[(String, String)],
    tools: &[ToolSpec],
    choice: &ToolChoice,
) -> Vec<(String, String)> {
    let Some(preamble) = tools::tools_to_prompt(tools, choice) else {
        return messages.to_vec();
    };
    let mut out = messages.to_vec();
    match out.iter_mut().find(|(role, _)| role == "system") {
        Some((_, content)) => *content = format!("{content}\n\n{preamble}"),
        None => out.insert(0, ("system".to_string(), preamble)),
    }
    out
}

/// The raw Jinja template to render, if one is available. A raw-Jinja override
/// wins; otherwise the model's embedded `tokenizer.chat_template`. Returns
/// `None` for a built-in-name override or a model with no embedded template —
/// both of which the legacy formatter handles.
fn jinja_template_source(model: &LlamaModel, override_tmpl: Option<&str>) -> Option<String> {
    match override_tmpl {
        Some(t) if looks_like_jinja(t) => Some(t.to_string()),
        Some(_) => None, // built-in template name — resolved by the legacy path
        None => model
            .meta_val_str("tokenizer.chat_template")
            .ok()
            .filter(|t| looks_like_jinja(t)),
    }
}

/// A template string is Jinja (rather than a built-in name like "chatml") if it
/// contains Jinja delimiters.
fn looks_like_jinja(t: &str) -> bool {
    t.contains("{{") || t.contains("{%")
}

/// Render `source` with minijinja, retrying with the system role folded into the
/// first user turn if the template rejects a standalone system message.
fn try_render_jinja(
    model: &LlamaModel,
    source: &str,
    messages: &[(String, String)],
    tools: Option<&serde_json::Value>,
) -> anyhow::Result<String> {
    let bos = special_token_text(model, model.token_bos());
    let eos = special_token_text(model, model.token_eos());

    match chat_template::render(source, messages, tools, &bos, &eos, true) {
        Ok(rendered) => Ok(rendered),
        Err(first_err) if has_system_message(messages) => {
            tracing::debug!(
                "Jinja template rejected the system role; folding it into the first user turn"
            );
            chat_template::render(source, &fold_system_into_user(messages), tools, &bos, &eos, true)
                .map_err(|retry_err| {
                    anyhow::anyhow!(
                        "{first_err} (also failed after folding the system role into the user turn: {retry_err})"
                    )
                })
        }
        Err(e) => Err(e),
    }
}

/// The textual form of a special token (e.g. `<s>`, `<|begin_of_text|>`) for the
/// Jinja context. `special = true` renders the token's special-token text rather
/// than plaintext. Empty if the model can't render it.
fn special_token_text(model: &LlamaModel, token: llama_cpp_2::token::LlamaToken) -> String {
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    model
        .token_to_piece(token, &mut decoder, true, None)
        .unwrap_or_default()
}

/// llama.cpp's built-in (non-Jinja) formatter: substring-detects the format or
/// resolves an explicit built-in template name, with a ChatML default and the
/// system-role fold recovery. The fallback when Jinja rendering isn't available.
fn apply_legacy_chat_template(
    model: &LlamaModel,
    messages: &[(String, String)],
    override_tmpl: Option<&str>,
) -> anyhow::Result<String> {
    use llama_cpp_2::model::LlamaChatTemplate;

    let tmpl = match override_tmpl {
        Some(t) => LlamaChatTemplate::new(t)
            .map_err(|e| anyhow::anyhow!("invalid chat-template override: {e}"))?,
        None => match model.chat_template(None) {
            Ok(t) => t,
            Err(_) => {
                tracing::debug!("model has no chat template, using ChatML fallback");
                LlamaChatTemplate::new("chatml")
                    .map_err(|e| anyhow::anyhow!("failed to create ChatML template: {e}"))?
            }
        },
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
