//! Rust FFI wrapper for the `MlxBridge` Swift static library.
//!
//! Mirrors the `Engine::generate` signature so the two backends are
//! drop-in substitutes for each other. Compiled only on
//! `aarch64-apple-darwin` with `--features mlx`.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::sync::mpsc;

use crate::engine::streaming::{GenerateParams, GenerateResult};
use super::traits::{BackendLoadParams, BackendModel, InferenceBackend};

// ---------------------------------------------------------------------------
// Raw C declarations (matches mlx_bridge.h)
// ---------------------------------------------------------------------------

// Opaque model handle — never dereferenced on the Rust side.
enum MlxModelHandle {}

unsafe extern "C" {
    fn mlx_model_load(model_path: *const c_char) -> *mut MlxModelHandle;
    fn mlx_model_free(handle: *mut MlxModelHandle);

    fn mlx_generate(
        handle:       *mut MlxModelHandle,
        prompt:       *const c_char,
        max_tokens:   u32,
        temperature:  f32,
        top_p:        f32,
        seed:         u32,
        callback:     unsafe extern "C" fn(*const c_char, *mut c_void) -> c_int,
        callback_ctx: *mut c_void,
    ) -> i32;
}

// ---------------------------------------------------------------------------
// Token callback — bridges from the Swift Task thread into a Rust channel
// ---------------------------------------------------------------------------

struct TokenSender {
    tx: mpsc::SyncSender<String>,
}

unsafe extern "C" fn token_callback(token: *const c_char, ctx: *mut c_void) -> c_int {
    let sender = unsafe { &*(ctx as *const TokenSender) };
    if token.is_null() {
        return 0;
    }
    let s = unsafe { CStr::from_ptr(token) }.to_string_lossy().into_owned();
    if sender.tx.send(s).is_err() { 0 } else { 1 }
}

// ---------------------------------------------------------------------------
// InferenceBackend — factory for loading MLX models
// ---------------------------------------------------------------------------

pub struct MlxBackend;

impl InferenceBackend for MlxBackend {
    fn load_model(
        &self,
        path: &Path,
        _params: BackendLoadParams,
    ) -> anyhow::Result<Box<dyn BackendModel>> {
        let engine = MlxSwiftEngine::load(path)?;
        tracing::info!(
            n_ctx = engine.model_n_ctx,
            size_bytes = engine.model_size_bytes,
            "MLX model loaded"
        );
        Ok(Box::new(engine))
    }

    fn name(&self) -> &str {
        "mlx"
    }
}

// ---------------------------------------------------------------------------
// Safe wrapper
// ---------------------------------------------------------------------------

/// MLX inference engine backed by `mlx-swift-lm` via a Swift static library.
///
/// Supports any architecture that `mlx-swift-lm` ships (Llama, Mistral,
/// Phi, Gemma, Qwen, DeepSeek, …). The model directory must be in
/// HuggingFace MLX Community format (safetensors + `config.json`).
pub struct MlxSwiftEngine {
    handle: *mut MlxModelHandle,
    pub model_n_ctx: u32,
    pub model_size_bytes: u64,
}

// The handle is an opaque pointer managed by Swift ARC; it is never aliased.
unsafe impl Send for MlxSwiftEngine {}
unsafe impl Sync for MlxSwiftEngine {}

impl MlxSwiftEngine {
    /// Load a model from a local MLX-format directory.
    pub fn load(model_path: &Path) -> anyhow::Result<Self> {
        let path_str = model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?;
        let c_path = CString::new(path_str)?;

        let handle = unsafe { mlx_model_load(c_path.as_ptr()) };
        if handle.is_null() {
            anyhow::bail!("mlx_model_load returned NULL for {:?}", model_path);
        }

        let model_n_ctx = std::fs::read_to_string(model_path.join("config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v.get("max_position_embeddings")?.as_u64())
            .unwrap_or(4096) as u32;

        let model_size_bytes: u64 = std::fs::read_dir(model_path)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        e.path()
                            .extension()
                            .is_some_and(|ext| ext == "safetensors")
                    })
                    .filter_map(|e| std::fs::metadata(e.path()).ok())
                    .map(|m| m.len())
                    .sum()
            })
            .unwrap_or(0);

        Ok(Self {
            handle,
            model_n_ctx,
            model_size_bytes,
        })
    }

    /// Generate tokens from `prompt`, calling `on_token` for each text chunk.
    ///
    /// Return `false` from `on_token` to stop early.
    /// MLX generation runs on a background thread; this call blocks until done.
    pub fn generate(
        &self,
        prompt: &str,
        params: &GenerateParams,
        mut on_token: impl FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_dyn(prompt, params, &mut on_token)
    }

    fn generate_dyn(
        &self,
        prompt: &str,
        params: &GenerateParams,
        on_token: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult> {
        let c_prompt = CString::new(prompt)?;

        // Bounded channel: Swift callback thread → this thread.
        let (tx, rx) = mpsc::sync_channel::<String>(64);
        let sender = Box::new(TokenSender { tx });

        // Transmit the two pointers as `usize` so the closure captures `Send + Copy`
        // values. We manually assert the safety: both pointers remain valid for the
        // lifetime of the spawned thread (guarded by `join.join()` below).
        let handle_addr: usize = self.handle as usize;
        let sender_addr: usize = Box::into_raw(sender) as usize;

        let max_tok = params.max_tokens;
        let temp    = params.temperature;
        let top_p   = params.top_p;
        let seed    = params.seed;

        // Spawn a thread so `mlx_generate`'s internal DispatchSemaphore.wait()
        // never blocks a Tokio worker thread.
        let join = std::thread::spawn(move || {
            let h = handle_addr as *mut MlxModelHandle;
            let s = sender_addr as *mut TokenSender;

            let result = unsafe {
                mlx_generate(
                    h,
                    c_prompt.as_ptr(),
                    max_tok,
                    temp,
                    top_p,
                    seed,
                    token_callback,
                    s as *mut c_void,
                )
            };
            // Drop the sender so the channel closes and the rx loop below ends.
            drop(unsafe { Box::from_raw(s) });
            result
        });

        let mut completion_tokens = 0u32;
        for token in &rx {
            completion_tokens += 1;
            if !on_token(&token) {
                // Dropping rx causes the next send() to fail, making the callback
                // return 0 and signalling Swift to stop generation.
                break;
            }
        }
        drop(rx);

        let raw_result = join
            .join()
            .map_err(|_| anyhow::anyhow!("MLX generation thread panicked"))?;

        if raw_result < 0 {
            anyhow::bail!("mlx_generate returned error ({})", raw_result);
        }

        Ok(GenerateResult {
            prompt_tokens:     0,
            completion_tokens,
            cache_hit:         false,
        })
    }
}

impl BackendModel for MlxSwiftEngine {
    fn generate(
        &self,
        prompt: &str,
        params: &GenerateParams,
        on_token: &mut dyn FnMut(&str) -> bool,
    ) -> anyhow::Result<GenerateResult> {
        self.generate_dyn(prompt, params, on_token)
    }

    fn apply_chat_template(
        &self,
        _messages: &[(String, String)],
    ) -> anyhow::Result<String> {
        anyhow::bail!("chat templates are not yet supported for MLX models")
    }

    fn n_ctx(&self) -> u32 {
        self.model_n_ctx
    }

    fn size_bytes(&self) -> u64 {
        self.model_size_bytes
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Drop for MlxSwiftEngine {
    fn drop(&mut self) {
        unsafe { mlx_model_free(self.handle) }
    }
}
