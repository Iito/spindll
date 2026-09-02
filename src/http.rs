// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

//! HTTP/SSE API for direct access from web frontends and CLI tools.
//!
//! Provides the same capabilities as the gRPC server in a browser-friendly
//! format. The `/chat` endpoint streams tokens via Server-Sent Events.

use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::extract::Path;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;

use crate::engine::{GenerateParams, ModelManager};
#[cfg(feature = "vision")]
use crate::engine::multimodal::{check_image_len, ContentPart, MAX_IMAGE_BYTES, MultimodalMessage};
use crate::engine::reasoning::{ReasoningCollector, ReasoningSplitter};
use crate::engine::residency::ensure_loaded;
use crate::model_store::registry::Registry;
use crate::model_store::ModelStore;

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) manager: Arc<ModelManager>,
    pub(crate) store: Arc<ModelStore>,
}

/// Build the HTTP API router without binding to a port.
/// Useful for embedding into a larger application server.
pub fn router(manager: Arc<ModelManager>, store: Arc<ModelStore>) -> Router {
    let state = AppState { manager, store };

    Router::new()
        .route("/health", get(health))
        .route("/models", get(models))
        .route("/chat", post(chat))
        .route("/models/{id}", delete(model_delete))
        .route("/models/{id}/unload", post(model_unload))
        .route("/load", post(load))
        .route("/pull", post(pull))
        // OpenAI-compatible API
        .route("/v1/models", get(oai_models))
        .route("/v1/models/{id}", get(oai_model_config))
        .route("/v1/chat/completions", post(oai_chat_completions))
        .route("/v1/completions", post(oai_completions))
        .route("/v1/embeddings", post(oai_embeddings))
        .route("/v1/status", get(oai_status))
        // Anthropic Messages API + OpenAI Responses API (agent clients)
        .route("/v1/messages", post(crate::http_anthropic::anthropic_messages))
        .route("/v1/responses", post(crate::http_responses::responses_create))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Start the HTTP/SSE server on the given port.
///
/// Runs alongside the gRPC server on a separate port, sharing the same
/// `ModelManager` and `ModelStore`.
pub async fn start_http_server(
    port: u16,
    manager: Arc<ModelManager>,
    store: Arc<ModelStore>,
) -> anyhow::Result<()> {
    let app = router(manager, store);
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(%addr, "HTTP server listening");
    axum::serve(listener, app).await?;
    Ok(())
}

// -- /health ------------------------------------------------------------------

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

// -- /models ------------------------------------------------------------------

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    size_bytes: u64,
    quantization: String,
    digest: String,
    loaded: bool,
    model_name: String,
    description: String,
    architecture: String,
    context_length: u32,
}

async fn models(State(state): State<AppState>) -> impl IntoResponse {
    let mut reg = match Registry::load(&state.store.registry_path()) {
        Ok(r) => r,
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };
    if reg.backfill_metadata() {
        let _ = reg.save(&state.store.registry_path());
    }

    let loaded: std::collections::HashMap<String, (u32, u32)> = state
        .manager
        .loaded_models()
        .into_iter()
        .map(|(name, _, _, _, n_ctx, n_ctx_train)| (name, (n_ctx, n_ctx_train)))
        .collect();

    let list: Vec<ModelInfo> = reg
        .models
        .iter()
        .map(|(key, entry)| {
            // If loaded, use the effective context size; otherwise use registry metadata.
            let context_length = loaded.get(key)
                .map(|(n_ctx, _)| *n_ctx)
                .unwrap_or(entry.context_length);
            ModelInfo {
                name: key.clone(),
                size_bytes: entry.size_bytes,
                quantization: String::new(),
                digest: entry.digest.clone(),
                loaded: loaded.contains_key(key),
                model_name: entry.model_name.clone(),
                description: entry.description.clone(),
                architecture: entry.architecture.clone(),
                context_length,
            }
        })
        .collect();

    Json(list).into_response()
}

// -- DELETE /models/{id} — remove from disk -----------------------------------

async fn model_delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    // Unload from memory first if loaded.
    let _ = state.manager.unload_model(&id);

    match state.store.remove(&id, false) {
        Ok(()) => Json(serde_json::json!({"status": "ok"})).into_response(),
        Err(e) => (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

// -- POST /models/{id}/unload — remove from memory ----------------------------

async fn model_unload(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    // Models are resident under their canonical key, so an alias has to be
    // resolved the same way `/load` resolves it. A name the registry doesn't
    // know still gets tried verbatim, so a model loaded from outside the
    // registry stays unloadable.
    let key = state.store.resolve_key(&id).unwrap_or_else(|_| id.clone());
    let manager = state.manager.clone();

    // unload re-warms the RAM cache, which reads the whole model file on the
    // way out — never on an async worker.
    let result =
        tokio::task::spawn_blocking(move || manager.unload_model_or_alias(&key, &id)).await;

    match result {
        Ok(Ok(())) => Json(serde_json::json!({"status": "ok"})).into_response(),
        Ok(Err(e)) => (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("unload task failed: {e}")})),
        )
            .into_response(),
    }
}

// -- /chat (SSE) --------------------------------------------------------------

#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    params: Option<ChatParams>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatParams {
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    seed: Option<u32>,
}

async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    let mgr = state.manager.clone();
    let store = state.store.clone();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);

    tokio::task::spawn_blocking(move || {
        let key = match ensure_loaded(&mgr, &store, &req.model) {
            Ok(k) => k,
            Err(e) => {
                let _ = tx.blocking_send(Ok(sse_data(&serde_json::json!({"type": "error", "error": e.to_string()}))));
                return;
            }
        };

        let messages: Vec<_> = req.messages.iter().map(|m| (m.role.clone(), m.content.clone())).collect();

        let params = match req.params {
            Some(p) => GenerateParams {
                max_tokens: p.max_tokens.unwrap_or(512),
                temperature: p.temperature.unwrap_or(0.8),
                top_p: p.top_p.unwrap_or(0.95),
                top_k: p.top_k.unwrap_or(40),
                seed: p.seed.unwrap_or(42),
                prefill_only: false,
                ..Default::default()
            },
            None => GenerateParams::default(),
        };

        let result = mgr.generate_chat(&key, &messages, &[], &crate::engine::tools::ToolChoice::None, &params, None, |token| {
            let payload = serde_json::json!({"type": "token", "content": token});
            tx.blocking_send(Ok(sse_data(&payload))).is_ok()
        });

        match result {
            Ok(_) => {
                let _ = tx.blocking_send(Ok(sse_data(&serde_json::json!({"type": "done"}))));
            }
            Err(e) => {
                let _ = tx.blocking_send(Ok(sse_data(&serde_json::json!({"type": "error", "error": e.to_string()}))));
            }
        }
    });

    Sse::new(ReceiverStream::new(rx))
}

fn sse_data(data: &serde_json::Value) -> Event {
    Event::default().data(data.to_string())
}

// -- /load --------------------------------------------------------------------

#[derive(Deserialize)]
struct LoadRequest {
    model: String,
    #[serde(default)]
    gpu_layers: Option<i32>,
}

#[derive(Serialize)]
struct LoadResponse {
    already_loaded: bool,
}

async fn load(
    State(state): State<AppState>,
    Json(req): Json<LoadRequest>,
) -> impl IntoResponse {
    let resolved = match state.store.resolve(&req.model) {
        Ok(r) => r,
        Err(e) => {
            return (
                axum::http::StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };

    // Check residency under the canonical key, not the alias the caller typed.
    if state.manager.is_loaded(&resolved.key) {
        return Json(LoadResponse { already_loaded: true }).into_response();
    }

    let gpu_layers = req.gpu_layers.and_then(|l| if l < 0 { None } else { Some(l as u32) });
    let manager = state.manager.clone();
    let opts = crate::engine::manager::LoadOptions {
        gpu_layers,
        digest: resolved.digest,
        #[cfg(feature = "vision")]
        mmproj_path: resolved.mmproj_path,
        ..Default::default()
    };

    // Loading is minutes of blocking mmap + GPU upload; keep it off the async
    // worker so in-flight streams on this thread don't stall.
    let result = tokio::task::spawn_blocking(move || {
        manager.load_model_with_options(&resolved.key, &resolved.path, opts)
    })
    .await;

    match result {
        Ok(Ok(())) => Json(LoadResponse { already_loaded: false }).into_response(),
        Ok(Err(e)) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("load task failed: {e}")})),
        )
            .into_response(),
    }
}

// -- /pull --------------------------------------------------------------------

#[derive(Deserialize)]
struct PullRequest {
    model: String,
    #[serde(default)]
    quantization: Option<String>,
}

async fn pull(
    State(state): State<AppState>,
    Json(req): Json<PullRequest>,
) -> impl IntoResponse {
    let store = state.store.clone();

    let result = tokio::task::spawn_blocking(move || {
        store.pull(&req.model, req.quantization.as_deref(), crate::model_store::FormatPreference::Auto)
    })
    .await;

    match result {
        Ok(Ok(_)) => Json(serde_json::json!({"status": "ok"})).into_response(),
        Ok(Err(e)) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

// =============================================================================
// OpenAI-compatible API (/v1)
// =============================================================================

// -- GET /v1/models -----------------------------------------------------------

async fn oai_models(State(state): State<AppState>) -> impl IntoResponse {
    let mut reg = match Registry::load(&state.store.registry_path()) {
        Ok(r) => r,
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": e.to_string(), "type": "server_error"}})),
            )
                .into_response();
        }
    };
    if reg.backfill_metadata() {
        let _ = reg.save(&state.store.registry_path());
    }

    let data: Vec<OaiModelInfo> = reg
        .models
        .iter()
        .map(|(key, entry)| {
            let format_str = match entry.format {
                crate::model_store::registry::ModelFormat::Gguf => "gguf",
                crate::model_store::registry::ModelFormat::Mlx => "mlx",
            };
            OaiModelInfo {
                id: key.clone(),
                object: "model".to_string(),
                owned_by: "spindll".to_string(),
                created: entry.downloaded_at,
                architecture: if entry.architecture.is_empty() { None } else { Some(entry.architecture.clone()) },
                context_length: if entry.context_length == 0 { None } else { Some(entry.context_length) },
                format: Some(format_str.to_string()),
                size_bytes: Some(entry.size_bytes),
                capabilities: Some(ModelCapabilities {
                    chat: true,
                    completions: true,
                    embeddings: entry.format == crate::model_store::registry::ModelFormat::Gguf || entry.format == crate::model_store::registry::ModelFormat::Mlx,
                }),
            }
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "data": data,
    }))
    .into_response()
}

// -- /v1/models enhanced ------------------------------------------------------

#[derive(Serialize)]
struct OaiModelInfo {
    id: String,
    object: String,
    owned_by: String,
    created: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    architecture: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    capabilities: Option<ModelCapabilities>,
}

#[derive(Serialize)]
struct ModelCapabilities {
    chat: bool,
    completions: bool,
    embeddings: bool,
}

async fn oai_model_config(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let mut reg = match Registry::load(&state.store.registry_path()) {
        Ok(r) => r,
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": e.to_string(), "type": "server_error"}})),
            )
                .into_response();
        }
    };
    if reg.backfill_metadata() {
        let _ = reg.save(&state.store.registry_path());
    }

    match reg.models.get(&id) {
        Some(entry) => {
            let format_str = match entry.format {
                crate::model_store::registry::ModelFormat::Gguf => "gguf",
                crate::model_store::registry::ModelFormat::Mlx => "mlx",
            };
            let model_info = OaiModelInfo {
                id: id.clone(),
                object: "model".to_string(),
                owned_by: "spindll".to_string(),
                created: entry.downloaded_at,
                architecture: if entry.architecture.is_empty() { None } else { Some(entry.architecture.clone()) },
                context_length: if entry.context_length == 0 { None } else { Some(entry.context_length) },
                format: Some(format_str.to_string()),
                size_bytes: Some(entry.size_bytes),
                capabilities: Some(ModelCapabilities {
                    chat: true,
                    completions: true,
                    embeddings: true,
                }),
            };
            Json(model_info).into_response()
        }
        None => (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": {"message": "model not found", "type": "invalid_request_error"}})),
        )
            .into_response(),
    }
}

async fn oai_status(State(state): State<AppState>) -> impl IntoResponse {
    let mut reg = match Registry::load(&state.store.registry_path()) {
        Ok(r) => r,
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": e.to_string(), "type": "server_error"}})),
            )
                .into_response();
        }
    };
    if reg.backfill_metadata() {
        let _ = reg.save(&state.store.registry_path());
    }

    let loaded_models: Vec<String> = state.manager.loaded_models().iter().map(|(name, _, _, _, _, _)| name.clone()).collect();
    let total_models = reg.models.len();

    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "models": {
            "total": total_models,
            "loaded": loaded_models.len(),
            "loaded_models": loaded_models,
        }
    }))
    .into_response()
}

// -- POST /v1/chat/completions ------------------------------------------------

#[derive(Deserialize)]
struct OaiChatRequest {
    model: String,
    messages: Vec<OaiMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    seed: Option<u32>,
    #[serde(default)]
    tools: Option<Vec<OaiTool>>,
    /// OpenAI `tool_choice`: `"none"` | `"auto"` | `"required"` | `{function}`.
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
}

#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize)]
struct OaiMessage {
    role: String,
    #[serde(default)]
    content: Option<OaiContent>,
    #[serde(default)]
    tool_calls: Option<Vec<OaiToolCallMessage>>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

/// OpenAI `content` field: either a plain string or an array of content parts.
#[derive(Deserialize, Clone)]
#[serde(untagged)]
enum OaiContent {
    Text(String),
    Parts(Vec<OaiContentPart>),
}

impl OaiContent {
    /// Flatten to a plain text string (ignoring images).
    fn as_text(&self) -> String {
        match self {
            OaiContent::Text(s) => s.clone(),
            OaiContent::Parts(parts) => parts.iter()
                .filter_map(|p| match p {
                    OaiContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<String>(),
        }
    }

    /// Returns `true` if any part is an image.
    fn has_images(&self) -> bool {
        match self {
            OaiContent::Text(_) => false,
            OaiContent::Parts(parts) => parts.iter().any(|p| matches!(p, OaiContentPart::ImageUrl { .. })),
        }
    }
}

#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
enum OaiContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    #[cfg_attr(not(feature = "vision"), allow(dead_code))]
    ImageUrl { image_url: OaiImageUrl },
}

#[derive(Deserialize, Clone)]
#[cfg_attr(not(feature = "vision"), allow(dead_code))]
struct OaiImageUrl {
    url: String,
}

/// MIME allow-list. Rejected before `MtmdBitmap::from_buffer` / MLX ImageIO.
#[cfg(feature = "vision")]
pub(crate) const ALLOWED_IMAGE_MEDIA: &[&str] = &[
    "image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp",
];

/// Decode `data:...;base64,...` URI. Rejects > MAX_IMAGE_BYTES.
#[cfg(feature = "vision")]
pub(crate) fn decode_data_uri(uri: &str) -> anyhow::Result<(Vec<u8>, Option<String>)> {
    use base64::Engine as _;

    let rest = uri.strip_prefix("data:")
        .ok_or_else(|| anyhow::anyhow!("image_url must be a data: URI (http(s) URLs not yet supported)"))?;
    let (header, b64) = rest.split_once(',')
        .ok_or_else(|| anyhow::anyhow!("malformed data URI: missing comma"))?;

    // Require base64 so raw/percent-encoded bodies aren't fed to the decoder.
    let Some(meta) = header.strip_suffix(";base64") else {
        anyhow::bail!("image_url data URI must be base64-encoded (missing \";base64\")");
    };
    // Strip `;param=value` tail; empty type → None.
    let media_type = {
        let mt = meta.split(';').next().unwrap_or(meta).trim().to_lowercase();
        if mt.is_empty() { None } else { Some(mt) }
    };

    // Pre-check encoded length (b64 ~4→3 bytes) → reject before alloc.
    if b64.len() / 4 * 3 > MAX_IMAGE_BYTES {
        anyhow::bail!(
            "image exceeds {} byte limit (encoded ~{} bytes)",
            MAX_IMAGE_BYTES,
            b64.len() / 4 * 3,
        );
    }

    let data = base64::engine::general_purpose::STANDARD.decode(b64)
        .map_err(|e| anyhow::anyhow!("base64 decode failed: {e}"))?;

    check_image_len(data.len())?;

    if let Some(mt) = &media_type
        && !ALLOWED_IMAGE_MEDIA.contains(&mt.as_str())
    {
        anyhow::bail!(
            "unsupported image media type {mt}; allowed: {}",
            ALLOWED_IMAGE_MEDIA.join(", ")
        );
    }

    Ok((data, media_type))
}

/// Convert OAI messages into multimodal messages, decoding data URIs.
#[cfg(feature = "vision")]
fn oai_to_multimodal(messages: &[OaiMessage]) -> anyhow::Result<Vec<MultimodalMessage>> {
    let mut out = Vec::with_capacity(messages.len());
    for msg in messages {
        let content = match &msg.content {
            None => vec![ContentPart::Text(String::new())],
            Some(OaiContent::Text(s)) => vec![ContentPart::Text(s.clone())],
            Some(OaiContent::Parts(parts)) => {
                let mut content_parts = Vec::with_capacity(parts.len());
                for part in parts {
                    match part {
                        OaiContentPart::Text { text } => {
                            content_parts.push(ContentPart::Text(text.clone()));
                        }
                        OaiContentPart::ImageUrl { image_url } => {
                            let (data, media_type) = decode_data_uri(&image_url.url)?;
                            content_parts.push(ContentPart::ImageBytes { data, media_type });
                        }
                    }
                }
                content_parts
            }
        };
        out.push(MultimodalMessage {
            role: msg.role.clone(),
            content,
        });
    }
    Ok(out)
}

#[derive(Deserialize, Serialize, Clone)]
struct OaiTool {
    r#type: String,
    function: OaiFunction,
}

#[derive(Deserialize, Serialize, Clone)]
struct OaiFunction {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<serde_json::Value>,
}

#[derive(Deserialize, Serialize, Clone)]
struct OaiToolCallMessage {
    id: String,
    r#type: String,
    function: OaiToolCallFunction,
}

#[derive(Deserialize, Serialize, Clone)]
struct OaiToolCallFunction {
    name: String,
    arguments: String,
}

/// Convert OpenAI tool definitions into the shared [`ToolSpec`] vocabulary.
fn oai_tools_to_specs(tools: &[OaiTool]) -> Vec<crate::engine::tools::ToolSpec> {
    tools
        .iter()
        .map(|t| crate::engine::tools::ToolSpec {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            parameters: t.function.parameters.clone(),
        })
        .collect()
}

/// Extract tool calls from model output, adapting the shared engine parser
/// (`engine::tools::parse_tool_calls`) to the OpenAI response shape. The shared
/// parser also understands the Hermes / Llama-3.1 / Mistral wrappers, so this
/// is strictly more capable than the old HTTP-local JSON scan it replaced.
fn parse_tool_calls(output: &str) -> (Vec<OaiToolCallMessage>, String) {
    let (calls, remaining) = crate::engine::tools::parse_tool_calls(output);
    let calls = calls
        .into_iter()
        .map(|c| OaiToolCallMessage {
            id: c.id,
            r#type: "function".to_string(),
            function: OaiToolCallFunction {
                name: c.name,
                arguments: c.arguments,
            },
        })
        .collect();
    (calls, remaining)
}

/// OpenAI `finish_reason` for a completed generation: `"length"` when the
/// token budget was exhausted, `"stop"` otherwise. Both backends stop *at* the
/// cap and report one decoded piece per token, so `>=` is the budget-hit
/// signal. (The MLX bridge skips empty pieces and can slightly under-count —
/// a budget hit may then still read as `"stop"`, never the reverse.)
fn oai_finish_reason(completion_tokens: u32, max_tokens: u32) -> &'static str {
    if max_tokens > 0 && completion_tokens >= max_tokens {
        "length"
    } else {
        "stop"
    }
}

/// OpenAI `usage` object. When a think block was split off, adds
/// `completion_tokens_details.reasoning_tokens` (a stream-piece approximation
/// of tokens) so clients can see how much of the budget the reasoning took.
fn oai_usage(stats: &crate::engine::GenerateResult, reasoning_pieces: u32) -> serde_json::Value {
    let mut usage = serde_json::json!({
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "total_tokens": stats.prompt_tokens + stats.completion_tokens,
    });
    if reasoning_pieces > 0 {
        usage["completion_tokens_details"] =
            serde_json::json!({ "reasoning_tokens": reasoning_pieces });
    }
    usage
}

/// Prepare messages for template application, injecting the pre-rendered tool
/// preamble (`engine::tools::tools_to_prompt`, built once by the caller and
/// `None` when tools are inactive) into the system turn. Returns `(role,
/// content)` pairs for the chat template.
fn prepare_messages_with_tools(
    messages: &[OaiMessage],
    tool_preamble: Option<&str>,
) -> Vec<(String, String)> {
    let mut result: Vec<(String, String)> = Vec::new();
    let mut system_injected = false;

    for msg in messages {
        let content = msg.content.as_ref().map(|c| c.as_text()).unwrap_or_default();

        if msg.role == "system" && !system_injected {
            if let Some(preamble) = tool_preamble {
                result.push(("system".to_string(), format!("{content}\n\n{preamble}")));
            } else {
                result.push(("system".to_string(), content));
            }
            system_injected = true;
        } else if msg.role == "tool" {
            // Tool results go as user messages with context
            let tool_id = msg.tool_call_id.as_deref().unwrap_or("unknown");
            result.push(("user".to_string(), format!("[Tool result for {tool_id}]: {content}")));
        } else if msg.role == "assistant" {
            if let Some(ref tc) = msg.tool_calls {
                // Serialize assistant tool calls back into the conversation
                let calls_json = serde_json::to_string(tc).unwrap_or_default();
                let full = if content.is_empty() {
                    calls_json
                } else {
                    format!("{content}\n{calls_json}")
                };
                result.push(("assistant".to_string(), full));
            } else {
                result.push(("assistant".to_string(), content));
            }
        } else {
            result.push((msg.role.clone(), content));
        }
    }

    // If there was no system message but we have tools, inject one
    if !system_injected
        && let Some(preamble) = tool_preamble {
            result.insert(0, ("system".to_string(), preamble.to_string()));
        }

    result
}

/// Check whether any OAI message contains image content parts.
fn oai_has_images(messages: &[OaiMessage]) -> bool {
    messages.iter().any(|m| m.content.as_ref().is_some_and(|c| c.has_images()))
}

/// Generate via the multimodal path (vision) or the text-only chat path,
/// depending on whether images are present.
///
/// Returns the `GenerateResult` from whichever path was taken.
///
/// `tool_preamble` is the rendered tool-calling instructions (when tools are
/// active). On the vision path it is injected into the multimodal messages so
/// the model sees the tools — without this the text path's preamble (carried in
/// `text_messages`) would be dropped and tool calling would silently no-op.
///
/// `key` must be the canonical registry key from [`ensure_loaded`], not the name the
/// caller typed — the manager keys its slots by the former.
#[cfg(feature = "vision")]
#[allow(clippy::too_many_arguments)]
fn generate_maybe_multimodal(
    mgr: &ModelManager,
    key: &str,
    oai_messages: &[OaiMessage],
    text_messages: &[(String, String)],
    tools: &[crate::engine::tools::ToolSpec],
    tool_choice: &crate::engine::tools::ToolChoice,
    params: &GenerateParams,
    on_token: &mut dyn FnMut(&str) -> bool,
) -> anyhow::Result<crate::engine::GenerateResult> {
    if oai_has_images(oai_messages) {
        // Vision path keeps tool injection: the multimodal generate path doesn't
        // render tools through a template yet, so fold the preamble into a system
        // turn (a no-op when there are no tools).
        let mut mm = oai_to_multimodal(oai_messages)?;
        if let Some(preamble) = crate::engine::tools::tools_to_prompt(tools, tool_choice) {
            crate::engine::multimodal::inject_system_text(&mut mm, &preamble);
        }
        mgr.generate_chat_multimodal(key, &mm, params, on_token)
    } else {
        mgr.generate_chat(key, text_messages, tools, tool_choice, params, None, on_token)
    }
}

async fn oai_chat_completions(
    State(state): State<AppState>,
    Json(req): Json<OaiChatRequest>,
) -> impl IntoResponse {
    let model_id = req.model.clone();
    let mgr = state.manager.clone();
    let store = state.store.clone();
    let tool_choice = crate::engine::tools::ToolChoice::from_oai(req.tool_choice.as_ref());
    // tool_choice "none" disables tools entirely: no preamble, no parsing.
    let has_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty())
        && !matches!(tool_choice, crate::engine::tools::ToolChoice::None);
    let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);

    // Tool specs (empty when tools are off) are passed to `generate_chat`, which
    // renders them through the model's native tool template or, as a fallback,
    // injects a preamble. The vision path still injects (see
    // `generate_maybe_multimodal`).
    let tool_specs: Vec<crate::engine::tools::ToolSpec> = if has_tools {
        oai_tools_to_specs(req.tools.as_deref().unwrap_or_default())
    } else {
        Vec::new()
    };

    #[cfg(not(feature = "vision"))]
    if oai_has_images(&req.messages) {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(oai_error("image_url content requires the vision feature")),
        )
            .into_response();
    }

    if req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);

        tokio::task::spawn_blocking(move || {
            let key = match ensure_loaded(&mgr, &store, &req.model) {
                Ok(k) => k,
                Err(e) => {
                    let _ = tx.blocking_send(Ok(sse_data(&oai_error(&e.to_string()))));
                    return;
                }
            };

            let forced_open = mgr.reasoning_forced_open(&key);
            let messages = prepare_messages_with_tools(&req.messages, None);
            let params = GenerateParams {
                max_tokens: req.max_tokens.unwrap_or(512),
                temperature: req.temperature.unwrap_or(0.8),
                top_p: req.top_p.unwrap_or(0.95),
                top_k: 40,
                seed: req.seed.unwrap_or(42),
                prefill_only: false,
                ..Default::default()
            };

            let completion_id = format!("chatcmpl-{:016x}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();

            // OpenAI's first streamed chunk announces the assistant role; emit it
            // before any content or tool-call deltas (either branch below) so
            // accumulating clients set the message role.
            let _ = tx.blocking_send(Ok(sse_data(&serde_json::json!({
                "id": &completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": &req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
            }))));

            if has_tools {
                // When tools are active, collect full output to parse tool
                // calls. The collector splits think-block reasoning off first,
                // so a call the model merely *plans* inside `<think>` is not
                // mistaken for one it made.
                let mut collector = ReasoningCollector::new(forced_open);
                #[cfg(feature = "vision")]
                let result = generate_maybe_multimodal(&mgr, &key, &req.messages, &messages, &tool_specs, &tool_choice, &params, &mut |token| {
                    collector.push(token);
                    true
                });
                #[cfg(not(feature = "vision"))]
                let result = mgr.generate_chat(&key, &messages, &tool_specs, &tool_choice, &params, None, |token| {
                    collector.push(token);
                    true
                });

                match result {
                    Ok(ref stats) => {
                        let split = collector.finish();
                        let (tool_calls, remaining) = parse_tool_calls(&split.content);
                        // One delta per chunk, keyed by `index`, matching OpenAI's
                        // streaming shape: id+name first, then arguments, so clients
                        // accumulate calls correctly. Arguments arrive in a single
                        // fragment — calls are parsed from the completed output, not
                        // token-by-token (no streaming parser on 0.1.150).
                        let emit = |delta: serde_json::Value| {
                            let chunk = serde_json::json!({
                                "id": &completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": &req.model,
                                "choices": [{"index": 0, "delta": delta, "finish_reason": null}]
                            });
                            let _ = tx.blocking_send(Ok(sse_data(&chunk)));
                        };
                        // Reasoning first (one delta — the tools path buffers
                        // the whole output anyway, and chunking a long think
                        // block here would only fragment an already-buffered
                        // string), then leftover prose as a content delta;
                        // with no calls that prose is the whole answer, and a
                        // blank one is dropped rather than streamed as stray
                        // whitespace.
                        if let Some(r) = &split.reasoning {
                            emit(serde_json::json!({ "reasoning_content": r }));
                        }
                        if !remaining.is_empty() {
                            emit(serde_json::json!({ "content": remaining }));
                        }
                        for (i, call) in tool_calls.iter().enumerate() {
                            emit(serde_json::json!({ "tool_calls": [{
                                "index": i,
                                "id": call.id,
                                "type": "function",
                                "function": { "name": call.function.name, "arguments": "" }
                            }]}));
                            emit(serde_json::json!({ "tool_calls": [{
                                "index": i,
                                "function": { "arguments": call.function.arguments }
                            }]}));
                        }
                        let finish = if !tool_calls.is_empty() {
                            "tool_calls"
                        } else {
                            oai_finish_reason(stats.completion_tokens, params.max_tokens)
                        };
                        let done_chunk = serde_json::json!({
                            "id": &completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": &req.model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]
                        });
                        let _ = tx.blocking_send(Ok(sse_data(&done_chunk)));
                        if include_usage {
                            let usage_chunk = serde_json::json!({
                                "id": &completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": &req.model,
                                "choices": [],
                                "usage": oai_usage(stats, split.reasoning_pieces),
                            });
                            let _ = tx.blocking_send(Ok(sse_data(&usage_chunk)));
                        }
                        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                    }
                    Err(e) => {
                        let _ = tx.blocking_send(Ok(sse_data(&oai_error(&e.to_string()))));
                    }
                }
            } else {
                // No tools — stream tokens directly, splitting think-block
                // reasoning into `delta.reasoning_content` as it arrives.
                let mut splitter = ReasoningSplitter::new(forced_open);
                let mut reasoning_pieces = 0u32;
                let send_delta = |delta: serde_json::Value| -> bool {
                    let chunk = serde_json::json!({
                        "id": &completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": &req.model,
                        "choices": [{
                            "index": 0,
                            "delta": delta,
                            "finish_reason": null,
                        }]
                    });
                    tx.blocking_send(Ok(sse_data(&chunk))).is_ok()
                };
                let mut on_tok = |token: &str| -> bool {
                    let (r, c) = splitter.push(token);
                    let mut alive = true;
                    if !r.is_empty() {
                        reasoning_pieces += 1;
                        alive &= send_delta(serde_json::json!({ "reasoning_content": r }));
                    }
                    if !c.is_empty() {
                        alive &= send_delta(serde_json::json!({ "content": c }));
                    }
                    alive
                };
                #[cfg(feature = "vision")]
                let result = generate_maybe_multimodal(&mgr, &key, &req.messages, &messages, &tool_specs, &tool_choice, &params, &mut on_tok);
                #[cfg(not(feature = "vision"))]
                let result = mgr.generate_chat(&key, &messages, &tool_specs, &tool_choice, &params, None, &mut on_tok);

                match result {
                    Ok(ref stats) => {
                        // Flush whatever the splitter still holds — a pending
                        // partial tag, or a think block max_tokens cut short.
                        let (r, c) = splitter.finish();
                        if !r.is_empty() {
                            reasoning_pieces += 1;
                            let _ = send_delta(serde_json::json!({ "reasoning_content": r }));
                        }
                        if !c.is_empty() {
                            let _ = send_delta(serde_json::json!({ "content": c }));
                        }
                        let done_chunk = serde_json::json!({
                            "id": &completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": &req.model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": oai_finish_reason(stats.completion_tokens, params.max_tokens),
                            }]
                        });
                        let _ = tx.blocking_send(Ok(sse_data(&done_chunk)));
                        if include_usage {
                            let usage_chunk = serde_json::json!({
                                "id": &completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": &req.model,
                                "choices": [],
                                "usage": oai_usage(stats, reasoning_pieces),
                            });
                            let _ = tx.blocking_send(Ok(sse_data(&usage_chunk)));
                        }
                        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                    }
                    Err(e) => {
                        let _ = tx.blocking_send(Ok(sse_data(&oai_error(&e.to_string()))));
                    }
                }
            }
            drop(tx);
        });

        Sse::new(ReceiverStream::new(rx)).into_response()
    } else {
        // Non-streaming: collect all tokens then return a single JSON response.
        let result = tokio::task::spawn_blocking(move || {
            let key = ensure_loaded(&mgr, &store, &req.model)?;

            let forced_open = mgr.reasoning_forced_open(&key);
            let messages = prepare_messages_with_tools(&req.messages, None);

            let params = GenerateParams {
                max_tokens: req.max_tokens.unwrap_or(512),
                temperature: req.temperature.unwrap_or(0.8),
                top_p: req.top_p.unwrap_or(0.95),
                top_k: 40,
                seed: req.seed.unwrap_or(42),
                prefill_only: false,
                ..Default::default()
            };

            let mut collector = ReasoningCollector::new(forced_open);
            #[cfg(feature = "vision")]
            let stats = generate_maybe_multimodal(&mgr, &key, &req.messages, &messages, &tool_specs, &tool_choice, &params, &mut |token| {
                collector.push(token);
                true
            })?;
            #[cfg(not(feature = "vision"))]
            let stats = mgr.generate_chat(&key, &messages, &tool_specs, &tool_choice, &params, None, |token| {
                collector.push(token);
                true
            })?;

            Ok::<_, anyhow::Error>((collector.finish(), stats, params.max_tokens))
        })
        .await;

        match result {
            Ok(Ok((split, stats, max_tokens))) => {
                let completion_id = format!("chatcmpl-{:016x}", std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();

                let (mut message, finish_reason) = if has_tools {
                    let (tool_calls, remaining) = parse_tool_calls(&split.content);
                    if !tool_calls.is_empty() {
                        let msg = if remaining.is_empty() {
                            serde_json::json!({"role": "assistant", "content": null, "tool_calls": tool_calls})
                        } else {
                            serde_json::json!({"role": "assistant", "content": remaining, "tool_calls": tool_calls})
                        };
                        (msg, "tool_calls")
                    } else {
                        // No call detected — return the parsed (call-free) content.
                        (serde_json::json!({"role": "assistant", "content": remaining}),
                         oai_finish_reason(stats.completion_tokens, max_tokens))
                    }
                } else {
                    (serde_json::json!({"role": "assistant", "content": split.content}),
                     oai_finish_reason(stats.completion_tokens, max_tokens))
                };
                if let Some(r) = &split.reasoning {
                    message["reasoning_content"] = serde_json::json!(r);
                }

                Json(serde_json::json!({
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }],
                    "usage": oai_usage(&stats, split.reasoning_pieces),
                }))
                .into_response()
            }
            Ok(Err(e)) => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(oai_error(&e.to_string())),
            )
                .into_response(),
            Err(e) => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(oai_error(&e.to_string())),
            )
                .into_response(),
        }
    }
}

// -- POST /v1/completions ----------------------------------------------------

#[derive(Deserialize)]
struct OaiCompletionRequest {
    model: String,
    prompt: String,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    seed: Option<u32>,
}

async fn oai_completions(
    State(state): State<AppState>,
    Json(req): Json<OaiCompletionRequest>,
) -> impl IntoResponse {
    let model_id = req.model.clone();
    let mgr = state.manager.clone();
    let store = state.store.clone();

    if req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);

        tokio::task::spawn_blocking(move || {
            let key = match ensure_loaded(&mgr, &store, &req.model) {
                Ok(k) => k,
                Err(e) => {
                    let _ = tx.blocking_send(Ok(sse_data(&oai_error(&e.to_string()))));
                    return;
                }
            };

            let params = GenerateParams {
                max_tokens: req.max_tokens.unwrap_or(512),
                temperature: req.temperature.unwrap_or(0.8),
                top_p: req.top_p.unwrap_or(0.95),
                top_k: 40,
                seed: req.seed.unwrap_or(42),
                prefill_only: false,
                ..Default::default()
            };

            let completion_id = format!("cmpl-{:016x}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();

            let result = mgr.generate(&key, &req.prompt, &params, None, |token| {
                let chunk = serde_json::json!({
                    "id": &completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": &req.model,
                    "choices": [{
                        "index": 0,
                        "text": token,
                        "finish_reason": null,
                    }]
                });
                tx.blocking_send(Ok(sse_data(&chunk))).is_ok()
            });

            match result {
                Ok(_) => {
                    let done_chunk = serde_json::json!({
                        "id": &completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": &req.model,
                        "choices": [{
                            "index": 0,
                            "text": "",
                            "finish_reason": "stop",
                        }]
                    });
                    let _ = tx.blocking_send(Ok(sse_data(&done_chunk)));
                    let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                }
                Err(e) => {
                    let _ = tx.blocking_send(Ok(sse_data(&oai_error(&e.to_string()))));
                }
            }
            drop(tx);
        });

        Sse::new(ReceiverStream::new(rx)).into_response()
    } else {
        let result = tokio::task::spawn_blocking(move || {
            let key = ensure_loaded(&mgr, &store, &req.model)?;

            let params = GenerateParams {
                max_tokens: req.max_tokens.unwrap_or(512),
                temperature: req.temperature.unwrap_or(0.8),
                top_p: req.top_p.unwrap_or(0.95),
                top_k: 40,
                seed: req.seed.unwrap_or(42),
                prefill_only: false,
                ..Default::default()
            };

            let mut output = String::new();
            let stats = mgr.generate(&key, &req.prompt, &params, None, |token| {
                output.push_str(token);
                true
            })?;

            Ok::<_, anyhow::Error>((output, stats))
        })
        .await;

        match result {
            Ok(Ok((text, stats))) => {
                let completion_id = format!("cmpl-{:016x}", std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
                Json(serde_json::json!({
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "text": text,
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": stats.prompt_tokens,
                        "completion_tokens": stats.completion_tokens,
                        "total_tokens": stats.prompt_tokens + stats.completion_tokens,
                    }
                }))
                .into_response()
            }
            Ok(Err(e)) => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(oai_error(&e.to_string())),
            )
                .into_response(),
            Err(e) => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(oai_error(&e.to_string())),
            )
                .into_response(),
        }
    }
}

// -- POST /v1/embeddings ----------------------------------------------------

#[derive(Deserialize)]
struct OaiEmbeddingRequest {
    model: String,
    input: EmbeddingInput,
    #[serde(default)]
    #[allow(dead_code)]
    user: Option<String>,
    #[serde(default)]
    encoding_format: Option<String>,
    /// OpenAI lets clients ask for a reduced output dimensionality.
    /// We don't support truncation yet — reject when supplied so callers
    /// don't silently get full-dimension embeddings.
    #[serde(default)]
    dimensions: Option<u32>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
    /// OpenAI-spec token-id inputs: `list[int]` (single) and `list[list[int]]`
    /// (batch). The order matters here — serde tries variants top-down, so
    /// these have to come after the string forms to avoid coercion. The
    /// payload is not used (rejected with 400 below) but is parsed so we
    /// return a structured error instead of a generic JSON decode failure.
    #[allow(dead_code)]
    TokenIds(Vec<i32>),
    #[allow(dead_code)]
    TokenIdsBatch(Vec<Vec<i32>>),
}

/// Hard cap on a single embedding input length, in characters/tokens, to bound
/// server-side memory before tokenization or matmul.
const MAX_EMBED_INPUT_LEN: usize = 32_768;
const MAX_EMBED_BATCH: usize = 2048;

/// Encode an embedding vector as base64'd little-endian f32 (OpenAI spec).
/// Stdlib-only implementation to avoid adding a base64 crate dependency for
/// a single encode site.
fn encode_embedding_base64(v: &[f32]) -> String {
    const A: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8) | bytes[i + 2] as u32;
        out.push(A[((n >> 18) & 0x3f) as usize] as char);
        out.push(A[((n >> 12) & 0x3f) as usize] as char);
        out.push(A[((n >> 6) & 0x3f) as usize] as char);
        out.push(A[(n & 0x3f) as usize] as char);
        i += 3;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        let n = (bytes[i] as u32) << 16;
        out.push(A[((n >> 18) & 0x3f) as usize] as char);
        out.push(A[((n >> 12) & 0x3f) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8);
        out.push(A[((n >> 18) & 0x3f) as usize] as char);
        out.push(A[((n >> 12) & 0x3f) as usize] as char);
        out.push(A[((n >> 6) & 0x3f) as usize] as char);
        out.push('=');
    }
    out
}

async fn oai_embeddings(
    State(state): State<AppState>,
    Json(req): Json<OaiEmbeddingRequest>,
) -> impl IntoResponse {
    // encoding_format: only "float" and "base64" are valid per OpenAI spec.
    let want_base64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(oai_error(&format!(
                    "encoding_format '{other}' not supported (use 'float' or 'base64')"
                ))),
            )
                .into_response();
        }
    };

    if req.dimensions.is_some() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(oai_error(
                "the `dimensions` parameter is not supported by this backend",
            )),
        )
            .into_response();
    }

    // Normalise all four input shapes into Vec<String>. Token-id inputs
    // currently can't bypass the tokenizer (no token-array embed path on
    // backends yet) — reject so clients don't get a confusing tokenisation.
    let texts: Vec<String> = match req.input {
        EmbeddingInput::Single(s) => vec![s],
        EmbeddingInput::Batch(v) => v,
        EmbeddingInput::TokenIds(_) | EmbeddingInput::TokenIdsBatch(_) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(oai_error(
                    "token-id input is not yet supported; pass a string or array of strings",
                )),
            )
                .into_response();
        }
    };

    if texts.is_empty() || texts.iter().any(|t| t.is_empty()) {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(oai_error(
                "input must be a non-empty string or array of non-empty strings",
            )),
        )
            .into_response();
    }

    if texts.len() > MAX_EMBED_BATCH {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(oai_error(&format!(
                "input array exceeds {MAX_EMBED_BATCH}-item limit ({} items)",
                texts.len()
            ))),
        )
            .into_response();
    }

    if let Some(over) = texts.iter().find(|t| t.len() > MAX_EMBED_INPUT_LEN) {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(oai_error(&format!(
                "input exceeds {MAX_EMBED_INPUT_LEN}-char limit ({} chars)",
                over.len()
            ))),
        )
            .into_response();
    }

    let model_id = req.model.clone();
    let mgr = state.manager.clone();
    let store = state.store.clone();

    let result = tokio::task::spawn_blocking(move || {
        let key = ensure_loaded(&mgr, &store, &req.model)?;

        let mut data = Vec::with_capacity(texts.len());
        let mut total_tokens = 0u32;

        for (i, text) in texts.iter().enumerate() {
            let r = mgr.embed(&key, text)?;
            total_tokens += r.prompt_tokens;
            let embedding_val = if want_base64 {
                serde_json::Value::String(encode_embedding_base64(&r.embedding))
            } else {
                serde_json::json!(r.embedding)
            };
            data.push(serde_json::json!({
                "object": "embedding",
                "index": i,
                "embedding": embedding_val,
            }));
        }

        Ok::<_, anyhow::Error>((data, total_tokens))
    })
    .await;

    match result {
        Ok(Ok((data, total_tokens))) => {
            Json(serde_json::json!({
                "object": "list",
                "data": data,
                "model": model_id,
                "usage": {
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens,
                }
            }))
            .into_response()
        }
        // Surface "model not loaded" / "embeddings not supported" as 400 so
        // clients can distinguish capability errors from infra failures.
        Ok(Err(e)) => {
            let msg = e.to_string();
            let status = if msg.contains("not loaded")
                || msg.contains("does not support")
                || msg.contains("not supported")
            {
                axum::http::StatusCode::BAD_REQUEST
            } else {
                axum::http::StatusCode::INTERNAL_SERVER_ERROR
            };
            (status, Json(oai_error(&msg))).into_response()
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(oai_error(&e.to_string())),
        )
            .into_response(),
    }
}

fn oai_error(msg: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": msg,
            "type": "server_error",
        }
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::backend::{BackendLoadParams, BackendModel, EmbedResult, InferenceBackend};
    use crate::engine::streaming::{GenerateParams as EngineParams, GenerateResult};
    use crate::model_store::registry::{ModelEntry, ModelFormat, ModelSource};
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    struct FakeBackend;
    impl InferenceBackend for FakeBackend {
        fn load_model(&self, _: &std::path::Path, _: BackendLoadParams) -> anyhow::Result<Box<dyn BackendModel>> {
            Ok(Box::new(FakeModel))
        }
        fn name(&self) -> &str { "llamacpp" }
    }
    struct FakeModel;
    impl BackendModel for FakeModel {
        fn generate(&self, _: &str, _params: &EngineParams, on_token: &mut dyn FnMut(&str) -> bool) -> anyhow::Result<GenerateResult> {
            for tok in &["Hello", " world"] {
                if !on_token(tok) { break; }
            }
            Ok(GenerateResult { prompt_tokens: 5, completion_tokens: 2, cache_hit: false })
        }
        fn apply_chat_template(&self, _: &[(String, String)], _: &[crate::engine::tools::ToolSpec], _: &crate::engine::tools::ToolChoice) -> anyhow::Result<String> { Ok("prompt".into()) }
        fn n_ctx(&self) -> u32 { 2048 }
        fn size_bytes(&self) -> u64 { 100 }
        fn kv_bytes_per_token(&self) -> u64 { 1 }
        fn as_any(&self) -> &dyn std::any::Any { self }
        fn embed(&self, _text: &str) -> anyhow::Result<EmbedResult> {
            Ok(EmbedResult {
                embedding: vec![0.6, 0.8],
                prompt_tokens: 3,
            })
        }
    }

    /// Backend whose model streams a fixed piece sequence and renders a fixed
    /// chat-template string — drives the reasoning-split scenarios.
    struct ScriptedBackend {
        template: &'static str,
        pieces: &'static [&'static str],
    }
    impl InferenceBackend for ScriptedBackend {
        fn load_model(&self, _: &std::path::Path, _: BackendLoadParams) -> anyhow::Result<Box<dyn BackendModel>> {
            Ok(Box::new(ScriptedModel { template: self.template, pieces: self.pieces }))
        }
        fn name(&self) -> &str { "llamacpp" }
    }
    struct ScriptedModel {
        template: &'static str,
        pieces: &'static [&'static str],
    }
    impl BackendModel for ScriptedModel {
        fn generate(&self, _: &str, _params: &EngineParams, on_token: &mut dyn FnMut(&str) -> bool) -> anyhow::Result<GenerateResult> {
            for tok in self.pieces {
                if !on_token(tok) { break; }
            }
            Ok(GenerateResult {
                prompt_tokens: 5,
                completion_tokens: self.pieces.len() as u32,
                cache_hit: false,
            })
        }
        fn apply_chat_template(&self, _: &[(String, String)], _: &[crate::engine::tools::ToolSpec], _: &crate::engine::tools::ToolChoice) -> anyhow::Result<String> {
            Ok(self.template.to_string())
        }
        fn n_ctx(&self) -> u32 { 2048 }
        fn size_bytes(&self) -> u64 { 100 }
        fn kv_bytes_per_token(&self) -> u64 { 1 }
        fn as_any(&self) -> &dyn std::any::Any { self }
    }

    pub(crate) fn setup_store_and_manager(dir: &std::path::Path) -> (Arc<ModelStore>, Arc<ModelManager>) {
        setup_with_backend(dir, Box::new(FakeBackend))
    }

    fn setup_with_backend(
        dir: &std::path::Path,
        backend: Box<dyn InferenceBackend>,
    ) -> (Arc<ModelStore>, Arc<ModelManager>) {
        let store = ModelStore::new(Some(dir.to_path_buf()));
        std::fs::create_dir_all(store.models_dir()).unwrap();

        let model_dir = store.models_dir().join("test-org/test-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        let model_file = model_dir.join("model.gguf");
        std::fs::write(&model_file, b"fake-gguf").unwrap();

        let mut reg = Registry::load(&store.registry_path()).unwrap();
        reg.add("test-org/test-model/model.gguf".into(), ModelEntry {
            repo: "test-org/test-model".into(),
            filename: "model.gguf".into(),
            path: model_file,
            size_bytes: 9,
            downloaded_at: 1,
            digest: String::new(),
            model_name: String::new(),
            description: String::new(),
            architecture: String::new(),
            context_length: 0,
            metadata_read: true,
            format: ModelFormat::Gguf,
            base_model: String::new(),
            source: ModelSource::HfSourceDownloaded,
            mmproj_path: None,
        });
        reg.save(&store.registry_path()).unwrap();

        let mgr = ModelManager::with_backends(vec![backend], 0);
        (Arc::new(store), Arc::new(mgr))
    }

    #[tokio::test]
    async fn oai_chat_completions_streams_sse_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_store_and_manager(dir.path());
        let app = router(mgr, store);

        let body = serde_json::json!({
            "model": "test-org/test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "max_tokens": 10
        });

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 200);

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("data: "), "should contain SSE data lines");
        assert!(text.contains("[DONE]"), "should end with [DONE] sentinel");
    }

    #[tokio::test]
    async fn oai_chat_completions_non_stream_returns_json() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_store_and_manager(dir.path());
        let app = router(mgr, store);

        let body = serde_json::json!({
            "model": "test-org/test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false,
            "max_tokens": 10
        });

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 200);

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert!(json["choices"][0]["message"]["content"].as_str().unwrap().contains("Hello"));
    }

    async fn post_json(app: Router, uri: &str, body: serde_json::Value) -> (axum::http::StatusCode, String) {
        let req = axum::http::Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn oai_chat_splits_explicit_think_block_in_stream() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_with_backend(dir.path(), Box::new(ScriptedBackend {
            template: "prompt",
            pieces: &["<think>", "why", "</think>", "\n\n", "hi"],
        }));
        let (status, text) = post_json(router(mgr, store), "/v1/chat/completions", serde_json::json!({
            "model": "test-org/test-model",
            "messages": [{"role": "user", "content": "q"}],
            "stream": true,
            "max_tokens": 10,
            "stream_options": {"include_usage": true}
        })).await;

        assert_eq!(status, 200);
        assert!(text.contains(r#""reasoning_content":"why""#), "{text}");
        assert!(text.contains(r#""content":"hi""#), "{text}");
        assert!(!text.contains("<think>"), "think tags must not leak: {text}");
        assert!(text.contains(r#""finish_reason":"stop""#), "{text}");
        assert!(text.contains(r#""reasoning_tokens":1"#), "{text}");
    }

    #[tokio::test]
    async fn oai_chat_forced_open_template_splits_from_first_piece() {
        // Qwen3-thinking style: the template leaves the block open, the stream
        // carries only the closing tag. The load-time probe must catch it.
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_with_backend(dir.path(), Box::new(ScriptedBackend {
            template: "<|im_start|>assistant\n<think>\n",
            pieces: &["I am reasoning.", "</think>", "\n42"],
        }));
        let (status, text) = post_json(router(mgr, store), "/v1/chat/completions", serde_json::json!({
            "model": "test-org/test-model",
            "messages": [{"role": "user", "content": "q"}],
            "stream": true,
            "max_tokens": 10
        })).await;

        assert_eq!(status, 200);
        assert!(text.contains(r#""reasoning_content":"I am reasoning.""#), "{text}");
        assert!(text.contains(r#""content":"42""#), "{text}");
        assert!(!text.contains("</think>"), "close tag must not leak: {text}");
    }

    #[tokio::test]
    async fn oai_chat_non_stream_returns_reasoning_content_and_usage_detail() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_with_backend(dir.path(), Box::new(ScriptedBackend {
            template: "prompt",
            pieces: &["<think>", "why", "</think>", "\n\n", "hi"],
        }));
        let (status, text) = post_json(router(mgr, store), "/v1/chat/completions", serde_json::json!({
            "model": "test-org/test-model",
            "messages": [{"role": "user", "content": "q"}],
            "stream": false,
            "max_tokens": 10
        })).await;

        assert_eq!(status, 200);
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        let msg = &json["choices"][0]["message"];
        assert_eq!(msg["reasoning_content"], "why");
        assert_eq!(msg["content"], "hi");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["completion_tokens_details"]["reasoning_tokens"], 1);
    }

    #[tokio::test]
    async fn oai_chat_reports_length_when_budget_exhausted() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_store_and_manager(dir.path());
        // FakeModel reports completion_tokens = 2; cap the request at 2.
        let (status, text) = post_json(router(mgr, store), "/v1/chat/completions", serde_json::json!({
            "model": "test-org/test-model",
            "messages": [{"role": "user", "content": "q"}],
            "stream": false,
            "max_tokens": 2
        })).await;

        assert_eq!(status, 200);
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn oai_chat_exhausted_mid_think_is_reasoning_only_with_length() {
        // Issue #75's failure mode: the whole budget went to the think block.
        // The response must expose it as reasoning + "length", not as content.
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_with_backend(dir.path(), Box::new(ScriptedBackend {
            template: "<|im_start|>assistant\n<think>\n",
            pieces: &["thinking forever"],
        }));
        let (status, text) = post_json(router(mgr, store), "/v1/chat/completions", serde_json::json!({
            "model": "test-org/test-model",
            "messages": [{"role": "user", "content": "q"}],
            "stream": false,
            "max_tokens": 1
        })).await;

        assert_eq!(status, 200);
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        let msg = &json["choices"][0]["message"];
        assert_eq!(msg["reasoning_content"], "thinking forever");
        assert_eq!(msg["content"], "");
        assert_eq!(json["choices"][0]["finish_reason"], "length");
    }

    #[cfg(not(feature = "vision"))]
    #[tokio::test]
    async fn oai_chat_completions_rejects_images_without_vision() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_store_and_manager(dir.path());
        let app = router(mgr, store);

        let body = serde_json::json!({
            "model": "test-org/test-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                ]
            }]
        });

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn oai_embeddings_single_input() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_store_and_manager(dir.path());
        let app = router(mgr, store);

        let body = serde_json::json!({
            "model": "test-org/test-model",
            "input": "hello world"
        });

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 200);

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["object"], "embedding");
        assert_eq!(json["data"][0]["index"], 0);
        let emb = json["data"][0]["embedding"].as_array().unwrap();
        assert_eq!(emb.len(), 2);
        assert_eq!(json["usage"]["prompt_tokens"], 3);
    }

    #[tokio::test]
    async fn oai_embeddings_batch_input() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_store_and_manager(dir.path());
        let app = router(mgr, store);

        let body = serde_json::json!({
            "model": "test-org/test-model",
            "input": ["hello", "world"]
        });

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 200);

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["data"].as_array().unwrap().len(), 2);
        assert_eq!(json["data"][1]["index"], 1);
        assert_eq!(json["usage"]["prompt_tokens"], 6);
    }

    #[tokio::test]
    async fn oai_embeddings_rejects_empty_input() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_store_and_manager(dir.path());
        let app = router(mgr, store);

        let body = serde_json::json!({
            "model": "test-org/test-model",
            "input": ""
        });

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 400);
    }

    #[tokio::test]
    async fn oai_embeddings_rejects_too_many_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_store_and_manager(dir.path());
        let app = router(mgr, store);

        let body = serde_json::json!({
            "model": "test-org/test-model",
            "input": vec!["x"; MAX_EMBED_BATCH + 1],
        });

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 400);
    }

    #[tokio::test]
    async fn oai_embeddings_rejects_unknown_encoding_format() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_store_and_manager(dir.path());
        let app = router(mgr, store);

        let body = serde_json::json!({
            "model": "test-org/test-model",
            "input": "hello",
            "encoding_format": "binary"  // not "float" or "base64"
        });

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 400);
    }

    #[tokio::test]
    async fn oai_embeddings_rejects_dimensions_param() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_store_and_manager(dir.path());
        let app = router(mgr, store);

        let body = serde_json::json!({
            "model": "test-org/test-model",
            "input": "hello",
            "dimensions": 256,
        });

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 400);
    }

    #[test]
    fn tool_call_parsing_extracts_json_objects() {
        let output = r#"The model output {"name": "search", "arguments": "{\"query\": \"hello\"}"} and some more text"#;
        let (calls, remaining) = parse_tool_calls(output);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
        assert_eq!(calls[0].r#type, "function");
        assert!(calls[0].id.starts_with("call_"));
        assert!(!remaining.is_empty());
    }

    #[test]
    fn tool_call_parsing_handles_multiple_calls() {
        let output = r#"First {"name": "search", "arguments": "{\"q\": \"a\"}"} and then {"name": "calculate", "arguments": "{\"expr\": \"1+1\"}"}"#;
        let (calls, _) = parse_tool_calls(output);

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "search");
        assert_eq!(calls[1].function.name, "calculate");
    }

    #[test]
    fn tool_call_parsing_fallback_when_no_calls() {
        let output = "This is just regular model output with no tool calls";
        let (calls, remaining) = parse_tool_calls(output);

        assert_eq!(calls.len(), 0);
        assert_eq!(remaining, output);
    }

    #[test]
    fn tool_call_response_format_is_openai_compatible() {
        let call = OaiToolCallMessage {
            id: "call_abc123".to_string(),
            r#type: "function".to_string(),
            function: OaiToolCallFunction {
                name: "search".to_string(),
                arguments: r#"{"query":"hello"}"#.to_string(),
            },
        };

        let json = serde_json::json!({
            "id": call.id,
            "type": call.r#type,
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments
            }
        });

        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "search");
        assert!(json["id"].as_str().unwrap().starts_with("call_"));
    }

    #[tokio::test]
    async fn oai_chat_completions_accepts_functions_parameter() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_store_and_manager(dir.path());
        let app = router(mgr, store);

        let body = serde_json::json!({
            "model": "test-org/test-model",
            "messages": [{"role": "user", "content": "use search tool"}],
            "stream": false,
            "max_tokens": 10,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            }]
        });

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert!(resp.status().as_u16() >= 200 && resp.status().as_u16() < 500);
    }

    #[tokio::test]
    async fn oai_chat_completions_streams_tool_calls_with_correct_format() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_store_and_manager(dir.path());
        let app = router(mgr, store);

        let body = serde_json::json!({
            "model": "test-org/test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "max_tokens": 10,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]
        });

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 200);

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(text.contains("data: "));
        assert!(text.contains("[DONE]"));
        assert!(text.contains("finish_reason"));
    }

    #[tokio::test]
    async fn oai_chat_completions_non_stream_returns_tool_calls() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_store_and_manager(dir.path());
        let app = router(mgr, store);

        let body = serde_json::json!({
            "model": "test-org/test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false,
            "max_tokens": 10,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]
        });

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 200);

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();

        assert!(json["choices"].is_array());
        assert!(json["choices"][0]["message"].is_object());

        let message = &json["choices"][0]["message"];
        let has_content = message.get("content").is_some();
        let has_tool_calls = message.get("tool_calls").is_some();
        assert!(has_content || has_tool_calls);

        if let Some(tool_calls) = message.get("tool_calls") {
            assert!(tool_calls.is_array());
            if let Some(first_call) = tool_calls.get(0) {
                assert_eq!(first_call["type"], "function");
                assert!(first_call["id"].is_string());
                assert!(first_call["function"]["name"].is_string());
                assert!(first_call["function"]["arguments"].is_string());
            }
        }
    }

    #[tokio::test]
    async fn oai_chat_completions_fallback_without_tool_calls() {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = setup_store_and_manager(dir.path());
        let app = router(mgr, store);

        let body = serde_json::json!({
            "model": "test-org/test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false,
            "max_tokens": 10
        });

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 200);

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();

        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert!(json["choices"][0]["message"]["content"].is_string());
    }

    #[test]
    fn oai_chat_request_defaults_stream_to_false() {
        // OpenAI spec: `stream` defaults to false when the field is omitted.
        let req: OaiChatRequest =
            serde_json::from_str(r#"{"model":"m","messages":[]}"#).unwrap();
        assert!(!req.stream, "omitted stream must default to false");
    }

    #[test]
    fn oai_chat_request_honors_explicit_stream() {
        let on: OaiChatRequest =
            serde_json::from_str(r#"{"model":"m","messages":[],"stream":true}"#).unwrap();
        assert!(on.stream);
        let off: OaiChatRequest =
            serde_json::from_str(r#"{"model":"m","messages":[],"stream":false}"#).unwrap();
        assert!(!off.stream);
    }

    #[test]
    fn oai_completion_request_defaults_stream_to_false() {
        let req: OaiCompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":"hi"}"#).unwrap();
        assert!(!req.stream, "omitted stream must default to false");
    }
}

#[cfg(all(test, feature = "vision"))]
mod vision_tests {
    use super::*;

    #[test]
    fn decode_data_uri_decodes_base64_with_media_type() {
        let (bytes, media) = decode_data_uri("data:image/png;base64,aGVsbG8=").unwrap();
        assert_eq!(bytes.as_slice(), b"hello");
        assert_eq!(media.as_deref(), Some("image/png"));
    }

    #[test]
    fn decode_data_uri_rejects_uri_without_base64_marker() {
        let err = decode_data_uri("data:image/png,aGVsbG8=").unwrap_err();
        assert!(err.to_string().contains("base64"), "got: {err}");
    }

    #[test]
    fn decode_data_uri_rejects_disallowed_media_type() {
        let err = decode_data_uri("data:image/tiff;base64,aGVsbG8=").unwrap_err();
        assert!(err.to_string().contains("unsupported"), "got: {err}");
    }

    #[test]
    fn decode_data_uri_rejects_invalid_base64() {
        let err = decode_data_uri("data:image/png;base64,@@@").unwrap_err();
        assert!(err.to_string().contains("base64 decode"), "got: {err}");
    }
}
