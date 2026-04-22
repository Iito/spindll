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
use crate::model_store::registry::Registry;
use crate::model_store::ModelStore;

#[derive(Clone)]
struct AppState {
    manager: Arc<ModelManager>,
    store: Arc<ModelStore>,
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
    let state = AppState { manager, store };

    let app = Router::new()
        .route("/health", get(health))
        .route("/models", get(models))
        .route("/chat", post(chat))
        .route("/models/{id}", delete(model_delete))
        .route("/models/{id}/unload", post(model_unload))
        .route("/load", post(load))
        .route("/pull", post(pull))
        // OpenAI-compatible API
        .route("/v1/models", get(oai_models))
        .route("/v1/chat/completions", post(oai_chat_completions))
        .route("/v1/completions", post(oai_completions))
        .layer(CorsLayer::permissive())
        .with_state(state);

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

    let loaded: std::collections::HashSet<String> = state
        .manager
        .loaded_models()
        .into_iter()
        .map(|(name, _, _, _)| name)
        .collect();

    let list: Vec<ModelInfo> = reg
        .models
        .iter()
        .map(|(key, entry)| ModelInfo {
            name: key.clone(),
            size_bytes: entry.size_bytes,
            quantization: String::new(),
            digest: entry.digest.clone(),
            loaded: loaded.contains(key),
            model_name: entry.model_name.clone(),
            description: entry.description.clone(),
            architecture: entry.architecture.clone(),
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

    match state.store.remove(&id) {
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
    match state.manager.unload_model(&id) {
        Ok(()) => Json(serde_json::json!({"status": "ok"})).into_response(),
        Err(e) => (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": e.to_string()})),
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
        if let Err(e) = auto_load(&mgr, &store, &req.model) {
            let _ = tx.blocking_send(Ok(sse_data(&serde_json::json!({"type": "error", "error": e.to_string()}))));
            return;
        }

        let messages: Vec<_> = req.messages.iter().map(|m| (m.role.clone(), m.content.clone())).collect();
        let prompt = match mgr.apply_chat_template(&req.model, &messages) {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.blocking_send(Ok(sse_data(&serde_json::json!({"type": "error", "error": format!("chat template error: {e}")}))));
                return;
            }
        };

        let params = match req.params {
            Some(p) => GenerateParams {
                max_tokens: p.max_tokens.unwrap_or(512),
                temperature: p.temperature.unwrap_or(0.8),
                top_p: p.top_p.unwrap_or(0.95),
                top_k: p.top_k.unwrap_or(40),
                seed: p.seed.unwrap_or(42),
                prefill_only: false,
            },
            None => GenerateParams::default(),
        };

        let result = mgr.generate(&req.model, &prompt, &params, None, |token| {
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
    if state.manager.is_loaded(&req.model) {
        return Json(LoadResponse { already_loaded: true }).into_response();
    }

    let path = match state.store.resolve_model_path(&req.model) {
        Ok(p) => p,
        Err(e) => {
            return (
                axum::http::StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };
    let digest = state.store.resolve_model_digest(&req.model).unwrap_or_default();
    let gpu_layers = req.gpu_layers.and_then(|l| if l < 0 { None } else { Some(l as u32) });

    match state.manager.load_model_with_digest(&req.model, &path, gpu_layers, digest) {
        Ok(()) => Json(LoadResponse { already_loaded: false }).into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
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
        store.pull(&req.model, req.quantization.as_deref())
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

    let data: Vec<serde_json::Value> = reg
        .models
        .keys()
        .map(|key| {
            serde_json::json!({
                "id": key,
                "object": "model",
                "owned_by": "spindll",
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "data": data,
    }))
    .into_response()
}

// -- POST /v1/chat/completions ------------------------------------------------

#[derive(Deserialize)]
struct OaiChatRequest {
    model: String,
    messages: Vec<OaiMessage>,
    #[serde(default = "default_true")]
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
    /// Accepted for API compatibility; not yet used for constrained selection.
    #[serde(default)]
    #[allow(dead_code)]
    tool_choice: Option<serde_json::Value>,
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
struct OaiMessage {
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OaiToolCallMessage>>,
    #[serde(default)]
    tool_call_id: Option<String>,
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

/// Build a system prompt section describing available tools.
fn format_tools_for_prompt(tools: &[OaiTool]) -> String {
    let mut out = String::from(
        "You have access to the following tools. To call a tool, respond with a JSON object \
         in this exact format:\n\n\
         {\"name\": \"function_name\", \"arguments\": {\"param\": \"value\"}}\n\n\
         Available tools:\n\n",
    );
    for tool in tools {
        out.push_str(&format!("### {}\n", tool.function.name));
        if let Some(desc) = &tool.function.description {
            out.push_str(&format!("{desc}\n"));
        }
        if let Some(params) = &tool.function.parameters {
            out.push_str(&format!("Parameters: {}\n", serde_json::to_string(params).unwrap_or_default()));
        }
        out.push('\n');
    }
    out
}

/// Try to extract tool calls from model output.
///
/// Looks for JSON objects containing `"name"` and `"arguments"` keys, which is the
/// format most tool-calling models produce. Returns extracted calls and any remaining
/// text content.
fn parse_tool_calls(output: &str) -> (Vec<OaiToolCallMessage>, String) {
    let mut calls = Vec::new();
    let mut remaining = String::new();
    let trimmed = output.trim();

    // Try to find JSON objects in the output
    let mut search_from = 0;
    while search_from < trimmed.len() {
        if let Some(start) = trimmed[search_from..].find('{') {
            let abs_start = search_from + start;
            // Try increasingly larger slices to find valid JSON
            if let Some(call) = extract_tool_call_at(trimmed, abs_start) {
                remaining.push_str(&trimmed[search_from..abs_start]);
                let json_len = find_json_end(trimmed, abs_start).unwrap_or(trimmed.len()) - abs_start;
                search_from = abs_start + json_len;
                calls.push(call);
                continue;
            }
        }
        remaining.push_str(&trimmed[search_from..]);
        break;
    }

    (calls, remaining.trim().to_string())
}

fn extract_tool_call_at(text: &str, start: usize) -> Option<OaiToolCallMessage> {
    let end = find_json_end(text, start)?;
    let candidate = &text[start..end];
    let parsed: serde_json::Value = serde_json::from_str(candidate).ok()?;
    let obj = parsed.as_object()?;

    let name = obj.get("name")?.as_str()?;
    let arguments = obj.get("arguments")?;

    let call_id = format!("call_{:016x}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());

    Some(OaiToolCallMessage {
        id: call_id,
        r#type: "function".to_string(),
        function: OaiToolCallFunction {
            name: name.to_string(),
            arguments: if arguments.is_string() {
                arguments.as_str().unwrap().to_string()
            } else {
                serde_json::to_string(arguments).unwrap_or_default()
            },
        },
    })
}

/// Find the end of a balanced JSON object starting at `start`.
fn find_json_end(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(start)? != &b'{' {
        return None;
    }
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for i in start..bytes.len() {
        let ch = bytes[i];
        if escape {
            escape = false;
            continue;
        }
        if ch == b'\\' && in_string {
            escape = true;
            continue;
        }
        if ch == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if ch == b'{' {
            depth += 1;
        } else if ch == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(i + 1);
            }
        }
    }
    None
}

/// Prepare messages for template application, injecting tool descriptions when present.
fn prepare_messages_with_tools(
    messages: &[OaiMessage],
    tools: &Option<Vec<OaiTool>>,
) -> Vec<(String, String)> {
    let mut result: Vec<(String, String)> = Vec::new();

    // If tools are provided, prepend tool descriptions to the system message.
    let tool_preamble = tools.as_ref()
        .filter(|t| !t.is_empty())
        .map(|t| format_tools_for_prompt(t));

    let mut system_injected = false;

    for msg in messages {
        let content = msg.content.clone().unwrap_or_default();

        if msg.role == "system" && !system_injected {
            if let Some(ref preamble) = tool_preamble {
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
    if !system_injected {
        if let Some(preamble) = tool_preamble {
            result.insert(0, ("system".to_string(), preamble));
        }
    }

    result
}

async fn oai_chat_completions(
    State(state): State<AppState>,
    Json(req): Json<OaiChatRequest>,
) -> impl IntoResponse {
    let model_id = req.model.clone();
    let mgr = state.manager.clone();
    let store = state.store.clone();
    let has_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty());

    if req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);

        tokio::task::spawn_blocking(move || {
            if let Err(e) = auto_load(&mgr, &store, &req.model) {
                let _ = tx.blocking_send(Ok(sse_data(&oai_error(&e.to_string()))));
                return;
            }

            let messages = prepare_messages_with_tools(&req.messages, &req.tools);
            let prompt = match mgr.apply_chat_template(&req.model, &messages) {
                Ok(p) => p,
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
            };

            let completion_id = format!("chatcmpl-{:016x}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();

            if has_tools {
                // When tools are active, collect full output to parse tool calls.
                let mut output = String::new();
                let result = mgr.generate(&req.model, &prompt, &params, None, |token| {
                    output.push_str(token);
                    true
                });

                match result {
                    Ok(_) => {
                        let (tool_calls, remaining) = parse_tool_calls(&output);
                        if !tool_calls.is_empty() {
                            // Send tool calls as a single chunk
                            let delta: serde_json::Value = if remaining.is_empty() {
                                serde_json::json!({"tool_calls": tool_calls})
                            } else {
                                serde_json::json!({"content": remaining, "tool_calls": tool_calls})
                            };
                            let chunk = serde_json::json!({
                                "id": &completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": &req.model,
                                "choices": [{"index": 0, "delta": delta, "finish_reason": null}]
                            });
                            let _ = tx.blocking_send(Ok(sse_data(&chunk)));
                        } else {
                            // No tool calls detected — send as regular content
                            let chunk = serde_json::json!({
                                "id": &completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": &req.model,
                                "choices": [{"index": 0, "delta": {"content": output}, "finish_reason": null}]
                            });
                            let _ = tx.blocking_send(Ok(sse_data(&chunk)));
                        }
                        let finish = if !tool_calls.is_empty() { "tool_calls" } else { "stop" };
                        let done_chunk = serde_json::json!({
                            "id": &completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": &req.model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]
                        });
                        let _ = tx.blocking_send(Ok(sse_data(&done_chunk)));
                        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
                    }
                    Err(e) => {
                        let _ = tx.blocking_send(Ok(sse_data(&oai_error(&e.to_string()))));
                    }
                }
            } else {
                // No tools — stream tokens directly as before.
                let result = mgr.generate(&req.model, &prompt, &params, None, |token| {
                    let chunk = serde_json::json!({
                        "id": &completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": &req.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": null,
                        }]
                    });
                    tx.blocking_send(Ok(sse_data(&chunk))).is_ok()
                });

                match result {
                    Ok(_) => {
                        let done_chunk = serde_json::json!({
                            "id": &completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": &req.model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
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
            }
            drop(tx);
        });

        Sse::new(ReceiverStream::new(rx)).into_response()
    } else {
        // Non-streaming: collect all tokens then return a single JSON response.
        let result = tokio::task::spawn_blocking(move || {
            auto_load(&mgr, &store, &req.model)?;

            let messages = prepare_messages_with_tools(&req.messages, &req.tools);
            let prompt = mgr.apply_chat_template(&req.model, &messages)?;

            let params = GenerateParams {
                max_tokens: req.max_tokens.unwrap_or(512),
                temperature: req.temperature.unwrap_or(0.8),
                top_p: req.top_p.unwrap_or(0.95),
                top_k: 40,
                seed: req.seed.unwrap_or(42),
                prefill_only: false,
            };

            let mut output = String::new();
            let stats = mgr.generate(&req.model, &prompt, &params, None, |token| {
                output.push_str(token);
                true
            })?;

            Ok::<_, anyhow::Error>((output, stats))
        })
        .await;

        match result {
            Ok(Ok((content, stats))) => {
                let completion_id = format!("chatcmpl-{:016x}", std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();

                let (message, finish_reason) = if has_tools {
                    let (tool_calls, remaining) = parse_tool_calls(&content);
                    if !tool_calls.is_empty() {
                        let msg = if remaining.is_empty() {
                            serde_json::json!({"role": "assistant", "content": null, "tool_calls": tool_calls})
                        } else {
                            serde_json::json!({"role": "assistant", "content": remaining, "tool_calls": tool_calls})
                        };
                        (msg, "tool_calls")
                    } else {
                        (serde_json::json!({"role": "assistant", "content": content}), "stop")
                    }
                } else {
                    (serde_json::json!({"role": "assistant", "content": content}), "stop")
                };

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

// -- POST /v1/completions ----------------------------------------------------

#[derive(Deserialize)]
struct OaiCompletionRequest {
    model: String,
    prompt: String,
    #[serde(default = "default_true")]
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
            if let Err(e) = auto_load(&mgr, &store, &req.model) {
                let _ = tx.blocking_send(Ok(sse_data(&oai_error(&e.to_string()))));
                return;
            }

            let params = GenerateParams {
                max_tokens: req.max_tokens.unwrap_or(512),
                temperature: req.temperature.unwrap_or(0.8),
                top_p: req.top_p.unwrap_or(0.95),
                top_k: 40,
                seed: req.seed.unwrap_or(42),
                prefill_only: false,
            };

            let completion_id = format!("cmpl-{:016x}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();

            let result = mgr.generate(&req.model, &req.prompt, &params, None, |token| {
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
            auto_load(&mgr, &store, &req.model)?;

            let params = GenerateParams {
                max_tokens: req.max_tokens.unwrap_or(512),
                temperature: req.temperature.unwrap_or(0.8),
                top_p: req.top_p.unwrap_or(0.95),
                top_k: 40,
                seed: req.seed.unwrap_or(42),
                prefill_only: false,
            };

            let mut output = String::new();
            let stats = mgr.generate(&req.model, &req.prompt, &params, None, |token| {
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

fn oai_error(msg: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": msg,
            "type": "server_error",
        }
    })
}

/// Auto-load a model if not already in memory.
fn auto_load(
    mgr: &ModelManager,
    store: &ModelStore,
    model: &str,
) -> anyhow::Result<()> {
    if mgr.is_loaded(model) {
        return Ok(());
    }
    let path = store.resolve_model_path(model)?;
    let digest = store.resolve_model_digest(model).unwrap_or_default();
    mgr.load_model_with_digest(model, &path, None, digest)?;
    Ok(())
}
