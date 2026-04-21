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
}

async fn models(State(state): State<AppState>) -> impl IntoResponse {
    let reg = match Registry::load(&state.store.registry_path()) {
        Ok(r) => r,
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };

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
        // Auto-load model if needed.
        if !mgr.is_loaded(&req.model) {
            let path = match store.resolve_model_path(&req.model) {
                Ok(p) => p,
                Err(e) => {
                    let _ = tx.blocking_send(Ok(sse_data(&serde_json::json!({"type": "error", "error": format!("model '{}' not found: {e}", req.model)}))));
                    return;
                }
            };
            let digest = store.resolve_model_digest(&req.model).unwrap_or_default();
            if let Err(e) = mgr.load_model_with_digest(&req.model, &path, None, digest) {
                let _ = tx.blocking_send(Ok(sse_data(&serde_json::json!({"type": "error", "error": format!("failed to load model: {e}")}))));
                return;
            }
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
        .keep_alive(axum::response::sse::KeepAlive::default())
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
