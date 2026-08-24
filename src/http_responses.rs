//! OpenAI Responses API compatibility (`POST /v1/responses`) — the stateless
//! subset Codex CLI uses with `store: false`.
//!
//! Covered: `input` as a string or item array (`message` — typed or the bare
//! `{role, content}` form — plus `function_call` / `function_call_output`),
//! `instructions`, flat function tools, `tool_choice`, `max_output_tokens`,
//! and the item-based SSE grammar (`response.created` →
//! `response.output_item.added` → `response.output_text.delta` /
//! `response.function_call_arguments.*` → a terminal `response.completed`,
//! `response.incomplete` (token cap), or `response.failed` (error), every
//! event carrying a monotonic `sequence_number`). Stateful features are out of
//! scope: `previous_response_id` is rejected with a clear 400, `store` is
//! accepted and ignored (nothing is persisted), reasoning items are dropped.
//! `input_image` parts are not supported yet (400).

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::sse::{Event, Sse};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_stream::wrappers::ReceiverStream;

use crate::engine::streaming::GenerateParams;
use crate::engine::tools::{ToolChoice, ToolSpec};
use crate::http::{AppState, auto_load};

// -- Request types -----------------------------------------------------------

#[derive(Deserialize)]
pub(crate) struct ResponsesRequest {
    model: String,
    #[serde(default)]
    input: Option<RespInput>,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    tools: Option<Vec<RespTool>>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    previous_response_id: Option<String>,
}

/// `input`: a bare string (one user turn) or a list of items.
#[derive(Deserialize)]
#[serde(untagged)]
enum RespInput {
    Text(String),
    Items(Vec<RespItem>),
}

/// One input item. The API allows both the typed form (`{"type": "message",
/// ...}`) and the bare `{role, content}` message form, so this is an untagged
/// wrapper over the two.
#[derive(Deserialize)]
#[serde(untagged)]
enum RespItem {
    Typed(TypedRespItem),
    Easy { role: String, content: RespContent },
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum TypedRespItem {
    #[serde(rename = "message")]
    Message { role: String, content: RespContent },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        call_id: String,
        output: RespFnOutput,
    },
    /// reasoning / item_reference / anything newer: accepted, dropped —
    /// Codex replays reasoning items when resending a transcript.
    #[serde(other)]
    Unknown,
}

/// Message `content`: a plain string or a list of typed parts.
#[derive(Deserialize)]
#[serde(untagged)]
enum RespContent {
    Text(String),
    Parts(Vec<RespPart>),
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum RespPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(rename = "input_image")]
    InputImage {},
    #[serde(other)]
    Unknown,
}

/// `function_call_output.output`: a string or a list of output parts.
#[derive(Deserialize)]
#[serde(untagged)]
enum RespFnOutput {
    Text(String),
    Parts(Vec<RespPart>),
}

/// Responses tool definitions are flat (`{"type": "function", "name", ...}`),
/// unlike chat completions' nested `{"function": {...}}`.
#[derive(Deserialize)]
struct RespTool {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<Value>,
}

// -- Mapping helpers ---------------------------------------------------------

fn parts_text(parts: &[RespPart]) -> String {
    let texts: Vec<&str> = parts
        .iter()
        .filter_map(|p| match p {
            RespPart::InputText { text } | RespPart::OutputText { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    texts.join("\n")
}

fn content_text(content: &RespContent) -> String {
    match content {
        RespContent::Text(s) => s.clone(),
        RespContent::Parts(parts) => parts_text(parts),
    }
}

fn fn_output_text(output: &RespFnOutput) -> String {
    match output {
        RespFnOutput::Text(s) => s.clone(),
        RespFnOutput::Parts(parts) => parts_text(parts),
    }
}

fn content_has_images(content: &RespContent) -> bool {
    matches!(content, RespContent::Parts(parts)
        if parts.iter().any(|p| matches!(p, RespPart::InputImage {})))
}

fn input_has_images(input: Option<&RespInput>) -> bool {
    let Some(RespInput::Items(items)) = input else { return false };
    items.iter().any(|item| match item {
        RespItem::Easy { content, .. } => content_has_images(content),
        RespItem::Typed(TypedRespItem::Message { content, .. }) => content_has_images(content),
        RespItem::Typed(_) => false,
    })
}

/// Lower `instructions` + `input` to the engine's `(role, content)` pairs.
///
/// `developer` maps to `system` (the Responses-API name for the same turn).
/// `function_call` items are serialized in the same OpenAI-shaped JSON the
/// chat-completions path feeds the template, keyed by `call_id`, so the
/// round-trip through `function_call_output` lines up.
fn responses_to_pairs(
    instructions: Option<&str>,
    input: Option<&RespInput>,
) -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    if let Some(instr) = instructions {
        pairs.push(("system".to_string(), instr.to_string()));
    }

    match input {
        None => {}
        Some(RespInput::Text(s)) => pairs.push(("user".to_string(), s.clone())),
        Some(RespInput::Items(items)) => {
            for item in items {
                match item {
                    RespItem::Easy { role, content }
                    | RespItem::Typed(TypedRespItem::Message { role, content }) => {
                        let role = if role == "developer" { "system" } else { role.as_str() };
                        pairs.push((role.to_string(), content_text(content)));
                    }
                    RespItem::Typed(TypedRespItem::FunctionCall { call_id, name, arguments }) => {
                        let calls = json!([{
                            "id": call_id,
                            "type": "function",
                            "function": { "name": name, "arguments": arguments }
                        }]);
                        pairs.push(("assistant".to_string(), calls.to_string()));
                    }
                    RespItem::Typed(TypedRespItem::FunctionCallOutput { call_id, output }) => {
                        pairs.push((
                            "user".to_string(),
                            format!("[Tool result for {call_id}]: {}", fn_output_text(output)),
                        ));
                    }
                    RespItem::Typed(TypedRespItem::Unknown) => {}
                }
            }
        }
    }
    pairs
}

/// Only `function` tools map onto the engine; hosted tool types
/// (`web_search`, `local_shell`, …) are skipped — spindll has no
/// server-side executors for them.
fn resp_tools_to_specs(tools: &[RespTool]) -> Vec<ToolSpec> {
    tools
        .iter()
        .filter(|t| t.kind.as_deref() == Some("function"))
        .filter_map(|t| {
            t.name.as_ref().map(|name| ToolSpec {
                name: name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            })
        })
        .collect()
}

/// Responses `tool_choice`: `"auto" | "none" | "required"` or
/// `{"type": "function", "name": "f"}` (name at the top level).
fn resp_tool_choice(value: Option<&Value>) -> ToolChoice {
    match value {
        None => ToolChoice::Auto,
        Some(Value::String(s)) => match s.as_str() {
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            _ => ToolChoice::Auto,
        },
        Some(Value::Object(obj)) => obj
            .get("name")
            .or_else(|| obj.get("function").and_then(|f| f.get("name")))
            .and_then(Value::as_str)
            .map(|n| ToolChoice::Named(n.to_string()))
            .unwrap_or(ToolChoice::Auto),
        _ => ToolChoice::Auto,
    }
}

// -- Response helpers --------------------------------------------------------

/// Terminal SSE event for a finished response: the API signals an
/// `incomplete` status with its own `response.incomplete` event type, not a
/// `response.completed` carrying the status (Codex parses the event name).
fn terminal_event(status: &str) -> &'static str {
    if status == "incomplete" { "response.incomplete" } else { "response.completed" }
}

/// Response object for a `response.failed` event. Codex surfaces
/// `response.error.code` / `.message` from this shape; a bare `error` event
/// would be silently ignored and reported as a closed stream instead.
fn failed_response(id: &str, model: &str, msg: &str) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": unix_secs(),
        "status": "failed",
        "incomplete_details": null,
        "error": { "code": "server_error", "message": msg },
        "model": model,
        "output": [],
        "usage": null,
    })
}

fn resp_error_body(kind: &str, msg: &str) -> Value {
    json!({ "error": { "type": kind, "message": msg } })
}

fn resp_error(status: StatusCode, kind: &str, msg: &str) -> axum::response::Response {
    (status, Json(resp_error_body(kind, msg))).into_response()
}

fn new_id(prefix: &str) -> String {
    format!(
        "{prefix}_{:016x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    )
}

fn unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Build the `output` item array from the finished text: an assistant message
/// item for leftover prose, one `function_call` item per parsed call.
fn build_output_items(output: &str, has_tools: bool) -> (Vec<Value>, bool) {
    let mut items: Vec<Value> = Vec::new();
    let (calls, remaining) = if has_tools {
        crate::engine::tools::parse_tool_calls(output)
    } else {
        (Vec::new(), output.to_string())
    };

    if !remaining.is_empty() || calls.is_empty() {
        items.push(json!({
            "type": "message",
            "id": new_id("msg"),
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": remaining, "annotations": [] }],
        }));
    }
    let has_calls = !calls.is_empty();
    for call in calls {
        items.push(json!({
            "type": "function_call",
            "id": new_id("fc"),
            "call_id": call.id,
            "name": call.name,
            "arguments": call.arguments,
            "status": "completed",
        }));
    }
    (items, has_calls)
}

/// Assemble the full response object. `status` is `incomplete` only when the
/// token cap was hit without producing a tool call.
fn build_response(
    id: &str,
    model: &str,
    output: Vec<Value>,
    status: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": unix_secs(),
        "status": status,
        "incomplete_details": if status == "incomplete" {
            json!({ "reason": "max_output_tokens" })
        } else {
            Value::Null
        },
        "error": null,
        "model": model,
        "output": output,
        "parallel_tool_calls": true,
        "usage": {
            "input_tokens": prompt_tokens,
            "input_tokens_details": { "cached_tokens": 0 },
            "output_tokens": completion_tokens,
            "output_tokens_details": { "reasoning_tokens": 0 },
            "total_tokens": prompt_tokens + completion_tokens,
        }
    })
}

/// Sequenced SSE emitter: every Responses event carries the `type` field and
/// a monotonic `sequence_number`.
struct Emitter {
    tx: tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
    seq: u64,
}

impl Emitter {
    fn send(&mut self, event_type: &str, mut extra: Value) -> bool {
        let obj = extra.as_object_mut().expect("event payload must be an object");
        obj.insert("type".into(), json!(event_type));
        obj.insert("sequence_number".into(), json!(self.seq));
        self.seq += 1;
        self.tx
            .blocking_send(Ok(Event::default().event(event_type).data(extra.to_string())))
            .is_ok()
    }
}

// -- Handler -----------------------------------------------------------------

pub(crate) async fn responses_create(
    State(state): State<AppState>,
    Json(req): Json<ResponsesRequest>,
) -> axum::response::Response {
    if req.previous_response_id.is_some() {
        return resp_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "previous_response_id is not supported: spindll serves the stateless subset \
             (Codex: this is the store:false path and needs no configuration)",
        );
    }
    if req.input.is_none() && req.instructions.is_none() {
        return resp_error(StatusCode::BAD_REQUEST, "invalid_request_error", "input: field required");
    }
    if input_has_images(req.input.as_ref()) {
        return resp_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "input_image parts are not supported on /v1/responses yet",
        );
    }

    let mgr = state.manager.clone();
    let store = state.store.clone();
    let tool_choice = resp_tool_choice(req.tool_choice.as_ref());
    let tool_specs = resp_tools_to_specs(req.tools.as_deref().unwrap_or_default());
    let has_tools = !tool_specs.is_empty() && tool_choice != ToolChoice::None;
    let tool_specs = if has_tools { tool_specs } else { Vec::new() };
    let max_tokens = req.max_output_tokens.unwrap_or(512);

    let params = GenerateParams {
        max_tokens,
        temperature: req.temperature.unwrap_or(0.8),
        top_p: req.top_p.unwrap_or(0.95),
        top_k: 40,
        seed: 42,
        prefill_only: false,
        ..Default::default()
    };

    let pairs = responses_to_pairs(req.instructions.as_deref(), req.input.as_ref());
    let model = req.model.clone();

    if req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);

        tokio::task::spawn_blocking(move || {
            let mut em = Emitter { tx, seq: 0 };
            let resp_id = new_id("resp");
            let key = match auto_load(&mgr, &store, &model) {
                Ok(k) => k,
                Err(e) => {
                    em.send("response.failed", json!({
                        "response": failed_response(&resp_id, &model, &e.to_string())
                    }));
                    return;
                }
            };

            let skeleton = build_response(&resp_id, &model, Vec::new(), "in_progress", 0, 0);
            em.send("response.created", json!({ "response": skeleton }));
            em.send("response.in_progress", json!({ "response": skeleton }));

            if has_tools {
                // Buffer the full output so calls can be parsed, then replay
                // it through the item grammar.
                let mut output = String::new();
                let result = mgr.generate_chat(&key, &pairs, &tool_specs, &tool_choice, &params, None, |t| {
                    output.push_str(t);
                    true
                });
                match result {
                    Ok(stats) => {
                        let (items, has_calls) = build_output_items(&output, true);
                        for (i, item) in items.iter().enumerate() {
                            stream_item(&mut em, i, item);
                        }
                        let status = if !has_calls && stats.completion_tokens >= max_tokens {
                            "incomplete"
                        } else {
                            "completed"
                        };
                        let full = build_response(&resp_id, &model, items, status,
                            stats.prompt_tokens, stats.completion_tokens);
                        em.send(terminal_event(status), json!({ "response": full }));
                    }
                    Err(e) => {
                        em.send("response.failed", json!({
                            "response": failed_response(&resp_id, &model, &e.to_string())
                        }));
                    }
                }
            } else {
                // Pure text: one message item, live token deltas.
                let item_id = new_id("msg");
                em.send("response.output_item.added", json!({
                    "output_index": 0,
                    "item": {
                        "type": "message", "id": &item_id, "status": "in_progress",
                        "role": "assistant", "content": []
                    }
                }));
                em.send("response.content_part.added", json!({
                    "item_id": &item_id, "output_index": 0, "content_index": 0,
                    "part": { "type": "output_text", "text": "", "annotations": [] }
                }));

                let mut text = String::new();
                let result = mgr.generate_chat(&key, &pairs, &tool_specs, &tool_choice, &params, None, |token| {
                    text.push_str(token);
                    em.send("response.output_text.delta", json!({
                        "item_id": &item_id, "output_index": 0, "content_index": 0,
                        "delta": token
                    }))
                });

                match result {
                    Ok(stats) => {
                        em.send("response.output_text.done", json!({
                            "item_id": &item_id, "output_index": 0, "content_index": 0,
                            "text": &text
                        }));
                        em.send("response.content_part.done", json!({
                            "item_id": &item_id, "output_index": 0, "content_index": 0,
                            "part": { "type": "output_text", "text": &text, "annotations": [] }
                        }));
                        let item = json!({
                            "type": "message", "id": &item_id, "status": "completed",
                            "role": "assistant",
                            "content": [{ "type": "output_text", "text": &text, "annotations": [] }]
                        });
                        em.send("response.output_item.done", json!({ "output_index": 0, "item": item }));
                        let status = if stats.completion_tokens >= max_tokens { "incomplete" } else { "completed" };
                        let full = build_response(&resp_id, &model, vec![item], status,
                            stats.prompt_tokens, stats.completion_tokens);
                        em.send(terminal_event(status), json!({ "response": full }));
                    }
                    Err(e) => {
                        em.send("response.failed", json!({
                            "response": failed_response(&resp_id, &model, &e.to_string())
                        }));
                    }
                }
            }
        });

        Sse::new(ReceiverStream::new(rx)).into_response()
    } else {
        let result = tokio::task::spawn_blocking(move || {
            let key = auto_load(&mgr, &store, &model)?;
            let mut output = String::new();
            let stats = mgr.generate_chat(&key, &pairs, &tool_specs, &tool_choice, &params, None, |t| {
                output.push_str(t);
                true
            })?;
            Ok::<_, anyhow::Error>((output, stats, model))
        })
        .await;

        match result {
            Ok(Ok((output, stats, model))) => {
                let (items, has_calls) = build_output_items(&output, has_tools);
                let status = if !has_calls && stats.completion_tokens >= max_tokens {
                    "incomplete"
                } else {
                    "completed"
                };
                Json(build_response(&new_id("resp"), &model, items, status,
                    stats.prompt_tokens, stats.completion_tokens))
                .into_response()
            }
            Ok(Err(e)) => resp_error(StatusCode::INTERNAL_SERVER_ERROR, "api_error", &e.to_string()),
            Err(e) => resp_error(StatusCode::INTERNAL_SERVER_ERROR, "api_error", &e.to_string()),
        }
    }
}

/// Replay one finished output item through the streaming grammar (used on the
/// buffered tools path; arguments arrive as a single fragment).
fn stream_item(em: &mut Emitter, index: usize, item: &Value) {
    match item["type"].as_str() {
        Some("message") => {
            let item_id = item["id"].clone();
            let text = item["content"][0]["text"].clone();
            em.send("response.output_item.added", json!({
                "output_index": index,
                "item": {
                    "type": "message", "id": item_id, "status": "in_progress",
                    "role": "assistant", "content": []
                }
            }));
            em.send("response.content_part.added", json!({
                "item_id": item_id, "output_index": index, "content_index": 0,
                "part": { "type": "output_text", "text": "", "annotations": [] }
            }));
            em.send("response.output_text.delta", json!({
                "item_id": item_id, "output_index": index, "content_index": 0,
                "delta": text
            }));
            em.send("response.output_text.done", json!({
                "item_id": item_id, "output_index": index, "content_index": 0,
                "text": text
            }));
            em.send("response.content_part.done", json!({
                "item_id": item_id, "output_index": index, "content_index": 0,
                "part": { "type": "output_text", "text": text, "annotations": [] }
            }));
        }
        Some("function_call") => {
            let item_id = item["id"].clone();
            let arguments = item["arguments"].clone();
            let mut added = item.clone();
            added["status"] = json!("in_progress");
            added["arguments"] = json!("");
            em.send("response.output_item.added", json!({
                "output_index": index, "item": added
            }));
            em.send("response.function_call_arguments.delta", json!({
                "item_id": item_id, "output_index": index, "delta": arguments
            }));
            em.send("response.function_call_arguments.done", json!({
                "item_id": item_id, "output_index": index, "arguments": arguments
            }));
        }
        _ => return,
    }
    em.send("response.output_item.done", json!({ "output_index": index, "item": item }));
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_req(body: serde_json::Value) -> ResponsesRequest {
        serde_json::from_value(body).unwrap()
    }

    #[test]
    fn string_input_becomes_user_turn_after_instructions() {
        let req = parse_req(json!({ "model": "m", "input": "hi", "instructions": "be brief" }));
        let pairs = responses_to_pairs(req.instructions.as_deref(), req.input.as_ref());
        assert_eq!(pairs, vec![
            ("system".into(), "be brief".into()),
            ("user".into(), "hi".into()),
        ]);
    }

    #[test]
    fn item_forms_lower_to_pairs() {
        let req = parse_req(json!({ "model": "m", "input": [
            {"role": "developer", "content": "rules"},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "question"}
            ]},
            {"type": "reasoning", "summary": []},
            {"type": "function_call", "call_id": "call_1", "name": "ls", "arguments": "{\"path\":\".\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "README.md"}
        ]}));
        let pairs = responses_to_pairs(None, req.input.as_ref());
        assert_eq!(pairs.len(), 4);
        assert_eq!(pairs[0], ("system".into(), "rules".into()));
        assert_eq!(pairs[1], ("user".into(), "question".into()));
        assert_eq!(pairs[2].0, "assistant");
        assert!(pairs[2].1.contains("\"call_1\""));
        assert!(pairs[2].1.contains("\"ls\""));
        assert_eq!(pairs[3], ("user".into(), "[Tool result for call_1]: README.md".into()));
    }

    #[test]
    fn function_tools_map_and_hosted_tools_are_skipped() {
        let req = parse_req(json!({ "model": "m", "input": "x", "tools": [
            {"type": "function", "name": "grep", "description": "search", "parameters": {"type": "object"}},
            {"type": "web_search"},
            {"type": "local_shell"}
        ]}));
        let specs = resp_tools_to_specs(req.tools.as_deref().unwrap());
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "grep");
        assert_eq!(specs[0].description.as_deref(), Some("search"));
    }

    #[test]
    fn tool_choice_mapping_covers_both_object_forms() {
        assert_eq!(resp_tool_choice(None), ToolChoice::Auto);
        assert_eq!(resp_tool_choice(Some(&json!("none"))), ToolChoice::None);
        assert_eq!(resp_tool_choice(Some(&json!("required"))), ToolChoice::Required);
        assert_eq!(
            resp_tool_choice(Some(&json!({"type": "function", "name": "f"}))),
            ToolChoice::Named("f".into())
        );
        assert_eq!(
            resp_tool_choice(Some(&json!({"type": "function", "function": {"name": "g"}}))),
            ToolChoice::Named("g".into())
        );
    }

    #[test]
    fn output_items_split_prose_and_calls() {
        let (items, has_calls) = build_output_items("plain answer", false);
        assert!(!has_calls);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "message");
        assert_eq!(items[0]["content"][0]["text"], "plain answer");
        assert_eq!(items[0]["content"][0]["type"], "output_text");
    }

    // -- Endpoint tests against the fake backend ----------------------------

    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    async fn post_json(body: serde_json::Value) -> (u16, String) {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = crate::http::tests::setup_store_and_manager(dir.path());
        let app = crate::http::router(mgr, store);
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status().as_u16();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn responses_non_streaming_returns_output_items_and_usage() {
        let (status, text) = post_json(json!({
            "model": "test-org/test-model",
            "input": "hi",
            "max_output_tokens": 10
        })).await;
        assert_eq!(status, 200);
        let v: Value = serde_json::from_str(&text).unwrap();
        assert!(v["id"].as_str().unwrap().starts_with("resp_"));
        assert_eq!(v["object"], "response");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["output"][0]["type"], "message");
        assert_eq!(v["output"][0]["role"], "assistant");
        assert_eq!(v["output"][0]["content"][0]["text"], "Hello world");
        assert_eq!(v["usage"]["input_tokens"], 5);
        assert_eq!(v["usage"]["output_tokens"], 2);
        assert_eq!(v["usage"]["total_tokens"], 7);
    }

    #[tokio::test]
    async fn responses_hitting_cap_reports_incomplete() {
        let (status, text) = post_json(json!({
            "model": "test-org/test-model",
            "input": "hi",
            "max_output_tokens": 2
        })).await;
        assert_eq!(status, 200);
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["status"], "incomplete");
        assert_eq!(v["incomplete_details"]["reason"], "max_output_tokens");
    }

    #[tokio::test]
    async fn responses_streaming_emits_item_grammar_in_order() {
        let (status, text) = post_json(json!({
            "model": "test-org/test-model",
            "input": "hi",
            "stream": true,
            "max_output_tokens": 10
        })).await;
        assert_eq!(status, 200);
        let order = [
            "event: response.created",
            "event: response.in_progress",
            "event: response.output_item.added",
            "event: response.content_part.added",
            "event: response.output_text.delta",
            "event: response.output_text.done",
            "event: response.content_part.done",
            "event: response.output_item.done",
            "event: response.completed",
        ];
        let mut last = 0;
        for marker in order {
            let pos = text[last..].find(marker)
                .unwrap_or_else(|| panic!("missing or out of order: {marker}\n{text}"));
            last += pos;
        }
        assert!(text.contains("\"sequence_number\":0"));
        assert!(!text.contains("[DONE]"), "Responses SSE has no [DONE] sentinel");
    }

    #[tokio::test]
    async fn responses_streaming_cap_ends_with_incomplete_event() {
        let (status, text) = post_json(json!({
            "model": "test-org/test-model",
            "input": "hi",
            "stream": true,
            "max_output_tokens": 2
        })).await;
        assert_eq!(status, 200);
        assert!(text.contains("event: response.incomplete"), "terminal event must be response.incomplete\n{text}");
        assert!(!text.contains("event: response.completed"), "must not also send response.completed\n{text}");
        assert!(text.contains("\"reason\":\"max_output_tokens\""));
    }

    #[tokio::test]
    async fn responses_streaming_error_emits_response_failed() {
        let (status, text) = post_json(json!({
            "model": "no-such/model",
            "input": "hi",
            "stream": true
        })).await;
        assert_eq!(status, 200);
        assert!(text.contains("event: response.failed"), "load failure must emit response.failed\n{text}");
        assert!(text.contains("\"status\":\"failed\""));
        assert!(text.contains("\"code\":\"server_error\""));
        assert!(!text.contains("event: response.completed"));
    }

    #[tokio::test]
    async fn responses_rejects_previous_response_id() {
        let (status, text) = post_json(json!({
            "model": "test-org/test-model",
            "input": "hi",
            "previous_response_id": "resp_123"
        })).await;
        assert_eq!(status, 400);
        assert!(text.contains("previous_response_id"));
    }

    #[tokio::test]
    async fn responses_rejects_image_parts() {
        let (status, text) = post_json(json!({
            "model": "test-org/test-model",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
            ]}]
        })).await;
        assert_eq!(status, 400);
        assert!(text.contains("input_image"));
    }
}
