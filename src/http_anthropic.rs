//! Anthropic Messages API compatibility (`POST /v1/messages`).
//!
//! Translates the Anthropic dialect onto the shared engine chat path so
//! Anthropic SDK clients (Claude Code among them) can point
//! `ANTHROPIC_BASE_URL` at spindll. Text, image (behind `vision`),
//! `tool_use`/`tool_result` blocks, and `stop_sequences` are mapped; thinking
//! and other unknown block types are accepted and dropped. Streaming follows
//! the Messages SSE grammar — message_start → content_block_start /
//! content_block_delta / content_block_stop → message_delta → message_stop —
//! with named `event:` lines and no `[DONE]` sentinel.

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
use crate::engine::residency::ensure_loaded;
use crate::http::AppState;

// -- Request types -----------------------------------------------------------

#[derive(Deserialize)]
pub(crate) struct AnthMessagesRequest {
    model: String,
    /// Required by the Anthropic API; `Option` here so the missing-field error
    /// comes back in the Anthropic error shape instead of a serde 422.
    #[serde(default)]
    max_tokens: Option<u32>,
    messages: Vec<AnthMessage>,
    #[serde(default)]
    system: Option<AnthSystem>,
    #[serde(default)]
    tools: Option<Vec<AnthTool>>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    stop_sequences: Vec<String>,
}

#[derive(Deserialize)]
struct AnthMessage {
    role: String,
    content: AnthContent,
}

/// Anthropic `system`: a plain string or a list of text blocks.
#[derive(Deserialize)]
#[serde(untagged)]
enum AnthSystem {
    Text(String),
    Blocks(Vec<AnthBlock>),
}

/// Message `content`: a plain string or a list of typed blocks.
#[derive(Deserialize)]
#[serde(untagged)]
enum AnthContent {
    Text(String),
    Blocks(Vec<AnthBlock>),
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum AnthBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    #[cfg_attr(not(feature = "vision"), allow(dead_code))]
    Image { source: AnthImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<AnthToolResultContent>,
        #[serde(default)]
        is_error: bool,
    },
    /// thinking / redacted_thinking / document / anything newer: accepted,
    /// dropped. Rejecting them would break clients that replay history.
    #[serde(other)]
    Unknown,
}

/// `tool_result.content`: a plain string or nested blocks (text/image).
#[derive(Deserialize)]
#[serde(untagged)]
enum AnthToolResultContent {
    Text(String),
    Blocks(Vec<AnthBlock>),
}

#[derive(Deserialize)]
#[cfg_attr(not(feature = "vision"), allow(dead_code))]
struct AnthImageSource {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    media_type: Option<String>,
    #[serde(default)]
    data: Option<String>,
    #[serde(default)]
    url: Option<String>,
}

#[derive(Deserialize)]
struct AnthTool {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    input_schema: Option<Value>,
}

// -- Mapping helpers ---------------------------------------------------------

/// Flatten a block list to text: text blocks joined with `\n`, other types
/// skipped.
fn blocks_text(blocks: &[AnthBlock]) -> String {
    let texts: Vec<&str> = blocks
        .iter()
        .filter_map(|b| match b {
            AnthBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    texts.join("\n")
}

fn system_text(system: Option<&AnthSystem>) -> Option<String> {
    match system {
        None => None,
        Some(AnthSystem::Text(s)) => Some(s.clone()),
        Some(AnthSystem::Blocks(blocks)) => Some(blocks_text(blocks)),
    }
}

fn tool_result_text(content: Option<&AnthToolResultContent>) -> String {
    match content {
        None => String::new(),
        Some(AnthToolResultContent::Text(s)) => s.clone(),
        Some(AnthToolResultContent::Blocks(blocks)) => blocks_text(blocks),
    }
}

/// Lower the Anthropic message list to the engine's `(role, content)` pairs.
///
/// - assistant `tool_use` blocks are serialized in the same OpenAI-shaped JSON
///   the chat-completions path feeds the template, so both dialects render
///   identically through the model's tool template;
/// - `tool_result` blocks become their own `user` turns, in block order;
/// - image and unknown blocks contribute nothing here (images ride the
///   multimodal path instead).
fn anth_to_pairs(system: Option<&AnthSystem>, messages: &[AnthMessage]) -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    if let Some(sys) = system_text(system) {
        pairs.push(("system".to_string(), sys));
    }

    for msg in messages {
        match &msg.content {
            AnthContent::Text(s) => pairs.push((msg.role.clone(), s.clone())),
            AnthContent::Blocks(blocks) => {
                let mut text = String::new();
                let mut calls: Vec<Value> = Vec::new();
                for block in blocks {
                    match block {
                        AnthBlock::Text { text: t } => {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(t);
                        }
                        AnthBlock::ToolUse { id, name, input } => {
                            calls.push(json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": input.to_string(),
                                }
                            }));
                        }
                        AnthBlock::ToolResult { tool_use_id, content, is_error } => {
                            let body = tool_result_text(content.as_ref());
                            let label = if *is_error { "Tool error" } else { "Tool result" };
                            pairs.push((
                                "user".to_string(),
                                format!("[{label} for {tool_use_id}]: {body}"),
                            ));
                        }
                        AnthBlock::Image { .. } | AnthBlock::Unknown => {}
                    }
                }
                if !calls.is_empty() {
                    let calls_json = serde_json::to_string(&calls).unwrap_or_default();
                    let full = if text.is_empty() {
                        calls_json
                    } else {
                        format!("{text}\n{calls_json}")
                    };
                    pairs.push((msg.role.clone(), full));
                } else if !text.is_empty() {
                    pairs.push((msg.role.clone(), text));
                }
            }
        }
    }
    pairs
}

fn anth_tools_to_specs(tools: &[AnthTool]) -> Vec<ToolSpec> {
    tools
        .iter()
        .map(|t| ToolSpec {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.input_schema.clone(),
        })
        .collect()
}

/// Anthropic `tool_choice`: `{type: "auto" | "any" | "tool" | "none", name?}`.
fn anth_tool_choice(value: Option<&Value>) -> ToolChoice {
    let Some(obj) = value.and_then(Value::as_object) else {
        return ToolChoice::Auto;
    };
    match obj.get("type").and_then(Value::as_str) {
        Some("none") => ToolChoice::None,
        Some("any") => ToolChoice::Required,
        Some("tool") => obj
            .get("name")
            .and_then(Value::as_str)
            .map(|n| ToolChoice::Named(n.to_string()))
            .unwrap_or(ToolChoice::Auto),
        _ => ToolChoice::Auto,
    }
}

/// `tool_use.input` must be an object; the model's arguments may not parse.
/// Invalid JSON is preserved under `_raw_arguments` rather than dropped.
fn parse_tool_input(arguments: &str) -> Value {
    serde_json::from_str::<Value>(arguments)
        .ok()
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({ "_raw_arguments": arguments }))
}

fn anth_has_images(messages: &[AnthMessage]) -> bool {
    messages.iter().any(|m| match &m.content {
        AnthContent::Text(_) => false,
        AnthContent::Blocks(blocks) => {
            blocks.iter().any(|b| matches!(b, AnthBlock::Image { .. }))
        }
    })
}

// -- Stop sequences ----------------------------------------------------------

/// Earliest occurrence of any stop sequence in `text`:
/// `(byte_pos, matched_sequence)`.
fn find_stop<'a>(text: &str, stops: &'a [String]) -> Option<(usize, &'a str)> {
    stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()).map(|pos| (pos, s.as_str())))
        .min_by_key(|(pos, _)| *pos)
}

/// Incremental stop-sequence scanner for the streaming path.
///
/// Holds back up to `longest_stop - 1` bytes so a stop string split across
/// token boundaries is caught before any of it is emitted; emitted prefixes
/// are snapped down to a `char` boundary so held-back bytes never split a
/// code point.
struct StopScanner {
    stops: Vec<String>,
    tail: String,
    holdback: usize,
}

enum Scan {
    /// Emit this text (possibly empty) and keep generating.
    Continue(String),
    /// A stop sequence matched: emit this final text, record `matched`, halt.
    Stop { emit: String, matched: String },
}

impl StopScanner {
    fn new(stops: &[String]) -> Self {
        let holdback = stops.iter().map(|s| s.len()).max().unwrap_or(0).saturating_sub(1);
        StopScanner { stops: stops.to_vec(), tail: String::new(), holdback }
    }

    fn feed(&mut self, token: &str) -> Scan {
        if self.stops.is_empty() {
            return Scan::Continue(token.to_string());
        }
        self.tail.push_str(token);
        if let Some((pos, matched)) = find_stop(&self.tail, &self.stops) {
            let emit = self.tail[..pos].to_string();
            let matched = matched.to_string();
            self.tail.clear();
            return Scan::Stop { emit, matched };
        }
        let safe = self.tail.len().saturating_sub(self.holdback);
        let mut cut = safe;
        while cut > 0 && !self.tail.is_char_boundary(cut) {
            cut -= 1;
        }
        let emit: String = self.tail.drain(..cut).collect();
        Scan::Continue(emit)
    }

    /// Natural end of generation: everything still held back is clean.
    fn flush(&mut self) -> String {
        std::mem::take(&mut self.tail)
    }
}

// -- Response helpers --------------------------------------------------------

fn anth_error_body(kind: &str, msg: &str) -> Value {
    json!({ "type": "error", "error": { "type": kind, "message": msg } })
}

fn anth_error(status: StatusCode, kind: &str, msg: &str) -> axum::response::Response {
    (status, Json(anth_error_body(kind, msg))).into_response()
}

fn new_message_id() -> String {
    format!(
        "msg_{:016x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    )
}

fn sse_named(event: &str, data: &Value) -> Event {
    Event::default().event(event).data(data.to_string())
}

/// Assemble the `content` array and `stop_reason` from the finished output.
///
/// Order of precedence: a matched stop sequence beats tool parsing (the API
/// halts generation at the stop string), tool calls beat max_tokens, and
/// hitting the token cap beats `end_turn`.
fn build_content(
    output: &str,
    has_tools: bool,
    stop_hit: Option<&str>,
    completion_tokens: u32,
    max_tokens: u32,
) -> (Vec<Value>, &'static str, Option<String>) {
    let mut content: Vec<Value> = Vec::new();

    if stop_hit.is_none() && has_tools {
        let (calls, remaining) = crate::engine::tools::parse_tool_calls(output);
        if !calls.is_empty() {
            if !remaining.is_empty() {
                content.push(json!({ "type": "text", "text": remaining }));
            }
            for call in &calls {
                content.push(json!({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": parse_tool_input(&call.arguments),
                }));
            }
            return (content, "tool_use", None);
        }
        if !remaining.is_empty() {
            content.push(json!({ "type": "text", "text": remaining }));
        }
    } else if !output.is_empty() {
        content.push(json!({ "type": "text", "text": output }));
    }

    if let Some(seq) = stop_hit {
        return (content, "stop_sequence", Some(seq.to_string()));
    }
    if completion_tokens >= max_tokens {
        return (content, "max_tokens", None);
    }
    (content, "end_turn", None)
}

// -- Vision path -------------------------------------------------------------

#[cfg(feature = "vision")]
fn anth_to_multimodal(
    system: Option<&AnthSystem>,
    messages: &[AnthMessage],
) -> anyhow::Result<Vec<crate::engine::multimodal::MultimodalMessage>> {
    use crate::engine::multimodal::{ContentPart, MultimodalMessage};

    let mut out: Vec<MultimodalMessage> = Vec::new();
    if let Some(sys) = system_text(system) {
        out.push(MultimodalMessage { role: "system".into(), content: vec![ContentPart::Text(sys)] });
    }
    for msg in messages {
        let mut parts: Vec<ContentPart> = Vec::new();
        match &msg.content {
            AnthContent::Text(s) => parts.push(ContentPart::Text(s.clone())),
            AnthContent::Blocks(blocks) => {
                for block in blocks {
                    match block {
                        AnthBlock::Text { text } => parts.push(ContentPart::Text(text.clone())),
                        AnthBlock::Image { source } => {
                            let (data, media_type) = decode_image_source(source)?;
                            parts.push(ContentPart::ImageBytes { data, media_type });
                        }
                        AnthBlock::ToolResult { tool_use_id, content, is_error } => {
                            let body = tool_result_text(content.as_ref());
                            let label = if *is_error { "Tool error" } else { "Tool result" };
                            parts.push(ContentPart::Text(format!("[{label} for {tool_use_id}]: {body}")));
                        }
                        AnthBlock::ToolUse { .. } | AnthBlock::Unknown => {}
                    }
                }
            }
        }
        if !parts.is_empty() {
            out.push(MultimodalMessage { role: msg.role.clone(), content: parts });
        }
    }
    Ok(out)
}

/// Decode an Anthropic image source. `base64` sources carry the bytes
/// directly; `url` sources are only accepted as `data:` URIs (spindll never
/// fetches remote URLs on behalf of a request).
#[cfg(feature = "vision")]
fn decode_image_source(source: &AnthImageSource) -> anyhow::Result<(Vec<u8>, Option<String>)> {
    use base64::Engine as _;
    match source.kind.as_str() {
        "base64" => {
            let b64 = source.data.as_deref()
                .ok_or_else(|| anyhow::anyhow!("image source.data missing"))?;
            let data = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|e| anyhow::anyhow!("image base64 decode failed: {e}"))?;
            crate::engine::multimodal::check_image_len(data.len())?;
            let media_type = source.media_type.clone();
            if let Some(mt) = &media_type
                && !crate::http::ALLOWED_IMAGE_MEDIA.contains(&mt.as_str())
            {
                anyhow::bail!("unsupported image media_type: {mt}");
            }
            Ok((data, media_type))
        }
        "url" => {
            let url = source.url.as_deref()
                .ok_or_else(|| anyhow::anyhow!("image source.url missing"))?;
            crate::http::decode_data_uri(url)
        }
        other => anyhow::bail!("unsupported image source type: {other}"),
    }
}

// -- Handler -----------------------------------------------------------------

pub(crate) async fn anthropic_messages(
    State(state): State<AppState>,
    Json(req): Json<AnthMessagesRequest>,
) -> axum::response::Response {
    let Some(max_tokens) = req.max_tokens else {
        return anth_error(StatusCode::BAD_REQUEST, "invalid_request_error", "max_tokens: field required");
    };

    #[cfg(not(feature = "vision"))]
    if anth_has_images(&req.messages) {
        return anth_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "image blocks require the vision feature (rebuild with --features vision)",
        );
    }

    let mgr = state.manager.clone();
    let store = state.store.clone();
    let tool_choice = anth_tool_choice(req.tool_choice.as_ref());
    let has_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty())
        && tool_choice != ToolChoice::None;
    let tool_specs: Vec<ToolSpec> = if has_tools {
        anth_tools_to_specs(req.tools.as_deref().unwrap_or_default())
    } else {
        Vec::new()
    };

    let params = GenerateParams {
        max_tokens,
        temperature: req.temperature.unwrap_or(0.8),
        top_p: req.top_p.unwrap_or(0.95),
        top_k: req.top_k.map(|k| k as i32).unwrap_or(40),
        seed: 42,
        prefill_only: false,
        ..Default::default()
    };

    if req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);

        tokio::task::spawn_blocking(move || {
            let key = match ensure_loaded(&mgr, &store, &req.model) {
                Ok(k) => k,
                Err(e) => {
                    let _ = tx.blocking_send(Ok(sse_named("error", &anth_error_body("api_error", &e.to_string()))));
                    return;
                }
            };

            let msg_id = new_message_id();
            let _ = tx.blocking_send(Ok(sse_named("message_start", &json!({
                "type": "message_start",
                "message": {
                    "id": &msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": &req.model,
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": { "input_tokens": 0, "output_tokens": 0 }
                }
            }))));

            // Tools buffer the full output (call parsing needs it); pure text
            // streams through the stop scanner token by token.
            if has_tools {
                let mut output = String::new();
                let result = generate(&mgr, &key, &req, &tool_specs, &tool_choice, &params, &mut |t| {
                    output.push_str(t);
                    true
                });
                match result {
                    Ok(stats) => {
                        let stop_hit = find_stop(&output, &req.stop_sequences)
                            .map(|(pos, seq)| { output.truncate(pos); seq.to_string() });
                        let (content, stop_reason, stop_sequence) = build_content(
                            &output, true, stop_hit.as_deref(), stats.completion_tokens, max_tokens);
                        for (i, block) in content.iter().enumerate() {
                            stream_block(&tx, i, block);
                        }
                        finish_stream(&tx, stop_reason, stop_sequence.as_deref(), stats.prompt_tokens, stats.completion_tokens);
                    }
                    Err(e) => {
                        let _ = tx.blocking_send(Ok(sse_named("error", &anth_error_body("api_error", &e.to_string()))));
                    }
                }
            } else {
                let _ = tx.blocking_send(Ok(sse_named("content_block_start", &json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "text", "text": "" }
                }))));

                let mut scanner = StopScanner::new(&req.stop_sequences);
                let mut stop_hit: Option<String> = None;
                let result = generate(&mgr, &key, &req, &tool_specs, &tool_choice, &params, &mut |token| {
                    if stop_hit.is_some() {
                        return false;
                    }
                    match scanner.feed(token) {
                        Scan::Continue(text) => {
                            if text.is_empty() {
                                true
                            } else {
                                tx.blocking_send(Ok(sse_named("content_block_delta", &json!({
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": { "type": "text_delta", "text": text }
                                })))).is_ok()
                            }
                        }
                        Scan::Stop { emit, matched } => {
                            if !emit.is_empty() {
                                let _ = tx.blocking_send(Ok(sse_named("content_block_delta", &json!({
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": { "type": "text_delta", "text": emit }
                                }))));
                            }
                            stop_hit = Some(matched);
                            false
                        }
                    }
                });

                match result {
                    Ok(stats) => {
                        if stop_hit.is_none() {
                            let rest = scanner.flush();
                            if !rest.is_empty() {
                                let _ = tx.blocking_send(Ok(sse_named("content_block_delta", &json!({
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": { "type": "text_delta", "text": rest }
                                }))));
                            }
                        }
                        let _ = tx.blocking_send(Ok(sse_named("content_block_stop", &json!({
                            "type": "content_block_stop", "index": 0
                        }))));
                        let stop_reason = if stop_hit.is_some() {
                            "stop_sequence"
                        } else if stats.completion_tokens >= max_tokens {
                            "max_tokens"
                        } else {
                            "end_turn"
                        };
                        finish_stream(&tx, stop_reason, stop_hit.as_deref(), stats.prompt_tokens, stats.completion_tokens);
                    }
                    Err(e) => {
                        let _ = tx.blocking_send(Ok(sse_named("error", &anth_error_body("api_error", &e.to_string()))));
                    }
                }
            }
            drop(tx);
        });

        Sse::new(ReceiverStream::new(rx)).into_response()
    } else {
        let model_id = req.model.clone();
        let stop_sequences = req.stop_sequences.clone();
        let result = tokio::task::spawn_blocking(move || {
            let key = ensure_loaded(&mgr, &store, &req.model)?;
            let mut output = String::new();
            let stats = generate(&mgr, &key, &req, &tool_specs, &tool_choice, &params, &mut |t| {
                output.push_str(t);
                true
            })?;
            Ok::<_, anyhow::Error>((output, stats))
        })
        .await;

        match result {
            Ok(Ok((mut output, stats))) => {
                let stop_hit = find_stop(&output, &stop_sequences)
                    .map(|(pos, seq)| { output.truncate(pos); seq.to_string() });
                let (content, stop_reason, stop_sequence) = build_content(
                    &output, has_tools, stop_hit.as_deref(), stats.completion_tokens, max_tokens);
                Json(json!({
                    "id": new_message_id(),
                    "type": "message",
                    "role": "assistant",
                    "model": model_id,
                    "content": content,
                    "stop_reason": stop_reason,
                    "stop_sequence": stop_sequence,
                    "usage": {
                        "input_tokens": stats.prompt_tokens,
                        "output_tokens": stats.completion_tokens,
                    }
                }))
                .into_response()
            }
            Ok(Err(e)) => anth_error(StatusCode::INTERNAL_SERVER_ERROR, "api_error", &e.to_string()),
            Err(e) => anth_error(StatusCode::INTERNAL_SERVER_ERROR, "api_error", &e.to_string()),
        }
    }
}

/// Run generation on the vision path when image blocks are present, else the
/// text chat path — mirrors the chat-completions handler's split.
/// `key` is the canonical registry key the model is resident under — not
/// `req.model`, which may be an alias.
fn generate(
    mgr: &crate::engine::manager::ModelManager,
    key: &str,
    req: &AnthMessagesRequest,
    tool_specs: &[ToolSpec],
    tool_choice: &ToolChoice,
    params: &GenerateParams,
    on_token: &mut dyn FnMut(&str) -> bool,
) -> anyhow::Result<crate::engine::GenerateResult> {
    #[cfg(feature = "vision")]
    if anth_has_images(&req.messages) {
        let mut mm = anth_to_multimodal(req.system.as_ref(), &req.messages)?;
        if let Some(preamble) = crate::engine::tools::tools_to_prompt(tool_specs, tool_choice) {
            crate::engine::multimodal::inject_system_text(&mut mm, &preamble);
        }
        return mgr.generate_chat_multimodal(key, &mm, params, on_token);
    }
    let pairs = anth_to_pairs(req.system.as_ref(), &req.messages);
    mgr.generate_chat(key, &pairs, tool_specs, tool_choice, params, None, on_token)
}

/// Emit one finished content block through the streaming grammar. Text blocks
/// send a single text_delta; tool_use blocks send their arguments as one
/// input_json_delta (calls are parsed from the completed output, not
/// token-by-token).
fn stream_block(
    tx: &tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
    index: usize,
    block: &Value,
) {
    match block["type"].as_str() {
        Some("text") => {
            let _ = tx.blocking_send(Ok(sse_named("content_block_start", &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": { "type": "text", "text": "" }
            }))));
            let _ = tx.blocking_send(Ok(sse_named("content_block_delta", &json!({
                "type": "content_block_delta",
                "index": index,
                "delta": { "type": "text_delta", "text": block["text"] }
            }))));
        }
        Some("tool_use") => {
            let _ = tx.blocking_send(Ok(sse_named("content_block_start", &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": block["id"],
                    "name": block["name"],
                    "input": {}
                }
            }))));
            let partial = block["input"].to_string();
            let _ = tx.blocking_send(Ok(sse_named("content_block_delta", &json!({
                "type": "content_block_delta",
                "index": index,
                "delta": { "type": "input_json_delta", "partial_json": partial }
            }))));
        }
        _ => return,
    }
    let _ = tx.blocking_send(Ok(sse_named("content_block_stop", &json!({
        "type": "content_block_stop", "index": index
    }))));
}

fn finish_stream(
    tx: &tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
    stop_reason: &str,
    stop_sequence: Option<&str>,
    input_tokens: u32,
    output_tokens: u32,
) {
    let _ = tx.blocking_send(Ok(sse_named("message_delta", &json!({
        "type": "message_delta",
        "delta": { "stop_reason": stop_reason, "stop_sequence": stop_sequence },
        "usage": { "input_tokens": input_tokens, "output_tokens": output_tokens }
    }))));
    let _ = tx.blocking_send(Ok(sse_named("message_stop", &json!({ "type": "message_stop" }))));
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_req(body: serde_json::Value) -> AnthMessagesRequest {
        serde_json::from_value(body).unwrap()
    }

    #[test]
    fn string_and_block_content_lower_to_pairs() {
        let req = parse_req(json!({
            "model": "m", "max_tokens": 10,
            "system": "be brief",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
                {"role": "user", "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"}
                ]}
            ]
        }));
        let pairs = anth_to_pairs(req.system.as_ref(), &req.messages);
        assert_eq!(pairs, vec![
            ("system".into(), "be brief".into()),
            ("user".into(), "hi".into()),
            ("assistant".into(), "hello".into()),
            ("user".into(), "part one\npart two".into()),
        ]);
    }

    #[test]
    fn system_blocks_and_unknown_block_types_are_tolerated() {
        let req = parse_req(json!({
            "model": "m", "max_tokens": 10,
            "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "...", "signature": "x"},
                    {"type": "text", "text": "answer"}
                ]}
            ]
        }));
        let pairs = anth_to_pairs(req.system.as_ref(), &req.messages);
        assert_eq!(pairs, vec![
            ("system".into(), "a\nb".into()),
            ("assistant".into(), "answer".into()),
        ]);
    }

    #[test]
    fn tool_use_and_tool_result_round_trip_through_pairs() {
        let req = parse_req(json!({
            "model": "m", "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F"}
                ]}
            ]
        }));
        let pairs = anth_to_pairs(None, &req.messages);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "assistant");
        assert!(pairs[0].1.starts_with("checking\n"));
        assert!(pairs[0].1.contains("\"get_weather\""));
        assert!(pairs[0].1.contains("{\\\"city\\\":\\\"Paris\\\"}"));
        assert_eq!(pairs[1], ("user".into(), "[Tool result for toolu_1]: 72F".into()));
    }

    #[test]
    fn tool_result_block_content_and_error_flag() {
        let req = parse_req(json!({
            "model": "m", "max_tokens": 10,
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t2", "is_error": true,
                     "content": [{"type": "text", "text": "boom"}]}
                ]}
            ]
        }));
        let pairs = anth_to_pairs(None, &req.messages);
        assert_eq!(pairs, vec![("user".into(), "[Tool error for t2]: boom".into())]);
    }

    #[test]
    fn tool_choice_mapping() {
        assert_eq!(anth_tool_choice(None), ToolChoice::Auto);
        assert_eq!(anth_tool_choice(Some(&json!({"type": "auto"}))), ToolChoice::Auto);
        assert_eq!(anth_tool_choice(Some(&json!({"type": "any"}))), ToolChoice::Required);
        assert_eq!(anth_tool_choice(Some(&json!({"type": "none"}))), ToolChoice::None);
        assert_eq!(
            anth_tool_choice(Some(&json!({"type": "tool", "name": "f"}))),
            ToolChoice::Named("f".into())
        );
    }

    #[test]
    fn tool_input_parses_or_preserves_raw() {
        assert_eq!(parse_tool_input("{\"a\":1}"), json!({"a": 1}));
        assert_eq!(parse_tool_input("not json"), json!({"_raw_arguments": "not json"}));
        assert_eq!(parse_tool_input("[1,2]"), json!({"_raw_arguments": "[1,2]"}));
    }

    #[test]
    fn find_stop_picks_earliest_match() {
        let stops = vec!["END".to_string(), "STOP".to_string()];
        assert_eq!(find_stop("abSTOPcdEND", &stops), Some((2, "STOP")));
        assert_eq!(find_stop("no match", &stops), None);
    }

    #[test]
    fn stop_scanner_catches_sequence_split_across_tokens() {
        let mut s = StopScanner::new(&["STOP".to_string()]);
        let Scan::Continue(a) = s.feed("hello ST") else { panic!("early stop") };
        assert_eq!(a, "hello");
        let Scan::Stop { emit, matched } = s.feed("OP tail") else { panic!("missed stop") };
        assert_eq!(emit, " ");
        assert_eq!(matched, "STOP");
    }

    #[test]
    fn stop_scanner_flush_returns_held_tail() {
        let mut s = StopScanner::new(&["XYZ".to_string()]);
        let Scan::Continue(a) = s.feed("abc") else { panic!() };
        assert_eq!(a, "a");
        assert_eq!(s.flush(), "bc");
    }

    #[test]
    fn stop_scanner_respects_char_boundaries() {
        let mut s = StopScanner::new(&["終わり".to_string()]);
        // Holdback lands mid-codepoint; the emitted prefix must stay valid UTF-8.
        let Scan::Continue(a) = s.feed("日本語のテキスト") else { panic!() };
        let held = s.flush();
        assert_eq!(format!("{a}{held}"), "日本語のテキスト");
    }

    #[test]
    fn build_content_maps_stop_reasons() {
        let (c, reason, seq) = build_content("hello", false, None, 2, 10);
        assert_eq!(c, vec![json!({"type": "text", "text": "hello"})]);
        assert_eq!((reason, seq), ("end_turn", None));

        let (_, reason, _) = build_content("hello", false, None, 10, 10);
        assert_eq!(reason, "max_tokens");

        let (c, reason, seq) = build_content("hel", false, Some("STOP"), 4, 10);
        assert_eq!(c, vec![json!({"type": "text", "text": "hel"})]);
        assert_eq!(reason, "stop_sequence");
        assert_eq!(seq.as_deref(), Some("STOP"));
    }

    // -- Endpoint tests against the fake backend ----------------------------

    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    async fn post_json(uri: &str, body: serde_json::Value) -> (u16, String) {
        let dir = tempfile::tempdir().unwrap();
        let (store, mgr) = crate::http::tests::setup_store_and_manager(dir.path());
        let app = crate::http::router(mgr, store);
        let req = axum::http::Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status().as_u16();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn messages_non_streaming_returns_anthropic_shape() {
        let (status, text) = post_json("/v1/messages", json!({
            "model": "test-org/test-model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}]
        })).await;
        assert_eq!(status, 200);
        let v: Value = serde_json::from_str(&text).unwrap();
        assert!(v["id"].as_str().unwrap().starts_with("msg_"));
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["content"], json!([{"type": "text", "text": "Hello world"}]));
        assert_eq!(v["stop_reason"], "end_turn");
        assert_eq!(v["usage"], json!({"input_tokens": 5, "output_tokens": 2}));
    }

    #[tokio::test]
    async fn messages_hitting_token_cap_reports_max_tokens() {
        let (status, text) = post_json("/v1/messages", json!({
            "model": "test-org/test-model",
            "max_tokens": 2,
            "messages": [{"role": "user", "content": "hi"}]
        })).await;
        assert_eq!(status, 200);
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["stop_reason"], "max_tokens");
    }

    #[tokio::test]
    async fn messages_stop_sequence_truncates_and_reports() {
        let (status, text) = post_json("/v1/messages", json!({
            "model": "test-org/test-model",
            "max_tokens": 10,
            "stop_sequences": [" world"],
            "messages": [{"role": "user", "content": "hi"}]
        })).await;
        assert_eq!(status, 200);
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["content"], json!([{"type": "text", "text": "Hello"}]));
        assert_eq!(v["stop_reason"], "stop_sequence");
        assert_eq!(v["stop_sequence"], " world");
    }

    #[tokio::test]
    async fn messages_streaming_emits_event_grammar_in_order() {
        let (status, text) = post_json("/v1/messages", json!({
            "model": "test-org/test-model",
            "max_tokens": 10,
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        })).await;
        assert_eq!(status, 200);
        let order = [
            "event: message_start",
            "event: content_block_start",
            "event: content_block_delta",
            "event: content_block_stop",
            "event: message_delta",
            "event: message_stop",
        ];
        let mut last = 0;
        for marker in order {
            let pos = text[last..].find(marker)
                .unwrap_or_else(|| panic!("missing or out of order: {marker}\n{text}"));
            last += pos;
        }
        assert!(text.contains("\"text_delta\""));
        assert!(text.contains("\"stop_reason\":\"end_turn\""));
        assert!(!text.contains("[DONE]"), "Anthropic SSE has no [DONE] sentinel");
    }

    #[tokio::test]
    async fn messages_missing_max_tokens_is_anthropic_error() {
        let (status, text) = post_json("/v1/messages", json!({
            "model": "test-org/test-model",
            "messages": [{"role": "user", "content": "hi"}]
        })).await;
        assert_eq!(status, 400);
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    #[cfg(not(feature = "vision"))]
    #[tokio::test]
    async fn messages_rejects_images_without_vision() {
        let (status, _) = post_json("/v1/messages", json!({
            "model": "test-org/test-model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
            ]}]
        })).await;
        assert_eq!(status, 400);
    }
}
