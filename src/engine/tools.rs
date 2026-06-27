//! Backend-agnostic tool-calling support.
//!
//! Spindll owns tool-call *emission* (prompt rendering + parsing); the
//! *execution* loop belongs to the consumer. This module is the shared
//! vocabulary so the HTTP (`/v1/chat/completions`) and gRPC (`Chat`) surfaces
//! speak the same types and behave identically.
//!
//! Tool calls are driven by **prompt injection**: the specs are rendered into
//! the system prompt and the model is asked to emit `<tool_call>{…}</tool_call>`
//! JSON, which [`parse_tool_calls`] extracts. Model-native, grammar-constrained
//! decoding is **not** wired today: `llama-cpp-2` exposed an OpenAI-compatible
//! template + GBNF-grammar helper through 0.1.146 but **removed it in 0.1.150**,
//! so there is currently no upstream way to derive a grammar from the tool
//! schemas. Revisit if a later llama-cpp-2 restores that API.
//!
//! - [`ToolSpec`] / [`ToolChoice`] / [`ToolCall`] — the neutral request/response
//!   types each API surface converts into.
//! - [`tools_to_prompt`] — render specs + `tool_choice` into the system-prompt
//!   preamble used for injection (the one place either surface describes tools).
//! - [`tools_to_oai_json`] — render specs into the OpenAI `tools` JSON array
//!   (embedded in that preamble; also handy for logging).
//! - [`parse_tool_calls`] — extract calls from raw model output. Aware of the
//!   common model wrappers (Hermes `<tool_call>`, Llama-3.1 `<|python_tag|>`,
//!   Mistral `[TOOL_CALLS]`) with a balanced-JSON scan as the fallback. Source
//!   of truth for extraction on every backend.

use serde_json::Value;

/// A tool the model may call. Mirrors one entry of OpenAI's `tools` array,
/// flattened to the `function` fields (the only `type` we support today).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: Option<String>,
    /// JSON Schema for the function arguments (OpenAI `parameters`).
    pub parameters: Option<Value>,
}

/// Caller control over whether/which tool is invoked. Parsed from OpenAI's
/// `tool_choice` field, which is either a string or a `{type, function}` object.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ToolChoice {
    /// No tool call — plain completion. OpenAI `"none"`.
    None,
    /// Model decides. OpenAI `"auto"` (and the default when tools are present).
    #[default]
    Auto,
    /// Model must call some tool. OpenAI `"required"`.
    Required,
    /// Model must call this specific function.
    Named(String),
}

impl ToolChoice {
    /// Parse OpenAI's `tool_choice` value. Absent / unrecognized → [`Auto`]
    /// when tools exist (the caller decides what "absent" means; this only maps
    /// the wire value). `{"type":"function","function":{"name":"f"}}` → `Named`.
    ///
    /// [`Auto`]: ToolChoice::Auto
    pub fn from_oai(value: Option<&Value>) -> Self {
        match value {
            None => ToolChoice::Auto,
            Some(Value::String(s)) => match s.as_str() {
                "none" => ToolChoice::None,
                "required" => ToolChoice::Required,
                "auto" => ToolChoice::Auto,
                // A bare function name isn't standard, but be lenient.
                other => ToolChoice::Named(other.to_string()),
            },
            Some(Value::Object(obj)) => obj
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .map(|n| ToolChoice::Named(n.to_string()))
                .unwrap_or(ToolChoice::Auto),
            _ => ToolChoice::Auto,
        }
    }
}

/// A single tool call emitted by the model. `arguments` is always a JSON
/// *string* (OpenAI's shape), even when the model produced a JSON object.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Generate an OpenAI-style call id (`call_<hex>`). Unique even when several
/// calls are minted in the same nanosecond — a coarse clock (e.g. Windows) can
/// repeat, so a process-wide sequence is mixed in.
pub fn new_call_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("call_{nanos:016x}{seq:08x}")
}

/// Render specs into the OpenAI-compatible `tools` JSON array string. Embedded
/// in the prompt by [`tools_to_prompt`] (and handy for logging/clients).
/// Returns `None` when there are no tools.
pub fn tools_to_oai_json(tools: &[ToolSpec]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }
    let arr: Vec<Value> = tools
        .iter()
        .map(|t| {
            let mut func = serde_json::Map::new();
            func.insert("name".into(), Value::String(t.name.clone()));
            if let Some(desc) = &t.description {
                func.insert("description".into(), Value::String(desc.clone()));
            }
            // Default to an empty-object schema; some templates require the key.
            func.insert(
                "parameters".into(),
                t.parameters.clone().unwrap_or_else(|| {
                    serde_json::json!({ "type": "object", "properties": {} })
                }),
            );
            serde_json::json!({ "type": "function", "function": Value::Object(func) })
        })
        .collect();
    serde_json::to_string(&arr).ok()
}

/// Render a system-prompt preamble describing the available tools and how to
/// call them, honoring `tool_choice`. This is the prompt-injection path used by
/// every backend (see the module docs — model-native grammar isn't available on
/// llama-cpp-2 0.1.150). Returns `None` when there are no tools or `tool_choice`
/// is [`ToolChoice::None`] — the caller then injects nothing and treats the turn
/// as a plain completion.
pub fn tools_to_prompt(tools: &[ToolSpec], choice: &ToolChoice) -> Option<String> {
    if matches!(choice, ToolChoice::None) {
        return None;
    }
    let tools_json = tools_to_oai_json(tools)?;
    let directive = match choice {
        ToolChoice::Required => {
            "You MUST call one of the tools above before answering.".to_string()
        }
        ToolChoice::Named(name) => format!("You MUST call the tool \"{name}\"."),
        _ => "Call a tool only when it helps answer the request; otherwise answer normally."
            .to_string(),
    };
    Some(format!(
        "You have access to the following tools (OpenAI JSON):\n{tools_json}\n\
         {directive}\n\
         To call a tool, emit a block exactly like \
         <tool_call>{{\"name\": <tool name>, \"arguments\": <json object>}}</tool_call>. \
         Multiple <tool_call> blocks are allowed."
    ))
}

/// Extract tool calls from raw model output.
///
/// Returns the parsed calls plus any non-call text that surrounded them
/// (trimmed). When no call is found, `calls` is empty and the full text is
/// returned as the remaining content.
pub fn parse_tool_calls(output: &str) -> (Vec<ToolCall>, String) {
    let trimmed = output.trim();

    // 1. Hermes / Qwen: one or more `<tool_call> ... </tool_call>` blocks.
    if trimmed.contains("<tool_call>")
        && let Some(calls) = parse_wrapped(trimmed, "<tool_call>", "</tool_call>")
        && !calls.is_empty()
    {
        return (calls, strip_wrapped(trimmed, "<tool_call>", "</tool_call>"));
    }

    // 2. Mistral `[TOOL_CALLS]` / Llama-3.1 `<|python_tag|>` prefixes: the
    //    remainder is a JSON object or array of calls. The prefix is an explicit
    //    tool-call signal, so the scan below treats a missing `arguments`
    //    leniently. Without a prefix the scan stays strict (a stray `{"name": …}`
    //    in prose must not be misread as a call).
    let stripped = trimmed
        .strip_prefix("[TOOL_CALLS]")
        .or_else(|| trimmed.strip_prefix("<|python_tag|>"))
        .map(str::trim);
    let require_args = stripped.is_none();
    let body = stripped.unwrap_or(trimmed);

    // 2a. After an explicit prefix the body is commonly a JSON *array* of calls
    //     (`[{…}, …]`, the documented Mistral shape). Parse it as a whole so the
    //     array's brackets and commas don't leak into the remaining content.
    if stripped.is_some()
        && let Some(result) = parse_json_array_calls(body)
    {
        return result;
    }

    // 3. Balanced-JSON scan: pull every top-level `{...}` that looks like a
    //    call. Handles a bare object, several objects, or text + object.
    scan_json_calls(body, require_args)
}

/// Parse a JSON array of call objects (`[{…}, …]`) — the Mistral / Llama-3.1
/// prefix form. Returns `Some((calls, trailing_text))` when `text` begins with a
/// JSON array that yields at least one call (any text after the array becomes
/// content); `None` otherwise, so the caller falls back to the object scan. The
/// prefix is an explicit tool-call signal, so a missing `arguments` defaults to
/// `{}` (lenient), matching the wrapped/prefix object paths.
fn parse_json_array_calls(text: &str) -> Option<(Vec<ToolCall>, String)> {
    let mut stream = serde_json::Deserializer::from_str(text).into_iter::<Value>();
    let Value::Array(items) = stream.next()?.ok()? else {
        return None;
    };
    let calls: Vec<ToolCall> = items
        .iter()
        .filter_map(|item| call_from_value(item, false))
        .collect();
    if calls.is_empty() {
        return None;
    }
    let trailing = text[stream.byte_offset()..].trim().to_string();
    Some((calls, trailing))
}

/// Parse the JSON inside each `<open>...</close>` block.
fn parse_wrapped(text: &str, open: &str, close: &str) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(open) {
        let after = &rest[start + open.len()..];
        let Some(end) = after.find(close) else {
            // Unclosed block (e.g. the output was truncated mid-call): stop and
            // keep the calls already parsed rather than discarding all of them.
            break;
        };
        let inner = after[..end].trim();
        // Explicit wrapper ⇒ lenient: a missing `arguments` defaults to `{}`.
        if let Some(call) = call_from_json_str(inner, false) {
            calls.push(call);
        }
        rest = &after[end + close.len()..];
    }
    Some(calls)
}

/// Everything outside the `<open>...</close>` blocks, trimmed.
fn strip_wrapped(text: &str, open: &str, close: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(open) {
        out.push_str(&rest[..start]);
        let after = &rest[start + open.len()..];
        match after.find(close) {
            Some(end) => rest = &after[end + close.len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

/// Scan for balanced top-level JSON objects that parse as `{name, arguments}`.
/// `require_args` is forwarded to [`call_from_json_str`]: `true` for the bare
/// fallback (an object must carry `arguments` to count as a call), `false` after
/// an explicit `[TOOL_CALLS]` / `<|python_tag|>` prefix.
fn scan_json_calls(text: &str, require_args: bool) -> (Vec<ToolCall>, String) {
    let mut calls = Vec::new();
    let mut remaining = String::new();
    let mut search_from = 0;

    while let Some(rel) = text[search_from..].find('{') {
        let abs = search_from + rel;
        let Some(end) = find_json_end(text, abs) else {
            // No balanced object starts here, so the rest of the text has
            // unbalanced braces. Keep it as plain text and stop: re-scanning
            // every following `{` of an unterminated tail is O(n²) and a wall
            // of `{` from a degenerate decode would stall the request thread.
            break;
        };
        remaining.push_str(&text[search_from..abs]);
        match call_from_json_str(&text[abs..end], require_args) {
            Some(call) => calls.push(call),
            // A complete JSON object that isn't a call: keep it verbatim and
            // resume *past* it. Re-entering would let a nested `{"name",
            // "arguments"}` be misread as a call and leave broken JSON behind.
            None => remaining.push_str(&text[abs..end]),
        }
        search_from = end;
    }
    remaining.push_str(&text[search_from..]);

    (calls, remaining.trim().to_string())
}

/// Build a [`ToolCall`] from a JSON string holding `{"name", "arguments"}`.
/// `arguments` may be an object (serialized to a string) or already a string.
///
/// `require_args` controls the missing-`arguments` case. Explicit wrapper /
/// prefix contexts pass `false` so a no-argument call such as `{"name": "now"}`
/// still parses (arguments default to `{}`). The bare-JSON fallback scan passes
/// `true` so an arbitrary `{"name": …}` object in prose isn't misread as a call.
fn call_from_json_str(candidate: &str, require_args: bool) -> Option<ToolCall> {
    let parsed: Value = serde_json::from_str(candidate).ok()?;
    call_from_value(&parsed, require_args)
}

/// Build a [`ToolCall`] from an already-parsed JSON value. Shares the
/// `require_args` contract documented on [`call_from_json_str`]; used directly by
/// the JSON-array path so each element isn't re-serialized and re-parsed.
fn call_from_value(parsed: &Value, require_args: bool) -> Option<ToolCall> {
    let obj = parsed.as_object()?;
    let name = obj.get("name")?.as_str()?;
    let arguments = match obj.get("arguments") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
        None if require_args => return None,
        None => "{}".to_string(),
    };
    Some(ToolCall {
        id: new_call_id(),
        name: name.to_string(),
        arguments,
    })
}

/// Index just past the `}` that closes the object starting at `start`, honoring
/// strings and escapes. `None` if `start` isn't `{` or the object is unbalanced.
fn find_json_end(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(start)? != &b'{' {
        return None;
    }
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &ch) in bytes.iter().enumerate().skip(start) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_choice_from_oai_strings() {
        assert_eq!(ToolChoice::from_oai(None), ToolChoice::Auto);
        assert_eq!(ToolChoice::from_oai(Some(&json!("auto"))), ToolChoice::Auto);
        assert_eq!(ToolChoice::from_oai(Some(&json!("none"))), ToolChoice::None);
        assert_eq!(
            ToolChoice::from_oai(Some(&json!("required"))),
            ToolChoice::Required
        );
    }

    #[test]
    fn tool_choice_named_function() {
        let v = json!({"type": "function", "function": {"name": "get_weather"}});
        assert_eq!(
            ToolChoice::from_oai(Some(&v)),
            ToolChoice::Named("get_weather".to_string())
        );
    }

    #[test]
    fn tool_choice_unknown_object_is_auto() {
        assert_eq!(ToolChoice::from_oai(Some(&json!({}))), ToolChoice::Auto);
    }

    #[test]
    fn tools_to_oai_json_shape() {
        let tools = vec![ToolSpec {
            name: "get_weather".into(),
            description: Some("Get weather".into()),
            parameters: Some(json!({"type": "object", "properties": {"city": {"type": "string"}}})),
        }];
        let s = tools_to_oai_json(&tools).expect("some");
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v[0]["type"], "function");
        assert_eq!(v[0]["function"]["name"], "get_weather");
        assert_eq!(v[0]["function"]["parameters"]["properties"]["city"]["type"], "string");
    }

    #[test]
    fn tools_to_oai_json_defaults_parameters() {
        let tools = vec![ToolSpec { name: "ping".into(), description: None, parameters: None }];
        let s = tools_to_oai_json(&tools).unwrap();
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn tools_to_oai_json_empty_is_none() {
        assert_eq!(tools_to_oai_json(&[]), None);
    }

    #[test]
    fn tools_to_prompt_none_and_empty_are_none() {
        let tools = vec![ToolSpec { name: "f".into(), description: None, parameters: None }];
        assert_eq!(tools_to_prompt(&tools, &ToolChoice::None), None);
        assert_eq!(tools_to_prompt(&[], &ToolChoice::Auto), None);
    }

    #[test]
    fn tools_to_prompt_named_and_required_demand_a_call() {
        let tools = vec![ToolSpec { name: "get_weather".into(), description: None, parameters: None }];
        let named = tools_to_prompt(&tools, &ToolChoice::Named("get_weather".into())).unwrap();
        assert!(named.contains("get_weather") && named.contains("<tool_call>"));
        let required = tools_to_prompt(&tools, &ToolChoice::Required).unwrap();
        assert!(required.contains("MUST call"));
    }

    #[test]
    fn parse_bare_object() {
        let (calls, rest) = parse_tool_calls(r#"{"name": "f", "arguments": {"x": 1}}"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].arguments, r#"{"x":1}"#);
        assert!(rest.is_empty());
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn parse_string_arguments_passthrough() {
        let (calls, _) = parse_tool_calls(r#"{"name": "f", "arguments": "{\"x\":1}"}"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, r#"{"x":1}"#);
    }

    #[test]
    fn parse_multiple_objects() {
        let out = r#"{"name": "a", "arguments": {}} {"name": "b", "arguments": {"k": "v"}}"#;
        let (calls, _) = parse_tool_calls(out);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn parse_hermes_wrapped() {
        let out = "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"rust\"}}\n</tool_call>";
        let (calls, rest) = parse_tool_calls(out);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert!(rest.is_empty());
    }

    #[test]
    fn parse_hermes_multiple_blocks() {
        let out = "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\
                   <tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>";
        let (calls, _) = parse_tool_calls(out);
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn parse_wrapped_keeps_valid_block_when_later_block_is_truncated() {
        // Output cut off mid-second-call must not discard the first, complete one.
        let out = "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"b\"";
        let (calls, _rest) = parse_tool_calls(out);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "a");
    }

    #[test]
    fn parse_python_tag_prefix() {
        let out = r#"<|python_tag|>{"name": "f", "arguments": {"a": true}}"#;
        let (calls, _) = parse_tool_calls(out);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
    }

    #[test]
    fn parse_mistral_prefix() {
        let out = r#"[TOOL_CALLS]{"name": "f", "arguments": {}}"#;
        let (calls, _) = parse_tool_calls(out);
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn parse_text_then_call_keeps_text() {
        let (calls, rest) = parse_tool_calls(r#"Let me check. {"name": "f", "arguments": {}}"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(rest, "Let me check.");
    }

    #[test]
    fn parse_plain_prose_no_calls() {
        let (calls, rest) = parse_tool_calls("Just a normal answer with no tools.");
        assert!(calls.is_empty());
        assert_eq!(rest, "Just a normal answer with no tools.");
    }

    #[test]
    fn parse_object_without_name_is_not_a_call() {
        let (calls, rest) = parse_tool_calls(r#"{"foo": "bar"}"#);
        assert!(calls.is_empty());
        assert_eq!(rest, r#"{"foo": "bar"}"#);
    }

    #[test]
    fn parse_wrapped_call_without_arguments_defaults_to_empty_object() {
        // A no-argument tool inside an explicit wrapper is still a call.
        let (calls, _) = parse_tool_calls("<tool_call>{\"name\": \"get_time\"}</tool_call>");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_time");
        assert_eq!(calls[0].arguments, "{}");
    }

    #[test]
    fn parse_prefix_call_without_arguments_defaults_to_empty_object() {
        let (calls, _) = parse_tool_calls(r#"[TOOL_CALLS]{"name": "ping"}"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, "{}");
    }

    #[test]
    fn parse_bare_name_object_without_arguments_is_not_a_call() {
        // Without an explicit wrapper/prefix, a `{"name": …}` object lacking
        // `arguments` is ordinary prose (e.g. a record the model is discussing),
        // not a tool call — it must be left untouched.
        let input = r#"{"name": "Alice", "city": "Paris"}"#;
        let (calls, rest) = parse_tool_calls(input);
        assert!(calls.is_empty());
        assert_eq!(rest, input);
    }

    #[test]
    fn parse_skips_noncall_object_before_call() {
        // A non-call JSON object must not abort the scan and swallow a later call.
        let (calls, _) =
            parse_tool_calls(r#"{"foo": "bar"} {"name": "f", "arguments": {"x": 1}}"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].arguments, r#"{"x":1}"#);
    }

    #[test]
    fn parse_mistral_array_form_has_no_bracket_junk() {
        // Mistral's documented shape is a JSON array after the prefix; the array
        // delimiters must not leak into the remaining content.
        let out = r#"[TOOL_CALLS][{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"k": "v"}}]"#;
        let (calls, rest) = parse_tool_calls(out);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].arguments, r#"{"k":"v"}"#);
        assert!(rest.is_empty(), "array punctuation leaked as content: {rest:?}");
    }

    #[test]
    fn parse_python_tag_array_form() {
        let out = r#"<|python_tag|>[{"name": "f", "arguments": {"a": 1}}]"#;
        let (calls, rest) = parse_tool_calls(out);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
        assert!(rest.is_empty());
    }

    #[test]
    fn parse_array_form_keeps_trailing_text() {
        let out = r#"[TOOL_CALLS][{"name": "f", "arguments": {}}] all done"#;
        let (calls, rest) = parse_tool_calls(out);
        assert_eq!(calls.len(), 1);
        assert_eq!(rest, "all done");
    }

    #[test]
    fn parse_nested_call_shaped_object_is_not_a_call() {
        // A bare object that merely *contains* a {name, arguments} sub-object is
        // not a call; it must be returned verbatim, not mined for the inner
        // object (which would also leave broken JSON in the content).
        let input = r#"{"data": {"name": "f", "arguments": {}}}"#;
        let (calls, rest) = parse_tool_calls(input);
        assert!(calls.is_empty());
        assert_eq!(rest, input);
    }

    #[test]
    fn parse_unbalanced_braces_terminates_without_calls() {
        // A wall of opening braces (degenerate decode) must not be read as calls
        // and must return promptly — the scan stops at the first unterminated
        // object instead of rescanning every brace (O(n²)).
        let input = "{".repeat(4096);
        let (calls, rest) = parse_tool_calls(&input);
        assert!(calls.is_empty());
        assert_eq!(rest, input);
    }

    #[test]
    fn new_call_id_is_unique_per_call() {
        // Back-to-back ids must differ even within a single nanosecond.
        let a = new_call_id();
        let b = new_call_id();
        assert_ne!(a, b);
        assert!(a.starts_with("call_") && b.starts_with("call_"));
    }

    #[test]
    fn find_json_end_handles_nested_and_strings() {
        let s = r#"{"a": {"b": "}"}, "c": 1}xyz"#;
        let end = find_json_end(s, 0).unwrap();
        assert_eq!(&s[..end], r#"{"a": {"b": "}"}, "c": 1}"#);
    }
}
