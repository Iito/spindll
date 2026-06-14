//! Backend-agnostic tool-calling support.
//!
//! Spindll owns tool-call *emission* (templating + constrained decoding +
//! parsing); the *execution* loop belongs to the consumer. This module is the
//! shared vocabulary so the HTTP (`/v1/chat/completions`) and gRPC (`Chat`)
//! surfaces speak the same types instead of each re-implementing parsing.
//!
//! - [`ToolSpec`] / [`ToolChoice`] / [`ToolCall`] — the neutral request/response
//!   types each API surface converts into.
//! - [`tools_to_oai_json`] — render specs into the OpenAI-compatible tool array
//!   string that `llama_cpp_2`'s `apply_chat_template_with_tools_oaicompat`
//!   consumes (it returns a model-correct prompt + GBNF grammar).
//! - [`parse_tool_calls`] — extract calls from raw model output. Aware of the
//!   common model wrappers (Hermes `<tool_call>`, Llama-3.1 `<|python_tag|>`,
//!   Mistral `[TOOL_CALLS]`) with a balanced-JSON scan as the fallback. This is
//!   the source of truth for extraction on every backend; on llama.cpp the
//!   grammar makes the body reliable, but the wrapper tokens are model-specific.

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

/// Generate an OpenAI-style call id (`call_<hex>`).
pub fn new_call_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("call_{nanos:016x}")
}

/// Render specs into the OpenAI-compatible `tools` JSON array string consumed
/// by `llama_cpp_2::LlamaModel::apply_chat_template_with_tools_oaicompat`.
/// Returns `None` when there are no tools (so the caller passes `None` through
/// and gets the plain, no-grammar template).
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
    //    remainder is a JSON object or array of calls. Strip and fall through.
    let body = trimmed
        .strip_prefix("[TOOL_CALLS]")
        .or_else(|| trimmed.strip_prefix("<|python_tag|>"))
        .map(str::trim)
        .unwrap_or(trimmed);

    // 3. Balanced-JSON scan: pull every top-level `{...}` that looks like a
    //    call. Handles a bare object, several objects, or text + object.
    scan_json_calls(body)
}

/// Parse the JSON inside each `<open>...</close>` block.
fn parse_wrapped(text: &str, open: &str, close: &str) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(open) {
        let after = &rest[start + open.len()..];
        let end = after.find(close)?;
        let inner = after[..end].trim();
        if let Some(call) = call_from_json_str(inner) {
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
fn scan_json_calls(text: &str) -> (Vec<ToolCall>, String) {
    let mut calls = Vec::new();
    let mut remaining = String::new();
    let mut search_from = 0;

    while search_from < text.len() {
        if let Some(rel) = text[search_from..].find('{') {
            let abs = search_from + rel;
            if let Some(end) = find_json_end(text, abs)
                && let Some(call) = call_from_json_str(&text[abs..end])
            {
                remaining.push_str(&text[search_from..abs]);
                search_from = end;
                calls.push(call);
                continue;
            }
        }
        remaining.push_str(&text[search_from..]);
        break;
    }

    (calls, remaining.trim().to_string())
}

/// Build a [`ToolCall`] from a JSON string holding `{"name", "arguments"}`.
/// `arguments` may be an object (serialized to a string) or already a string.
fn call_from_json_str(candidate: &str) -> Option<ToolCall> {
    let parsed: Value = serde_json::from_str(candidate).ok()?;
    let obj = parsed.as_object()?;
    let name = obj.get("name")?.as_str()?;
    let arguments = obj.get("arguments")?;
    Some(ToolCall {
        id: new_call_id(),
        name: name.to_string(),
        arguments: match arguments {
            Value::String(s) => s.clone(),
            other => serde_json::to_string(other).unwrap_or_default(),
        },
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
    fn find_json_end_handles_nested_and_strings() {
        let s = r#"{"a": {"b": "}"}, "c": 1}xyz"#;
        let end = find_json_end(s, 0).unwrap();
        assert_eq!(&s[..end], r#"{"a": {"b": "}"}, "c": 1}"#);
    }
}
