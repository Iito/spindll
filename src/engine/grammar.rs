//! GBNF grammar construction for constrained tool-call decoding.
//!
//! Behind the `grammar` feature, which pulls in llama-cpp-2's `common` library
//! for [`llama_cpp_2::json_schema_to_grammar`] and the lazy grammar sampler.
//! When `tool_choice` demands a call, [`tool_call_grammar`] builds a GBNF
//! grammar forcing the model to emit a syntactically valid tool call — a real
//! tool `name` plus an `arguments` object. The streaming sampler applies it
//! *lazily*, keyed on the model's tool-call opener ([`TOOL_CALL_TRIGGERS`]), so
//! ordinary prose stays unconstrained until the model commits to a call.

use super::tools::{ToolChoice, ToolSpec};

/// Tool-call openers across the common model formats. The lazy grammar sampler
/// activates the grammar only once one of these appears in the output, leaving
/// plain-text turns unconstrained.
pub(crate) const TOOL_CALL_TRIGGERS: &[&str] = &["<tool_call>", "[TOOL_CALLS]", "<|python_tag|>"];

/// Build a GBNF grammar constraining output to a valid tool call, or `None` when
/// `tool_choice` doesn't demand one (`Auto`/`None`) or no schema is derivable.
///
/// For [`ToolChoice::Named`] the call is pinned to that function and its argument
/// schema; for [`ToolChoice::Required`] any listed tool name is allowed with a
/// generic object for arguments.
pub(crate) fn tool_call_grammar(tools: &[ToolSpec], choice: &ToolChoice) -> Option<String> {
    let (names, arguments): (Vec<&str>, serde_json::Value) = match choice {
        ToolChoice::Named(name) => {
            let spec = tools.iter().find(|t| &t.name == name)?;
            let args = spec
                .parameters
                .clone()
                .unwrap_or_else(|| serde_json::json!({ "type": "object" }));
            (vec![name.as_str()], args)
        }
        ToolChoice::Required => {
            if tools.is_empty() {
                return None;
            }
            (
                tools.iter().map(|t| t.name.as_str()).collect(),
                serde_json::json!({ "type": "object" }),
            )
        }
        ToolChoice::Auto | ToolChoice::None => return None,
    };

    // Constrain to the {"name": <one of the tools>, "arguments": {…}} object the
    // tool-call wrappers carry. json_schema_to_grammar emits a `root` rule.
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string", "enum": names },
            "arguments": arguments,
        },
        "required": ["name", "arguments"],
        "additionalProperties": false,
    });

    llama_cpp_2::json_schema_to_grammar(&schema.to_string()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            })),
        }
    }

    #[test]
    fn auto_and_none_are_unconstrained() {
        assert!(tool_call_grammar(&[spec("f")], &ToolChoice::Auto).is_none());
        assert!(tool_call_grammar(&[spec("f")], &ToolChoice::None).is_none());
    }

    #[test]
    fn named_builds_a_root_grammar() {
        let g = tool_call_grammar(&[spec("get_weather")], &ToolChoice::Named("get_weather".into()))
            .expect("a named choice yields a grammar");
        assert!(g.contains("root"), "GBNF should define a root rule, got: {g}");
    }

    #[test]
    fn named_missing_tool_is_none() {
        assert!(tool_call_grammar(&[spec("f")], &ToolChoice::Named("ghost".into())).is_none());
    }

    #[test]
    fn required_with_tools_builds_a_grammar() {
        assert!(tool_call_grammar(&[spec("a"), spec("b")], &ToolChoice::Required).is_some());
    }

    #[test]
    fn required_without_tools_is_none() {
        assert!(tool_call_grammar(&[], &ToolChoice::Required).is_none());
    }
}
