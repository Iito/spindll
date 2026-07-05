//! Faithful chat-template rendering for GGUF models.
//!
//! GGUF bakes the model's exact training template into the `tokenizer.chat_template`
//! metadata key — a Hugging Face Jinja template. llama.cpp's legacy
//! `llama_chat_apply_template` only substring-detects ~55 hard-coded formats and
//! returns `-1` on anything else, so it cannot render an arbitrary embedded
//! template (its own header says it *"does not use a jinja parser"*). We render
//! the real template with [minijinja] instead — the path llama-cpp-2's own docs
//! recommend — which matches what the MLX backend already does via
//! swift-transformers, closing the GGUF/MLX parity gap.
//!
//! [`render`] is intentionally free of any llama.cpp types so it can be unit
//! tested against real template strings without loading a model. The caller
//! ([`super::apply_chat_template_with_fallback`]) supplies the `bos`/`eos` token
//! strings and falls back to the legacy formatter when a model ships no template
//! or uses a Jinja feature minijinja can't handle.
//!
//! [minijinja]: https://github.com/mitsuhiko/minijinja

use minijinja::{Environment, Error, ErrorKind, Value, context};

/// Render a Hugging Face Jinja chat template to a prompt string.
///
/// `messages` are `(role, content)` pairs; `bos_token`/`eos_token` are the
/// model's special-token strings (templates reference them directly, e.g.
/// `{{ bos_token }}`). With `add_generation_prompt` the template appends the
/// opening of an assistant turn so the model continues from there.
pub(crate) fn render(
    template: &str,
    messages: &[(String, String)],
    bos_token: &str,
    eos_token: &str,
    add_generation_prompt: bool,
) -> anyhow::Result<String> {
    let mut env = Environment::new();

    // Match Hugging Face's `apply_chat_template`, which renders with
    // `trim_blocks=True, lstrip_blocks=True`. Without these the whitespace of
    // the rendered prompt diverges from what the model was trained on.
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);

    // HF templates are written for Python Jinja2 and call Python str/list
    // methods (`.strip()`, `.startswith()`, `.split()`, …). pycompat routes
    // those method calls to Python-compatible implementations.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

    // Templates call `raise_exception(...)` to reject unsupported message
    // shapes (e.g. a standalone system role). Surface it as a render error.
    env.add_function("raise_exception", raise_exception);

    env.add_template("chat", template)
        .map_err(|e| anyhow::anyhow!("invalid chat template: {}", render_error_chain(&e)))?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| anyhow::anyhow!("chat template unavailable: {e}"))?;

    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|(role, content)| serde_json::json!({ "role": role, "content": content }))
        .collect();

    tmpl.render(context! {
        messages => Value::from_serialize(&msgs),
        add_generation_prompt => add_generation_prompt,
        bos_token => bos_token,
        eos_token => eos_token,
    })
    .map_err(|e| anyhow::anyhow!("chat template render failed: {}", render_error_chain(&e)))
}

/// Jinja `raise_exception(msg)` — templates call it to reject inputs they can't
/// represent. Maps to a minijinja error so [`render`] returns `Err`.
fn raise_exception(msg: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, msg))
}

/// minijinja errors carry their real cause in `.source()`; the top-level
/// `Display` alone is often just "invalid operation". Walk the chain so the
/// fallback log says *why* a template failed.
fn render_error_chain(err: &Error) -> String {
    use std::error::Error as _;
    let mut out = err.to_string();
    let mut src = err.source();
    while let Some(e) = src {
        out.push_str(&format!(": {e}"));
        src = e.source();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal ChatML template (Qwen2.5 / many others ship this shape).
    const CHATML: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    fn msgs(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(r, c)| (r.to_string(), c.to_string()))
            .collect()
    }

    #[test]
    fn renders_chatml_with_generation_prompt() {
        let out = render(
            CHATML,
            &msgs(&[("system", "You are helpful."), ("user", "Hi")]),
            "<s>",
            "</s>",
            true,
        )
        .unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn omits_generation_prompt_when_disabled() {
        let out = render(CHATML, &msgs(&[("user", "Hi")]), "<s>", "</s>", false).unwrap();
        assert_eq!(out, "<|im_start|>user\nHi<|im_end|>\n");
    }

    #[test]
    fn pycompat_supplies_python_string_methods() {
        // `.strip()` and `.upper()` are Python str methods, not native Jinja.
        let out = render(
            "{{ messages[0]['content'].strip().upper() }}",
            &msgs(&[("user", "  hi there  ")]),
            "<s>",
            "</s>",
            false,
        )
        .unwrap();
        assert_eq!(out, "HI THERE");
    }

    #[test]
    fn bos_token_is_available_to_template() {
        let out = render(
            "{{ bos_token }}{{ messages[0]['content'] }}",
            &msgs(&[("user", "x")]),
            "<|begin_of_text|>",
            "</s>",
            false,
        )
        .unwrap();
        assert_eq!(out, "<|begin_of_text|>x");
    }

    #[test]
    fn raise_exception_becomes_render_error() {
        let tmpl = "{% if messages[0]['role'] == 'system' %}\
                    {{ raise_exception('System role not supported') }}{% endif %}ok";
        let err = render(tmpl, &msgs(&[("system", "hello")]), "<s>", "</s>", false).unwrap_err();
        assert!(
            err.to_string().contains("System role not supported"),
            "error should carry the template's message, got: {err}"
        );
    }

    #[test]
    fn trim_blocks_removes_block_line_newlines() {
        // trim_blocks (a Hugging Face default) drops the newline immediately
        // after a `%}`, so block tags on their own lines don't inject blank
        // lines between turns.
        let tmpl = "{% for m in messages %}\n{{ m['content'] }}\n{% endfor %}";
        let out = render(tmpl, &msgs(&[("user", "a"), ("assistant", "b")]), "<s>", "</s>", false)
            .unwrap();
        assert_eq!(out, "a\nb\n");
    }

    #[test]
    fn lstrip_blocks_strips_indent_before_block_tags() {
        // lstrip_blocks (a Hugging Face default) strips leading whitespace before
        // a `{%` block tag — but NOT before a `{{` expression — so indented
        // control blocks contribute no spaces to the output.
        let out = render(
            "x\n    {% if true %}y{% endif %}",
            &msgs(&[("user", "unused")]),
            "<s>",
            "</s>",
            false,
        )
        .unwrap();
        assert_eq!(out, "x\ny");
    }

    #[test]
    fn invalid_template_is_an_error_not_a_panic() {
        let err = render("{% for x in %}", &msgs(&[("user", "hi")]), "<s>", "</s>", false)
            .unwrap_err();
        assert!(err.to_string().contains("chat template"));
    }
}
