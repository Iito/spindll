// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

//! Split think-block reasoning out of generated text (issue #75).
//!
//! Thinking models (Qwen3 / Qwen3.5, DeepSeek-R1, GLM, …) emit a reasoning
//! pass delimited by `<think>` / `</think>` before the visible answer. Both
//! backends stream those delimiters as literal text (llama.cpp's
//! `token_to_piece` and the MLX bridge's `NaiveStreamingDetokenizer` decode
//! special tokens verbatim), so the split can happen here, backend-agnostic,
//! on the decoded stream.
//!
//! Two delimiter conventions exist in the wild:
//!
//! 1. **Explicit** — the model emits `<think>` itself as its first output.
//! 2. **Forced-open** — the chat template ends the generation prompt inside
//!    an already-opened block (`…assistant\n<think>\n`), so the stream starts
//!    mid-reasoning and only `</think>` ever appears. Detected once per model
//!    at load by probing the rendered template ([`prompt_opens_reasoning`]).
//!
//! Only a single leading think block is recognized — matching what the target
//! model families emit. A `<think>` appearing after visible content streams
//! through as ordinary text.

/// Opening delimiter of a reasoning block.
pub const THINK_OPEN: &str = "<think>";
/// Closing delimiter of a reasoning block.
pub const THINK_CLOSE: &str = "</think>";

/// True when a rendered generation prompt leaves a think block open, i.e. the
/// model will start its output mid-reasoning with no opening tag of its own
/// (Qwen3-thinking style templates end with `<|im_start|>assistant\n<think>\n`).
///
/// A template that *closes* the block itself (Qwen3's no-think mode renders a
/// trailing `<think>\n\n</think>`) ends with `</think>` and stays `false`.
pub fn prompt_opens_reasoning(rendered_prompt: &str) -> bool {
    rendered_prompt.trim_end().ends_with(THINK_OPEN)
}

enum State {
    /// Start of stream: leading whitespace and a possible explicit `<think>`.
    Detect,
    /// Inside a think block, watching for `</think>`.
    Reasoning,
    /// Past the block (or there never was one): pass text through. `skip_ws`
    /// swallows the whitespace the model emits right after `</think>` so the
    /// visible answer doesn't start with stray blank lines.
    Content { skip_ws: bool },
}

/// Incremental splitter: feed decoded stream pieces, get back
/// `(reasoning_delta, content_delta)` pairs suitable for
/// `delta.reasoning_content` / `delta.content` SSE chunks.
///
/// Delimiters may arrive split across pieces; the splitter withholds at most a
/// partial tag's worth of characters until they are decided.
pub struct ReasoningSplitter {
    state: State,
    buf: String,
}

impl ReasoningSplitter {
    /// `forced_open` — start inside a think block (template force-opened it).
    pub fn new(forced_open: bool) -> Self {
        Self {
            state: if forced_open { State::Reasoning } else { State::Detect },
            buf: String::new(),
        }
    }

    /// Feed one decoded piece; returns the reasoning and content text that
    /// became definite with it (either or both may be empty).
    pub fn push(&mut self, piece: &str) -> (String, String) {
        self.buf.push_str(piece);
        let mut reasoning = String::new();
        let mut content = String::new();

        loop {
            match &mut self.state {
                State::Detect => {
                    let after_ws = self.buf.trim_start();
                    if after_ws.is_empty() {
                        break; // only whitespace so far — undecided
                    }
                    if after_ws.starts_with(THINK_OPEN) {
                        // Drop the leading whitespace + tag, enter the block.
                        let cut = self.buf.len() - after_ws.len() + THINK_OPEN.len();
                        self.buf.drain(..cut);
                        self.state = State::Reasoning;
                        continue;
                    }
                    if THINK_OPEN.starts_with(after_ws) {
                        break; // could still complete into the opening tag
                    }
                    // Not a think block — everything (whitespace included)
                    // is ordinary content.
                    self.state = State::Content { skip_ws: false };
                    continue;
                }
                State::Reasoning => {
                    if let Some(i) = self.buf.find(THINK_CLOSE) {
                        reasoning.push_str(&self.buf[..i]);
                        self.buf.drain(..i + THINK_CLOSE.len());
                        self.state = State::Content { skip_ws: true };
                        continue;
                    }
                    // Emit all but the longest tail that could still grow into
                    // the closing tag (tags are ASCII, so byte-safe slicing).
                    let hold = longest_suffix_matching_prefix(&self.buf, THINK_CLOSE);
                    let emit_to = self.buf.len() - hold;
                    reasoning.push_str(&self.buf[..emit_to]);
                    self.buf.drain(..emit_to);
                    break;
                }
                State::Content { skip_ws } => {
                    if *skip_ws {
                        let after_ws = self.buf.trim_start();
                        if after_ws.is_empty() {
                            self.buf.clear();
                            break;
                        }
                        let cut = self.buf.len() - after_ws.len();
                        self.buf.drain(..cut);
                        *skip_ws = false;
                    }
                    content.push_str(&self.buf);
                    self.buf.clear();
                    break;
                }
            }
        }
        (reasoning, content)
    }

    /// End of generation: flush whatever is still undecided. A never-closed
    /// block (the `max_tokens`-exhausted case) flushes as reasoning; a pending
    /// partial tag that never completed flushes as content.
    pub fn finish(mut self) -> (String, String) {
        let rest = std::mem::take(&mut self.buf);
        match self.state {
            State::Reasoning => (rest, String::new()),
            State::Detect | State::Content { .. } => (String::new(), rest),
        }
    }
}

/// Longest `k` such that the last `k` bytes of `haystack` equal the first `k`
/// bytes of `tag` (with `k < tag.len()`), i.e. the tail that might yet grow
/// into the full tag once more pieces arrive.
fn longest_suffix_matching_prefix(haystack: &str, tag: &str) -> usize {
    let max = tag.len().saturating_sub(1).min(haystack.len());
    (1..=max)
        .rev()
        .find(|&k| haystack.ends_with(&tag[..k]))
        .unwrap_or(0)
}

/// Collected result of a full generation, split once at the end.
pub struct SplitOutput {
    /// Trimmed reasoning text; `None` when no think block was found.
    pub reasoning: Option<String>,
    /// The visible answer.
    pub content: String,
    /// Stream pieces that carried reasoning (≈ tokens: both backends emit one
    /// decoded piece per sampled token).
    pub reasoning_pieces: u32,
    /// Stream pieces that carried visible content.
    pub content_pieces: u32,
}

/// Accumulating wrapper for the non-streaming paths: feed every piece, then
/// [`finish`](ReasoningCollector::finish). Because nothing is emitted until
/// the end, it can also recover the forced-open case *retroactively* when the
/// load-time probe missed it: a bare `</think>` with no opener proves the
/// prefix was reasoning.
pub struct ReasoningCollector {
    splitter: ReasoningSplitter,
    reasoning: String,
    content: String,
    reasoning_pieces: u32,
    content_pieces: u32,
    /// Pieces the splitter withheld while deciding whether their bytes begin a
    /// delimiter. Attributed once the text resolves to one side or the other.
    pending_pieces: u32,
}

impl ReasoningCollector {
    pub fn new(forced_open: bool) -> Self {
        Self {
            splitter: ReasoningSplitter::new(forced_open),
            reasoning: String::new(),
            content: String::new(),
            reasoning_pieces: 0,
            content_pieces: 0,
            pending_pieces: 0,
        }
    }

    pub fn push(&mut self, piece: &str) {
        let (r, c) = self.splitter.push(piece);

        // A withheld piece emits nothing yet but was still a sampled token.
        // Counting only pieces that produced text under-reports reasoning, and
        // `completion_tokens - reasoning_tokens` then looks like answer tokens
        // that were generated and thrown away. That arithmetic is what #81 read
        // as dropped content.
        if r.is_empty() && c.is_empty() {
            if !piece.is_empty() {
                self.pending_pieces += 1;
            }
            return;
        }

        if !r.is_empty() {
            self.reasoning_pieces += 1 + std::mem::take(&mut self.pending_pieces);
            self.reasoning.push_str(&r);
        }
        if !c.is_empty() {
            self.content_pieces += 1 + std::mem::take(&mut self.pending_pieces);
            self.content.push_str(&c);
        }
    }

    pub fn finish(mut self) -> SplitOutput {
        let (r, c) = self.splitter.finish();
        if !r.is_empty() {
            self.reasoning_pieces += 1 + std::mem::take(&mut self.pending_pieces);
            self.reasoning.push_str(&r);
        }
        if !c.is_empty() {
            self.content_pieces += 1 + std::mem::take(&mut self.pending_pieces);
            self.content.push_str(&c);
        }
        // Nothing flushed at all: attribute the stragglers to whichever side
        // actually saw text, so no sampled token goes unaccounted for.
        if self.pending_pieces > 0 {
            let pending = std::mem::take(&mut self.pending_pieces);
            if self.content.is_empty() {
                self.reasoning_pieces += pending;
            } else {
                self.content_pieces += pending;
            }
        }

        // Retroactive forced-open recovery: no reasoning was recognized, but
        // the stream carries a closing tag with no opener before it.
        if self.reasoning.is_empty()
            && let Some(i) = self.content.find(THINK_CLOSE)
            && !self.content[..i].contains(THINK_OPEN)
        {
            let total = self.content_pieces.max(1);
            let share = i as f64 / self.content.len().max(1) as f64;
            self.reasoning_pieces = (total as f64 * share).round() as u32;
            self.content_pieces = total - self.reasoning_pieces;
            let after = self.content[i + THINK_CLOSE.len()..].trim_start().to_string();
            self.reasoning = self.content[..i].to_string();
            self.content = after;
        }

        let reasoning = self.reasoning.trim();
        SplitOutput {
            reasoning: (!reasoning.is_empty()).then(|| reasoning.to_string()),
            content: self.content,
            reasoning_pieces: self.reasoning_pieces,
            content_pieces: self.content_pieces,
        }
    }
}

/// One-shot split of a fully collected output string.
pub fn split_reasoning(text: &str, forced_open: bool) -> SplitOutput {
    let mut c = ReasoningCollector::new(forced_open);
    c.push(text);
    c.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive the streaming splitter over `pieces`, joining the deltas.
    fn stream(forced_open: bool, pieces: &[&str]) -> (String, String) {
        let mut s = ReasoningSplitter::new(forced_open);
        let (mut r, mut c) = (String::new(), String::new());
        for p in pieces {
            let (dr, dc) = s.push(p);
            r.push_str(&dr);
            c.push_str(&dc);
        }
        let (dr, dc) = s.finish();
        r.push_str(&dr);
        c.push_str(&dc);
        (r, c)
    }

    #[test]
    fn explicit_block_splits() {
        let (r, c) = stream(false, &["<think>", "plan", "ning", "</think>", "\n\nanswer"]);
        assert_eq!(r, "planning");
        assert_eq!(c, "answer");
    }

    #[test]
    fn tags_split_across_pieces() {
        let (r, c) = stream(false, &["<th", "ink>abc</th", "ink> hi"]);
        assert_eq!(r, "abc");
        assert_eq!(c, "hi");
    }

    #[test]
    fn forced_open_stream_needs_no_opening_tag() {
        let (r, c) = stream(true, &["The user", " wants X.", "</think>", "\nAnswer: 42"]);
        assert_eq!(r, "The user wants X.");
        assert_eq!(c, "Answer: 42");
    }

    #[test]
    fn plain_stream_passes_through_verbatim() {
        let (r, c) = stream(false, &[" leading", " text"]);
        assert_eq!(r, "");
        assert_eq!(c, " leading text");
    }

    #[test]
    fn budget_exhausted_mid_think_is_all_reasoning() {
        // The issue-#75 failure mode: max_tokens ran out before `</think>`.
        let (r, c) = stream(true, &["step 1…", " step 2…"]);
        assert_eq!(r, "step 1… step 2…");
        assert_eq!(c, "");
    }

    #[test]
    fn partial_open_tag_that_never_completes_is_content() {
        let (r, c) = stream(false, &["<t", "able>x"]);
        assert_eq!(r, "");
        assert_eq!(c, "<table>x");

        // …including when the stream ends mid-prefix.
        let (r, c) = stream(false, &["<thi"]);
        assert_eq!(r, "");
        assert_eq!(c, "<thi");
    }

    #[test]
    fn whitespace_before_explicit_tag_is_allowed() {
        let (r, c) = stream(false, &["\n", "<think>", "r", "</think>", "c"]);
        assert_eq!(r, "r");
        assert_eq!(c, "c");
    }

    #[test]
    fn empty_think_block_yields_no_reasoning() {
        let out = split_reasoning("<think></think>ok", false);
        assert_eq!(out.reasoning, None);
        assert_eq!(out.content, "ok");
    }

    #[test]
    fn whitespace_after_close_is_swallowed_once() {
        let (r, c) = stream(false, &["<think>r</think>", "\n", "\n\nanswer\nmore"]);
        assert_eq!(r, "r");
        assert_eq!(c, "answer\nmore");
    }

    #[test]
    fn reasoning_streams_promptly_not_only_at_close() {
        let mut s = ReasoningSplitter::new(true);
        let (r, _) = s.push("a long reasoning fragment");
        assert_eq!(r, "a long reasoning fragment", "must not buffer whole block");
    }

    #[test]
    fn one_shot_retro_recovers_missed_forced_open() {
        // Probe said not-forced, but the stream carries a bare close tag.
        let out = split_reasoning("reasoning here</think>\n\nanswer", false);
        assert_eq!(out.reasoning.as_deref(), Some("reasoning here"));
        assert_eq!(out.content, "answer");
        assert!(out.reasoning_pieces + out.content_pieces >= 1);
    }

    #[test]
    fn inline_think_after_content_is_not_split() {
        // A think block after visible content is out of scope — passthrough.
        let text = "abc <think>x</think> y";
        let out = split_reasoning(text, false);
        assert_eq!(out.reasoning, None);
        assert_eq!(out.content, text);
    }

    #[test]
    fn literal_close_tag_without_open_in_forced_mode() {
        let out = split_reasoning("thoughts</think>done", true);
        assert_eq!(out.reasoning.as_deref(), Some("thoughts"));
        assert_eq!(out.content, "done");
    }

    #[test]
    fn collector_counts_pieces_per_side() {
        let pieces = ["<think>", "a", "b", "</think>", "x", "y"];
        let mut c = ReasoningCollector::new(false);
        for p in pieces {
            c.push(p);
        }
        let out = c.finish();
        assert_eq!(out.reasoning.as_deref(), Some("ab"));
        assert_eq!(out.content, "xy");
        // Delimiters are sampled tokens too, so they count. `<think>` lands on
        // reasoning; `</think>` is attributed to the side that follows it,
        // which costs one token of precision and buys the invariant below.
        assert_eq!(out.reasoning_pieces, 3);
        assert_eq!(out.content_pieces, 3);
    }

    #[test]
    fn every_piece_is_accounted_for_on_one_side_or_the_other() {
        // The property #81 turned on: with pieces going uncounted whenever the
        // splitter withheld them, `completion_tokens - reasoning_tokens` left a
        // phantom remainder that read as answer tokens generated and discarded.
        for (forced, pieces) in [
            (false, &["<think>", "a", "b", "</think>", "x", "y"][..]),
            (false, &["<think>", "never closed"][..]),
            (true, &["mid", "thought", "</think>", "\n\n", "answer"][..]),
            (true, &["</th", "ink>", "answer"][..]),
            (false, &["no tags at all"][..]),
            (false, &["<thi", "nk>", "r", "</thi", "nk>", "c"][..]),
        ] {
            let mut c = ReasoningCollector::new(forced);
            for p in pieces {
                c.push(p);
            }
            let out = c.finish();
            assert_eq!(
                out.reasoning_pieces + out.content_pieces,
                pieces.len() as u32,
                "unaccounted pieces for {pieces:?} (forced_open={forced})"
            );
        }
    }

    #[test]
    fn multibyte_text_streams_safely() {
        let (r, c) = stream(true, &["思考", "中…", "</th", "ink>", "答え"]);
        assert_eq!(r, "思考中…");
        assert_eq!(c, "答え");
    }

    #[test]
    fn prompt_probe_detects_forced_open_tail() {
        assert!(prompt_opens_reasoning("<|im_start|>assistant\n<think>\n"));
        assert!(prompt_opens_reasoning("…<｜Assistant｜><think>"));
        // Qwen3 no-think mode closes the block itself.
        assert!(!prompt_opens_reasoning("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert!(!prompt_opens_reasoning("<|im_start|>assistant\n"));
        assert!(!prompt_opens_reasoning(""));
    }
}
