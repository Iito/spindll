// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

//! C API over spindll's [`Scorer`], for apps that link spindll as a static
//! library: on iOS, through the SpindllKit Swift package. The header, and the
//! contract for every pointer, is `include/spindll.h`.
//!
//! Functions that can fail return NULL or a negative number and leave a
//! message for `spindll_last_error()`, which is per thread. No panic crosses
//! into C: each entry point catches it and reports it as an error. A scorer
//! serializes its own calls, so it can be used from any thread.

// The pointer contracts live in include/spindll.h, the only documentation C
// callers read; repeating them per function here would only drift.
#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{anyhow, ensure};
use spindll::engine::{Scorer, ScorerConfig};

/// Mirrors `SpindllScorerConfig` in spindll.h.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SpindllScorerConfig {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub parallel: u32,
    pub max_context_tokens: u32,
    pub max_candidate_tokens: u32,
    pub threads: i32,
    pub gpu_layers: u32,
}

impl From<&ScorerConfig> for SpindllScorerConfig {
    fn from(c: &ScorerConfig) -> Self {
        Self {
            n_ctx: c.n_ctx,
            n_batch: c.n_batch,
            parallel: c.parallel,
            max_context_tokens: c.max_context_tokens as u32,
            max_candidate_tokens: c.max_candidate_tokens as u32,
            threads: c.threads,
            gpu_layers: c.gpu_layers,
        }
    }
}

impl From<&SpindllScorerConfig> for ScorerConfig {
    fn from(c: &SpindllScorerConfig) -> Self {
        Self {
            n_ctx: c.n_ctx,
            n_batch: c.n_batch,
            parallel: c.parallel,
            max_context_tokens: c.max_context_tokens as usize,
            max_candidate_tokens: c.max_candidate_tokens as usize,
            threads: c.threads,
            gpu_layers: c.gpu_layers,
        }
    }
}

/// The handle behind `SpindllScorer *`.
pub struct SpindllScorer {
    scorer: Scorer,
    /// Token texts, fetched from the worker once, on first use.
    pieces: OnceLock<Vec<String>>,
}

impl SpindllScorer {
    fn pieces(&self) -> anyhow::Result<&[String]> {
        if self.pieces.get().is_none() {
            let _ = self.pieces.set(self.scorer.pieces()?);
        }
        Ok(self.pieces.get().expect("set above"))
    }
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(message: &str) {
    let message = CString::new(message.replace('\0', " ")).expect("NUL bytes were replaced");
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(message));
}

/// Run `f`, turning an error or a panic into `on_error` and a message for
/// `spindll_last_error()`.
fn guard<T>(on_error: T, f: impl FnOnce() -> anyhow::Result<T>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => value,
        Ok(Err(e)) => {
            set_error(&format!("{e:#}"));
            on_error
        }
        Err(panic) => {
            let what = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_owned());
            set_error(&format!("internal error: {what}"));
            on_error
        }
    }
}

unsafe fn handle<'a>(scorer: *const SpindllScorer) -> anyhow::Result<&'a SpindllScorer> {
    unsafe { scorer.as_ref() }.ok_or_else(|| anyhow!("scorer is NULL"))
}

unsafe fn text<'a>(ptr: *const c_char, what: &str) -> anyhow::Result<&'a str> {
    ensure!(!ptr.is_null(), "{what} is NULL");
    unsafe { CStr::from_ptr(ptr) }.to_str().map_err(|_| anyhow!("{what} is not valid UTF-8"))
}

fn length(n: i32, what: &str) -> anyhow::Result<usize> {
    usize::try_from(n).map_err(|_| anyhow!("{what} is negative"))
}

unsafe fn slice<'a, T>(ptr: *const T, len: usize, what: &str) -> anyhow::Result<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    ensure!(!ptr.is_null(), "{what} is NULL");
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

unsafe fn slice_mut<'a, T>(ptr: *mut T, len: usize, what: &str) -> anyhow::Result<&'a mut [T]> {
    if len == 0 {
        return Ok(&mut []);
    }
    ensure!(!ptr.is_null(), "{what} is NULL");
    Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn spindll_scorer_config_default(config: *mut SpindllScorerConfig) {
    if let Some(config) = unsafe { config.as_mut() } {
        *config = (&ScorerConfig::default()).into();
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn spindll_scorer_load(
    path: *const c_char,
    config: *const SpindllScorerConfig,
) -> *mut SpindllScorer {
    guard(std::ptr::null_mut(), || {
        let path = unsafe { text(path, "path") }?;
        let config = unsafe { config.as_ref() }.map_or_else(ScorerConfig::default, ScorerConfig::from);
        let scorer = Scorer::load(Path::new(path), config)?;
        Ok(Box::into_raw(Box::new(SpindllScorer { scorer, pieces: OnceLock::new() })))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn spindll_scorer_free(scorer: *mut SpindllScorer) {
    guard((), || {
        if !scorer.is_null() {
            // Dropping stops the worker thread, which frees the model.
            drop(unsafe { Box::from_raw(scorer) });
        }
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn spindll_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ref().map_or(std::ptr::null(), |s| s.as_ptr()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn spindll_scorer_n_vocab(scorer: *const SpindllScorer) -> i32 {
    guard(-1, || Ok(i32::try_from(unsafe { handle(scorer) }?.scorer.n_vocab())?))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn spindll_scorer_bos(scorer: *const SpindllScorer) -> i32 {
    guard(-1, || Ok(unsafe { handle(scorer) }?.scorer.bos()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn spindll_scorer_eos(scorer: *const SpindllScorer) -> i32 {
    guard(-1, || Ok(unsafe { handle(scorer) }?.scorer.eos()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn spindll_scorer_piece(
    scorer: *mut SpindllScorer,
    token: i32,
    buffer: *mut c_char,
    capacity: i32,
) -> i32 {
    guard(-1, || {
        let pieces = unsafe { handle(scorer) }?.pieces()?;
        let piece = usize::try_from(token)
            .ok()
            .and_then(|t| pieces.get(t))
            .ok_or_else(|| anyhow!("token {token} is out of range"))?
            .as_bytes();
        let capacity = length(capacity, "capacity")?;
        let n = piece.len().min(capacity);
        unsafe { slice_mut(buffer.cast::<u8>(), n, "buffer") }?.copy_from_slice(&piece[..n]);
        Ok(i32::try_from(piece.len())?)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn spindll_scorer_tokenize(
    scorer: *mut SpindllScorer,
    text_ptr: *const c_char,
    add_bos: bool,
    tokens: *mut i32,
    capacity: i32,
) -> i32 {
    guard(-1, || {
        let s = unsafe { handle(scorer) }?;
        let ids = s.scorer.tokenize(unsafe { text(text_ptr, "text") }?, add_bos)?;
        let n = ids.len().min(length(capacity, "capacity")?);
        unsafe { slice_mut(tokens, n, "tokens") }?.copy_from_slice(&ids[..n]);
        Ok(i32::try_from(ids.len())?)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn spindll_scorer_set_context(scorer: *mut SpindllScorer, text_ptr: *const c_char) -> i32 {
    guard(-1, || {
        unsafe { handle(scorer) }?.scorer.set_context(unsafe { text(text_ptr, "text") }?)?;
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn spindll_scorer_set_context_tokens(
    scorer: *mut SpindllScorer,
    tokens: *const i32,
    count: i32,
) -> i32 {
    guard(-1, || {
        let tokens = unsafe { slice(tokens, length(count, "count")?, "tokens") }?;
        unsafe { handle(scorer) }?.scorer.set_context_tokens(tokens)?;
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn spindll_scorer_next_logprobs(
    scorer: *mut SpindllScorer,
    tokens: *const i32,
    count: i32,
    logprobs: *mut f32,
) -> i32 {
    guard(-1, || {
        let count = length(count, "count")?;
        let tokens = unsafe { slice(tokens, count, "tokens") }?;
        let values = unsafe { handle(scorer) }?.scorer.next_logprobs(tokens)?;
        unsafe { slice_mut(logprobs, count, "logprobs") }?.copy_from_slice(&values);
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn spindll_scorer_top_next(
    scorer: *mut SpindllScorer,
    k: i32,
    allowed: *const bool,
    tokens: *mut i32,
    logprobs: *mut f32,
) -> i32 {
    guard(-1, || {
        let s = unsafe { handle(scorer) }?;
        let k = length(k, "k")?;
        let allowed = if allowed.is_null() {
            None
        } else {
            Some(unsafe { slice(allowed, s.scorer.n_vocab(), "allowed") }?)
        };
        let top = s.scorer.top_next(k, allowed)?;
        let out_tokens = unsafe { slice_mut(tokens, top.len(), "tokens") }?;
        let out_logprobs = unsafe { slice_mut(logprobs, top.len(), "logprobs") }?;
        for (i, (token, logprob)) in top.iter().enumerate() {
            out_tokens[i] = *token;
            out_logprobs[i] = *logprob;
        }
        Ok(i32::try_from(top.len())?)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn spindll_scorer_score(
    scorer: *mut SpindllScorer,
    texts: *const *const c_char,
    count: i32,
    terminators: *const i32,
    terminator_count: i32,
    scores: *mut f32,
) -> i32 {
    guard(-1, || {
        let count = length(count, "count")?;
        let texts = unsafe { slice(texts, count, "texts") }?
            .iter()
            .map(|&t| unsafe { text(t, "texts[i]") })
            .collect::<anyhow::Result<Vec<&str>>>()?;
        let terminators = unsafe { slice(terminators, length(terminator_count, "terminator_count")?, "terminators") }?;
        let values = unsafe { handle(scorer) }?.scorer.score(&texts, terminators)?;
        unsafe { slice_mut(scores, count, "scores") }?.copy_from_slice(&values);
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn spindll_scorer_extend_greedy(
    scorer: *mut SpindllScorer,
    starts: *const i32,
    count: i32,
    max_steps: i32,
    keep_going: *const bool,
    tokens: *mut i32,
    lengths: *mut i32,
) -> i32 {
    guard(-1, || {
        let s = unsafe { handle(scorer) }?;
        let count = length(count, "count")?;
        let max_steps = length(max_steps, "max_steps")?;
        let starts = unsafe { slice(starts, count, "starts") }?;
        let keep_going = unsafe { slice(keep_going, s.scorer.n_vocab(), "keep_going") }?;
        let extended = s.scorer.extend_greedy(starts, max_steps, keep_going)?;
        let stride = max_steps + 1;
        let out = unsafe { slice_mut(tokens, count * stride, "tokens") }?;
        let out_lengths = unsafe { slice_mut(lengths, count, "lengths") }?;
        for (i, seq) in extended.iter().enumerate() {
            out[i * stride..i * stride + seq.len()].copy_from_slice(seq);
            out_lengths[i] = i32::try_from(seq.len())?;
        }
        Ok(0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn last_error() -> String {
        let ptr = spindll_last_error();
        assert!(!ptr.is_null(), "expected an error message");
        unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
    }

    #[test]
    fn the_default_config_matches_the_scorer_default() {
        let mut config = SpindllScorerConfig {
            n_ctx: 0,
            n_batch: 0,
            parallel: 0,
            max_context_tokens: 0,
            max_candidate_tokens: 0,
            threads: 0,
            gpu_layers: 7,
        };
        unsafe { spindll_scorer_config_default(&mut config) };
        let back = ScorerConfig::from(&config);
        let expected = ScorerConfig::default();
        assert_eq!((back.n_ctx, back.n_batch, back.parallel), (expected.n_ctx, expected.n_batch, expected.parallel));
        assert_eq!(back.gpu_layers, 0);
    }

    #[test]
    fn a_bad_path_returns_null_and_says_why() {
        let scorer = unsafe { spindll_scorer_load(c"/nonexistent/model.gguf".as_ptr(), std::ptr::null()) };
        assert!(scorer.is_null());
        assert!(last_error().contains("model not found"), "{}", last_error());
    }

    #[test]
    fn null_arguments_are_errors_not_crashes() {
        assert_eq!(unsafe { spindll_scorer_set_context(std::ptr::null_mut(), c"x".as_ptr()) }, -1);
        assert!(last_error().contains("scorer is NULL"));
        assert_eq!(unsafe { spindll_scorer_n_vocab(std::ptr::null()) }, -1);
        let scorer = unsafe { spindll_scorer_load(std::ptr::null(), std::ptr::null()) };
        assert!(scorer.is_null());
        assert!(last_error().contains("path is NULL"));
        // Freeing NULL is a no-op, as with free().
        unsafe { spindll_scorer_free(std::ptr::null_mut()) };
    }

    /// Reads a small GGUF from `SPINDLL_SCORER_GGUF`:
    /// `cargo test -p spindll-ffi -- --ignored`.
    fn load() -> *mut SpindllScorer {
        let path = std::env::var("SPINDLL_SCORER_GGUF").expect("set SPINDLL_SCORER_GGUF to a small GGUF");
        let path = CString::new(path).unwrap();
        let scorer = unsafe { spindll_scorer_load(path.as_ptr(), std::ptr::null()) };
        assert!(!scorer.is_null(), "{}", last_error());
        scorer
    }

    fn tokenize(scorer: *mut SpindllScorer, text: &CStr, add_bos: bool) -> Vec<i32> {
        let mut tokens = vec![0; 64];
        let n = unsafe { spindll_scorer_tokenize(scorer, text.as_ptr(), add_bos, tokens.as_mut_ptr(), 64) };
        assert!(n >= 0, "{}", last_error());
        tokens.truncate(n as usize);
        tokens
    }

    #[test]
    #[ignore] // requires a downloaded model
    fn the_c_api_gives_the_rust_scorers_answers() {
        let scorer = load();
        let path = std::env::var("SPINDLL_SCORER_GGUF").unwrap();
        let rust = Scorer::load(Path::new(&path), ScorerConfig::default()).unwrap();

        let context = c"I would like to go to the";
        assert_eq!(unsafe { spindll_scorer_set_context(scorer, context.as_ptr()) }, 0);
        rust.set_context(context.to_str().unwrap()).unwrap();

        let words = [c" park", c" beach", c" moon"];
        let texts: Vec<*const c_char> = words.iter().map(|w| w.as_ptr()).collect();
        let terminators = tokenize(scorer, c" ,.", false);
        let mut scores = [0f32; 3];
        let rc = unsafe {
            spindll_scorer_score(
                scorer,
                texts.as_ptr(),
                3,
                terminators.as_ptr(),
                terminators.len() as i32,
                scores.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "{}", last_error());
        let expected = rust.score(&[" park", " beach", " moon"], &terminators).unwrap();
        assert_eq!(scores.to_vec(), expected);

        let mut top_tokens = [0i32; 5];
        let mut top_logprobs = [0f32; 5];
        let n = unsafe {
            spindll_scorer_top_next(scorer, 5, std::ptr::null(), top_tokens.as_mut_ptr(), top_logprobs.as_mut_ptr())
        };
        assert_eq!(n, 5, "{}", last_error());
        let expected: Vec<(i32, f32)> = rust.top_next(5, None).unwrap();
        assert_eq!(top_tokens.iter().copied().zip(top_logprobs).collect::<Vec<_>>(), expected);

        unsafe { spindll_scorer_free(scorer) };
    }

    #[test]
    #[ignore] // requires a downloaded model
    fn pieces_rebuild_the_tokenized_text() {
        let scorer = load();
        let tokens = tokenize(scorer, c"hello world", false);
        let mut text = Vec::new();
        for token in tokens {
            let mut buf = [0u8; 64];
            let n = unsafe { spindll_scorer_piece(scorer, token, buf.as_mut_ptr().cast(), 64) };
            assert!(n >= 0, "{}", last_error());
            text.extend_from_slice(&buf[..n as usize]);
        }
        assert_eq!(String::from_utf8(text).unwrap(), "hello world");
        unsafe { spindll_scorer_free(scorer) };
    }

    #[test]
    #[ignore] // requires a downloaded model
    fn greedy_extension_writes_each_run_at_its_stride() {
        let scorer = load();
        assert_eq!(unsafe { spindll_scorer_set_context(scorer, c"I would like to go to the".as_ptr()) }, 0);
        let n_vocab = unsafe { spindll_scorer_n_vocab(scorer) } as usize;
        let everything = vec![true; n_vocab];
        let starts = [tokenize(scorer, c" park", false)[0], tokenize(scorer, c" beach", false)[0]];
        let max_steps = 2;
        let mut tokens = vec![-1i32; starts.len() * (max_steps + 1)];
        let mut lengths = [0i32; 2];
        let rc = unsafe {
            spindll_scorer_extend_greedy(
                scorer,
                starts.as_ptr(),
                2,
                max_steps as i32,
                everything.as_ptr(),
                tokens.as_mut_ptr(),
                lengths.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "{}", last_error());
        assert_eq!(lengths, [3, 3], "with every token allowed, each run takes all its steps");
        assert_eq!(tokens[0], starts[0]);
        assert_eq!(tokens[3], starts[1]);
        unsafe { spindll_scorer_free(scorer) };
    }
}
