//! End-to-end tests for greedy speculative decoding.
//!
//! These tests exercise the draft+verify loop wired into `ModelManager`.
//! The functional correctness test uses self-speculation (same model loaded
//! twice as both target and draft): every draft token is always accepted by
//! the verifier, which lets us check the loop bookkeeping (KV rewind,
//! position tracking, emit ordering) without needing two real models or
//! shipping a tokenizer-compatible draft.
//!
//! Functional tests are `#[ignore]` and require a real GGUF model at the
//! canonical `~/.spindll/` location. Run with:
//!   cargo test --test spec_decoding -- --ignored

use std::path::PathBuf;

use spindll::engine::{GenerateParams, ModelManager};

/// Discover any local GGUF under `~/.spindll/models`. Falls back through
/// candidate paths so the test runs against whichever small model is pulled.
fn real_gguf_path() -> Option<PathBuf> {
    let home = std::env::var("HOME")
        .ok()
        .or_else(|| std::env::var("USERPROFILE").ok())?;
    let root = PathBuf::from(home).join(".spindll/models");
    if !root.exists() {
        return None;
    }
    // Recurse, return the smallest GGUF (fastest test).
    let mut best: Option<(PathBuf, u64)> = None;
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(u64::MAX);
                if best.as_ref().map_or(true, |(_, s)| size < *s) {
                    best = Some((path, size));
                }
            }
        }
    }
    best.map(|(p, _)| p)
}

#[test]
fn spec_decoding_rejects_non_greedy_sampling() {
    // No model required — failure is by params shape.
    let mgr = ModelManager::new(512, Some(0), 0).unwrap();
    let params = GenerateParams {
        max_tokens: 4,
        temperature: 0.8,
        draft_model_name: Some("draft".to_string()),
        n_draft: 4,
        ..Default::default()
    };
    let msg = match mgr.generate("target", "hello", &params, None, |_| true) {
        Ok(_) => panic!("expected temperature=0.0 enforcement to error"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("temperature") || msg.contains("greedy"),
        "expected temperature-related error, got: {msg}"
    );
}

#[test]
fn spec_decoding_rejects_missing_draft_model() {
    let mgr = ModelManager::new(512, Some(0), 0).unwrap();
    let params = GenerateParams {
        max_tokens: 4,
        temperature: 0.0,
        draft_model_name: Some("nonexistent_draft".to_string()),
        n_draft: 4,
        ..Default::default()
    };
    let msg = match mgr.generate("nonexistent_target", "hello", &params, None, |_| true) {
        Ok(_) => panic!("expected missing-model error"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("not loaded"),
        "expected 'not loaded' error, got: {msg}"
    );
}

#[test]
#[ignore] // requires downloaded model: cargo test --test spec_decoding -- --ignored
fn self_speculation_matches_plain_greedy_output() {
    let path = real_gguf_path().expect(
        "GGUF model not found — pull Qwen/Qwen2.5-3B-Instruct-GGUF first or skip",
    );

    let mgr = ModelManager::new(512, None, 0).unwrap();
    // Load the same file twice under different names: one acts as target,
    // the other as draft. Self-speculation always full-accepts every cycle.
    mgr.load_model("target", &path, None).unwrap();
    mgr.load_model("draft", &path, None).unwrap();

    let prompt = "The capital of France is";

    // Baseline: plain greedy generation (no draft).
    let mut baseline = String::new();
    let baseline_params = GenerateParams {
        max_tokens: 16,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        seed: 0,
        ..Default::default()
    };
    mgr.generate("target", prompt, &baseline_params, None, |t| {
        baseline.push_str(t);
        true
    })
    .unwrap();

    // Speculative: same target, draft = self. Output must match.
    let mut spec_out = String::new();
    let spec_params = GenerateParams {
        max_tokens: 16,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        seed: 0,
        draft_model_name: Some("draft".to_string()),
        n_draft: 4,
        ..Default::default()
    };
    mgr.generate("target", prompt, &spec_params, None, |t| {
        spec_out.push_str(t);
        true
    })
    .unwrap();

    assert_eq!(
        baseline, spec_out,
        "self-speculation must reproduce greedy output exactly"
    );
}

#[test]
#[ignore]
fn self_speculation_respects_max_tokens() {
    let path = real_gguf_path().expect("GGUF model not found");

    let mgr = ModelManager::new(512, None, 0).unwrap();
    mgr.load_model("target", &path, None).unwrap();
    mgr.load_model("draft", &path, None).unwrap();

    let params = GenerateParams {
        max_tokens: 7,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        seed: 0,
        draft_model_name: Some("draft".to_string()),
        // Deliberately oversized — verify the loop stops mid-cycle.
        n_draft: 8,
        ..Default::default()
    };
    let mut count = 0u32;
    let result = mgr
        .generate("target", "Hello", &params, None, |_| {
            count += 1;
            true
        })
        .unwrap();

    assert!(
        result.completion_tokens <= 7,
        "completion_tokens={} exceeded max_tokens=7",
        result.completion_tokens
    );
    assert_eq!(count, result.completion_tokens);
}

#[test]
#[ignore]
fn self_speculation_cancel_callback_stops_generation() {
    let path = real_gguf_path().expect("GGUF model not found");

    let mgr = ModelManager::new(512, None, 0).unwrap();
    mgr.load_model("target", &path, None).unwrap();
    mgr.load_model("draft", &path, None).unwrap();

    let params = GenerateParams {
        max_tokens: 100,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        seed: 0,
        draft_model_name: Some("draft".to_string()),
        n_draft: 4,
        ..Default::default()
    };
    let mut count = 0u32;
    let _ = mgr.generate("target", "Count to fifty", &params, None, |_| {
        count += 1;
        false
    });
    // The first cycle may emit up to (n_draft + 1) tokens before the
    // callback gets a chance to stop the loop.
    assert!(
        count <= 1 + params.n_draft,
        "expected cancellation within one verify cycle, got {count} tokens"
    );
}
