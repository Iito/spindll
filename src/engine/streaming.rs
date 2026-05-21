use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;

use super::kv_cache::{KvCache, load_state_from_disk, save_state_to_disk};
use super::kv_ram_cache::KvRamCache;

/// Sampling and generation parameters for a single inference request.
pub struct GenerateParams {
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature (higher = more random). 0.0 is greedy.
    pub temperature: f32,
    /// Nucleus sampling threshold.
    pub top_p: f32,
    /// Top-K sampling cutoff.
    pub top_k: i32,
    /// RNG seed for reproducible output.
    pub seed: u32,
    /// If `true`, only encode the prompt into the KV cache without generating tokens.
    pub prefill_only: bool,
    /// Name of a separately-loaded GGUF model to use as the draft for
    /// speculative decoding. The draft must share the target's vocabulary.
    /// Only takes effect on the GGUF direct path with greedy sampling
    /// (`temperature == 0.0`); other paths fall back to plain generation.
    pub draft_model_name: Option<String>,
    /// Number of tokens the draft model proposes per verification cycle.
    /// Ignored when `draft_model_name` is `None`.
    pub n_draft: u32,
    /// N-gram speculative draft cap per cycle (no separate draft model).
    /// `0` disables. `draft_model_name` takes precedence if both are set.
    pub n_gram_draft: u32,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            seed: 42,
            prefill_only: false,
            draft_model_name: None,
            n_draft: 5,
            n_gram_draft: 0,
        }
    }
}

/// 2-gram drafter: `(prev, curr) → next` from the most recent occurrence.
#[derive(Default)]
pub struct NGramDrafter {
    cache: std::collections::HashMap<(i32, i32), i32>,
}

impl NGramDrafter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, tokens: &[llama_cpp_2::token::LlamaToken]) {
        for w in tokens.windows(3) {
            self.cache.insert((w[0].0, w[1].0), w[2].0);
        }
    }

    pub fn propose(
        &self,
        tail: &[llama_cpp_2::token::LlamaToken],
        max_drafts: usize,
    ) -> Vec<llama_cpp_2::token::LlamaToken> {
        let mut drafts = Vec::with_capacity(max_drafts);
        if tail.len() < 2 || max_drafts == 0 {
            return drafts;
        }
        let mut a = tail[tail.len() - 2].0;
        let mut b = tail[tail.len() - 1].0;
        for _ in 0..max_drafts {
            match self.cache.get(&(a, b)) {
                Some(&c) => {
                    drafts.push(llama_cpp_2::token::LlamaToken(c));
                    a = b;
                    b = c;
                }
                None => break,
            }
        }
        drafts
    }
}

/// Truncate `tokens` to fit `n_ctx`, reserving `reserve_completion` tokens for
/// generation. Mirrors ollama's behavior: keeps the first `keep` tokens (BOS +
/// system prompt) and the most recent tail.
///
/// Returns `(truncated_tokens, was_truncated)`.
pub(crate) fn truncate_to_fit(
    tokens: Vec<llama_cpp_2::token::LlamaToken>,
    n_ctx: u32,
    reserve_completion: u32,
    keep: usize,
) -> (Vec<llama_cpp_2::token::LlamaToken>, bool) {
    // Reserve space for generation. Clamp into [256, n_ctx/4] so an unset
    // max_tokens (=0) still leaves room to generate, and an absurdly large
    // max_tokens doesn't wipe the prompt entirely.
    let reserve = reserve_completion
        .max(256)
        .min(n_ctx / 4)
        .max(64);
    let limit = n_ctx.saturating_sub(reserve).max(1) as usize;
    if tokens.len() <= limit {
        return (tokens, false);
    }
    let keep = keep.min(limit / 4);
    let tail = limit - keep;
    let mut out = Vec::with_capacity(limit);
    out.extend_from_slice(&tokens[..keep]);
    out.extend_from_slice(&tokens[tokens.len() - tail..]);
    (out, true)
}

/// Statistics returned after a generation request completes.
#[derive(Default)]
pub struct GenerateResult {
    /// Number of tokens in the encoded prompt.
    pub prompt_tokens: u32,
    /// Number of tokens generated (0 if `prefill_only` was set).
    pub completion_tokens: u32,
    /// Whether the KV cache was hit for this prompt.
    pub cache_hit: bool,
    /// Tokens proposed by the draft model (0 if speculative decoding was not used).
    pub spec_drafts_proposed: u32,
    /// Draft tokens accepted by the target model (0 if speculative decoding was not used).
    pub spec_drafts_accepted: u32,
    /// Number of verify cycles run during speculative decoding (0 if not used).
    pub spec_cycles: u32,
}

/// Generate tokens from a prompt, calling `on_token` for each piece of text produced.
#[tracing::instrument(skip_all, fields(prompt_tokens, completion_tokens))]
pub fn generate_streaming(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    prompt: &str,
    params: &GenerateParams,
    mut on_token: impl FnMut(&str) -> bool, // return false to stop
) -> anyhow::Result<GenerateResult> {
    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    if tokens.is_empty() {
        anyhow::bail!("prompt produced no tokens");
    }

    let n_ctx = ctx.n_ctx();
    let original_tokens = tokens.len();
    let (tokens, truncated) = truncate_to_fit(tokens, n_ctx, params.max_tokens, 5);
    tracing::info!(
        original_tokens,
        kept = tokens.len(),
        n_ctx,
        truncated,
        "tokenized prompt"
    );

    let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
    let last_idx = tokens.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens.iter()) {
        batch.add(*token, i, &[0], i == last_idx)?;
    }

    ctx.decode(&mut batch)?;

    let prompt_token_count = tokens.len() as u32;
    tracing::Span::current().record("prompt_tokens", prompt_token_count);

    if params.prefill_only {
        return Ok(GenerateResult {
            prompt_tokens: prompt_token_count,
            completion_tokens: 0,
            cache_hit: false,
            ..Default::default()
        });
    }

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(params.temperature),
        LlamaSampler::top_k(params.top_k),
        LlamaSampler::top_p(params.top_p, 1),
        LlamaSampler::dist(params.seed),
    ]);

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut n_cur = batch.n_tokens();
    let mut completion_tokens = 0u32;

    for _ in 0..params.max_tokens {
        let token = sampler.sample(ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        completion_tokens += 1;
        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| anyhow::anyhow!("token decode error: {e}"))?;

        if !on_token(&piece) {
            break;
        }

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        ctx.decode(&mut batch)?;
        n_cur += 1;
    }

    tracing::Span::current().record("completion_tokens", completion_tokens);

    Ok(GenerateResult {
        prompt_tokens: prompt_token_count,
        completion_tokens,
        cache_hit: false,
        ..Default::default()
    })
}

/// Snapshot the current context state into the RAM cache.
fn snapshot_to_ram(ctx: &LlamaContext, rc: &KvRamCache, hash: &str, tokens: &[i32]) {
    let state_size = ctx.get_state_size();
    let mut buf = vec![0u8; state_size];
    let actual = unsafe { ctx.copy_state_data(buf.as_mut_ptr()) };
    buf.truncate(actual);
    rc.insert(hash, &buf, tokens);
}

/// Generate with KV cache support. On cache hit, restores saved state instead
/// of re-encoding the prompt. On miss, saves the post-prompt state for next time.
///
/// Three-tier lookup: RAM cache → disk cache → full prefill.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, fields(prompt_tokens, completion_tokens, cache_hit))]
pub fn generate_streaming_cached(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    prompt: &str,
    params: &GenerateParams,
    model_name: &str,
    model_digest: &str,
    cache: &KvCache,
    ram_cache: Option<&KvRamCache>,
    encryption_key: Option<&[u8; 32]>,
    mut on_token: impl FnMut(&str) -> bool,
) -> anyhow::Result<GenerateResult> {
    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    if tokens.is_empty() {
        anyhow::bail!("prompt produced no tokens");
    }

    let n_ctx_u32 = ctx.n_ctx();
    let (tokens, truncated) = truncate_to_fit(tokens, n_ctx_u32, params.max_tokens, 5);
    if truncated {
        tracing::warn!(limit = n_ctx_u32, kept = tokens.len(), "truncating input prompt");
    }

    let prompt_token_count = tokens.len() as u32;
    tracing::Span::current().record("prompt_tokens", prompt_token_count);
    let n_ctx = n_ctx_u32 as usize;

    let token_ids: Vec<i32> = tokens.iter().map(|t| t.0).collect();
    let ram_hash = ram_cache
        .map(|_| KvRamCache::hash_key(prompt, model_name, model_digest, encryption_key));

    // ── Tier 1: RAM cache ──────────────────────────────────────────────
    let cache_hit = if let (Some(rc), Some(hash)) = (ram_cache, ram_hash.as_deref()) {
        if let Some((raw_state, cached_tokens)) = rc.lookup(hash) {
            if cached_tokens.len() == token_ids.len()
                && cached_tokens.iter().zip(token_ids.iter()).all(|(a, b)| a == b)
            {
                let read = unsafe { ctx.set_state_data(&raw_state) };
                // Require a full restore — partial reads leave ctx in a
                // half-loaded state and would silently skip prompt prefill.
                if read == raw_state.len() {
                    tracing::debug!("kv cache: ram hit");
                    true
                } else {
                    tracing::warn!(
                        read,
                        expected = raw_state.len(),
                        "kv ram cache: partial restore; falling back to prefill"
                    );
                    ctx.clear_kv_cache();
                    false
                }
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    };

    // ── Tier 2: Disk cache ─────────────────────────────────────────────
    let cache_hit = if cache_hit {
        true
    } else if let Some(cache_path) = cache.lookup(prompt, model_name, model_digest, encryption_key)
    {
        let loaded = if encryption_key.is_some() {
            load_state_from_disk(&cache_path, encryption_key).and_then(|plain| {
                let tmp = cache_path.with_extension("tmp");
                std::fs::write(&tmp, &plain).ok()?;
                let result = ctx.state_load_file(&tmp, n_ctx).ok();
                std::fs::remove_file(&tmp).ok();
                result
            })
        } else {
            load_state_from_disk(&cache_path, None)
                .and_then(|plain| {
                    let tmp = cache_path.with_extension("tmp");
                    std::fs::write(&tmp, &plain).ok()?;
                    let result = ctx.state_load_file(&tmp, n_ctx).ok();
                    std::fs::remove_file(&tmp).ok();
                    result
                })
                .or_else(|| ctx.state_load_file(&cache_path, n_ctx).ok())
        };

        match loaded {
            Some(cached_tokens)
                if cached_tokens.len() == tokens.len()
                    && cached_tokens.iter().zip(tokens.iter()).all(|(a, b)| a == b) =>
            {
                // Promote to RAM cache for next time.
                if let (Some(rc), Some(hash)) = (ram_cache, ram_hash.as_deref()) {
                    snapshot_to_ram(ctx, rc, hash, &token_ids);
                }
                true
            }
            Some(_) => {
                // state_load_file installed stale KV state before token
                // validation rejected it — clear before prefill so the new
                // decode runs on an empty cache instead of a contaminated one.
                ctx.clear_kv_cache();
                encode_prompt(model, ctx, &tokens)?;
                false
            }
            None => {
                encode_prompt(model, ctx, &tokens)?;
                false
            }
        }
    } else {
        encode_prompt(model, ctx, &tokens)?;
        false
    };

    tracing::Span::current().record("cache_hit", cache_hit);

    // ── Save on miss ───────────────────────────────────────────────────
    if !cache_hit {
        // Save to RAM cache.
        if let (Some(rc), Some(hash)) = (ram_cache, ram_hash.as_deref()) {
            snapshot_to_ram(ctx, rc, hash, &token_ids);
        }
        // Save to disk cache.
        let save_path = cache.save_path(prompt, model_name, model_digest, encryption_key);
        let tmp = save_path.with_extension("tmp");
        if ctx.state_save_file(&tmp, &tokens).is_ok() {
            if let Ok(raw) = std::fs::read(&tmp) {
                if save_state_to_disk(&save_path, &raw, encryption_key).is_ok() {
                    cache.register(prompt, model_name, model_digest, encryption_key);
                }
            }
            std::fs::remove_file(&tmp).ok();
        }
    }

    let prompt_token_count_result = prompt_token_count;

    if params.prefill_only {
        return Ok(GenerateResult {
            prompt_tokens: prompt_token_count_result,
            completion_tokens: 0,
            cache_hit,
            ..Default::default()
        });
    }

    // Sample tokens
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(params.temperature),
        LlamaSampler::top_k(params.top_k),
        LlamaSampler::top_p(params.top_p, 1),
        LlamaSampler::dist(params.seed),
    ]);

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut n_cur = tokens.len() as i32;
    let mut completion_tokens = 0u32;

    let mut batch = LlamaBatch::new(tokens.len().max(512), 1);

    // On cache miss, encode_prompt already decoded the full prompt batch and
    // logits sit at index tokens.len()-1.  On cache hit, restored state has
    // no logits so we re-eval the last prompt token in a fresh batch.
    let mut logit_idx = if cache_hit {
        let last_pos = (tokens.len() - 1) as u32;
        ctx.clear_kv_cache_seq(Some(0), Some(last_pos), Some(last_pos + 1))
            .map_err(|e| anyhow::anyhow!("failed to clear last KV position: {e}"))?;
        batch.add(tokens[tokens.len() - 1], n_cur - 1, &[0], true)?;
        ctx.decode(&mut batch)?;
        0
    } else {
        tokens.len() as i32 - 1
    };

    for _ in 0..params.max_tokens {
        let token = sampler.sample(ctx, logit_idx);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        completion_tokens += 1;
        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| anyhow::anyhow!("token decode error: {e}"))?;

        if !on_token(&piece) {
            break;
        }

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        ctx.decode(&mut batch)?;
        logit_idx = 0;
        n_cur += 1;
    }

    tracing::Span::current().record("completion_tokens", completion_tokens);

    Ok(GenerateResult {
        prompt_tokens: prompt_token_count_result,
        completion_tokens,
        cache_hit,
        ..Default::default()
    })
}

/// Encode the full prompt into the context via batch decode.
fn encode_prompt(
    _model: &LlamaModel,
    ctx: &mut LlamaContext,
    tokens: &[llama_cpp_2::token::LlamaToken],
) -> anyhow::Result<()> {
    let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
    let last_idx = tokens.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens.iter()) {
        batch.add(*token, i, &[0], i == last_idx)?;
    }
    ctx.decode(&mut batch)?;
    Ok(())
}

fn argmax_token(logits: &[f32]) -> llama_cpp_2::token::LlamaToken {
    let mut best_i = 0i32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i as i32;
        }
    }
    llama_cpp_2::token::LlamaToken(best_i)
}

/// Vocab compat check matching llama.cpp's `common_speculative_are_compatible`:
/// vocab type + BOS + EOS + token bytes equal for ids below `min(vocab_a, vocab_b)`.
/// Tolerates trailing padding-token differences (e.g. Qwen2.5-7B 152064 vs 0.5B 151936).
fn vocabs_compatible(target: &LlamaModel, draft: &LlamaModel) -> anyhow::Result<()> {
    if target.vocab_type() != draft.vocab_type() {
        anyhow::bail!("vocab_type mismatch");
    }
    if target.token_bos() != draft.token_bos() {
        anyhow::bail!("BOS token mismatch");
    }
    if target.token_eos() != draft.token_eos() {
        anyhow::bail!("EOS token mismatch");
    }
    let n_check = target.n_vocab().min(draft.n_vocab());
    for i in 0..n_check {
        let tok = llama_cpp_2::token::LlamaToken(i);
        let tb = target.token_to_piece_bytes(tok, 8, true, None).ok();
        let db = draft.token_to_piece_bytes(tok, 8, true, None).ok();
        if tb != db {
            anyhow::bail!("vocab token id {} differs between target and draft", i);
        }
    }
    Ok(())
}

/// Returns the top token and the margin between top-1 and top-2 logits.
/// Used by speculative decoding to short-circuit drafts the model is unsure
/// about — mirrors llama.cpp's `p_min` cutoff, the proxy for "this draft is
/// likely to be rejected by the larger target, so don't spend cycles on it."
fn argmax_token_with_margin(logits: &[f32]) -> (llama_cpp_2::token::LlamaToken, f32) {
    let mut best_i = 0i32;
    let mut best_v = f32::NEG_INFINITY;
    let mut second_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            second_v = best_v;
            best_v = v;
            best_i = i as i32;
        } else if v > second_v {
            second_v = v;
        }
    }
    (llama_cpp_2::token::LlamaToken(best_i), best_v - second_v)
}

/// Greedy spec decoding driven by an n-gram drafter (single model, no draft).
/// Caller must enforce greedy sampling (T==0.0).
#[tracing::instrument(skip_all, fields(prompt_tokens, completion_tokens, n_draft))]
pub fn generate_streaming_speculative_ngram(
    target_model: &LlamaModel,
    target_ctx: &mut LlamaContext,
    prompt: &str,
    params: &GenerateParams,
    n_draft: u32,
    mut on_token: impl FnMut(&str) -> bool,
) -> anyhow::Result<GenerateResult> {
    let tokens = target_model.str_to_token(prompt, AddBos::Always)?;
    if tokens.is_empty() {
        anyhow::bail!("prompt produced no tokens");
    }
    let n_ctx = target_ctx.n_ctx();
    let (tokens, truncated) = truncate_to_fit(tokens, n_ctx, params.max_tokens, 5);
    tracing::info!(
        kept = tokens.len(),
        n_ctx,
        truncated,
        "tokenized prompt (ngram-speculative)"
    );

    encode_prompt(target_model, target_ctx, &tokens)?;
    let prompt_token_count = tokens.len() as u32;
    tracing::Span::current().record("prompt_tokens", prompt_token_count);
    tracing::Span::current().record("n_draft", n_draft);

    if params.prefill_only {
        return Ok(GenerateResult {
            prompt_tokens: prompt_token_count,
            completion_tokens: 0,
            cache_hit: false,
            ..Default::default()
        });
    }

    let n_max = n_draft.max(1) as usize;
    let n_min: usize = 1;
    let mut dyn_n = n_max;
    let mut drafter = NGramDrafter::new();
    drafter.update(&tokens);

    let mut history: Vec<llama_cpp_2::token::LlamaToken> = tokens.clone();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut completion_tokens = 0u32;
    let mut spec_proposed = 0u32;
    let mut spec_accepted = 0u32;
    let mut spec_cycles = 0u32;
    let mut n_cur = tokens.len() as i32;
    let mut prev_target_idx: i32 = tokens.len() as i32 - 1;

    'outer: loop {
        if completion_tokens >= params.max_tokens {
            break;
        }

        let drafts = drafter.propose(&history, dyn_n);

        // No drafts → fall back to single-token greedy decode.
        if drafts.is_empty() {
            let tok = argmax_token(target_ctx.get_logits_ith(prev_target_idx));
            if target_model.is_eog_token(tok) {
                break;
            }
            completion_tokens += 1;
            let piece = target_model
                .token_to_piece(tok, &mut decoder, true, None)
                .map_err(|e| anyhow::anyhow!("token decode error: {e}"))?;
            if !on_token(&piece) {
                break;
            }
            history.push(tok);
            drafter.update(&history[history.len().saturating_sub(3)..]);
            let mut b = LlamaBatch::new(1, 1);
            b.add(tok, n_cur, &[0], true)?;
            target_ctx.decode(&mut b)?;
            n_cur += 1;
            prev_target_idx = 0;
            continue;
        }

        spec_cycles += 1;
        spec_proposed += drafts.len() as u32;

        let target_tok_0 = argmax_token(target_ctx.get_logits_ith(prev_target_idx));
        let mut accepted: usize = 0;
        let bonus: llama_cpp_2::token::LlamaToken;
        let mut target_decoded_drafts = false;

        if target_tok_0 != drafts[0] {
            bonus = target_tok_0;
        } else {
            accepted = 1;
            if drafts.len() == 1 {
                let mut b = LlamaBatch::new(1, 1);
                b.add(drafts[0], n_cur, &[0], true)?;
                target_ctx.decode(&mut b)?;
                target_decoded_drafts = true;
                bonus = argmax_token(target_ctx.get_logits_ith(0));
            } else {
                let mut tb = LlamaBatch::new(drafts.len(), 1);
                for (k, t) in drafts.iter().enumerate() {
                    tb.add(*t, n_cur + k as i32, &[0], true)?;
                }
                target_ctx.decode(&mut tb)?;
                target_decoded_drafts = true;
                let mut bonus_opt: Option<llama_cpp_2::token::LlamaToken> = None;
                for i in 0..(drafts.len() - 1) {
                    let pred = argmax_token(target_ctx.get_logits_ith(i as i32));
                    if pred == drafts[i + 1] {
                        accepted += 1;
                    } else {
                        bonus_opt = Some(pred);
                        break;
                    }
                }
                bonus = match bonus_opt {
                    Some(b) => b,
                    None => argmax_token(
                        target_ctx.get_logits_ith((drafts.len() - 1) as i32),
                    ),
                };
            }
        }
        spec_accepted += accepted as u32;

        for i in 0..accepted {
            let t = drafts[i];
            if target_model.is_eog_token(t) {
                break 'outer;
            }
            completion_tokens += 1;
            let piece = target_model
                .token_to_piece(t, &mut decoder, true, None)
                .map_err(|e| anyhow::anyhow!("token decode error: {e}"))?;
            if !on_token(&piece) {
                break 'outer;
            }
            history.push(t);
            if completion_tokens >= params.max_tokens {
                break 'outer;
            }
        }

        if target_model.is_eog_token(bonus) {
            break;
        }
        completion_tokens += 1;
        let piece = target_model
            .token_to_piece(bonus, &mut decoder, true, None)
            .map_err(|e| anyhow::anyhow!("token decode error: {e}"))?;
        if !on_token(&piece) {
            break;
        }
        history.push(bonus);

        let tail_start = history.len().saturating_sub(accepted + 3);
        drafter.update(&history[tail_start..]);

        let kept_end = n_cur + accepted as i32;
        let target_end = if target_decoded_drafts {
            n_cur + drafts.len() as i32
        } else {
            n_cur
        };
        if (kept_end as u32) < target_end as u32 {
            target_ctx
                .clear_kv_cache_seq(Some(0), Some(kept_end as u32), None)
                .map_err(|e| anyhow::anyhow!("target KV rewind failed: {e}"))?;
        }
        let mut tb = LlamaBatch::new(1, 1);
        tb.add(bonus, kept_end, &[0], true)?;
        target_ctx.decode(&mut tb)?;
        n_cur = kept_end + 1;
        prev_target_idx = 0;

        let full_accept = accepted == drafts.len();
        if full_accept {
            dyn_n = (dyn_n + 1).min(n_max);
        } else {
            dyn_n = (dyn_n / 2).max(n_min);
        }

        if completion_tokens >= params.max_tokens {
            break;
        }
    }

    tracing::Span::current().record("completion_tokens", completion_tokens);
    tracing::info!(
        spec_cycles,
        spec_proposed,
        spec_accepted,
        completion_tokens,
        "ngram-spec breakdown"
    );
    Ok(GenerateResult {
        prompt_tokens: prompt_token_count,
        completion_tokens,
        cache_hit: false,
        spec_drafts_proposed: spec_proposed,
        spec_drafts_accepted: spec_accepted,
        spec_cycles,
    })
}

/// Greedy speculative decoding. Draft proposes up to `n_draft` tokens per
/// cycle; target verifies them in a single batched decode and emits the
/// longest accepted prefix plus one bonus token sampled from the target.
///
/// Requires both models share the same vocabulary. Only correct under greedy
/// sampling — callers must restrict to `temperature == 0.0` paths.
#[tracing::instrument(skip_all, fields(prompt_tokens, completion_tokens, n_draft))]
pub fn generate_streaming_speculative(
    target_model: &LlamaModel,
    target_ctx: &mut LlamaContext,
    draft_model: &LlamaModel,
    draft_ctx: &mut LlamaContext,
    prompt: &str,
    params: &GenerateParams,
    n_draft: u32,
    mut on_token: impl FnMut(&str) -> bool,
) -> anyhow::Result<GenerateResult> {
    vocabs_compatible(target_model, draft_model)?;

    let tokens = target_model.str_to_token(prompt, AddBos::Always)?;
    if tokens.is_empty() {
        anyhow::bail!("prompt produced no tokens");
    }

    let n_ctx = target_ctx.n_ctx();
    let original_tokens = tokens.len();
    let (tokens, truncated) = truncate_to_fit(tokens, n_ctx, params.max_tokens, 5);
    tracing::info!(
        original_tokens,
        kept = tokens.len(),
        n_ctx,
        truncated,
        "tokenized prompt (speculative)"
    );

    let t0 = std::time::Instant::now();
    encode_prompt(target_model, target_ctx, &tokens)?;
    let target_prefill_us = t0.elapsed().as_micros() as u64;
    let t0 = std::time::Instant::now();
    encode_prompt(draft_model, draft_ctx, &tokens)?;
    let draft_prefill_us = t0.elapsed().as_micros() as u64;

    let prompt_token_count = tokens.len() as u32;
    tracing::Span::current().record("prompt_tokens", prompt_token_count);
    tracing::Span::current().record("n_draft", n_draft);

    if params.prefill_only {
        return Ok(GenerateResult {
            prompt_tokens: prompt_token_count,
            completion_tokens: 0,
            cache_hit: false,
            ..Default::default()
        });
    }

    let n_max = n_draft.max(1) as usize;
    let n_min: usize = 1;
    // Adaptive bound: shrinks on partial reject, grows on full accept.
    let mut dyn_n = n_max;
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut completion_tokens = 0u32;
    let mut spec_proposed = 0u32;
    let mut spec_accepted = 0u32;
    let mut spec_cycles = 0u32;
    let mut draft_decode_us = 0u64;
    let mut target_verify_us = 0u64;
    let mut bonus_decode_us = 0u64;
    let mut rewind_us = 0u64;
    // Next free KV position on both contexts.
    let mut n_cur = tokens.len() as i32;
    // Batch indices of pre-cycle logits (target = predictions for draft[0];
    // draft = next-token distribution for sampling draft[0]). After prefill
    // the logits live at batch index (prompt_len - 1); after a single-token
    // decode they live at index 0.
    let mut prev_target_idx: i32 = tokens.len() as i32 - 1;
    let mut prev_draft_idx: i32 = tokens.len() as i32 - 1;

    'outer: loop {
        if completion_tokens >= params.max_tokens {
            break;
        }

        // STEP 1: draft N tokens greedy, one at a time. Stop early on EOG
        // OR when the draft's top-1 vs top-2 logit gap is small — low margin
        // means the draft is guessing and the target will likely reject it,
        // so further drafts past this point waste target compute.
        const DRAFT_MARGIN_CUTOFF: f32 = 1.0; // logit units; calibrated empirically
        let mut drafts: Vec<llama_cpp_2::token::LlamaToken> = Vec::with_capacity(dyn_n);
        let mut draft_eog_at: Option<usize> = None;
        let t_draft = std::time::Instant::now();
        for k in 0..dyn_n {
            let logits = draft_ctx.get_logits_ith(prev_draft_idx);
            let (tok, margin) = argmax_token_with_margin(logits);
            // Always accept the first draft regardless of margin; we want at
            // least one proposal per cycle to make the verify batch useful.
            if k > 0 && margin < DRAFT_MARGIN_CUTOFF {
                break;
            }
            drafts.push(tok);
            if target_model.is_eog_token(tok) {
                draft_eog_at = Some(k);
                break;
            }
            let mut b = LlamaBatch::new(1, 1);
            b.add(tok, n_cur + k as i32, &[0], true)?;
            draft_ctx.decode(&mut b)?;
            prev_draft_idx = 0;
        }
        draft_decode_us += t_draft.elapsed().as_micros() as u64;
        if drafts.is_empty() {
            break;
        }
        spec_cycles += 1;
        spec_proposed += drafts.len() as u32;

        // STEP 2: verify draft[0] against target's pre-draft logits.
        // Argmax inline — don't copy the n_vocab-sized slice (~150 KB).
        let target_tok_0 = argmax_token(target_ctx.get_logits_ith(prev_target_idx));

        // If draft[0] is rejected, skip the target verify batch entirely.
        let mut accepted: usize = 0;
        let bonus: llama_cpp_2::token::LlamaToken;
        let mut target_decoded_drafts = false;

        if target_tok_0 != drafts[0] {
            bonus = target_tok_0;
        } else {
            accepted = 1;
            let t_verify = std::time::Instant::now();
            if drafts.len() == 1 {
                // Only one draft and it matched — need target logits at next
                // position to sample the bonus. Decode draft[0] on target.
                let mut b = LlamaBatch::new(1, 1);
                b.add(drafts[0], n_cur, &[0], true)?;
                target_ctx.decode(&mut b)?;
                target_decoded_drafts = true;
                let lg = target_ctx.get_logits_ith(0);
                bonus = argmax_token(lg);
                target_verify_us += t_verify.elapsed().as_micros() as u64;
            } else {
                // Batch-decode all drafts on target with logits everywhere.
                let mut tb = LlamaBatch::new(drafts.len(), 1);
                for (k, t) in drafts.iter().enumerate() {
                    tb.add(*t, n_cur + k as i32, &[0], true)?;
                }
                target_ctx.decode(&mut tb)?;
                target_verify_us += t_verify.elapsed().as_micros() as u64;
                target_decoded_drafts = true;

                // Verify draft[i+1] against logits[i] for i in 0..drafts.len()-1.
                let mut bonus_opt: Option<llama_cpp_2::token::LlamaToken> = None;
                for i in 0..(drafts.len() - 1) {
                    let lg = target_ctx.get_logits_ith(i as i32);
                    let pred = argmax_token(lg);
                    if pred == drafts[i + 1] {
                        accepted += 1;
                    } else {
                        bonus_opt = Some(pred);
                        break;
                    }
                }
                bonus = match bonus_opt {
                    Some(b) => b,
                    None => {
                        // Full accept — bonus from last verify logit.
                        let lg = target_ctx.get_logits_ith((drafts.len() - 1) as i32);
                        argmax_token(lg)
                    }
                };
            }
        }

        // If the draft hit EOG inside the window, cap accepted at the EOG
        // position (don't emit anything past EOG; ignore the would-be bonus).
        let mut emit_bonus = true;
        if let Some(eog_at) = draft_eog_at {
            if accepted > eog_at {
                accepted = eog_at;
                emit_bonus = false;
            }
        }
        spec_accepted += accepted as u32;

        // STEP 3: emit accepted draft tokens.
        for i in 0..accepted {
            let t = drafts[i];
            if target_model.is_eog_token(t) {
                break 'outer;
            }
            completion_tokens += 1;
            let piece = target_model
                .token_to_piece(t, &mut decoder, true, None)
                .map_err(|e| anyhow::anyhow!("token decode error: {e}"))?;
            if !on_token(&piece) {
                break 'outer;
            }
            if completion_tokens >= params.max_tokens {
                break 'outer;
            }
        }

        // STEP 4: emit bonus (unless EOG capped us).
        let kept_end = n_cur + accepted as i32;
        if emit_bonus {
            if target_model.is_eog_token(bonus) {
                break 'outer;
            }
            completion_tokens += 1;
            let piece = target_model
                .token_to_piece(bonus, &mut decoder, true, None)
                .map_err(|e| anyhow::anyhow!("token decode error: {e}"))?;
            if !on_token(&piece) {
                break 'outer;
            }
            if completion_tokens >= params.max_tokens {
                break 'outer;
            }
        } else {
            break 'outer;
        }

        // STEP 5: rewind both KVs to position `kept_end`, then commit bonus
        // at `kept_end` on both with logits=true so the next cycle has
        // verifiable pre-draft logits.
        let t_rewind = std::time::Instant::now();
        let draft_end = n_cur + drafts.len() as i32;
        if (kept_end as u32) < draft_end as u32 {
            draft_ctx
                .clear_kv_cache_seq(Some(0), Some(kept_end as u32), None)
                .map_err(|e| anyhow::anyhow!("draft KV rewind failed: {e}"))?;
        }
        let target_end = if target_decoded_drafts {
            n_cur + drafts.len() as i32
        } else {
            n_cur
        };
        if (kept_end as u32) < target_end as u32 {
            target_ctx
                .clear_kv_cache_seq(Some(0), Some(kept_end as u32), None)
                .map_err(|e| anyhow::anyhow!("target KV rewind failed: {e}"))?;
        }
        rewind_us += t_rewind.elapsed().as_micros() as u64;

        let t_bonus = std::time::Instant::now();
        let mut db = LlamaBatch::new(1, 1);
        db.add(bonus, kept_end, &[0], true)?;
        draft_ctx.decode(&mut db)?;
        let mut tb = LlamaBatch::new(1, 1);
        tb.add(bonus, kept_end, &[0], true)?;
        target_ctx.decode(&mut tb)?;
        bonus_decode_us += t_bonus.elapsed().as_micros() as u64;

        n_cur = kept_end + 1;
        prev_draft_idx = 0;
        prev_target_idx = 0;

        let full_accept = accepted == drafts.len() && draft_eog_at.is_none();
        if full_accept {
            dyn_n = (dyn_n + 1).min(n_max);
        } else {
            dyn_n = (dyn_n / 2).max(n_min);
        }
    }

    tracing::Span::current().record("completion_tokens", completion_tokens);
    tracing::info!(
        target_prefill_ms = target_prefill_us / 1000,
        draft_prefill_ms = draft_prefill_us / 1000,
        draft_decode_ms = draft_decode_us / 1000,
        target_verify_ms = target_verify_us / 1000,
        bonus_decode_ms = bonus_decode_us / 1000,
        rewind_ms = rewind_us / 1000,
        spec_cycles,
        spec_proposed,
        spec_accepted,
        completion_tokens,
        "spec phase breakdown"
    );
    Ok(GenerateResult {
        prompt_tokens: prompt_token_count,
        completion_tokens,
        cache_hit: false,
        spec_drafts_proposed: spec_proposed,
        spec_drafts_accepted: spec_accepted,
        spec_cycles,
    })
}
