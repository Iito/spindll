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
        }
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
