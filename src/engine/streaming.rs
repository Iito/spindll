use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;

use super::kv_cache::{KvCache, load_state_from_disk, save_state_to_disk};

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

/// Statistics returned after a generation request completes.
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

    let mut batch = LlamaBatch::new(512, 1);
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

/// Generate with KV cache support. On cache hit, restores saved state instead
/// of re-encoding the prompt. On miss, saves the post-prompt state for next time.
#[tracing::instrument(skip_all, fields(prompt_tokens, completion_tokens, cache_hit))]
pub fn generate_streaming_cached(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    prompt: &str,
    params: &GenerateParams,
    model_name: &str,
    model_digest: &str,
    cache: &KvCache,
    encryption_key: Option<&[u8; 32]>,
    mut on_token: impl FnMut(&str) -> bool,
) -> anyhow::Result<GenerateResult> {
    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    if tokens.is_empty() {
        anyhow::bail!("prompt produced no tokens");
    }

    let prompt_token_count = tokens.len() as u32;
    tracing::Span::current().record("prompt_tokens", prompt_token_count);
    let n_ctx = ctx.n_ctx() as usize;
    let cache_hit;

    // Try loading cached KV state
    if let Some(cache_path) = cache.lookup(prompt, model_name, model_digest, encryption_key) {
        let loaded = if encryption_key.is_some() {
            // Encrypted path: read+decrypt → temp file → state_load_file
            load_state_from_disk(&cache_path, encryption_key).and_then(|plain| {
                let tmp = cache_path.with_extension("tmp");
                std::fs::write(&tmp, &plain).ok()?;
                let result = ctx.state_load_file(&tmp, n_ctx).ok();
                std::fs::remove_file(&tmp).ok();
                result
            })
        } else {
            // Plaintext path: may or may not have magic byte wrapper
            load_state_from_disk(&cache_path, None).and_then(|plain| {
                let tmp = cache_path.with_extension("tmp");
                std::fs::write(&tmp, &plain).ok()?;
                let result = ctx.state_load_file(&tmp, n_ctx).ok();
                std::fs::remove_file(&tmp).ok();
                result
            }).or_else(|| {
                // Fallback: try direct load for legacy cache files without magic byte
                ctx.state_load_file(&cache_path, n_ctx).ok()
            })
        };

        match loaded {
            Some(cached_tokens)
                if cached_tokens.len() == tokens.len()
                    && cached_tokens.iter().zip(tokens.iter()).all(|(a, b)| a == b) =>
            {
                cache_hit = true;
            }
            _ => {
                cache_hit = false;
                encode_prompt(model, ctx, &tokens)?;
            }
        }
    } else {
        cache_hit = false;
        encode_prompt(model, ctx, &tokens)?;
    }

    tracing::Span::current().record("cache_hit", cache_hit);

    // Save KV state on miss
    if !cache_hit {
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

    // Need an initial logits position — on cache hit we need to re-eval the last token
    if cache_hit {
        let mut batch = LlamaBatch::new(1, 1);
        batch.add(tokens[tokens.len() - 1], n_cur - 1, &[0], true)?;
        ctx.decode(&mut batch)?;
    }

    for _ in 0..params.max_tokens {
        let token = sampler.sample(ctx, 0);
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

        let mut batch = LlamaBatch::new(1, 1);
        batch.add(token, n_cur, &[0], true)?;
        ctx.decode(&mut batch)?;
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
    let mut batch = LlamaBatch::new(512, 1);
    let last_idx = tokens.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens.iter()) {
        batch.add(*token, i, &[0], i == last_idx)?;
    }
    ctx.decode(&mut batch)?;
    Ok(())
}
