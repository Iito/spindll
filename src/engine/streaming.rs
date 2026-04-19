use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;

pub struct GenerateParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: u32,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            seed: 42,
        }
    }
}

pub struct GenerateResult {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// Generate tokens from a prompt, calling `on_token` for each piece of text produced.
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

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(params.temperature),
        LlamaSampler::top_k(params.top_k),
        LlamaSampler::top_p(params.top_p, 1),
        LlamaSampler::dist(params.seed),
    ]);

    let prompt_token_count = tokens.len() as u32;
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

    Ok(GenerateResult {
        prompt_tokens: prompt_token_count,
        completion_tokens,
    })
}
