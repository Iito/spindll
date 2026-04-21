use std::collections::HashMap;
use std::sync::mpsc;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;

use super::streaming::GenerateParams;

/// Event produced by the batch scheduler for a single sequence.
pub enum BatchEvent {
    /// A decoded text fragment.
    Token(String),
    /// Generation finished for this sequence.
    Done {
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    /// An error occurred processing this sequence.
    Error(String),
}

/// A request submitted to the batch scheduler.
pub struct BatchRequest {
    pub prompt: String,
    pub params: GenerateParams,
    pub response_tx: tokio::sync::mpsc::Sender<BatchEvent>,
}

/// Per-sequence state inside the scheduler.
struct Slot {
    seq_id: i32,
    /// Current position in the KV cache for this sequence.
    pos: i32,
    /// Token to feed in the next generation step (None during prefill).
    pending_token: Option<LlamaToken>,
    sampler: LlamaSampler,
    decoder: encoding_rs::Decoder,
    response_tx: tokio::sync::mpsc::Sender<BatchEvent>,
    max_tokens: u32,
    prompt_tokens: u32,
    completion_tokens: u32,
    prefill_only: bool,
    /// Prompt tokens waiting to be encoded. Cleared after first decode.
    prefill_buf: Vec<LlamaToken>,
}

/// Multiplexes concurrent inference requests through a single llama.cpp context
/// using per-request sequence IDs and batched forward passes.
pub struct BatchScheduler;

impl BatchScheduler {
    /// Run the decode loop on the calling thread (blocks forever until the
    /// request channel is dropped).
    ///
    /// * `max_sequences` — number of concurrent request slots.
    /// * `n_ctx` — total context window shared across all sequences.
    pub fn run(
        model: &LlamaModel,
        backend: &LlamaBackend,
        n_ctx: u32,
        max_sequences: usize,
        request_rx: mpsc::Receiver<BatchRequest>,
    ) -> anyhow::Result<()> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx);
        let mut ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("failed to create batch context: {e}"))?;

        let mut slots: HashMap<i32, Slot> = HashMap::new();
        let mut free_ids: Vec<i32> = (0..max_sequences as i32).rev().collect();

        loop {
            // Block when idle, waiting for the first request.
            if slots.is_empty() {
                match request_rx.recv() {
                    Ok(req) => try_accept(&mut slots, &mut free_ids, model, req),
                    Err(_) => break, // channel closed — shut down
                }
            }

            // Drain any additional queued requests into free slots.
            while !free_ids.is_empty() {
                match request_rx.try_recv() {
                    Ok(req) => try_accept(&mut slots, &mut free_ids, model, req),
                    Err(_) => break,
                }
            }

            if slots.is_empty() {
                continue;
            }

            // -- build batch --------------------------------------------------
            let mut batch = LlamaBatch::new(n_ctx as usize, 1);
            // (batch_index, seq_id) for every token added with logits = true
            let mut logit_map: Vec<(i32, i32)> = Vec::new();
            let mut batch_idx: i32 = 0;

            for slot in slots.values_mut() {
                if !slot.prefill_buf.is_empty() {
                    // Prefill: add all prompt tokens in one go.
                    let last = slot.prefill_buf.len() - 1;
                    for (i, &tok) in slot.prefill_buf.iter().enumerate() {
                        let logits = i == last;
                        if batch.add(tok, slot.pos + i as i32, &[slot.seq_id], logits).is_err() {
                            break; // batch capacity reached
                        }
                        if logits {
                            logit_map.push((batch_idx, slot.seq_id));
                        }
                        batch_idx += 1;
                    }
                    slot.prompt_tokens = slot.prefill_buf.len() as u32;
                    slot.pos += slot.prefill_buf.len() as i32;
                    slot.prefill_buf.clear();
                } else if let Some(tok) = slot.pending_token.take() {
                    // Generation: feed the previously sampled token.
                    if batch.add(tok, slot.pos, &[slot.seq_id], true).is_err() {
                        // Put it back — we'll try next iteration.
                        slot.pending_token = Some(tok);
                        continue;
                    }
                    logit_map.push((batch_idx, slot.seq_id));
                    slot.pos += 1;
                    batch_idx += 1;
                }
            }

            if batch.n_tokens() == 0 {
                continue;
            }

            // -- decode -------------------------------------------------------
            if let Err(e) = ctx.decode(&mut batch) {
                tracing::error!("batch decode failed: {e}");
                let ids: Vec<i32> = slots.keys().copied().collect();
                for id in ids {
                    finish_slot(&mut slots, &mut free_ids, &mut ctx, id,
                        Some(format!("decode error: {e}")));
                }
                continue;
            }

            // -- sample each sequence -----------------------------------------
            let mut finished: Vec<i32> = Vec::new();

            for &(bi, seq_id) in &logit_map {
                let slot = match slots.get_mut(&seq_id) {
                    Some(s) => s,
                    None => continue,
                };

                // Prefill-only requests are done after the first decode.
                if slot.prefill_only {
                    let _ = slot.response_tx.blocking_send(BatchEvent::Done {
                        prompt_tokens: slot.prompt_tokens,
                        completion_tokens: 0,
                    });
                    finished.push(seq_id);
                    continue;
                }

                let token = slot.sampler.sample(&ctx, bi);
                slot.sampler.accept(token);

                if model.is_eog_token(token) || slot.completion_tokens >= slot.max_tokens {
                    let _ = slot.response_tx.blocking_send(BatchEvent::Done {
                        prompt_tokens: slot.prompt_tokens,
                        completion_tokens: slot.completion_tokens,
                    });
                    finished.push(seq_id);
                    continue;
                }

                slot.completion_tokens += 1;
                match model.token_to_piece(token, &mut slot.decoder, true, None) {
                    Ok(piece) => {
                        if slot.response_tx.blocking_send(BatchEvent::Token(piece)).is_err() {
                            // Client disconnected.
                            finished.push(seq_id);
                            continue;
                        }
                    }
                    Err(e) => {
                        let _ = slot.response_tx.blocking_send(
                            BatchEvent::Error(format!("token decode: {e}")),
                        );
                        finished.push(seq_id);
                        continue;
                    }
                }

                slot.pending_token = Some(token);
            }

            // -- clean up finished sequences ----------------------------------
            for id in finished {
                finish_slot(&mut slots, &mut free_ids, &mut ctx, id, None);
            }
        }

        Ok(())
    }
}

/// Tokenize and accept a request into a free slot.
fn try_accept(
    slots: &mut HashMap<i32, Slot>,
    free_ids: &mut Vec<i32>,
    model: &LlamaModel,
    req: BatchRequest,
) {
    let seq_id = match free_ids.pop() {
        Some(id) => id,
        None => {
            // No free slots — reject immediately.
            let _ = req.response_tx.blocking_send(BatchEvent::Error(
                "all sequence slots are occupied".into(),
            ));
            return;
        }
    };

    let tokens = match model.str_to_token(&req.prompt, AddBos::Always) {
        Ok(t) if !t.is_empty() => t,
        Ok(_) => {
            let _ = req.response_tx.blocking_send(BatchEvent::Error(
                "prompt produced no tokens".into(),
            ));
            free_ids.push(seq_id);
            return;
        }
        Err(e) => {
            let _ = req.response_tx.blocking_send(BatchEvent::Error(
                format!("tokenization failed: {e}"),
            ));
            free_ids.push(seq_id);
            return;
        }
    };

    let sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(req.params.temperature),
        LlamaSampler::top_k(req.params.top_k),
        LlamaSampler::top_p(req.params.top_p, 1),
        LlamaSampler::dist(req.params.seed),
    ]);

    slots.insert(seq_id, Slot {
        seq_id,
        pos: 0,
        pending_token: None,
        sampler,
        decoder: encoding_rs::UTF_8.new_decoder(),
        response_tx: req.response_tx,
        max_tokens: req.params.max_tokens,
        prompt_tokens: 0,
        completion_tokens: 0,
        prefill_only: req.params.prefill_only,
        prefill_buf: tokens,
    });

    tracing::debug!(seq_id, "sequence slot claimed");
}

/// Remove a slot, free its sequence ID, and clear its KV cache entries.
fn finish_slot(
    slots: &mut HashMap<i32, Slot>,
    free_ids: &mut Vec<i32>,
    ctx: &mut LlamaContext,
    seq_id: i32,
    error: Option<String>,
) {
    if let Some(slot) = slots.remove(&seq_id) {
        if let Some(msg) = error {
            let _ = slot.response_tx.blocking_send(BatchEvent::Error(msg));
        }
        let _ = ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
        free_ids.push(seq_id);
        tracing::debug!(seq_id, "sequence slot released");
    }
}
