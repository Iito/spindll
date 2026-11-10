// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

//! Scoring text instead of generating it: the next-token distribution after a
//! context, the log-probability of candidate continuations, and short greedy
//! extensions. Built for keyboards, which re-score a growing context on every
//! keystroke under a tight memory limit.
//!
//! The context lives in KV sequence 0 and is reused between calls: setting a
//! new context only decodes the tokens past the longest prefix it shares with
//! the previous one. Candidates are scored side by side, each in its own
//! sequence copied from sequence 0, so they share the context's cache cells.
//!
//! A worker thread owns the model and its context, because the context
//! borrows the model and has to outlive any single call. A [`Scorer`] sends
//! it jobs and waits for each reply, so calls run one at a time and the
//! handle can be used from any thread.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::mpsc;
use std::thread::JoinHandle;

use anyhow::{Context as _, anyhow, ensure};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::LlamaToken;

use crate::backend::llamacpp::shared_backend;

/// Sizes for a scoring session. The defaults suit a keyboard extension
/// running Gemma 3 270M under iOS's ~76 MB limit.
#[derive(Debug, Clone)]
pub struct ScorerConfig {
    /// KV cache size in tokens: the context plus every candidate's tokens.
    pub n_ctx: u32,
    /// Tokens per decode. It also caps the logits buffer at `n_batch` rows
    /// of the vocabulary, one of the main memory costs after the weights.
    pub n_batch: u32,
    /// Candidates scored side by side, each in its own KV sequence.
    pub parallel: u32,
    /// Context tokens kept: the start-of-text token and the most recent ones.
    pub max_context_tokens: usize,
    /// Candidates longer than this, in tokens, score negative infinity.
    pub max_candidate_tokens: usize,
    pub threads: i32,
    pub gpu_layers: u32,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            n_ctx: 256,
            n_batch: 32,
            parallel: 10,
            max_context_tokens: 96,
            max_candidate_tokens: 10,
            threads: 2,
            gpu_layers: 0,
        }
    }
}

impl ScorerConfig {
    fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.n_batch > 0, "n_batch must be at least 1");
        ensure!(self.parallel > 0, "parallel must be at least 1");
        ensure!(self.max_context_tokens >= 2, "max_context_tokens must be at least 2");
        ensure!(self.max_candidate_tokens > 0, "max_candidate_tokens must be at least 1");
        // Candidates share the context's cells, so the cache needs the
        // context plus one full group of candidates.
        let needed = self.max_context_tokens + self.parallel as usize * self.max_candidate_tokens;
        ensure!(
            needed <= self.n_ctx as usize,
            "n_ctx {} is smaller than the {needed} tokens the context and a group of candidates need",
            self.n_ctx
        );
        Ok(())
    }
}

type Job = Box<dyn for<'m> FnOnce(&mut Session<'m>) + Send>;

/// A scoring session on one model. See the module docs.
pub struct Scorer {
    jobs: Option<mpsc::Sender<Job>>,
    worker: Option<JoinHandle<()>>,
    n_vocab: usize,
    bos: i32,
    eos: i32,
}

impl Scorer {
    /// Load `path` (a GGUF file) and start the session's worker thread.
    pub fn load(path: &Path, config: ScorerConfig) -> anyhow::Result<Self> {
        config.validate()?;
        let (jobs, inbox) = mpsc::channel::<Job>();
        let (ready_tx, ready) = mpsc::sync_channel::<anyhow::Result<(usize, i32, i32)>>(1);
        let path = path.to_path_buf();

        let worker = std::thread::Builder::new()
            .name("spindll-scorer".into())
            .spawn(move || {
                let params = LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers);
                let model = match LlamaModel::load_from_file(shared_backend(), &path, &params) {
                    Ok(model) => model,
                    Err(e) => {
                        let _ = ready_tx.send(Err(anyhow!("failed to load {}: {e}", path.display())));
                        return;
                    }
                };
                let mut session = match Session::new(&model, config) {
                    Ok(session) => session,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let vocab = model.vocab();
                let meta = (session.n_vocab, vocab.bos().0, vocab.eos().0);
                if ready_tx.send(Ok(meta)).is_err() {
                    return;
                }
                for job in inbox {
                    job(&mut session);
                }
            })
            .context("failed to start the scorer thread")?;

        let (n_vocab, bos, eos) = ready.recv().map_err(|_| anyhow!("scorer thread exited during load"))??;
        Ok(Self { jobs: Some(jobs), worker: Some(worker), n_vocab, bos, eos })
    }

    /// Run `f` on the worker and wait for its result.
    fn call<T: Send + 'static>(
        &self,
        f: impl for<'m> FnOnce(&mut Session<'m>) -> T + Send + 'static,
    ) -> anyhow::Result<T> {
        let (reply_tx, reply) = mpsc::sync_channel(1);
        let job: Job = Box::new(move |session| {
            let _ = reply_tx.send(f(session));
        });
        let stopped = || anyhow!("scorer thread stopped");
        self.jobs.as_ref().ok_or_else(stopped)?.send(job).map_err(|_| stopped())?;
        reply.recv().map_err(|_| stopped())
    }

    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    pub fn bos(&self) -> i32 {
        self.bos
    }

    pub fn eos(&self) -> i32 {
        self.eos
    }

    /// The text of every token, indexed by token id, for building the token
    /// sets that [`score`](Self::score) and [`extend_greedy`](Self::extend_greedy) take.
    pub fn pieces(&self) -> anyhow::Result<Vec<String>> {
        self.call(|s| s.pieces())
    }

    pub fn tokenize(&self, text: &str, add_bos: bool) -> anyhow::Result<Vec<i32>> {
        let text = text.to_owned();
        self.call(move |s| ids(&s.tokenize(&text, add_bos)))
    }

    /// Make `text` the context for the next calls. Only the tokens past the
    /// prefix it shares with the previous context are decoded.
    pub fn set_context(&self, text: &str) -> anyhow::Result<()> {
        let text = text.to_owned();
        self.call(move |s| {
            let tokens = s.tokenize(&text, true);
            s.set_context(tokens)
        })?
    }

    /// Like [`set_context`](Self::set_context), from token ids.
    pub fn set_context_tokens(&self, tokens: &[i32]) -> anyhow::Result<()> {
        let tokens: Vec<LlamaToken> = tokens.iter().map(|&t| LlamaToken(t)).collect();
        self.call(move |s| s.set_context(tokens))?
    }

    /// Log-probability of each token coming right after the context.
    pub fn next_logprobs(&self, tokens: &[i32]) -> anyhow::Result<Vec<f32>> {
        let tokens = tokens.to_vec();
        self.call(move |s| s.next_logprobs(&tokens))?
    }

    /// The `k` most likely next tokens, best first, among `allowed` if given
    /// (a flag per token id).
    pub fn top_next(&self, k: usize, allowed: Option<&[bool]>) -> anyhow::Result<Vec<(i32, f32)>> {
        let allowed = allowed.map(<[bool]>::to_vec);
        self.call(move |s| s.top_next(k, allowed.as_deref()))?
    }

    /// Log-probability of each text coming next, then one of `terminators`
    /// (e.g. the tokens that end a word). With no terminators, just the text.
    /// Texts longer than `max_candidate_tokens` score negative infinity.
    pub fn score(&self, texts: &[&str], terminators: &[i32]) -> anyhow::Result<Vec<f32>> {
        let texts: Vec<String> = texts.iter().map(|t| (*t).to_owned()).collect();
        let terminators = terminators.to_vec();
        self.call(move |s| s.score(&texts, &terminators))?
    }

    /// Extend each start token greedily, at most `max_steps` tokens, while
    /// the most likely next token is flagged in `keep_going`. Each result
    /// begins with its start token.
    pub fn extend_greedy(
        &self,
        starts: &[i32],
        max_steps: usize,
        keep_going: &[bool],
    ) -> anyhow::Result<Vec<Vec<i32>>> {
        let starts = starts.to_vec();
        let keep_going = keep_going.to_vec();
        self.call(move |s| s.extend_greedy(&starts, max_steps, &keep_going))?
    }
}

impl Drop for Scorer {
    fn drop(&mut self) {
        // Closing the channel ends the worker's loop; it frees the context
        // and then the model.
        self.jobs.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// One token to decode: where it goes and whether its logits are needed.
struct Item {
    token: LlamaToken,
    pos: i32,
    seq: i32,
    output: bool,
}

/// The worker's state. Lives on the worker thread only.
struct Session<'m> {
    model: &'m LlamaModel,
    ctx: LlamaContext<'m>,
    /// Built by `LlamaBatch::new`, so it owns its buffers and borrows nothing.
    batch: LlamaBatch<'static>,
    config: ScorerConfig,
    n_vocab: usize,
    /// Tokens in sequence 0.
    context: Vec<LlamaToken>,
    /// Log-probabilities of the token right after `context`.
    next: Vec<f32>,
}

impl<'m> Session<'m> {
    fn new(model: &'m LlamaModel, config: ScorerConfig) -> anyhow::Result<Self> {
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(config.n_ctx))
            .with_n_batch(config.n_batch)
            .with_n_ubatch(config.n_batch)
            .with_n_seq_max(config.parallel + 1)
            .with_kv_unified(true)
            .with_n_threads(config.threads)
            .with_n_threads_batch(config.threads);
        let ctx = model
            .new_context(shared_backend(), params)
            .map_err(|e| anyhow!("failed to create the scoring context: {e}"))?;
        let n_vocab = usize::try_from(model.vocab().n_tokens()).context("negative vocabulary size")?;
        Ok(Self {
            model,
            ctx,
            batch: LlamaBatch::new(config.n_batch as usize, 1),
            config,
            n_vocab,
            context: Vec::new(),
            next: Vec::new(),
        })
    }

    fn tokenize(&self, text: &str, add_bos: bool) -> Vec<LlamaToken> {
        self.model.vocab().tokenize(text.as_bytes(), add_bos, false)
    }

    fn pieces(&self) -> Vec<String> {
        let vocab = self.model.vocab();
        (0..self.n_vocab as i32)
            .map(|t| String::from_utf8_lossy(&vocab.token_to_piece(LlamaToken(t), false, None)).into_owned())
            .collect()
    }

    fn set_context(&mut self, tokens: Vec<LlamaToken>) -> anyhow::Result<()> {
        let tokens = keep_recent(tokens, self.config.max_context_tokens);
        ensure!(!tokens.is_empty(), "the context produced no tokens");

        let mut common = common_prefix(&self.context, &tokens);
        if common == tokens.len() && common == self.context.len() && !self.next.is_empty() {
            return Ok(());
        }
        // The last token always runs again: its output is the prediction.
        common = common.min(tokens.len() - 1);

        self.clear_candidates()?;
        self.ctx
            .clear_kv_cache_seq(Some(0), Some(common as u32), None)
            .map_err(|e| anyhow!("failed to trim the context: {e}"))?;
        self.context.truncate(common);
        self.next.clear();

        let last = tokens.len() - 1;
        let items: Vec<Item> = (common..tokens.len())
            .map(|i| Item { token: tokens[i], pos: i as i32, seq: 0, output: i == last })
            .collect();
        let n_vocab = self.n_vocab;
        let mut next = Vec::new();
        if let Err(e) = self.run(&items, |_, logits| next = log_softmax(&logits[..n_vocab])) {
            self.ctx.clear_kv_cache();
            self.context.clear();
            return Err(e);
        }
        self.context = tokens;
        self.next = next;
        Ok(())
    }

    fn require_context(&self) -> anyhow::Result<()> {
        ensure!(!self.next.is_empty(), "set a context first");
        Ok(())
    }

    fn next_logprobs(&self, tokens: &[i32]) -> anyhow::Result<Vec<f32>> {
        self.require_context()?;
        Ok(tokens.iter().map(|&t| self.logprob(t)).collect())
    }

    fn logprob(&self, token: i32) -> f32 {
        usize::try_from(token)
            .ok()
            .and_then(|t| self.next.get(t).copied())
            .unwrap_or(f32::NEG_INFINITY)
    }

    fn top_next(&self, k: usize, allowed: Option<&[bool]>) -> anyhow::Result<Vec<(i32, f32)>> {
        self.require_context()?;
        let mut top: Vec<(i32, f32)> = self
            .next
            .iter()
            .enumerate()
            .filter(|(t, _)| allowed.is_none_or(|a| a.get(*t).copied().unwrap_or(false)))
            .map(|(t, &lp)| (t as i32, lp))
            .collect();
        top.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        top.truncate(k);
        Ok(top)
    }

    fn score(&mut self, texts: &[String], terminators: &[i32]) -> anyhow::Result<Vec<f32>> {
        self.require_context()?;
        let mut results = vec![f32::NEG_INFINITY; texts.len()];
        let base = self.context.len() as i32;
        let n_vocab = self.n_vocab;
        let parallel = self.config.parallel as usize;

        for start in (0..texts.len()).step_by(parallel) {
            self.clear_candidates()?;
            let mut items = Vec::new();
            // For each item: the text it belongs to and its index in that text.
            let mut owners = Vec::new();
            let mut tokens_of: Vec<Vec<LlamaToken>> = vec![Vec::new(); texts.len()];

            for (slot, w) in (start..texts.len().min(start + parallel)).enumerate() {
                let tokens = self.tokenize(&texts[w], false);
                if tokens.is_empty() || tokens.len() > self.config.max_candidate_tokens {
                    continue;
                }
                results[w] = self.logprob(tokens[0].0);
                let seq = slot as i32 + 1;
                self.ctx
                    .copy_kv_cache_seq(0, seq, None, None)
                    .map_err(|e| anyhow!("failed to branch the context: {e}"))?;
                for (k, &token) in tokens.iter().enumerate() {
                    items.push(Item { token, pos: base + k as i32, seq, output: true });
                    owners.push((w, k));
                }
                tokens_of[w] = tokens;
            }

            self.run(&items, |i, logits| {
                let (w, k) = owners[i];
                let tokens = &tokens_of[w];
                let logits = &logits[..n_vocab];
                let norm = log_sum_exp(logits);
                if k + 1 < tokens.len() {
                    results[w] += logits[tokens[k + 1].0 as usize] - norm;
                } else if !terminators.is_empty() {
                    results[w] += log_sum_exp_of(logits, terminators) - norm;
                }
            })?;
        }
        self.clear_candidates()?;
        Ok(results)
    }

    fn extend_greedy(
        &mut self,
        starts: &[i32],
        max_steps: usize,
        keep_going: &[bool],
    ) -> anyhow::Result<Vec<Vec<i32>>> {
        self.require_context()?;
        let mut out: Vec<Vec<i32>> = starts.iter().map(|&s| vec![s]).collect();
        let base = self.context.len() as i32;
        let n_vocab = self.n_vocab;
        let parallel = self.config.parallel as usize;

        for start in (0..starts.len()).step_by(parallel) {
            self.clear_candidates()?;
            let group: Vec<usize> = (start..starts.len().min(start + parallel)).collect();
            for slot in 0..group.len() {
                self.ctx
                    .copy_kv_cache_seq(0, slot as i32 + 1, None, None)
                    .map_err(|e| anyhow!("failed to branch the context: {e}"))?;
            }
            let mut active = vec![true; group.len()];
            for step in 0..max_steps {
                let live: Vec<usize> = (0..group.len()).filter(|&slot| active[slot]).collect();
                if live.is_empty() {
                    break;
                }
                let items: Vec<Item> = live
                    .iter()
                    .map(|&slot| Item {
                        token: LlamaToken(*out[group[slot]].last().expect("starts with its start token")),
                        pos: base + step as i32,
                        seq: slot as i32 + 1,
                        output: true,
                    })
                    .collect();
                let mut best = vec![0usize; live.len()];
                self.run(&items, |i, logits| best[i] = argmax(&logits[..n_vocab]))?;
                for (i, &slot) in live.iter().enumerate() {
                    if keep_going.get(best[i]).copied().unwrap_or(false) {
                        out[group[slot]].push(best[i] as i32);
                    } else {
                        active[slot] = false;
                    }
                }
            }
        }
        self.clear_candidates()?;
        Ok(out)
    }

    fn clear_candidates(&mut self) -> anyhow::Result<()> {
        for seq in 1..=self.config.parallel {
            self.ctx
                .clear_kv_cache_seq(Some(seq), None, None)
                .map_err(|e| anyhow!("failed to clear candidate sequence {seq}: {e}"))?;
        }
        Ok(())
    }

    /// Decode `items` in batch-sized chunks. `on_output` gets each item's
    /// index and its logits, for the items that asked for output.
    fn run(&mut self, items: &[Item], mut on_output: impl FnMut(usize, &[f32])) -> anyhow::Result<()> {
        for (chunk_index, chunk) in items.chunks(self.config.n_batch as usize).enumerate() {
            let offset = chunk_index * self.config.n_batch as usize;
            self.batch.clear();
            let mut rows = Vec::new();
            for (k, item) in chunk.iter().enumerate() {
                self.batch
                    .add(item.token, item.pos, &[item.seq], item.output)
                    .map_err(|e| anyhow!("failed to fill the batch: {e}"))?;
                if item.output {
                    rows.push((offset + k, k as i32));
                }
            }
            self.ctx.decode(&mut self.batch).map_err(|e| anyhow!("decode failed: {e}"))?;
            for (item, row) in rows {
                on_output(item, self.ctx.get_logits_ith(row));
            }
        }
        Ok(())
    }
}

fn ids(tokens: &[LlamaToken]) -> Vec<i32> {
    tokens.iter().map(|t| t.0).collect()
}

/// Keep the first token (start of text) and the most recent ones.
fn keep_recent<T: Copy>(tokens: Vec<T>, max: usize) -> Vec<T> {
    if tokens.len() <= max {
        return tokens;
    }
    let mut kept = Vec::with_capacity(max);
    kept.push(tokens[0]);
    kept.extend_from_slice(&tokens[tokens.len() - (max - 1)..]);
    kept
}

fn common_prefix<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn log_sum_exp(xs: &[f32]) -> f32 {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return max;
    }
    max + xs.iter().map(|x| (x - max).exp()).sum::<f32>().ln()
}

fn log_sum_exp_of(logits: &[f32], tokens: &[i32]) -> f32 {
    let picked: Vec<f32> = tokens
        .iter()
        .filter_map(|&t| usize::try_from(t).ok().and_then(|t| logits.get(t).copied()))
        .collect();
    log_sum_exp(&picked)
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let norm = log_sum_exp(logits);
    logits.iter().map(|x| x - norm).collect()
}

/// Index of the largest value; the first one on ties.
fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_recent_keeps_the_first_token_and_the_tail() {
        assert_eq!(keep_recent(vec![1, 2, 3], 5), vec![1, 2, 3]);
        assert_eq!(keep_recent(vec![1, 2, 3, 4, 5, 6], 4), vec![1, 4, 5, 6]);
    }

    #[test]
    fn common_prefix_counts_matching_leading_items() {
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 4]), 2);
        assert_eq!(common_prefix(&[1, 2], &[1, 2, 3]), 2);
        assert_eq!(common_prefix::<i32>(&[], &[1]), 0);
    }

    #[test]
    fn log_sum_exp_matches_the_naive_sum_and_stays_finite() {
        let xs = [1.0f32, 2.0, 3.0];
        let naive = xs.iter().map(|x| x.exp()).sum::<f32>().ln();
        assert!((log_sum_exp(&xs) - naive).abs() < 1e-5);
        // Large values would overflow exp() without the max shift.
        assert!(log_sum_exp(&[1000.0, 1000.0]).is_finite());
        assert_eq!(log_sum_exp(&[]), f32::NEG_INFINITY);
    }

    #[test]
    fn log_softmax_is_a_distribution() {
        let total: f32 = log_softmax(&[0.5, -1.0, 2.0, 0.0]).iter().map(|x| x.exp()).sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn log_sum_exp_of_ignores_out_of_range_tokens() {
        let logits = [0.0f32, 1.0, 2.0];
        let expected = log_sum_exp(&[0.0, 2.0]);
        assert!((log_sum_exp_of(&logits, &[0, 2, 7, -1]) - expected).abs() < 1e-6);
    }

    #[test]
    fn argmax_takes_the_first_maximum() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn config_must_fit_the_context_and_a_group_of_candidates() {
        assert!(ScorerConfig::default().validate().is_ok());
        let tight = ScorerConfig { n_ctx: 100, ..ScorerConfig::default() };
        assert!(tight.validate().is_err());
    }

    /// Model-backed tests read a small GGUF (e.g. Gemma 3 270M) from
    /// `SPINDLL_SCORER_GGUF`: `cargo test scorer -- --ignored`.
    fn scorer() -> Scorer {
        let path = std::env::var("SPINDLL_SCORER_GGUF").expect("set SPINDLL_SCORER_GGUF to a small GGUF");
        Scorer::load(Path::new(&path), ScorerConfig::default()).unwrap()
    }

    fn assert_close(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            // Batch shape changes the CPU kernels' summation order, so a
            // token decoded alone and in a batch can differ in the last bits.
            assert!((x - y).abs() < 2e-2, "{what}[{i}]: {x} vs {y}");
        }
    }

    const CONTEXT: &str = "I would like to go to the";
    const WORDS: [&str; 4] = [" park", " beach", " moon", " xylophonist"];

    #[test]
    #[ignore] // requires a downloaded model
    fn reusing_a_prefix_gives_the_same_prediction_as_a_fresh_context() {
        let s = scorer();
        let probe: Vec<i32> = (0..s.n_vocab() as i32).step_by(97).collect();
        // Typing extends the context, a backspace-and-retype changes it
        // mid-way, and deleting shrinks it. Each must leave no stale cells.
        for text in ["I would like", CONTEXT, "I would like to see the", "I would like"] {
            s.set_context(text).unwrap();
            let fresh = scorer();
            fresh.set_context(text).unwrap();
            let reused = s.next_logprobs(&probe).unwrap();
            assert_close(&reused, &fresh.next_logprobs(&probe).unwrap(), text);
        }
    }

    #[test]
    #[ignore] // requires a downloaded model
    fn scores_match_decoding_each_token_in_turn() {
        let s = scorer();
        let context = s.tokenize(CONTEXT, true).unwrap();
        let terminators = s.tokenize(" ,.", false).unwrap();

        s.set_context_tokens(&context).unwrap();
        let scored = s.score(&WORDS, &terminators).unwrap();

        let mut expected = Vec::new();
        for word in WORDS {
            let tokens = s.tokenize(word, false).unwrap();
            let mut total = 0.0;
            let mut prefix = context.clone();
            for &token in &tokens {
                s.set_context_tokens(&prefix).unwrap();
                total += s.next_logprobs(&[token]).unwrap()[0];
                prefix.push(token);
            }
            s.set_context_tokens(&prefix).unwrap();
            let ends = s.next_logprobs(&terminators).unwrap();
            expected.push(total + log_sum_exp(&ends));
        }
        assert_close(&scored, &expected, "score");
    }

    #[test]
    #[ignore] // requires a downloaded model
    fn grouping_candidates_does_not_change_their_scores() {
        let s = scorer();
        s.set_context(CONTEXT).unwrap();
        // 15 candidates: more than one group of `parallel` (10).
        let words: Vec<String> = (0..15).map(|i| format!(" {}", ["park", "beach", "moon"][i % 3])).collect();
        let words: Vec<&str> = words.iter().map(String::as_str).collect();
        let together = s.score(&words, &[]).unwrap();
        let alone: Vec<f32> = words.iter().map(|w| s.score(&[w], &[]).unwrap()[0]).collect();
        assert_close(&together, &alone, "grouped");
        assert!(together[0] > together[2], "park should beat moon after \"go to the\"");
    }

    #[test]
    #[ignore] // requires a downloaded model
    fn a_long_context_keeps_the_start_token_and_the_most_recent_ones() {
        let s = scorer();
        let long = "The quick brown fox jumps over the lazy dog. ".repeat(20);
        let tokens = s.tokenize(&long, true).unwrap();
        assert!(tokens.len() > 96);

        s.set_context(&long).unwrap();
        let probe: Vec<i32> = (0..s.n_vocab() as i32).step_by(97).collect();
        let windowed = s.next_logprobs(&probe).unwrap();

        let kept = keep_recent(tokens, 96);
        assert_eq!(kept[0], s.bos());
        let reference = scorer();
        reference.set_context_tokens(&kept).unwrap();
        assert_close(&windowed, &reference.next_logprobs(&probe).unwrap(), "windowed");
    }

    #[test]
    #[ignore] // requires a downloaded model
    fn greedy_extension_follows_the_most_likely_token() {
        let s = scorer();
        s.set_context(CONTEXT).unwrap();
        let start = s.top_next(1, None).unwrap()[0].0;
        let everything = vec![true; s.n_vocab()];
        let extended = s.extend_greedy(&[start], 1, &everything).unwrap();
        assert_eq!(extended[0][0], start);

        let mut prefix = s.tokenize(CONTEXT, true).unwrap();
        prefix.push(start);
        s.set_context_tokens(&prefix).unwrap();
        assert_eq!(extended[0].get(1).copied(), Some(s.top_next(1, None).unwrap()[0].0));

        let nothing = vec![false; s.n_vocab()];
        s.set_context(CONTEXT).unwrap();
        assert_eq!(s.extend_greedy(&[start], 3, &nothing).unwrap()[0], vec![start]);
    }

    #[test]
    #[ignore] // requires a downloaded model
    fn scoring_before_a_context_is_an_error() {
        let s = scorer();
        assert!(s.score(&["x"], &[]).is_err());
        assert!(s.next_logprobs(&[0]).is_err());
    }
}
