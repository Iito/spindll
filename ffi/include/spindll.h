// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>
//
// spindll C API: score text with a small language model, for keyboards and
// other apps that rank text instead of generating it.
//
// Errors: functions return NULL or a negative number on failure, and
// spindll_last_error() describes the most recent failure on the calling
// thread. That string stays valid until the next failure on the same thread.
//
// Threads: a scorer runs its calls one at a time, so it can be shared across
// threads. Token ids are int32_t. A "token flags" array holds one bool per
// token id: spindll_scorer_n_vocab() entries.
//
// Pointers: every pointer must be valid for the length documented next to
// it. Pointers to zero-length arrays may be NULL.

#ifndef SPINDLL_H
#define SPINDLL_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SpindllScorer SpindllScorer;

typedef struct SpindllScorerConfig {
    /// KV cache size in tokens: the context plus every candidate's tokens.
    uint32_t n_ctx;
    /// Tokens per decode. Also caps the logits buffer at n_batch rows.
    uint32_t n_batch;
    /// Candidates scored side by side.
    uint32_t parallel;
    /// Context tokens kept: the start token and the most recent ones.
    uint32_t max_context_tokens;
    /// Candidates longer than this, in tokens, score -INFINITY.
    uint32_t max_candidate_tokens;
    int32_t threads;
    /// Layers on the GPU. 0 keeps everything on the CPU.
    uint32_t gpu_layers;
} SpindllScorerConfig;

/// Fills `config` with the defaults, which suit a keyboard extension running
/// Gemma 3 270M under iOS's ~76 MB limit.
void spindll_scorer_config_default(SpindllScorerConfig *config);

/// Loads a GGUF model. `config` may be NULL for the defaults.
/// Returns NULL on failure.
SpindllScorer *spindll_scorer_load(const char *path, const SpindllScorerConfig *config);

/// Frees the scorer and its model. NULL is a no-op.
void spindll_scorer_free(SpindllScorer *scorer);

/// The most recent failure on this thread, or NULL if there was none.
const char *spindll_last_error(void);

int32_t spindll_scorer_n_vocab(const SpindllScorer *scorer);
int32_t spindll_scorer_bos(const SpindllScorer *scorer);
int32_t spindll_scorer_eos(const SpindllScorer *scorer);

/// The text of `token`, as UTF-8 without a NUL terminator. Writes at most
/// `capacity` bytes to `buffer` and returns the full length, or -1.
int32_t spindll_scorer_piece(SpindllScorer *scorer, int32_t token, char *buffer, int32_t capacity);

/// Tokenizes NUL-terminated UTF-8 `text`. Writes at most `capacity` tokens to
/// `tokens` and returns the full count, or -1.
int32_t spindll_scorer_tokenize(SpindllScorer *scorer, const char *text, bool add_bos,
                                int32_t *tokens, int32_t capacity);

/// Makes `text` the context for later calls. Only the tokens past the prefix
/// it shares with the previous context are decoded. Returns 0, or -1.
int32_t spindll_scorer_set_context(SpindllScorer *scorer, const char *text);

/// Like spindll_scorer_set_context, from `count` token ids.
int32_t spindll_scorer_set_context_tokens(SpindllScorer *scorer, const int32_t *tokens, int32_t count);

/// Writes to `logprobs` the log-probability of each of `count` tokens coming
/// right after the context. Returns 0, or -1.
int32_t spindll_scorer_next_logprobs(SpindllScorer *scorer, const int32_t *tokens, int32_t count,
                                     float *logprobs);

/// The `k` most likely next tokens, best first, among `allowed` (token
/// flags, or NULL for every token). `tokens` and `logprobs` hold `k` entries.
/// Returns how many were written, or -1.
int32_t spindll_scorer_top_next(SpindllScorer *scorer, int32_t k, const bool *allowed,
                                int32_t *tokens, float *logprobs);

/// For each of `count` texts, writes to `scores` log P(text right after the
/// context), plus log P(any of `terminators` right after it) when
/// `terminator_count` > 0. Returns 0, or -1.
int32_t spindll_scorer_score(SpindllScorer *scorer, const char *const *texts, int32_t count,
                             const int32_t *terminators, int32_t terminator_count, float *scores);

/// Extends each of `count` start tokens greedily, at most `max_steps` tokens,
/// while the most likely next token is flagged in `keep_going` (token flags).
/// Run i is written to tokens[i * (max_steps + 1)] onward, starting with its
/// start token, and lengths[i] is its length. `tokens` holds
/// count * (max_steps + 1) entries. Returns 0, or -1.
int32_t spindll_scorer_extend_greedy(SpindllScorer *scorer, const int32_t *starts, int32_t count,
                                     int32_t max_steps, const bool *keep_going, int32_t *tokens,
                                     int32_t *lengths);

#ifdef __cplusplus
}
#endif

#endif
