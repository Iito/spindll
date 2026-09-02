// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

//! Fetch-versus-recompute policy for the KV prompt cache.
//!
//! A cache hit is only worth taking if materialising the blob beats prefilling
//! the same tokens again. That is not a given: on a fast GPU with a
//! zstd-gated disk tier, recomputing can genuinely be cheaper. The answer
//! depends on the model, the machine, and which tier the blob is in, so it has
//! to be measured rather than assumed.
//!
//! # The comparison
//!
//! For a cached prefix of `M` tokens:
//!
//! ```text
//! fetch_s     = M * kv_bytes_per_token / B_eff + latency
//! recompute_s = (M / prefill_tps) * (1 + M * n_layer * n_embd / params) * bias
//! ```
//!
//! Fetch when `fetch_s < recompute_s`. Dividing through by `M` turns this into
//! a comparison of two tokens-per-second figures, which is the useful mental
//! model: fetch if the tier delivers KV faster than the GPU can rebuild it.
//!
//! The `(1 + M * n_layer * n_embd / params)` factor is attention's O(M²) term.
//! For an 8B model `params / (n_layer * n_embd)` is about 61k, so the
//! correction is 3% at a 2k prefix, 13% at 8k, and 54% at 32k. It always
//! pushes toward fetching, because long prefixes are disproportionately
//! expensive to rebuild.
//!
//! # B_eff is not the device bandwidth
//!
//! Both tiers are zstd-compressed and the disk tier is also encrypted, and
//! those stages are serial with the read:
//!
//! ```text
//! 1/B_eff = 1/(ratio * B_link) + 1/B_zstd + 1/B_crypt
//! ```
//!
//! zstd decompresses at roughly 1-1.5 GB/s per core while an NVMe drive reads
//! at 3-7 GB/s, so on the disk tier the CPU is usually the bottleneck and a
//! faster drive buys nothing. This is why [`TierProfile`] holds one measured
//! end-to-end number instead of a device spec: only a real calibration run
//! captures the whole chain.

/// Measured cost of materialising KV from one storage tier.
///
/// `effective_bps` is end-to-end bytes of *usable* KV per second — after
/// decompression and decryption, not the raw device rate. Obtain it by timing
/// a real load, never from a datasheet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TierProfile {
    /// Usable KV bytes per second, end to end.
    pub effective_bps: f64,
    /// Fixed per-lookup cost: seek, index read, network round trip.
    pub latency_s: f64,
}

impl TierProfile {
    /// A tier that has not been calibrated yet. Always loses to recompute, so
    /// an uncalibrated cache degrades to today's behaviour rather than to a
    /// guess.
    pub const UNCALIBRATED: Self = Self {
        effective_bps: 0.0,
        latency_s: 0.0,
    };

    /// KV tokens per second this tier can deliver, ignoring fixed latency.
    fn tokens_per_second(&self, kv_bytes_per_token: u64) -> f64 {
        if kv_bytes_per_token == 0 {
            return 0.0;
        }
        self.effective_bps / kv_bytes_per_token as f64
    }
}

/// Per-model, per-machine constants. Computed once at load; none of it varies
/// per request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CachePolicy {
    /// From `backend::llamacpp::kv_bytes_per_token`.
    pub kv_bytes_per_token: u64,
    /// Measured prefill throughput, the `prompt_eval_tps` field of `bench --json`.
    pub prefill_tps: f64,
    /// Total model parameters, for the attention correction term.
    pub params: u64,
    pub n_layer: u32,
    pub n_embd: u32,
    /// Prefixes shorter than this are never fetched: below a few hundred
    /// tokens the index lookup and allocation swamp both sides of the
    /// comparison, so the arithmetic is measuring noise. Mirrors the MLX
    /// side's `minPrefix`.
    pub min_prefix_tokens: u32,
    /// Multiplies the recompute cost to account for the GPU time it steals.
    ///
    /// This is the knob that chooses what you are optimising. See
    /// [`throughput_bias`] — 1.0 gives the latency-optimal answer for a single
    /// request, higher values give the throughput-optimal one for a busy
    /// server.
    pub contention_bias: f64,
}

/// Recompute bias for a server optimising aggregate tokens per second.
///
/// The wall-clock comparison in this module answers "which is faster for this
/// one request", which is the right question only on an idle machine. On a
/// busy one the scarce resource is GPU time: a prefill occupies the GPU
/// exclusively, so every sequence currently decoding stalls for its duration,
/// while a fetch costs I/O bandwidth and a CPU core that were sitting idle
/// anyway. Measured in GPU-seconds a fetch is close to free.
///
/// First-order model: prefill blocks all `active_sequences + 1` sequences
/// sharing the device, so its throughput cost is that multiple of its solo
/// cost. Crude, but it has the right shape — the busier the server, the more
/// aggressively it should prefer fetching — and it beats a hardcoded constant.
///
/// Capped at 8.0 so a burst of traffic cannot justify an arbitrarily slow
/// fetch; past that point the fix is a faster cache, not a bigger multiplier.
pub fn throughput_bias(active_sequences: u32) -> f64 {
    (f64::from(active_sequences) + 1.0).min(8.0)
}

impl CachePolicy {
    /// Recompute cost in seconds for `tokens`, including the attention term
    /// and the contention bias.
    fn recompute_seconds(&self, tokens: u32) -> f64 {
        if self.prefill_tps <= 0.0 {
            return f64::INFINITY;
        }
        let m = f64::from(tokens);
        let linear = m / self.prefill_tps;
        let quadratic = if self.params == 0 {
            1.0
        } else {
            let per_token = f64::from(self.n_layer) * f64::from(self.n_embd);
            1.0 + (m * per_token) / self.params as f64
        };
        linear * quadratic * self.contention_bias.max(1.0)
    }

    /// Fetch cost in seconds for `tokens` from `tier`.
    fn fetch_seconds(&self, tier: &TierProfile, tokens: u32) -> f64 {
        if tier.effective_bps <= 0.0 {
            return f64::INFINITY;
        }
        let bytes = f64::from(tokens) * self.kv_bytes_per_token as f64;
        bytes / tier.effective_bps + tier.latency_s
    }

    /// Decide whether to materialise `cached_tokens` from `tier` or rebuild them.
    pub fn decide(&self, tier: &TierProfile, cached_tokens: u32) -> Decision {
        if cached_tokens < self.min_prefix_tokens {
            return Decision::Recompute {
                reason: RecomputeReason::BelowFloor,
                fetch_s: f64::INFINITY,
                recompute_s: self.recompute_seconds(cached_tokens),
            };
        }

        let fetch_s = self.fetch_seconds(tier, cached_tokens);
        let recompute_s = self.recompute_seconds(cached_tokens);

        if !fetch_s.is_finite() {
            return Decision::Recompute {
                reason: RecomputeReason::Uncalibrated,
                fetch_s,
                recompute_s,
            };
        }
        if fetch_s < recompute_s {
            Decision::Fetch {
                fetch_s,
                recompute_s,
            }
        } else {
            Decision::Recompute {
                reason: RecomputeReason::SlowerThanPrefill,
                fetch_s,
                recompute_s,
            }
        }
    }

    /// Shortest prefix for which fetching from `tier` wins, or `None` when it
    /// never can.
    ///
    /// `None` is the signal to stop consulting this tier entirely: if its
    /// per-token rate loses to prefill, no prefix length rescues it, because
    /// both sides grow at least linearly in `M`. Useful at load time to skip
    /// writing to a tier that will never be read back profitably.
    pub fn break_even_tokens(&self, tier: &TierProfile) -> Option<u32> {
        let fetch_tps = tier.tokens_per_second(self.kv_bytes_per_token);
        if fetch_tps <= 0.0 || self.prefill_tps <= 0.0 || fetch_tps <= self.prefill_tps {
            return None;
        }
        // Solving M/fetch_tps + latency = M/prefill_tps for M, ignoring the
        // quadratic term — it only ever helps fetching, so this is a
        // conservative (over-)estimate of the break-even point.
        let per_token_gain = 1.0 / self.prefill_tps - 1.0 / fetch_tps;
        let n = tier.latency_s / per_token_gain;
        let floored = n.ceil().max(f64::from(self.min_prefix_tokens));
        if floored > f64::from(u32::MAX) {
            return None;
        }
        Some(floored as u32)
    }
}

/// Why a prefix is being rebuilt instead of loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecomputeReason {
    /// Shorter than `min_prefix_tokens`.
    BelowFloor,
    /// The tier has no measured throughput yet.
    Uncalibrated,
    /// Measured, and slower than rebuilding.
    SlowerThanPrefill,
}

/// The outcome, carrying both estimates so callers can log what they compared.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decision {
    Fetch {
        fetch_s: f64,
        recompute_s: f64,
    },
    Recompute {
        reason: RecomputeReason,
        fetch_s: f64,
        recompute_s: f64,
    },
}

impl Decision {
    pub fn is_fetch(&self) -> bool {
        matches!(self, Decision::Fetch { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Llama-3-8B: 32 layers, d_model 4096, 8 KV heads, head_dim 128.
    /// kv_bytes_per_token = 2 * 32 * 8 * 128 * 2 = 131072.
    fn llama8b(prefill_tps: f64) -> CachePolicy {
        CachePolicy {
            kv_bytes_per_token: 131_072,
            prefill_tps,
            params: 8_030_000_000,
            n_layer: 32,
            n_embd: 4096,
            min_prefix_tokens: 512,
            contention_bias: 1.0,
        }
    }

    fn tier(gb_per_s: f64, latency_ms: f64) -> TierProfile {
        TierProfile {
            effective_bps: gb_per_s * 1e9,
            latency_s: latency_ms / 1000.0,
        }
    }

    #[test]
    fn below_the_floor_never_fetches() {
        let p = llama8b(1500.0);
        let d = p.decide(&tier(50.0, 0.01), 64);
        assert_eq!(
            d,
            Decision::Recompute {
                reason: RecomputeReason::BelowFloor,
                fetch_s: f64::INFINITY,
                recompute_s: p.recompute_seconds(64),
            }
        );
    }

    #[test]
    fn uncalibrated_tier_degrades_to_recompute() {
        let p = llama8b(1500.0);
        match p.decide(&TierProfile::UNCALIBRATED, 4096) {
            Decision::Recompute { reason, .. } => {
                assert_eq!(reason, RecomputeReason::Uncalibrated)
            }
            other => panic!("expected recompute, got {other:?}"),
        }
    }

    #[test]
    fn ram_tier_wins_easily() {
        // zstd-gated RAM: ~1.5 GB/s usable, negligible latency.
        let p = llama8b(1500.0);
        assert!(p.decide(&tier(1.5, 0.01), 2048).is_fetch());
    }

    #[test]
    fn slow_network_tier_never_wins() {
        // 1 GbE: 0.125 GB/s / 131072 B = ~950 tok/s, under a 1500 tok/s prefill.
        let p = llama8b(1500.0);
        assert!(!p.decide(&tier(0.125, 20.0), 8192).is_fetch());
        assert_eq!(p.break_even_tokens(&tier(0.125, 20.0)), None);
    }

    #[test]
    fn fast_gpu_flips_the_disk_tier_to_recompute() {
        // The same zstd-gated disk that wins on a laptop loses on a big GPU:
        // 0.8 GB/s is ~6100 tok/s, against 20k tok/s of prefill.
        let disk = tier(0.8, 0.2);
        assert!(llama8b(1500.0).decide(&disk, 4096).is_fetch());
        assert!(!llama8b(20_000.0).decide(&disk, 4096).is_fetch());
    }

    #[test]
    fn no_gqa_quadruples_kv_and_flips_the_decision() {
        let mut p = llama8b(3000.0);
        let disk = tier(0.8, 0.2);
        assert!(p.decide(&disk, 4096).is_fetch());
        // Same model without GQA: 32 KV heads instead of 8, so four times the
        // bytes to move and a quarter of the fetch throughput.
        p.kv_bytes_per_token = 131_072 * 4;
        assert!(!p.decide(&disk, 4096).is_fetch());
    }

    #[test]
    fn quadratic_term_favours_longer_prefixes() {
        // Same policy, same tier, opposite answers: attention's O(M^2) cost
        // makes a long prefix disproportionately expensive to rebuild.
        let p = llama8b(7500.0);
        let t = tier(0.8, 0.2);
        assert!(!p.decide(&t, 1024).is_fetch());
        assert!(p.decide(&t, 32_768).is_fetch());
    }

    #[test]
    fn contention_bias_pushes_a_marginal_case_to_fetch() {
        let t = tier(0.8, 0.2);
        let mut p = llama8b(8000.0);
        assert!(!p.decide(&t, 2048).is_fetch());
        // Other sequences are decoding, so prefill is dearer than it looks.
        p.contention_bias = 1.5;
        assert!(p.decide(&t, 2048).is_fetch());
    }

    #[test]
    fn break_even_respects_the_floor() {
        // RAM is so much faster than prefill that the raw break-even is a
        // handful of tokens; the floor must still win.
        let p = llama8b(1500.0);
        assert_eq!(p.break_even_tokens(&tier(1.5, 0.01)), Some(512));
    }

    #[test]
    fn break_even_grows_with_latency() {
        // A low floor, so the latency term is what moves the answer rather
        // than being clamped away by min_prefix_tokens.
        let mut p = llama8b(1500.0);
        p.min_prefix_tokens = 64;
        let near = p.break_even_tokens(&tier(1.25, 50.0)).unwrap();
        let far = p.break_even_tokens(&tier(1.25, 400.0)).unwrap();
        assert!(far > near, "expected {far} > {near}");
    }

    #[test]
    fn throughput_bias_scales_with_load_and_caps() {
        assert_eq!(throughput_bias(0), 1.0);
        assert_eq!(throughput_bias(3), 4.0);
        assert_eq!(throughput_bias(100), 8.0);
    }

    #[test]
    fn a_busy_server_fetches_where_an_idle_one_would_rebuild() {
        // The case the latency formula gets wrong: a tier that loses on
        // wall-clock still wins on GPU-seconds once other sequences are
        // decoding, because the prefill would stall all of them.
        let t = tier(0.8, 0.2);
        let mut p = llama8b(20_000.0);
        assert!(!p.decide(&t, 4096).is_fetch());
        p.contention_bias = throughput_bias(6);
        assert!(p.decide(&t, 4096).is_fetch());
    }

    #[test]
    fn decision_reports_both_estimates() {
        let p = llama8b(1500.0);
        match p.decide(&tier(1.5, 0.01), 4096) {
            Decision::Fetch {
                fetch_s,
                recompute_s,
            } => {
                assert!(fetch_s < recompute_s);
                assert!(fetch_s > 0.0);
            }
            other => panic!("expected fetch, got {other:?}"),
        }
    }
}
