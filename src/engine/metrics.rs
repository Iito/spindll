use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Lightweight operational metrics for the inference engine.
/// All counters are monotonically increasing.
pub struct Metrics {
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub total_prompt_tokens: AtomicU64,
    pub total_completion_tokens: AtomicU64,
    pub total_generate_time_us: AtomicU64,
    pub total_prefill_time_us: AtomicU64,
    pub generate_requests: AtomicU64,
    pub generate_errors: AtomicU64,
}

/// Point-in-time snapshot of all metrics counters.
pub struct MetricsSnapshot {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub total_generate_time_us: u64,
    pub total_prefill_time_us: u64,
    pub generate_requests: u64,
    pub generate_errors: u64,
}

impl Metrics {
    /// Create a new metrics instance with all counters at zero.
    pub fn new() -> Self {
        Self {
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            total_completion_tokens: AtomicU64::new(0),
            total_generate_time_us: AtomicU64::new(0),
            total_prefill_time_us: AtomicU64::new(0),
            generate_requests: AtomicU64::new(0),
            generate_errors: AtomicU64::new(0),
        }
    }

    /// Take a point-in-time snapshot of all counters.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            cache_hits: self.cache_hits.load(Relaxed),
            cache_misses: self.cache_misses.load(Relaxed),
            total_prompt_tokens: self.total_prompt_tokens.load(Relaxed),
            total_completion_tokens: self.total_completion_tokens.load(Relaxed),
            total_generate_time_us: self.total_generate_time_us.load(Relaxed),
            total_prefill_time_us: self.total_prefill_time_us.load(Relaxed),
            generate_requests: self.generate_requests.load(Relaxed),
            generate_errors: self.generate_errors.load(Relaxed),
        }
    }

    /// Record a completed generation request.
    pub fn record_generate(&self, prompt_tokens: u64, completion_tokens: u64, elapsed_us: u64, cache_hit: bool) {
        self.generate_requests.fetch_add(1, Relaxed);
        self.total_prompt_tokens.fetch_add(prompt_tokens, Relaxed);
        self.total_completion_tokens.fetch_add(completion_tokens, Relaxed);
        self.total_generate_time_us.fetch_add(elapsed_us, Relaxed);
        if cache_hit {
            self.cache_hits.fetch_add(1, Relaxed);
        } else {
            self.cache_misses.fetch_add(1, Relaxed);
        }
    }

    /// Record a prefill-only request (no tokens generated).
    pub fn record_prefill(&self, prompt_tokens: u64, elapsed_us: u64, cache_hit: bool) {
        self.generate_requests.fetch_add(1, Relaxed);
        self.total_prompt_tokens.fetch_add(prompt_tokens, Relaxed);
        self.total_prefill_time_us.fetch_add(elapsed_us, Relaxed);
        if cache_hit {
            self.cache_hits.fetch_add(1, Relaxed);
        } else {
            self.cache_misses.fetch_add(1, Relaxed);
        }
    }

    /// Increment the error counter.
    pub fn record_error(&self) {
        self.generate_errors.fetch_add(1, Relaxed);
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsSnapshot {
    /// KV cache hit rate as a fraction in `[0.0, 1.0]`. Returns 0.0 if no requests recorded.
    pub fn cache_hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 { 0.0 } else { self.cache_hits as f32 / total as f32 }
    }

    /// Average completion tokens per second across all generation requests.
    pub fn avg_tokens_per_second(&self) -> f32 {
        if self.total_generate_time_us == 0 { 0.0 }
        else { self.total_completion_tokens as f32 / (self.total_generate_time_us as f32 / 1_000_000.0) }
    }
}
