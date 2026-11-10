// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Iito <https://github.com/Iito> and sarmientoF <https://github.com/sarmientoF>

//! In-memory KV state cache for the llama.cpp backend.
//!
//! Entries are keyed by exact prompt hash, so hits require the same prompt and
//! do not reuse partial prefixes. The shared `--kv-ram-cache` budget flag also
//! budgets MLX prefix caching, but MLX performs prefix reuse internally.
//!
//! Entries are opaque [`SeqState`] snapshots of sequence 0, taken with
//! `LlamaContext::state_seq_get` and restored with `state_seq_set`. The
//! restore hands llama.cpp the buffer length, so a snapshot that does not fit
//! the context it is restored into (another model under the same name, a
//! smaller context, a different KV type) is rejected instead of being read
//! past its end. Snapshots are stored uncompressed: their bytes are private to
//! `llama-cpp-2`, which is what guarantees they only ever came from llama.cpp.

use llama_cpp_2::SeqState;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Monotonic counter for LRU ordering (avoids timestamp granularity issues).
static ACCESS_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_access_id() -> u64 {
    ACCESS_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// A cached KV snapshot, charged against the budget at its byte length.
///
/// The cache only ever holds [`SeqState`] in production. The trait exists so
/// the LRU and budget logic can be tested without loading a model, since a
/// `SeqState` can only be produced by a live context.
pub trait KvSnapshot {
    fn byte_len(&self) -> usize;
}

impl KvSnapshot for SeqState {
    fn byte_len(&self) -> usize {
        SeqState::byte_len(self)
    }
}

struct KvRamEntry<S> {
    /// Shared so a hit hands out the snapshot without copying it under the lock.
    state: Arc<S>,
    /// Token IDs associated with this state, for validation on restore.
    tokens: Vec<i32>,
    /// Snapshot size in bytes.
    size_bytes: u64,
    /// Monotonic access ID for LRU ordering.
    last_used: u64,
}

/// In-memory LRU cache of KV snapshots.
///
/// Sits in front of the disk-backed [`super::KvCache`] to eliminate all I/O on
/// a RAM hit.
pub struct KvRamCache<S = SeqState> {
    entries: Mutex<HashMap<String, KvRamEntry<S>>>,
    max_bytes: u64,
}

impl KvRamCache {
    /// Compute the RAM cache key.
    ///
    /// `encryption_key` is mixed into the hash even though RAM entries are
    /// stored plaintext — without it, two callers with the same prompt but
    /// different keys would share a cache slot, leaking plaintext KV state
    /// across encryption (tenant) boundaries. Matches `KvCache::hash_prompt`.
    pub fn hash_key(
        prompt: &str,
        model_name: &str,
        model_digest: &str,
        encryption_key: Option<&[u8; 32]>,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(model_name.as_bytes());
        hasher.update(b"\x00");
        hasher.update(model_digest.as_bytes());
        hasher.update(b"\x00");
        if let Some(key) = encryption_key {
            hasher.update(b"enc:");
            hasher.update(key);
            hasher.update(b"\x00");
        }
        hasher.update(prompt.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

impl<S: KvSnapshot> KvRamCache<S> {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            max_bytes,
        }
    }

    /// Look up a cached entry. Returns `(snapshot, token IDs)` on hit,
    /// updating the LRU position, or `None` on miss.
    pub fn lookup(&self, hash: &str) -> Option<(Arc<S>, Vec<i32>)> {
        let mut entries = self.entries.lock().unwrap();
        let entry = entries.get_mut(hash)?;
        entry.last_used = next_access_id();
        tracing::debug!(hash = &hash[..hash.len().min(12)], "kv ram cache hit");
        Some((Arc::clone(&entry.state), entry.tokens.clone()))
    }

    /// Insert a snapshot, evicting LRU entries if the budget would be exceeded.
    /// A snapshot larger than the whole budget is skipped.
    pub fn insert(&self, hash: &str, state: S, tokens: &[i32]) {
        let size = state.byte_len() as u64;

        if size > self.max_bytes {
            tracing::debug!("kv ram cache: entry too large for budget, skipping");
            return;
        }

        let mut entries = self.entries.lock().unwrap();
        // Remove existing entry for this key (if any) before eviction accounting.
        entries.remove(hash);
        Self::evict_for(&mut entries, self.max_bytes, size);

        tracing::debug!(
            hash = &hash[..hash.len().min(12)],
            size_mb = size / 1_048_576,
            "kv ram cache insert"
        );
        entries.insert(
            hash.to_string(),
            KvRamEntry {
                state: Arc::new(state),
                tokens: tokens.to_vec(),
                size_bytes: size,
                last_used: next_access_id(),
            },
        );
    }

    /// Returns `(entry_count, used_bytes, max_bytes)`.
    pub fn stats(&self) -> (usize, u64, u64) {
        let entries = self.entries.lock().unwrap();
        let used = Self::total_bytes(&entries);
        (entries.len(), used, self.max_bytes)
    }

    /// Remove all entries.
    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }

    fn total_bytes(entries: &HashMap<String, KvRamEntry<S>>) -> u64 {
        entries.values().map(|e| e.size_bytes).sum()
    }

    fn evict_for(entries: &mut HashMap<String, KvRamEntry<S>>, max_bytes: u64, needed: u64) {
        loop {
            let used = Self::total_bytes(entries);
            if used + needed <= max_bytes {
                return;
            }
            let lru = entries
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| k.clone());
            match lru {
                Some(key) => {
                    tracing::debug!(hash = &key[..key.len().min(12)], "kv ram cache: evicting");
                    entries.remove(&key);
                }
                None => return,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stand-in for `SeqState`, which only a live context can produce.
    #[derive(Debug, PartialEq)]
    struct FakeState(Vec<u8>);

    impl KvSnapshot for FakeState {
        fn byte_len(&self) -> usize {
            self.0.len()
        }
    }

    fn make_state(size: usize) -> FakeState {
        FakeState((0..size).map(|i| (i % 251) as u8).collect())
    }

    fn cache(max_bytes: u64) -> KvRamCache<FakeState> {
        KvRamCache::new(max_bytes)
    }

    #[test]
    fn insert_and_lookup_roundtrip() {
        let cache = cache(1_048_576); // 1 MB
        let tokens = vec![1i32, 2, 3, 42];
        let hash = KvRamCache::hash_key("hello", "model", "digest", None);

        cache.insert(&hash, make_state(4096), &tokens);
        let (loaded, loaded_tokens) = cache.lookup(&hash).expect("should hit");

        assert_eq!(*loaded, make_state(4096));
        assert_eq!(tokens, loaded_tokens);
    }

    #[test]
    fn lookup_miss_returns_none() {
        assert!(cache(1_048_576).lookup("nonexistent").is_none());
    }

    #[test]
    fn lookup_shares_the_snapshot_instead_of_copying_it() {
        let cache = cache(1_048_576);
        cache.insert("k", make_state(1024), &[1]);

        let (a, _) = cache.lookup("k").unwrap();
        let (b, _) = cache.lookup("k").unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn lru_eviction() {
        // Snapshots are charged at their byte length: this budget fits exactly 2.
        let cache = cache(2048);

        cache.insert("a", make_state(1024), &[1]);
        cache.insert("b", make_state(1024), &[2]);
        cache.insert("c", make_state(1024), &[3]); // should evict "a"

        assert!(cache.lookup("a").is_none(), "a should be evicted");
        assert!(cache.lookup("b").is_some());
        assert!(cache.lookup("c").is_some());
    }

    #[test]
    fn lru_promotes_on_access() {
        let cache = cache(2048);

        cache.insert("a", make_state(1024), &[1]);
        cache.insert("b", make_state(1024), &[2]);

        // Access "a" to make it most-recently-used.
        cache.lookup("a");

        // Insert "c" — should evict "b" (the LRU), not "a".
        cache.insert("c", make_state(1024), &[3]);

        assert!(cache.lookup("a").is_some(), "a should survive (MRU)");
        assert!(cache.lookup("b").is_none(), "b should be evicted (LRU)");
        assert!(cache.lookup("c").is_some());
    }

    #[test]
    fn entry_too_large_for_budget_is_skipped() {
        let cache = cache(16); // tiny budget

        cache.insert("big", make_state(4096), &[1]);
        assert!(cache.lookup("big").is_none());

        let (count, used, _) = cache.stats();
        assert_eq!(count, 0);
        assert_eq!(used, 0);
    }

    #[test]
    fn stats_reports_correctly() {
        let cache = cache(1_048_576);

        cache.insert("x", make_state(1024), &[1]);
        let (count, used, max) = cache.stats();

        assert_eq!(count, 1);
        assert_eq!(used, 1024);
        assert_eq!(max, 1_048_576);
    }

    #[test]
    fn clear_empties_cache() {
        let cache = cache(1_048_576);
        cache.insert("x", make_state(1024), &[1]);
        cache.insert("y", make_state(1024), &[2]);

        cache.clear();

        let (count, used, _) = cache.stats();
        assert_eq!(count, 0);
        assert_eq!(used, 0);
    }

    #[test]
    fn hash_key_includes_encryption() {
        // Two callers with no encryption key share the same slot — expected.
        let h_a = KvRamCache::hash_key("prompt", "model", "digest", None);
        let h_b = KvRamCache::hash_key("prompt", "model", "digest", None);
        assert_eq!(h_a, h_b);

        // Different keys must hash differently — otherwise plaintext KV state
        // leaks across encryption boundaries.
        let h_k1 = KvRamCache::hash_key("prompt", "model", "digest", Some(&[0xAA; 32]));
        let h_k2 = KvRamCache::hash_key("prompt", "model", "digest", Some(&[0xBB; 32]));
        assert_ne!(h_a, h_k1);
        assert_ne!(h_k1, h_k2);
    }

    #[test]
    fn replace_existing_key() {
        let cache = cache(1_048_576);

        cache.insert("k", make_state(1024), &[1]);
        cache.insert("k", make_state(2048), &[2, 3]);

        let (loaded, tokens) = cache.lookup("k").unwrap();
        assert_eq!(*loaded, make_state(2048));
        assert_eq!(tokens, vec![2, 3]);

        let (count, used, _) = cache.stats();
        assert_eq!(count, 1);
        assert_eq!(used, 2048);
    }
}
