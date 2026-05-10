use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Monotonic counter for LRU ordering (avoids timestamp granularity issues).
static ACCESS_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_access_id() -> u64 {
    ACCESS_COUNTER.fetch_add(1, Ordering::Relaxed)
}

struct KvRamEntry {
    /// zstd-compressed KV state blob (from `copy_state_data`, then compressed).
    compressed_state: Vec<u8>,
    /// Token IDs associated with this state, for validation on restore.
    tokens: Vec<i32>,
    /// Compressed size in bytes (== `compressed_state.len()`).
    size_bytes: u64,
    /// Monotonic access ID for LRU ordering.
    last_used: u64,
}

/// In-memory LRU cache of zstd-compressed GGUF KV state blobs.
///
/// Sits in front of the disk-backed [`super::KvCache`] to eliminate all I/O on
/// a RAM hit. State is stored compressed (zstd level 3) to minimise footprint
/// in system RAM. On a hit, `lookup` decompresses on the fly and returns raw
/// bytes ready for `set_state_data`.
pub struct KvRamCache {
    entries: Mutex<HashMap<String, KvRamEntry>>,
    max_bytes: u64,
}


impl KvRamCache {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            max_bytes,
        }
    }

    /// Compute the RAM cache key (no encryption_key — RAM entries are plaintext).
    pub fn hash_key(prompt: &str, model_name: &str, model_digest: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(model_name.as_bytes());
        hasher.update(b"\x00");
        hasher.update(model_digest.as_bytes());
        hasher.update(b"\x00");
        hasher.update(prompt.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Look up a cached entry. Returns `(decompressed raw state, token IDs)` on
    /// hit, updating the LRU timestamp. Returns `None` on miss or if
    /// decompression fails.
    pub fn lookup(&self, hash: &str) -> Option<(Vec<u8>, Vec<i32>)> {
        let mut entries = self.entries.lock().unwrap();
        let entry = entries.get_mut(hash)?;
        entry.last_used = next_access_id();
        let decompressed =
            zstd::decode_all(std::io::Cursor::new(&entry.compressed_state)).ok()?;
        tracing::debug!(hash = &hash[..12], "kv ram cache hit");
        Some((decompressed, entry.tokens.clone()))
    }

    /// Insert a new entry. `raw_state` is the uncompressed output from
    /// `copy_state_data`; it is compressed with zstd before storing. Evicts LRU
    /// entries if the budget would be exceeded.
    pub fn insert(&self, hash: &str, raw_state: &[u8], tokens: &[i32]) {
        let compressed = match zstd::encode_all(std::io::Cursor::new(raw_state), 3) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "kv ram cache: zstd compress failed, skipping insert");
                return;
            }
        };
        let size = compressed.len() as u64;

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
            compressed_mb = size / 1_048_576,
            "kv ram cache insert"
        );
        entries.insert(
            hash.to_string(),
            KvRamEntry {
                compressed_state: compressed,
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

    fn total_bytes(entries: &HashMap<String, KvRamEntry>) -> u64 {
        entries.values().map(|e| e.size_bytes).sum()
    }

    fn evict_for(entries: &mut HashMap<String, KvRamEntry>, max_bytes: u64, needed: u64) {
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

    fn make_state(size: usize) -> Vec<u8> {
        (0..size)
            .map(|i| ((i as f32 * 0.0137).sin() * 127.0) as u8)
            .collect()
    }

    #[test]
    fn insert_and_lookup_roundtrip() {
        let cache = KvRamCache::new(1_048_576); // 1 MB
        let state = make_state(4096);
        let tokens = vec![1i32, 2, 3, 42];
        let hash = KvRamCache::hash_key("hello", "model", "digest");

        cache.insert(&hash, &state, &tokens);
        let (loaded, loaded_tokens) = cache.lookup(&hash).expect("should hit");

        assert_eq!(state, loaded);
        assert_eq!(tokens, loaded_tokens);
    }

    #[test]
    fn lookup_miss_returns_none() {
        let cache = KvRamCache::new(1_048_576);
        assert!(cache.lookup("nonexistent").is_none());
    }

    #[test]
    fn lru_eviction() {
        let state = make_state(1024);
        let compressed_size = zstd::encode_all(std::io::Cursor::new(&state), 3)
            .unwrap()
            .len() as u64;

        // Budget fits exactly 2 entries.
        let cache = KvRamCache::new(compressed_size * 2);

        cache.insert("a", &state, &[1]);
        cache.insert("b", &state, &[2]);
        cache.insert("c", &state, &[3]); // should evict "a"

        assert!(cache.lookup("a").is_none(), "a should be evicted");
        assert!(cache.lookup("b").is_some());
        assert!(cache.lookup("c").is_some());
    }

    #[test]
    fn lru_promotes_on_access() {
        let state = make_state(1024);
        let compressed_size = zstd::encode_all(std::io::Cursor::new(&state), 3)
            .unwrap()
            .len() as u64;

        let cache = KvRamCache::new(compressed_size * 2);

        cache.insert("a", &state, &[1]);
        cache.insert("b", &state, &[2]);

        // Access "a" to make it most-recently-used.
        cache.lookup("a");

        // Insert "c" — should evict "b" (the LRU), not "a".
        cache.insert("c", &state, &[3]);

        assert!(cache.lookup("a").is_some(), "a should survive (MRU)");
        assert!(cache.lookup("b").is_none(), "b should be evicted (LRU)");
        assert!(cache.lookup("c").is_some());
    }

    #[test]
    fn entry_too_large_for_budget_is_skipped() {
        let cache = KvRamCache::new(16); // tiny budget
        let state = make_state(4096);

        cache.insert("big", &state, &[1]);
        assert!(cache.lookup("big").is_none());

        let (count, used, _) = cache.stats();
        assert_eq!(count, 0);
        assert_eq!(used, 0);
    }

    #[test]
    fn stats_reports_correctly() {
        let cache = KvRamCache::new(1_048_576);
        let state = make_state(1024);

        cache.insert("x", &state, &[1]);
        let (count, used, max) = cache.stats();

        assert_eq!(count, 1);
        assert!(used > 0);
        assert_eq!(max, 1_048_576);
    }

    #[test]
    fn clear_empties_cache() {
        let cache = KvRamCache::new(1_048_576);
        cache.insert("x", &make_state(1024), &[1]);
        cache.insert("y", &make_state(1024), &[2]);

        cache.clear();

        let (count, used, _) = cache.stats();
        assert_eq!(count, 0);
        assert_eq!(used, 0);
    }

    #[test]
    fn hash_key_excludes_encryption() {
        let h1 = KvRamCache::hash_key("prompt", "model", "digest");
        let h2 = KvRamCache::hash_key("prompt", "model", "digest");
        assert_eq!(h1, h2);

        // Different from KvCache::hash_prompt which includes encryption_key.
        let h_disk = super::super::kv_cache::KvCache::hash_prompt(
            "prompt",
            "model",
            "digest",
            Some(&[0xAA; 32]),
        );
        assert_ne!(h1, h_disk);
    }

    #[test]
    fn replace_existing_key() {
        let cache = KvRamCache::new(1_048_576);
        let state1 = make_state(1024);
        let state2 = make_state(2048);

        cache.insert("k", &state1, &[1]);
        cache.insert("k", &state2, &[2, 3]);

        let (loaded, tokens) = cache.lookup("k").unwrap();
        assert_eq!(loaded, state2);
        assert_eq!(tokens, vec![2, 3]);

        let (count, _, _) = cache.stats();
        assert_eq!(count, 1);
    }
}
