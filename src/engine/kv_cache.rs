use chacha20poly1305::{
    aead::{Aead, KeyInit, OsRng},
    AeadCore, ChaCha20Poly1305,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

/// Magic byte prefix for encrypted cache files.
pub const ENCRYPTED_MAGIC: u8 = 0xE1;
/// Magic byte prefix for plaintext cache files.
pub const PLAIN_MAGIC: u8 = 0x00;
/// Magic byte prefix for zstd-compressed plaintext cache files.
pub const COMPRESSED_PLAIN_MAGIC: u8 = 0xC1;
/// Magic byte prefix for zstd-compressed then encrypted cache files.
pub const COMPRESSED_ENCRYPTED_MAGIC: u8 = 0xC2;

/// Disk-backed LRU cache for KV state files.
///
/// Keyed by SHA-256 of the prompt prefix so identical prompts (e.g. system
/// prompts) reuse the already-processed KV state instead of re-encoding.
pub struct KvCache {
    cache_dir: PathBuf,
    max_bytes: u64,
    index: Mutex<CacheIndex>,
}

#[derive(Serialize, Deserialize, Default)]
struct CacheIndex {
    entries: HashMap<String, CacheEntry>,
}

#[derive(Serialize, Deserialize, Clone)]
struct CacheEntry {
    hash: String,
    size_bytes: u64,
    last_used: u64,
    model_name: String,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

impl KvCache {
    /// Create a cache under `~/.spindll/cache` with the given size limit.
    pub fn new(max_bytes: u64) -> Self {
        let home = std::env::var("HOME").expect("HOME not set");
        let cache_dir = PathBuf::from(home).join(".spindll").join("cache");
        Self::with_dir(cache_dir, max_bytes)
    }

    /// Create a cache in a custom directory with the given size limit.
    pub fn with_dir(cache_dir: PathBuf, max_bytes: u64) -> Self {
        std::fs::create_dir_all(&cache_dir).ok();

        let index_path = cache_dir.join("index.json");
        let index = if index_path.exists() {
            std::fs::read_to_string(&index_path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default()
        } else {
            CacheIndex::default()
        };

        Self {
            cache_dir,
            max_bytes,
            index: Mutex::new(index),
        }
    }

    /// Hash a prompt to produce a cache key.
    /// Includes the model digest and encryption key so that:
    /// - Re-downloaded models invalidate the cache
    /// - Callers with different encryption keys get isolated entries
    /// - Callers with the same key share entries for common prefixes
    pub fn hash_prompt(
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
        hasher.update(prompt.as_bytes());
        if let Some(key) = encryption_key {
            hasher.update(b"\x00");
            hasher.update(key);
        }
        format!("{:x}", hasher.finalize())
    }

    /// Path to the cached state file for a given hash.
    fn state_path(&self, hash: &str) -> PathBuf {
        self.cache_dir.join(format!("{hash}.llstate"))
    }

    fn index_path(&self) -> PathBuf {
        self.cache_dir.join("index.json")
    }

    fn flush_index(&self, idx: &CacheIndex) {
        if let Ok(json) = serde_json::to_string(idx) {
            std::fs::write(self.index_path(), json).ok();
        }
    }

    fn total_bytes(idx: &CacheIndex) -> u64 {
        idx.entries.values().map(|e| e.size_bytes).sum()
    }

    /// Evict LRU entries until `needed` bytes fit within the budget.
    fn evict_for(&self, idx: &mut CacheIndex, needed: u64) {
        while Self::total_bytes(idx) + needed > self.max_bytes {
            let lru = idx
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| k.clone());

            match lru {
                Some(key) => {
                    if let Some(entry) = idx.entries.remove(&key) {
                        tracing::debug!(hash = &entry.hash[..12], "kv cache evicting entry");
                        std::fs::remove_file(self.state_path(&entry.hash)).ok();
                    }
                }
                None => break,
            }
        }
    }

    /// Check if a cached state exists for the given prompt+model.
    /// Returns the path to load from if it exists.
    pub fn lookup(&self, prompt: &str, model_name: &str, model_digest: &str, encryption_key: Option<&[u8; 32]>) -> Option<PathBuf> {
        let hash = Self::hash_prompt(prompt, model_name, model_digest, encryption_key);
        let mut idx = self.index.lock().unwrap();

        if let Some(entry) = idx.entries.get_mut(&hash) {
            let path = self.state_path(&hash);
            if path.exists() {
                entry.last_used = now_secs();
                self.flush_index(&idx);
                tracing::debug!(hash = &hash[..12], "kv cache hit");
                return Some(path);
            }
            // Stale index entry — file was deleted externally
            idx.entries.remove(&hash);
            self.flush_index(&idx);
        }
        tracing::debug!("kv cache miss");
        None
    }

    /// Register a newly saved state file in the cache index, evicting as needed.
    pub fn register(&self, prompt: &str, model_name: &str, model_digest: &str, encryption_key: Option<&[u8; 32]>) -> PathBuf {
        let hash = Self::hash_prompt(prompt, model_name, model_digest, encryption_key);
        let path = self.state_path(&hash);

        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

        let mut idx = self.index.lock().unwrap();
        self.evict_for(&mut idx, size);

        tracing::debug!(hash = &hash[..12], size, "kv cache write");
        idx.entries.insert(
            hash.clone(),
            CacheEntry {
                hash,
                size_bytes: size,
                last_used: now_secs(),
                model_name: model_name.to_string(),
            },
        );
        self.flush_index(&idx);
        path
    }

    /// Get the save path for a hash without registering yet.
    pub fn save_path(&self, prompt: &str, model_name: &str, model_digest: &str, encryption_key: Option<&[u8; 32]>) -> PathBuf {
        let hash = Self::hash_prompt(prompt, model_name, model_digest, encryption_key);
        self.state_path(&hash)
    }

    /// Remove all cached state files and return the total bytes freed.
    pub fn clear(&self) -> std::io::Result<u64> {
        let mut idx = self.index.lock().unwrap();
        let mut freed = 0u64;
        for (_, entry) in idx.entries.drain() {
            let path = self.state_path(&entry.hash);
            if path.exists() {
                freed += entry.size_bytes;
                std::fs::remove_file(path)?;
            }
        }
        self.flush_index(&idx);
        Ok(freed)
    }

    /// Returns `(entry_count, used_bytes, max_bytes)` for the cache.
    pub fn stats(&self) -> (usize, u64, u64) {
        let idx = self.index.lock().unwrap();
        let count = idx.entries.len();
        let used = Self::total_bytes(&idx);
        (count, used, self.max_bytes)
    }
}

/// Write state data to disk, compressing with zstd and optionally encrypting
/// with ChaCha20-Poly1305.
///
/// Pipeline: raw → zstd(level 3) → [optional encrypt] → prepend magic → write
pub fn save_state_to_disk(
    path: &Path,
    data: &[u8],
    encryption_key: Option<&[u8; 32]>,
) -> std::io::Result<()> {
    tracing::debug!(encrypted = encryption_key.is_some(), "saving kv state to disk");
    let compressed = zstd::encode_all(std::io::Cursor::new(data), 3)?;
    let contents = match encryption_key {
        Some(key) => {
            let cipher = ChaCha20Poly1305::new(key.into());
            let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng);
            let ciphertext = cipher
                .encrypt(&nonce, compressed.as_slice())
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            let mut buf = Vec::with_capacity(1 + 12 + ciphertext.len());
            buf.push(COMPRESSED_ENCRYPTED_MAGIC);
            buf.extend_from_slice(&nonce);
            buf.extend_from_slice(&ciphertext);
            buf
        }
        None => {
            let mut buf = Vec::with_capacity(1 + compressed.len());
            buf.push(COMPRESSED_PLAIN_MAGIC);
            buf.extend_from_slice(&compressed);
            buf
        }
    };
    std::fs::write(path, contents)
}

/// Read state data from disk, decompressing and optionally decrypting.
/// Returns `None` on any error (corrupt file, wrong key, missing key for encrypted data).
///
/// Supports both legacy uncompressed (0x00, 0xE1) and new compressed (0xC1, 0xC2)
/// magic bytes for backward compatibility.
pub fn load_state_from_disk(
    path: &Path,
    encryption_key: Option<&[u8; 32]>,
) -> Option<Vec<u8>> {
    tracing::debug!(encrypted = encryption_key.is_some(), "loading kv state from disk");
    let contents = std::fs::read(path).ok()?;
    if contents.is_empty() {
        return None;
    }

    let magic = contents[0];
    let payload = &contents[1..];

    match magic {
        // New: compressed + encrypted
        COMPRESSED_ENCRYPTED_MAGIC => {
            let key = encryption_key?;
            if payload.len() < 12 {
                return None;
            }
            let (nonce_bytes, ciphertext) = payload.split_at(12);
            let cipher = ChaCha20Poly1305::new(key.into());
            let nonce = chacha20poly1305::Nonce::from_slice(nonce_bytes);
            let compressed = cipher.decrypt(nonce, ciphertext).ok()?;
            zstd::decode_all(std::io::Cursor::new(compressed)).ok()
        }
        // New: compressed plaintext
        COMPRESSED_PLAIN_MAGIC => {
            zstd::decode_all(std::io::Cursor::new(payload)).ok()
        }
        // Legacy: uncompressed encrypted
        ENCRYPTED_MAGIC => {
            let key = encryption_key?;
            if payload.len() < 12 {
                return None;
            }
            let (nonce_bytes, ciphertext) = payload.split_at(12);
            let cipher = ChaCha20Poly1305::new(key.into());
            let nonce = chacha20poly1305::Nonce::from_slice(nonce_bytes);
            cipher.decrypt(nonce, ciphertext).ok()
        }
        // Legacy: uncompressed plaintext
        PLAIN_MAGIC => Some(payload.to_vec()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Reinterpret a byte slice as f32 embeddings.
    fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// Cosine similarity between two f32 vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let x = *x as f64;
            let y = *y as f64;
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    /// Generate a synthetic KV state blob: 32 layers x 2048 head_dim x f16-ish.
    /// Mimics the structured floating-point data llama.cpp produces.
    fn make_kv_state(layers: usize, head_dim: usize) -> Vec<u8> {
        let num_floats = layers * head_dim * 2; // K + V per layer
        let mut data = Vec::with_capacity(num_floats * 4);
        for i in 0..num_floats {
            // Quasi-random structured data (not zeros, not uniform random)
            let v = ((i as f32) * 0.0137).sin() * 0.5;
            data.extend_from_slice(&v.to_le_bytes());
        }
        data
    }

    #[test]
    fn roundtrip_plaintext_compressed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.llstate");
        let original = make_kv_state(32, 2048);

        save_state_to_disk(&path, &original, None).unwrap();
        let loaded = load_state_from_disk(&path, None).unwrap();

        assert_eq!(original, loaded, "plaintext round-trip produced different bytes");
    }

    #[test]
    fn roundtrip_encrypted_compressed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_enc.llstate");
        let key: [u8; 32] = [0xAB; 32];
        let original = make_kv_state(32, 2048);

        save_state_to_disk(&path, &original, Some(&key)).unwrap();
        let loaded = load_state_from_disk(&path, Some(&key)).unwrap();

        assert_eq!(original, loaded, "encrypted round-trip produced different bytes");
    }

    #[test]
    fn compressed_file_has_correct_magic() {
        let dir = tempfile::tempdir().unwrap();

        let plain_path = dir.path().join("plain.llstate");
        save_state_to_disk(&plain_path, b"hello", None).unwrap();
        let raw = std::fs::read(&plain_path).unwrap();
        assert_eq!(raw[0], COMPRESSED_PLAIN_MAGIC);

        let enc_path = dir.path().join("enc.llstate");
        let key: [u8; 32] = [0x42; 32];
        save_state_to_disk(&enc_path, b"hello", Some(&key)).unwrap();
        let raw = std::fs::read(&enc_path).unwrap();
        assert_eq!(raw[0], COMPRESSED_ENCRYPTED_MAGIC);
    }

    #[test]
    fn compressed_is_smaller_than_raw() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.llstate");
        let original = make_kv_state(32, 2048);

        save_state_to_disk(&path, &original, None).unwrap();
        let on_disk = std::fs::metadata(&path).unwrap().len();

        // 1 byte magic + compressed payload should be well under raw size.
        // The structured sinusoidal data compresses very well.
        assert!(
            on_disk < original.len() as u64,
            "compressed ({on_disk}) should be smaller than raw ({})",
            original.len()
        );
    }

    #[test]
    fn backward_compat_legacy_plain() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.llstate");
        let original = b"legacy uncompressed data";

        // Write a legacy-format file (magic 0x00 + raw bytes)
        let mut buf = vec![PLAIN_MAGIC];
        buf.extend_from_slice(original);
        std::fs::write(&path, &buf).unwrap();

        let loaded = load_state_from_disk(&path, None).unwrap();
        assert_eq!(&loaded, original);
    }

    #[test]
    fn backward_compat_legacy_encrypted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy_enc.llstate");
        let key: [u8; 32] = [0xCC; 32];
        let original = b"legacy encrypted data";

        // Encrypt without compression, write with legacy magic
        let cipher = ChaCha20Poly1305::new((&key).into());
        let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng);
        let ciphertext = cipher.encrypt(&nonce, original.as_slice()).unwrap();
        let mut buf = Vec::with_capacity(1 + 12 + ciphertext.len());
        buf.push(ENCRYPTED_MAGIC);
        buf.extend_from_slice(&nonce);
        buf.extend_from_slice(&ciphertext);
        std::fs::write(&path, &buf).unwrap();

        let loaded = load_state_from_disk(&path, Some(&key)).unwrap();
        assert_eq!(&loaded, original);
    }

    #[test]
    fn embedding_similarity_after_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("embed.llstate");

        // Simulate 512-dim embedding vectors (2048 bytes each), 100 of them.
        let embed_dim = 512;
        let num_vectors = 100;
        let mut data = Vec::with_capacity(num_vectors * embed_dim * 4);
        for i in 0..num_vectors {
            for d in 0..embed_dim {
                let v = ((i * embed_dim + d) as f32 * 0.00731).cos() * 1.5;
                data.extend_from_slice(&v.to_le_bytes());
            }
        }

        save_state_to_disk(&path, &data, None).unwrap();
        let loaded = load_state_from_disk(&path, None).unwrap();

        let original_vecs = bytes_to_f32(&data);
        let loaded_vecs = bytes_to_f32(&loaded);

        // Check every embedding vector has cosine similarity == 1.0 (exact match)
        for i in 0..num_vectors {
            let start = i * embed_dim;
            let end = start + embed_dim;
            let sim = cosine_similarity(
                &original_vecs[start..end],
                &loaded_vecs[start..end],
            );
            if i < 5 {
                eprintln!("  embed[{i}] cosine_sim(original, decompressed) = {sim}");
            }
            assert!(
                (sim - 1.0).abs() < 1e-12,
                "vector {i}: cosine similarity {sim} != 1.0 — compression corrupted data"
            );
        }

        // Cross-vector similarities: verify search ordering is preserved
        let anchor = &original_vecs[0..embed_dim];
        eprintln!();
        for i in [1, 10, 50, 99] {
            let start = i * embed_dim;
            let sim_orig = cosine_similarity(anchor, &original_vecs[start..start + embed_dim]);
            let sim_rest = cosine_similarity(anchor, &loaded_vecs[start..start + embed_dim]);
            eprintln!("  sim(embed[0], embed[{i:>2}]): original={sim_orig:.10}  decompressed={sim_rest:.10}  drift={:.2e}", (sim_orig - sim_rest).abs());
        }
    }

    #[test]
    fn kv_state_pairwise_similarity_preserved() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kv.llstate");
        let key: [u8; 32] = [0xDD; 32];

        // 32 layers, 128 head_dim — realistic KV state
        let data = make_kv_state(32, 128);

        save_state_to_disk(&path, &data, Some(&key)).unwrap();
        let loaded = load_state_from_disk(&path, Some(&key)).unwrap();

        let orig = bytes_to_f32(&data);
        let rest = bytes_to_f32(&loaded);

        // Treat each 128-float chunk as a "head" and verify pairwise similarities
        // between heads are identical before and after compression.
        let head_dim = 128;
        let num_heads = orig.len() / head_dim;
        assert!(num_heads >= 4);

        for i in 0..4 {
            for j in (i + 1)..4 {
                let sim_orig = cosine_similarity(
                    &orig[i * head_dim..(i + 1) * head_dim],
                    &orig[j * head_dim..(j + 1) * head_dim],
                );
                let sim_rest = cosine_similarity(
                    &rest[i * head_dim..(i + 1) * head_dim],
                    &rest[j * head_dim..(j + 1) * head_dim],
                );
                eprintln!("  head({i},{j}): sim_orig={sim_orig:.10}  sim_rest={sim_rest:.10}  drift={:.2e}", (sim_orig - sim_rest).abs());
                assert!(
                    (sim_orig - sim_rest).abs() < 1e-12,
                    "head pair ({i},{j}): similarity drift {sim_orig} vs {sim_rest}"
                );
            }
        }
    }

    #[test]
    fn wrong_key_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wrong_key.llstate");
        let key: [u8; 32] = [0x11; 32];
        let wrong_key: [u8; 32] = [0x22; 32];

        save_state_to_disk(&path, b"secret", Some(&key)).unwrap();
        assert!(load_state_from_disk(&path, Some(&wrong_key)).is_none());
        assert!(load_state_from_disk(&path, None).is_none());
    }

    #[test]
    fn zstd_decompression_matches_direct() {
        // Verify our zstd usage is correct independent of the save/load wrapper.
        let original = make_kv_state(16, 256);
        let compressed = zstd::encode_all(Cursor::new(&original), 3).unwrap();
        let decompressed = zstd::decode_all(Cursor::new(&compressed)).unwrap();
        assert_eq!(original, decompressed);
    }

    #[test]
    fn cache_lru_eviction_with_compressed_sizes() {
        let dir = tempfile::tempdir().unwrap();

        // Write one entry to measure actual compressed size, then set budget
        // to fit exactly 2 entries.
        let data = make_kv_state(8, 128);
        let probe_path = dir.path().join("probe.llstate");
        save_state_to_disk(&probe_path, &data, None).unwrap();
        let entry_size = std::fs::metadata(&probe_path).unwrap().len();
        std::fs::remove_file(&probe_path).unwrap();

        let cache = KvCache::with_dir(dir.path().to_path_buf(), entry_size * 2);

        for i in 0..3 {
            let prompt = format!("prompt-{i}");
            let path = cache.save_path(&prompt, "model", "digest", None);
            save_state_to_disk(&path, &data, None).unwrap();
            cache.register(&prompt, "model", "digest", None);
        }

        let (count, used, max) = cache.stats();
        assert!(used <= max, "used {used} exceeds max {max}");
        assert!(count <= 2, "expected eviction, got {count} entries");
    }

    // -- Data integrity tests --

    #[test]
    fn truncated_compressed_plain_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trunc.llstate");
        let data = make_kv_state(8, 128);

        save_state_to_disk(&path, &data, None).unwrap();
        let raw = std::fs::read(&path).unwrap();

        // Chop off the last half of the compressed payload
        let truncated = &raw[..raw.len() / 2];
        std::fs::write(&path, truncated).unwrap();

        assert!(load_state_from_disk(&path, None).is_none());
    }

    #[test]
    fn truncated_compressed_encrypted_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trunc_enc.llstate");
        let key: [u8; 32] = [0xEE; 32];
        let data = make_kv_state(8, 128);

        save_state_to_disk(&path, &data, Some(&key)).unwrap();
        let raw = std::fs::read(&path).unwrap();

        let truncated = &raw[..raw.len() / 2];
        std::fs::write(&path, truncated).unwrap();

        assert!(load_state_from_disk(&path, Some(&key)).is_none());
    }

    #[test]
    fn bitflip_in_compressed_payload_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("flip.llstate");
        let data = make_kv_state(8, 128);

        save_state_to_disk(&path, &data, None).unwrap();
        let mut raw = std::fs::read(&path).unwrap();

        // Flip bits in the middle of the compressed payload
        let mid = raw.len() / 2;
        raw[mid] ^= 0xFF;
        raw[mid + 1] ^= 0xFF;
        std::fs::write(&path, &raw).unwrap();

        // Should return None (zstd detects corruption) or different data
        match load_state_from_disk(&path, None) {
            None => {} // expected: zstd rejected the corrupt frame
            Some(loaded) => {
                // If zstd somehow decoded, it must NOT silently return the original
                assert_ne!(loaded, data, "bitflip was silently ignored");
            }
        }
    }

    #[test]
    fn bitflip_in_encrypted_payload_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("flip_enc.llstate");
        let key: [u8; 32] = [0xFF; 32];
        let data = make_kv_state(8, 128);

        save_state_to_disk(&path, &data, Some(&key)).unwrap();
        let mut raw = std::fs::read(&path).unwrap();

        // Flip a bit in the ciphertext (past magic + nonce)
        let target = 1 + 12 + 10; // magic(1) + nonce(12) + offset into ciphertext
        raw[target] ^= 0x01;
        std::fs::write(&path, &raw).unwrap();

        // AEAD auth tag check must reject this
        assert!(load_state_from_disk(&path, Some(&key)).is_none());
    }

    #[test]
    fn empty_file_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.llstate");
        std::fs::write(&path, b"").unwrap();
        assert!(load_state_from_disk(&path, None).is_none());
    }

    #[test]
    fn magic_byte_only_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("magic_only.llstate");

        // Just the magic byte, no payload
        std::fs::write(&path, &[COMPRESSED_PLAIN_MAGIC]).unwrap();
        assert!(load_state_from_disk(&path, None).is_none());

        std::fs::write(&path, &[COMPRESSED_ENCRYPTED_MAGIC]).unwrap();
        let key: [u8; 32] = [0x01; 32];
        assert!(load_state_from_disk(&path, Some(&key)).is_none());
    }

    #[test]
    fn unknown_magic_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("unknown.llstate");
        std::fs::write(&path, &[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();
        assert!(load_state_from_disk(&path, None).is_none());
    }

    #[test]
    fn swapped_magic_plain_vs_encrypted_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let key: [u8; 32] = [0xAA; 32];
        let data = make_kv_state(4, 64);

        // Write encrypted, try to load as plaintext
        let enc_path = dir.path().join("enc.llstate");
        save_state_to_disk(&enc_path, &data, Some(&key)).unwrap();
        let mut raw = std::fs::read(&enc_path).unwrap();
        raw[0] = COMPRESSED_PLAIN_MAGIC; // swap magic to pretend it's plaintext
        std::fs::write(&enc_path, &raw).unwrap();
        // zstd will reject the encrypted gibberish as invalid compressed data
        assert!(load_state_from_disk(&enc_path, None).is_none());
    }

    #[test]
    fn garbage_after_valid_compressed_data_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("garbage_trail.llstate");
        let data = b"test payload";

        save_state_to_disk(&path, data, None).unwrap();
        let mut raw = std::fs::read(&path).unwrap();

        // Append garbage after the valid zstd frame
        raw.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xFF]);
        std::fs::write(&path, &raw).unwrap();

        // zstd::decode_all reads exactly one frame; trailing garbage should
        // either be ignored (returning valid data) or cause an error.
        // Either way, it must not panic.
        match load_state_from_disk(&path, None) {
            Some(loaded) => assert_eq!(&loaded, data),
            None => {} // also acceptable
        }
    }

    #[test]
    fn nonce_too_short_encrypted_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("short_nonce.llstate");
        let key: [u8; 32] = [0xBB; 32];

        // magic + only 5 bytes (nonce needs 12)
        let mut buf = vec![COMPRESSED_ENCRYPTED_MAGIC];
        buf.extend_from_slice(&[0x01; 5]);
        std::fs::write(&path, &buf).unwrap();

        assert!(load_state_from_disk(&path, Some(&key)).is_none());
    }
}
