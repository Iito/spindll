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

/// Write state data to disk, optionally encrypting with ChaCha20-Poly1305.
pub fn save_state_to_disk(
    path: &Path,
    data: &[u8],
    encryption_key: Option<&[u8; 32]>,
) -> std::io::Result<()> {
    tracing::debug!(encrypted = encryption_key.is_some(), "saving kv state to disk");
    let contents = match encryption_key {
        Some(key) => {
            let cipher = ChaCha20Poly1305::new(key.into());
            let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng);
            let ciphertext = cipher
                .encrypt(&nonce, data)
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            let mut buf = Vec::with_capacity(1 + 12 + ciphertext.len());
            buf.push(ENCRYPTED_MAGIC);
            buf.extend_from_slice(&nonce);
            buf.extend_from_slice(&ciphertext);
            buf
        }
        None => {
            let mut buf = Vec::with_capacity(1 + data.len());
            buf.push(PLAIN_MAGIC);
            buf.extend_from_slice(data);
            buf
        }
    };
    std::fs::write(path, contents)
}

/// Read state data from disk, optionally decrypting.
/// Returns `None` on any error (corrupt file, wrong key, missing key for encrypted data).
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
        PLAIN_MAGIC => Some(payload.to_vec()),
        _ => None,
    }
}
