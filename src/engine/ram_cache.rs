use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

struct RamCacheEntry {
    path: PathBuf,
    #[allow(dead_code)]
    data: Vec<u8>,
    size_bytes: u64,
    evicted_at: Instant,
}

/// Holds recently-evicted model file data in RAM so re-loading skips disk I/O.
///
/// When a model is unloaded from GPU, its GGUF file is read into a userspace
/// buffer. The OS page cache keeps these pages resident, so a subsequent
/// `LlamaModel::load_from_file` on the same path hits RAM instead of disk.
/// The buffer is dropped (and pages released) when evicted from this cache.
pub struct RamCache {
    entries: Mutex<HashMap<String, RamCacheEntry>>,
    max_bytes: u64,
}

impl RamCache {
    pub fn new(max_bytes: u64) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            max_bytes,
        }
    }

    /// Read a model file into RAM to keep its pages warm in the OS page cache.
    /// Silently drops the oldest entry if the budget would be exceeded.
    pub fn warm(&self, name: &str, path: &Path) {
        let size = match std::fs::metadata(path) {
            Ok(m) => m.len(),
            Err(_) => return,
        };

        if size > self.max_bytes {
            return;
        }

        let mut buf = Vec::new();
        let Ok(mut file) = File::open(path) else { return };
        if file.read_to_end(&mut buf).is_err() {
            return;
        }

        let mut entries = self.entries.lock().unwrap();
        Self::evict_for(&mut entries, self.max_bytes, size);

        tracing::debug!(model = name, size_mb = size / 1_048_576, "ram cache: warmed");
        entries.insert(
            name.to_string(),
            RamCacheEntry {
                path: path.to_path_buf(),
                data: buf,
                size_bytes: size,
                evicted_at: Instant::now(),
            },
        );
    }

    /// Check whether a model is in the RAM cache (pages are warm).
    /// Returns the cached path so the caller can pass it to `load_from_file`.
    pub fn get(&self, name: &str) -> Option<PathBuf> {
        let entries = self.entries.lock().unwrap();
        entries.get(name).map(|e| {
            tracing::debug!(model = name, "ram cache: hit");
            e.path.clone()
        })
    }

    /// Remove a model from the RAM cache (e.g. after it has been re-loaded).
    pub fn remove(&self, name: &str) {
        let mut entries = self.entries.lock().unwrap();
        if entries.remove(name).is_some() {
            tracing::debug!(model = name, "ram cache: removed after reload");
        }
    }

    /// Returns `(entry_count, used_bytes, max_bytes)`.
    pub fn stats(&self) -> (usize, u64, u64) {
        let entries = self.entries.lock().unwrap();
        let used: u64 = entries.values().map(|e| e.size_bytes).sum();
        (entries.len(), used, self.max_bytes)
    }

    fn evict_for(entries: &mut HashMap<String, RamCacheEntry>, max_bytes: u64, needed: u64) {
        loop {
            let used: u64 = entries.values().map(|e| e.size_bytes).sum();
            if used + needed <= max_bytes {
                return;
            }
            let lru = entries
                .iter()
                .min_by_key(|(_, e)| e.evicted_at)
                .map(|(k, _)| k.clone());
            match lru {
                Some(key) => {
                    tracing::debug!(model = key, "ram cache: evicting");
                    entries.remove(&key);
                }
                None => return,
            }
        }
    }
}
