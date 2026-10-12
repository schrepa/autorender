//! The index of a host's job output cache, the same on every host: what it
//! keeps (outputs of jobs the engine marks keepable, and what a host keeps
//! of its own work, as a page's transcoded textures) under a hash of the
//! job worker's build, the job's kind and its identity, so the same build
//! hands them back without running their jobs, in this session or a later
//! one; the least recently used beyond a byte budget go. Verifying runs
//! every job anyway and compares. Hosts keep the bytes (the origin's
//! private file system, a folder beside the program) and the index's JSON
//! beside them; the index says which files to read, write and remove.

use engine_math::collections::HashMap;
use sha2::{Digest, Sha256};

/// A kept output: its size, and when it was last used, in milliseconds
/// since the epoch.
#[derive(Clone, Copy, Debug)]
struct Entry {
    bytes: u64,
    used: f64,
}

/// What a host's job output cache keeps, and its counters.
#[derive(Debug)]
pub struct JobCache {
    // The job worker build's hash, which every key starts from.
    version: String,
    budget: u64,
    verify: bool,
    entries: HashMap<String, Entry>,
    total: u64,
    hits: u64,
    misses: u64,
    mismatches: u64,
    // The index changed since the host last saved it.
    changed: bool,
    // Writing failed: it keeps nothing more.
    broken: bool,
    // Keys whose bytes the host removes.
    removed: Vec<String>,
}

impl JobCache {
    /// The cache of the job worker build `build` (its bytes, which name the
    /// cache's version), as the host saved its `index` (JSON), within
    /// `budget` bytes and half of what storage the host has `available`
    /// (bytes beside those the index holds), where it knows; `verify` runs
    /// every job anyway and compares. The least recently used beyond the
    /// budget go at once.
    pub fn open(
        build: &[u8],
        budget: u64,
        available: Option<u64>,
        verify: bool,
        index: Option<&str>,
    ) -> Self {
        let entries = index.map(read_index).unwrap_or_default();
        let total: u64 = entries.values().map(|entry| entry.bytes).sum();
        let budget = available.map_or(budget, |free| budget.min(free.saturating_add(total) / 2));
        let mut cache = Self {
            version: hex(&Sha256::digest(build)),
            budget,
            verify,
            entries,
            total,
            hits: 0,
            misses: 0,
            mismatches: 0,
            changed: false,
            broken: false,
            removed: Vec::new(),
        };
        cache.evict();
        cache
    }

    /// Whether every job runs anyway, its output compared with the one
    /// kept.
    pub fn verify(&self) -> bool {
        self.verify
    }

    /// Outputs handed back, jobs it lacked, its bytes and verified outputs
    /// that differed from a fresh run.
    pub fn counters(&self) -> (u64, u64, u64, u64) {
        (self.hits, self.misses, self.total, self.mismatches)
    }

    /// The key of the work of `kind` on `identity` from this build.
    pub fn key(&self, kind: &str, identity: &[u8]) -> String {
        let mut hash = Sha256::new();
        hash.update(self.version.as_bytes());
        hash.update([0]);
        hash.update(kind.as_bytes());
        hash.update([0]);
        hash.update(identity);
        hex(&hash.finalize())
    }

    /// Whether the index lists the file `name`: the host's other files in
    /// the cache's folder, as those written just before a session ended,
    /// are no one's.
    pub fn lists(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Whether it keeps an output as `key`, to read.
    pub fn holds(&self, key: &str) -> bool {
        !self.broken && self.entries.contains_key(key)
    }

    /// The host read `bytes` bytes kept as `key` (none where the read
    /// failed): a hit where that is the size kept, else the output goes and
    /// it counts a miss. Whether it hit.
    pub fn hit(&mut self, key: &str, bytes: Option<u64>) -> bool {
        match self.entries.get_mut(key) {
            Some(entry) if bytes == Some(entry.bytes) && !self.broken => {
                entry.used = wall_ms();
                self.changed = true;
                self.hits += 1;
                true
            }
            _ => {
                self.remove(key);
                self.misses += 1;
                false
            }
        }
    }

    /// Counts work it lacked.
    pub fn miss(&mut self) {
        self.misses += 1;
    }

    /// Takes an output of `bytes` bytes as `key`, the least recently used
    /// beyond the budget going: whether the host writes it.
    pub fn store(&mut self, key: &str, bytes: u64) -> bool {
        if self.broken || bytes > self.budget {
            return false;
        }
        let replaced = self.entries.insert(
            key.to_owned(),
            Entry {
                bytes,
                used: wall_ms(),
            },
        );
        self.total = self.total + bytes - replaced.map_or(0, |entry| entry.bytes);
        self.changed = true;
        self.evict();
        true
    }

    /// Counts a verified output that differed from the one kept, which
    /// goes.
    pub fn mismatch(&mut self, key: &str) {
        self.mismatches += 1;
        self.remove(key);
    }

    /// Writing failed: it keeps nothing more.
    pub fn failed(&mut self) {
        self.broken = true;
    }

    /// The keys whose bytes the host removes, since the last call.
    pub fn take_removed(&mut self) -> Vec<String> {
        std::mem::take(&mut self.removed)
    }

    /// The index as the host saves it (JSON: each key's size and last use),
    /// where it changed since the last call.
    pub fn index(&mut self) -> Option<String> {
        if !self.changed {
            return None;
        }
        self.changed = false;
        let index: serde_json::Map<String, serde_json::Value> = self
            .entries
            .iter()
            .map(|(key, entry)| (key.clone(), serde_json::json!([entry.bytes, entry.used])))
            .collect();
        Some(serde_json::Value::Object(index).to_string())
    }

    /// Drops the least recently used outputs beyond the budget.
    fn evict(&mut self) {
        if self.total <= self.budget {
            return;
        }
        let mut oldest: Vec<(String, f64)> = self
            .entries
            .iter()
            .map(|(key, entry)| (key.clone(), entry.used))
            .collect();
        oldest.sort_by(|a, b| a.1.total_cmp(&b.1));
        for (key, _) in oldest {
            if self.total <= self.budget {
                break;
            }
            self.remove(&key);
        }
    }

    fn remove(&mut self, key: &str) {
        if let Some(entry) = self.entries.remove(key) {
            self.total -= entry.bytes;
            self.changed = true;
            self.removed.push(key.to_owned());
        }
    }
}

/// The entries of a saved `index`; none where it is unreadable.
fn read_index(index: &str) -> HashMap<String, Entry> {
    let Ok(index) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(index)
    else {
        return HashMap::default();
    };
    index
        .into_iter()
        .filter_map(|(key, value)| {
            let valid = key.len() == 64
                && key
                    .bytes()
                    .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
            let bytes = value.get(0)?.as_u64()?;
            let used = value.get(1)?.as_f64()?;
            valid.then_some((key, Entry { bytes, used }))
        })
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Milliseconds since the epoch, which kept outputs' last use is in.
fn wall_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Date::now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |elapsed| elapsed.as_secs_f64() * 1000.0)
    }
}
