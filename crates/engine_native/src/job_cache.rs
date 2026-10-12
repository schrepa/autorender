//! The native job output cache, as the page's: outputs of jobs the engine
//! marks keepable (terrain tiles, their far objects, walks and bakes) kept
//! in a folder beside the program under a hash of the program, the job's
//! kind and its identity, so the same build hands them back without running
//! their jobs, in this run or a later one. The least recently used beyond a
//! byte budget go. Verifying runs every job anyway and compares.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

const INDEX: &str = "index.json";
/// The hosts' default budget in bytes.
pub const DEFAULT_BUDGET: u64 = engine::host::policy::JOB_CACHE_MIB * 1024 * 1024;

struct Entry {
    bytes: u64,
    // Milliseconds since the epoch.
    used: f64,
}

/// Kept job outputs.
pub struct JobCache {
    folder: PathBuf,
    // The program's hash, which every key starts from.
    version: String,
    budget: u64,
    verify: bool,
    entries: HashMap<String, Entry>,
    total: u64,
    hits: u64,
    misses: u64,
    mismatches: u64,
    // The index changed since it was written.
    changed: bool,
    // Writing failed: it keeps nothing more.
    broken: bool,
}

/// Milliseconds since the epoch.
fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64() * 1000.0)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

impl JobCache {
    /// The cache beside the running program, within `budget` bytes;
    /// `verify` runs every job anyway and compares. `None` where the folder
    /// or the program cannot be read.
    pub fn open(budget: u64, verify: bool) -> Option<Self> {
        let opened = (|| {
            let program = std::env::current_exe()?;
            let folder = program
                .parent()
                .ok_or(std::io::ErrorKind::NotFound)?
                .join("engine-job-cache");
            std::fs::create_dir_all(&folder)?;
            let version = hex(&Sha256::digest(std::fs::read(&program)?));
            let entries = read_index(&folder).unwrap_or_default();
            // Files the index does not list, as those written just before a
            // run ended, are no one's.
            for file in std::fs::read_dir(&folder)?.flatten() {
                let name = file.file_name().to_string_lossy().into_owned();
                if name != INDEX && !entries.contains_key(&name) {
                    let _ = std::fs::remove_file(file.path());
                }
            }
            let total = entries.values().map(|entry| entry.bytes).sum();
            let mut cache = Self {
                folder,
                version,
                budget,
                verify,
                entries,
                total,
                hits: 0,
                misses: 0,
                mismatches: 0,
                changed: false,
                broken: false,
            };
            cache.evict();
            Ok::<_, std::io::Error>(cache)
        })();
        match opened {
            Ok(cache) => Some(cache),
            Err(error) => {
                eprintln!("The job cache is off: {error}");
                None
            }
        }
    }

    pub fn verify(&self) -> bool {
        self.verify
    }

    /// Outputs handed back, jobs it lacked, its bytes and verified outputs
    /// that differed from a fresh run.
    pub fn counters(&self) -> (u64, u64, u64, u64) {
        (self.hits, self.misses, self.total, self.mismatches)
    }

    /// The key of a job of `kind` with `identity` from this build.
    pub fn key(&self, kind: &str, identity: &[u8]) -> String {
        let mut hash = Sha256::new();
        hash.update(self.version.as_bytes());
        hash.update([0]);
        hash.update(kind.as_bytes());
        hash.update([0]);
        hash.update(identity);
        hex(&hash.finalize())
    }

    /// The output kept as `key`, if any.
    pub fn lookup(&mut self, key: &str) -> Option<Vec<u8>> {
        let Some(entry) = self.entries.get_mut(key).filter(|_| !self.broken) else {
            self.misses += 1;
            return None;
        };
        match std::fs::read(self.folder.join(key)) {
            Ok(bytes) if bytes.len() as u64 == entry.bytes => {
                entry.used = now();
                self.changed = true;
                self.hits += 1;
                Some(bytes)
            }
            _ => {
                self.remove(key);
                self.misses += 1;
                None
            }
        }
    }

    /// Keeps `output` as `key`.
    pub fn store(&mut self, key: &str, output: &[u8]) {
        if self.broken || output.len() as u64 > self.budget {
            return;
        }
        let written = (|| {
            // Written whole before it takes the key's name.
            let partial = self.folder.join(format!("{key}.partial"));
            std::fs::File::create(&partial)?.write_all(output)?;
            std::fs::rename(&partial, self.folder.join(key))
        })();
        if let Err(error) = written {
            self.broken = true;
            eprintln!("The job cache stopped keeping outputs: {error}");
            return;
        }
        let bytes = output.len() as u64;
        let replaced = self
            .entries
            .insert(key.to_owned(), Entry { bytes, used: now() });
        self.total = self.total + bytes - replaced.map_or(0, |entry| entry.bytes);
        self.changed = true;
        self.evict();
    }

    /// Counts a verified output that differed from the one kept.
    pub fn mismatch(&mut self, key: &str) {
        self.mismatches += 1;
        eprintln!("A kept job output differs from a fresh run: {key}");
        self.remove(key);
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
            let _ = std::fs::remove_file(self.folder.join(key));
        }
    }

    /// Writes the index, when it changed.
    pub fn save(&mut self) {
        if !self.changed {
            return;
        }
        let index: serde_json::Map<String, serde_json::Value> = self
            .entries
            .iter()
            .map(|(key, entry)| (key.clone(), serde_json::json!([entry.bytes, entry.used])))
            .collect();
        match std::fs::write(
            self.folder.join(INDEX),
            serde_json::Value::Object(index).to_string(),
        ) {
            Ok(()) => self.changed = false,
            Err(error) => eprintln!("The job cache's index was not written: {error}"),
        }
    }
}

/// The index's entries; `None` when it is missing or unreadable.
fn read_index(folder: &std::path::Path) -> Option<HashMap<String, Entry>> {
    let text = std::fs::read_to_string(folder.join(INDEX)).ok()?;
    let index: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&text).ok()?;
    let entries = index
        .into_iter()
        .filter_map(|(key, value)| {
            let valid = key.len() == 64 && key.bytes().all(|byte| byte.is_ascii_hexdigit());
            let bytes = value.get(0)?.as_u64()?;
            let used = value.get(1)?.as_f64()?;
            valid.then_some((key, Entry { bytes, used }))
        })
        .collect();
    Some(entries)
}
