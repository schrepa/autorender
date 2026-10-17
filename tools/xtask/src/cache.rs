//! A byte-bounded disposable store shared by a repository's worktrees.
//! OS locks pin active entries and disappear on process exit. Results belong
//! outside this store. Only this version's marked directory is ever collected.

use crate::files;
use serde_json::{Value, json};
use std::{
    fs::{self, File, TryLockError},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const MARKER: &str = "engine-cache-v1";
static LIVE: std::sync::Mutex<Vec<Entry>> = std::sync::Mutex::new(Vec::new());
static ADMISSION: std::sync::Mutex<Option<Admission>> = std::sync::Mutex::new(None);

struct Admission {
    root: PathBuf,
    limit: u64,
    at: Instant,
    bytes: u64,
}
/// Initial ceiling, adjustable with ENGINE_CACHE_MAX_BYTES or cache configure.
/// This is an idle logical-byte bound, not a reservation or an OS disk quota.
pub const DEFAULT_MAX_BYTES: u64 = 48 * 1024 * 1024 * 1024;

/// Shared disposable storage. Does not own ordinary target or result folders.
#[derive(Clone)]
pub struct Store {
    root: PathBuf,
    max_bytes: u64,
}

/// An entry is exclusively leased until dropped, including during execution.
pub struct Entry {
    store: Store,
    path: PathBuf,
    lock: File,
    pub wait_ms: u128,
}

fn lock_file(path: &Path) -> Result<File, String> {
    File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))
}

impl Store {
    /// A short per-user cache shared by worktrees. Source paths may be deep;
    /// MSVC and offline analyzers still limit intermediate path lengths.
    /// An override is also a parent of our explicitly marked directory.
    pub fn open(_workspace: &Path) -> Result<Self, String> {
        let parent = if let Some(path) = std::env::var_os("ENGINE_CACHE_DIR") {
            PathBuf::from(path)
        } else {
            user_cache()?
        };
        let mut store = Self::at(&parent, DEFAULT_MAX_BYTES)?;
        let config = store.root.join("config.json");
        if config.exists() {
            store.max_bytes = files::json_file(&config)?["max_bytes"]
                .as_u64()
                .ok_or("invalid cache max_bytes")?;
        }
        if let Ok(limit) = std::env::var("ENGINE_CACHE_MAX_BYTES") {
            store.max_bytes = limit
                .parse()
                .map_err(|_| "ENGINE_CACHE_MAX_BYTES must be an integer")?;
        }
        Ok(store)
    }

    /// Open an isolated marked store, also used by the storage selfcheck.
    pub fn at(parent: &Path, max_bytes: u64) -> Result<Self, String> {
        let root = parent.join(MARKER);
        fs::create_dir_all(&root).map_err(|e| e.to_string())?;
        files::reject_link(&root)?;
        let root = files::canonical(&root)?;
        let marker = root.join("owner.json");
        let owner = json!({"kind": MARKER, "version": 1});
        let init = lock_file(&root.join("init.lock"))?;
        init.lock().map_err(|e| e.to_string())?;
        if marker.exists() {
            if files::json_file(&marker)? != owner {
                return Err("cache ownership marker differs".into());
            }
        } else {
            if fs::read_dir(&root).map_err(|e| e.to_string())?.count() != 1 {
                return Err("refusing to adopt a nonempty cache directory".into());
            }
            files::atomic_json(&marker, &owner)?;
        }
        for part in ["entries", "locks"] {
            fs::create_dir_all(root.join(part)).map_err(|e| e.to_string())?;
        }
        Ok(Self { root, max_bytes })
    }

    /// Store location, suitable for inspection without traversing user files.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// A per-key OS lease, opened under the collection lock. A waiting caller
    /// closes its handle before retrying, so collection may safely remove a
    /// lock file without leaving waiters attached to an unlinked inode.
    pub fn entry(&self, category: &str, identity: &str) -> Result<Entry, String> {
        let key = files::digest(format!("{category}\0{identity}").as_bytes());
        let started = Instant::now();
        let lock = self.lease(&key[..32], false)?;
        let path = self.root.join("entries").join(&key[..32]);
        fs::create_dir_all(&path).map_err(|e| e.to_string())?;
        let requested = json!({"category": category, "identity": identity});
        let metadata = path.join("entry.json");
        if metadata.exists() && files::json_file(&metadata)? != requested {
            return Err("cache key collision".into());
        }
        files::atomic_json(&metadata, &requested)?;
        fs::write(path.join("used"), []).map_err(|e| e.to_string())?;
        Ok(Entry {
            store: self.clone(),
            path,
            lock,
            wait_ms: started.elapsed().as_millis(),
        })
    }

    fn lease(&self, key: &str, shared: bool) -> Result<File, String> {
        let started = Instant::now();
        loop {
            let collection = lock_file(&self.root.join("collect.lock"))?;
            collection.lock().map_err(|e| e.to_string())?;
            let file = lock_file(&self.root.join("locks").join(key))?;
            let attempt = if shared {
                file.try_lock_shared()
            } else {
                file.try_lock()
            };
            match attempt {
                Ok(()) => return Ok(file),
                Err(TryLockError::Error(error)) => return Err(error.to_string()),
                Err(TryLockError::WouldBlock) => {}
            }
            drop(file);
            drop(collection);
            if started.elapsed() > std::time::Duration::from_secs(30 * 60) {
                return Err("cache lease deadline exceeded".into());
            }
            if std::env::var_os("ENGINE_CANCEL_FILE").is_some_and(|p| Path::new(&p).exists()) {
                return Err("cancelled waiting for cache lease".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
    }

    /// Share an existing immutable result; no writer can alter it while any
    /// consumer holds this lease. A miss releases the lease before computing.
    pub fn read(&self, category: &str, identity: &str) -> Result<Option<Entry>, String> {
        let key = files::digest(format!("{category}\0{identity}").as_bytes());
        let started = Instant::now();
        let lock = self.lease(&key[..32], true)?;
        let path = self.root.join("entries").join(&key[..32]);
        let entry = Entry {
            store: self.clone(),
            path,
            lock,
            wait_ms: started.elapsed().as_millis(),
        };
        if !entry.valid()? {
            return Ok(None);
        }
        let metadata = files::json_file(&entry.path.join("entry.json"))?;
        if metadata != json!({"category": category, "identity": identity}) {
            return Err("cache key collision".into());
        }
        fs::write(entry.path.join("used"), []).map_err(|e| e.to_string())?;
        Ok(Some(entry))
    }

    /// Byte usage includes incomplete writes and failed deletions. A locked
    /// entry is counted even though collection cannot reclaim it.
    pub fn status(&self) -> Result<Value, String> {
        Ok(
            json!({"version": 1, "root": self.root, "logical_bytes": files::bytes(&self.root)?, "max_bytes": self.max_bytes, "limit": "idle logical bytes; active compiler peaks require an OS quota for a hard bound"}),
        )
    }

    /// Configure a persistent idle budget; reports and foreign targets stay out.
    pub fn configure(&mut self, max_bytes: u64) -> Result<Value, String> {
        if max_bytes == 0 {
            return Err("cache budget must be greater than zero".into());
        }
        files::atomic_json(
            &self.root.join("config.json"),
            &json!({"max_bytes": max_bytes}),
        )?;
        self.max_bytes = max_bytes;
        self.collect(0)
    }

    /// Reclaim least recently used inactive entries, accounting again after
    /// deletion. Never claim a failed Windows deletion freed its bytes.
    pub fn collect(&self, reserve: u64) -> Result<Value, String> {
        let limit = self.max_bytes.saturating_sub(reserve);
        // Inventory active compiler trees without blocking every cache lease.
        // An under-budget snapshot needs no mutation. Concurrent active writes
        // may peak above it; their owners collect again when they finish.
        let snapshot = files::bytes(&self.root)?;
        if snapshot <= limit {
            return Ok(json!({"logical_bytes":snapshot, "max_bytes":self.max_bytes,
                "reserved_bytes":reserve, "within_budget":true, "removed_entries":0, "delete_errors":[]}));
        }
        let collection = lock_file(&self.root.join("collect.lock"))?;
        collection.lock().map_err(|e| e.to_string())?;
        // Another collector may have run since the initial snapshot. Measure
        // again under the collection lock before deciding what to evict.
        let mut bytes = files::bytes(&self.root)?;
        let mut removed = 0;
        let mut failures = Vec::new();
        for (_, path) in self.oldest()? {
            if bytes <= limit {
                break;
            }
            let key = path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or("invalid cache key")?;
            if key.len() != 32 || !key.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("unknown entry in managed cache".into());
            }
            let lock_path = self.root.join("locks").join(key);
            let lock = lock_file(&lock_path)?;
            match lock.try_lock() {
                Err(TryLockError::WouldBlock) => continue,
                Err(TryLockError::Error(error)) => return Err(error.to_string()),
                Ok(()) => {}
            }
            // The lease is acquired before resolving/removing this directory.
            // Reject links rather than following them outside the owned root.
            let entry_bytes = files::bytes(&path)?;
            if let Err(error) = remove_source(&path) {
                failures.push(error);
                continue;
            }
            match fs::remove_dir_all(&path) {
                Ok(()) => {
                    removed += 1;
                    bytes = bytes.saturating_sub(entry_bytes);
                }
                Err(error) => failures.push(format!("{}: {error}", path.display())),
            }
            drop(lock);
            if !path.exists() {
                fs::remove_file(lock_path).map_err(|e| e.to_string())?;
            }
        }
        // Scan the whole store once more, including any concurrently growing
        // active builds and partial failed deletions. Re-scanning every Cargo
        // target after each small shader eviction makes collection quadratic.
        bytes = files::bytes(&self.root)?;
        Ok(
            json!({"logical_bytes": bytes, "max_bytes": self.max_bytes, "reserved_bytes": reserve, "within_budget": bytes <= limit, "removed_entries": removed, "delete_errors": failures}),
        )
    }

    fn oldest(&self) -> Result<Vec<(SystemTime, PathBuf)>, String> {
        let mut paths = Vec::new();
        for entry in fs::read_dir(self.root.join("entries")).map_err(|e| e.to_string())? {
            let path = entry.map_err(|e| e.to_string())?.path();
            let used = fs::metadata(path.join("used"))
                .and_then(|m| m.modified())
                .unwrap_or(UNIX_EPOCH);
            paths.push((used, path));
        }
        paths.sort();
        Ok(paths)
    }
}

fn remove_source(entry: &Path) -> Result<(), String> {
    let source = entry.join("source");
    if !source.exists() {
        return Ok(());
    }
    let metadata = files::json_file(&entry.join("entry.json"))?;
    if metadata["category"] != "source" {
        return Err("unknown source directory in managed entry".into());
    }
    let git = |args: &[&str]| -> Result<String, String> {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(&source)
            .output()
            .map_err(|e| e.to_string())?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).into_owned());
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().into())
    };
    if !git(&["status", "--porcelain"])?.is_empty() {
        return Err(format!("keeping source edits in {}", source.display()));
    }
    let common = git(&["rev-parse", "--path-format=absolute", "--git-common-dir"])?;
    let output = std::process::Command::new("git")
        .arg("--git-dir")
        .arg(common)
        .args(["worktree", "remove", "--force"])
        .arg(&source)
        .current_dir(entry)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "removing owned worktree {}: {}",
            source.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

impl Entry {
    /// Downgrade a completed immutable generation before exposing it to users.
    pub fn shared(self) -> Result<Self, String> {
        let collection = lock_file(&self.store.root.join("collect.lock"))?;
        collection.lock().map_err(|e| e.to_string())?;
        self.lock.unlock().map_err(|e| e.to_string())?;
        self.lock.lock_shared().map_err(|e| e.to_string())?;
        Ok(self)
    }
    /// Mutable state must be accessed while this entry is alive.
    pub fn path(&self) -> &Path {
        &self.path
    }
    /// Immutable, validated outputs live separately from publication scratch.
    pub fn data(&self) -> PathBuf {
        self.path.join("data")
    }
    /// Return an admitted result only when every output matches its manifest.
    pub fn valid(&self) -> Result<bool, String> {
        let manifest = self.path.join("manifest.json");
        if !manifest.is_file() || !self.data().is_dir() {
            return Ok(false);
        }
        let Ok(manifest) = files::json_file(&manifest) else {
            return Ok(false);
        };
        if manifest.is_array() {
            return files::verify(&self.data(), &manifest);
        }
        if manifest["version"] != 1 {
            return Ok(false);
        }
        let Some(mutable) = manifest["mutable_directories"].as_array() else {
            return Ok(false);
        };
        let mutable = mutable
            .iter()
            .map(|value| files::relative(value.as_str().ok_or("invalid runtime directory")?))
            .collect::<Result<Vec<_>, String>>()?;
        Ok(files::manifest_except(&self.data(), &mutable)? == manifest["files"])
    }
    /// Published preparation identity, excluding declared mutable runtime state.
    pub fn manifest(&self) -> Result<Value, String> {
        files::json_file(&self.path.join("manifest.json"))
    }
    /// Empty output generation, after holding the entry's exclusive lease.
    pub fn staging(&self) -> Result<PathBuf, String> {
        let path = self.path.join("staging");
        if path.exists() {
            files::list(&path)?;
            fs::remove_dir_all(&path).map_err(|e| e.to_string())?;
        }
        fs::create_dir_all(&path).map_err(|e| e.to_string())?;
        Ok(path)
    }
    /// Publish complete validated output. The manifest is written last.
    pub fn publish(&self) -> Result<(), String> {
        self.publish_mutable(&[])
    }
    /// Declare runtime-only directories alongside verified preparation files.
    /// They are absent at publication, but may grow while the entry is leased.
    /// Their bytes count against the same budget; arbitrary extra files do not.
    pub fn publish_mutable(&self, mutable: &[&str]) -> Result<(), String> {
        let staging = self.path.join("staging");
        for directory in mutable {
            let directory = files::relative(directory)?;
            if staging.join(directory).exists() {
                return Err("runtime directory must be absent during publication".into());
            }
        }
        let prepared = files::manifest(&staging)?;
        let manifest = if mutable.is_empty() {
            prepared
        } else {
            json!({"version":1, "files":prepared, "mutable_directories":mutable})
        };
        self.require_capacity(0)?;
        if self.data().exists() {
            files::list(&self.data())?;
            fs::remove_dir_all(self.data()).map_err(|e| e.to_string())?;
        }
        fs::rename(&staging, self.data())
            .map_err(|e| format!("publishing {}: {e}", staging.display()))?;
        files::atomic_json(&self.path.join("manifest.json"), &manifest)
    }
    /// Fail admission when inactive eviction cannot make room. Retained
    /// results and active entries are never sacrificed for a new request.
    pub fn require_capacity(&self, additional: u64) -> Result<(), String> {
        // Hundreds of parallel shader publications must not each traverse
        // every compiler target. Admission can use a snapshot for one second,
        // conservatively charging this entire entry again against headroom.
        // Active writers already permit transient peaks; explicit collection
        // and command completion always measure the full store again.
        let mut admission = ADMISSION.lock().map_err(|e| e.to_string())?;
        let margin = additional.saturating_add(files::bytes(&self.path)?);
        if admission.as_ref().is_some_and(|last| {
            last.root == self.store.root
                && last.limit == self.store.max_bytes
                && last.at.elapsed() < std::time::Duration::from_secs(1)
                && last.bytes.saturating_add(margin) <= last.limit
        }) {
            return Ok(());
        }
        let report = self.store.collect(additional)?;
        *admission = Some(Admission {
            root: self.store.root.clone(),
            limit: self.store.max_bytes,
            at: Instant::now(),
            bytes: report["logical_bytes"]
                .as_u64()
                .ok_or("missing cache accounting")?,
        });
        if report["within_budget"] != true {
            return Err(format!(
                "cache capacity exceeded; active entries remain pinned: {report}"
            ));
        }
        Ok(())
    }
}

impl Drop for Entry {
    fn drop(&mut self) {
        if let Err(error) = self.lock.unlock() {
            eprintln!("cache lease: {error}");
        }
    }
}

/// Keep outputs used by existing path-returning APIs alive for this command.
/// Shared leases permit repeated reads and vanish on normal exit or a crash.
pub fn retain(entry: Entry) -> Result<PathBuf, String> {
    let entry = entry.shared()?;
    let path = entry.data();
    LIVE.lock().map_err(|e| e.to_string())?.push(entry);
    Ok(path)
}

/// At command completion, release path-returning API leases before collection.
pub fn finish(root: &Path) {
    if let Ok(mut live) = LIVE.lock() {
        live.clear();
    }
    match Store::open(root).and_then(|store| store.collect(0)) {
        Ok(report) if report["within_budget"] == false => {
            eprintln!("cache capacity remains pinned: {report}")
        }
        Err(error) => eprintln!("cache collection: {error}"),
        _ => {}
    }
}

fn user_cache() -> Result<PathBuf, String> {
    #[cfg(windows)]
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(not(windows))]
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".cache")));
    base.map(|path| path.join("engine"))
        .ok_or_else(|| "cannot locate user cache; set ENGINE_CACHE_DIR".into())
}
