//! The native job output cache's folder, beside the program, as the page
//! keeps its cache in the origin's private file system: the bytes of each
//! output kept and the index's JSON. Which outputs it keeps, under which
//! keys, and which go beyond its byte budget is the job worker's index
//! (`engine_worker::JobCache`), the page's too.

use std::io::Write;
use std::path::PathBuf;

use engine_worker::JobCache;

const INDEX: &str = "index.json";
/// The hosts' default budget in bytes.
pub const DEFAULT_BUDGET: u64 = engine::host::policy::JOB_CACHE_MIB * 1024 * 1024;

/// Kept job outputs in a folder, and their index.
pub struct JobStore {
    folder: PathBuf,
    index: JobCache,
}

impl JobStore {
    /// The cache beside the running program, which names its version,
    /// within `budget` bytes; `verify` runs every job anyway and compares.
    /// `None` where the folder or the program cannot be read.
    pub fn open(budget: u64, verify: bool) -> Option<Self> {
        let opened = (|| {
            let program = std::env::current_exe()?;
            let folder = program
                .parent()
                .ok_or(std::io::ErrorKind::NotFound)?
                .join("engine-job-cache");
            std::fs::create_dir_all(&folder)?;
            let saved = std::fs::read_to_string(folder.join(INDEX)).ok();
            let index = JobCache::open(
                &std::fs::read(&program)?,
                budget,
                None,
                verify,
                saved.as_deref(),
            );
            let mut store = Self { folder, index };
            // Files the index does not list, as those written just before a
            // run ended, are no one's.
            for file in std::fs::read_dir(&store.folder)?.flatten() {
                let name = file.file_name().to_string_lossy().into_owned();
                if name != INDEX && !store.index.lists(&name) {
                    let _ = std::fs::remove_file(file.path());
                }
            }
            store.removals();
            Ok::<_, std::io::Error>(store)
        })();
        match opened {
            Ok(store) => Some(store),
            Err(error) => {
                eprintln!("The job cache is off: {error}");
                None
            }
        }
    }

    pub fn verify(&self) -> bool {
        self.index.verify()
    }

    /// Outputs handed back, work it lacked, its bytes and verified outputs
    /// that differed from a fresh run.
    pub fn counters(&self) -> (u64, u64, u64, u64) {
        self.index.counters()
    }

    /// The key of the work of `kind` on `identity`, and the output kept as
    /// it, if any.
    pub fn lookup(&mut self, kind: &str, identity: &[u8]) -> (String, Option<Vec<u8>>) {
        let key = self.index.key(kind, identity);
        if !self.index.holds(&key) {
            self.index.miss();
            return (key, None);
        }
        let bytes = std::fs::read(self.folder.join(&key)).ok();
        let hit = self
            .index
            .hit(&key, bytes.as_ref().map(|bytes| bytes.len() as u64));
        self.removals();
        (key, bytes.filter(|_| hit))
    }

    /// Keeps `output` as `key`, where the index takes it.
    pub fn keep(&mut self, key: &str, output: &[u8]) {
        if self.index.store(key, output.len() as u64) {
            let written = (|| {
                // Written whole before it takes the key's name.
                let partial = self.folder.join(format!("{key}.partial"));
                std::fs::File::create(&partial)?.write_all(output)?;
                std::fs::rename(&partial, self.folder.join(key))
            })();
            if let Err(error) = written {
                self.index.failed();
                eprintln!("The job cache stopped keeping outputs: {error}");
            }
        }
        self.removals();
    }

    /// A verified output differed from the one kept as `key`, which goes.
    pub fn mismatch(&mut self, key: &str) {
        eprintln!("A kept job output differs from a fresh run: {key}");
        self.index.mismatch(key);
        self.removals();
    }

    /// Writes the index, where it changed.
    pub fn save(&mut self) {
        if let Some(index) = self.index.index()
            && let Err(error) = std::fs::write(self.folder.join(INDEX), index)
        {
            eprintln!("The job cache's index was not written: {error}");
        }
    }

    /// Removes the files of outputs the index dropped.
    fn removals(&mut self) {
        for key in self.index.take_removed() {
            let _ = std::fs::remove_file(self.folder.join(key));
        }
    }
}
