//! The native host's records, as the page keeps its own: the cook's packs,
//! read by byte range from a folder of them, then a folder beside the
//! program keeping what jobs made and the engine handed over, with its
//! index's JSON. Which records the folder keeps and which go beyond its
//! byte budget is the job worker's index (`engine_worker::JobCache`), the
//! page's too; keys are the engine's, so what is kept outlives rebuilds
//! that leave the code making it alone.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use engine_worker::{JobCache, PACK_HEADER, PackIndex, pack_directory, unpack};

const INDEX: &str = "index.json";
/// The hosts' default budget in bytes.
pub const DEFAULT_BUDGET: u64 = engine::host::policy::JOB_CACHE_MIB * 1024 * 1024;

/// A pack open for reading.
struct Pack {
    path: PathBuf,
    file: File,
    index: PackIndex,
}

/// The cook's packs, the records kept in a folder, and the folder's index.
pub struct JobStore {
    folder: PathBuf,
    index: JobCache,
    packs: Vec<Pack>,
    // Records the packs handed back.
    pack_hits: u64,
}

impl JobStore {
    /// The packs in the folder `packs` (every `.pack` file there, read in
    /// name order) and the store beside the running program, within
    /// `budget` bytes; `verify` runs every job anyway and compares. `None`
    /// where the store's folder cannot be read; a pack that cannot be read
    /// is left out, and said so.
    pub fn open(budget: u64, verify: bool, packs: Option<&Path>) -> Option<Self> {
        let opened = (|| {
            let program = std::env::current_exe()?;
            let folder = program
                .parent()
                .ok_or(std::io::ErrorKind::NotFound)?
                .join("engine-job-cache");
            std::fs::create_dir_all(&folder)?;
            let saved = std::fs::read_to_string(folder.join(INDEX)).ok();
            let index = JobCache::open(budget, None, verify, saved.as_deref());
            let mut store = Self {
                folder,
                index,
                packs: packs.map(open_packs).unwrap_or_default(),
                pack_hits: 0,
            };
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
                eprintln!("The record store is off: {error}");
                None
            }
        }
    }

    pub fn verify(&self) -> bool {
        self.index.verify()
    }

    /// Records handed back (from the packs or the folder), records lacked,
    /// the folder's bytes and verified records that differed from a fresh
    /// run.
    pub fn counters(&self) -> (u64, u64, u64, u64) {
        let (hits, misses, bytes, mismatches) = self.index.counters();
        (hits + self.pack_hits, misses, bytes, mismatches)
    }

    /// The packs' records, over all of them.
    pub fn packed(&self) -> usize {
        self.packs.iter().map(|pack| pack.index.len()).sum()
    }

    /// The record kept as `key`, if a pack or the folder holds it.
    pub fn lookup(&mut self, key: &str) -> Option<Vec<u8>> {
        for pack in &mut self.packs {
            let Some(entry) = pack.index.find(key) else {
                continue;
            };
            let mut stored = vec![0; entry.stored as usize];
            let read = pack
                .file
                .seek(SeekFrom::Start(entry.offset))
                .and_then(|_| pack.file.read_exact(&mut stored))
                .map_err(|error| error.to_string())
                .and_then(|()| unpack(&entry, &stored));
            match read {
                Ok(record) => {
                    self.pack_hits += 1;
                    return Some(record);
                }
                Err(error) => {
                    eprintln!("A record of {} does not read: {error}", pack.path.display());
                }
            }
        }
        if !self.index.holds(key) {
            self.index.miss();
            return None;
        }
        let bytes = std::fs::read(self.folder.join(key)).ok();
        let hit = self
            .index
            .hit(key, bytes.as_ref().map(|bytes| bytes.len() as u64));
        self.removals();
        bytes.filter(|_| hit)
    }

    /// Keeps `output` as the record `key`, where the index takes it.
    pub fn keep(&mut self, key: &str, output: &[u8]) {
        if self.index.store(key, output.len() as u64) {
            let written = (|| {
                // Written whole before it takes the key's name.
                let partial = self.folder.join(format!("{key}.partial"));
                File::create(&partial)?.write_all(output)?;
                std::fs::rename(&partial, self.folder.join(key))
            })();
            if let Err(error) = written {
                self.index.failed();
                eprintln!("The record store stopped keeping records: {error}");
            }
        }
        self.removals();
    }

    /// A verified record differed from the one kept as `key`, which goes.
    pub fn mismatch(&mut self, key: &str) {
        eprintln!("A kept record differs from a fresh run: {key}");
        self.index.mismatch(key);
        self.removals();
    }

    /// Closes its packs, so a pack written in their place may replace one.
    pub fn close_packs(&mut self) {
        self.packs.clear();
    }

    /// Writes the index, where it changed.
    pub fn save(&mut self) {
        if let Some(index) = self.index.index()
            && let Err(error) = std::fs::write(self.folder.join(INDEX), index)
        {
            eprintln!("The record store's index was not written: {error}");
        }
    }

    /// Removes the files of records the index dropped.
    fn removals(&mut self) {
        for key in self.index.take_removed() {
            let _ = std::fs::remove_file(self.folder.join(key));
        }
    }
}

/// Every pack in `folder`, in name order; those that do not read are said
/// so and left out. None where the folder is missing.
fn open_packs(folder: &Path) -> Vec<Pack> {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "pack")
        })
        .collect();
    paths.sort();
    paths
        .into_iter()
        .filter_map(|path| match open_pack(&path) {
            Ok(pack) => Some(pack),
            Err(error) => {
                eprintln!("The pack {} is left out: {error}", path.display());
                None
            }
        })
        .collect()
}

fn open_pack(path: &Path) -> Result<Pack, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut header = [0; PACK_HEADER];
    file.read_exact(&mut header)
        .map_err(|error| error.to_string())?;
    let (offset, stored, _) = pack_directory(&header)?;
    let mut directory = vec![0; usize::try_from(stored).map_err(|error| error.to_string())?];
    file.seek(SeekFrom::Start(offset))
        .and_then(|_| file.read_exact(&mut directory))
        .map_err(|error| error.to_string())?;
    let index = PackIndex::open(&header, &directory)?;
    Ok(Pack {
        path: path.to_owned(),
        file,
        index,
    })
}
