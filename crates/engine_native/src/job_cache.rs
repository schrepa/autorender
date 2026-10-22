//! The native host's records, as the page keeps its own: the cook's packs,
//! read by byte range from a folder of them, then a folder beside the
//! program keeping what jobs made and the engine handed over, with its
//! index's JSON. Which records the folder keeps and which go beyond its
//! byte budget is the job worker's index (`engine_worker::JobCache`), the
//! page's too; keys are the engine's, so what is kept outlives rebuilds
//! that leave the code making it alone. A lookup finds where a record lies
//! ([`JobStore::sources`]); a worker reads and unpacks it ([`read`]), as
//! the page's workers do, and the store counts what came back
//! ([`JobStore::answered`]).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use engine_worker::{JobCache, PACK_HEADER, PackEntry, PackIndex, pack_directory, unpack};

const INDEX: &str = "index.json";
/// The hosts' default budget in bytes.
pub const DEFAULT_BUDGET: u64 = engine::host::policy::JOB_CACHE_MIB * 1024 * 1024;

/// A pack open for reading: its path and directory.
struct Pack {
    path: PathBuf,
    index: PackIndex,
}

/// Where a record may be read from: a pack's byte range, or the file the
/// store's folder keeps it in.
#[derive(Clone, Debug)]
pub enum Source {
    Packed { path: PathBuf, entry: PackEntry },
    Kept { path: PathBuf },
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

    /// Where the record kept as `key` may be read from, in order: each pack
    /// holding it, then the folder where the index holds it. None, counted
    /// a miss, where nothing does.
    pub fn sources(&mut self, key: &str) -> Vec<Source> {
        let mut sources: Vec<Source> = self
            .packs
            .iter()
            .filter_map(|pack| {
                Some(Source::Packed {
                    path: pack.path.clone(),
                    entry: pack.index.find(key)?,
                })
            })
            .collect();
        if self.index.holds(key) {
            sources.push(Source::Kept {
                path: self.folder.join(key),
            });
        }
        if sources.is_empty() {
            self.index.miss();
        }
        sources
    }

    /// Counts what reading the record kept as `key` from `sources` came to
    /// (`read`: the place among them of the one that read, and its bytes;
    /// none where none did): a pack's record is a hit, as is the folder's
    /// where it holds the bytes the index says, and anything else a miss,
    /// the folder's file going. The record, where it was a hit.
    pub fn answered(
        &mut self,
        key: &str,
        sources: &[Source],
        read: Option<(usize, Vec<u8>)>,
    ) -> Option<Vec<u8>> {
        let kept = sources
            .iter()
            .any(|source| matches!(source, Source::Kept { .. }));
        let answer = match read {
            Some((at, bytes)) => match sources.get(at) {
                Some(Source::Packed { .. }) => {
                    self.pack_hits += 1;
                    Some(bytes)
                }
                Some(Source::Kept { .. }) => self
                    .index
                    .hit(key, Some(bytes.len() as u64))
                    .then_some(bytes),
                None => None,
            },
            None if kept => {
                self.index.hit(key, None);
                None
            }
            None => {
                self.index.miss();
                None
            }
        };
        self.removals();
        answer
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

/// The record `sources` hold, read from the first of them that reads (a
/// pack's byte range, unpacked, or the folder's file): the place of the
/// one that read and the record's bytes, none where none did; a pack's
/// record that does not read is said so. Workers read what the host looks
/// up with it, and a host without workers in place.
pub fn read(sources: &[Source]) -> Option<(usize, Vec<u8>)> {
    sources.iter().enumerate().find_map(|(at, source)| {
        let read = match source {
            Source::Packed { path, entry } => {
                let read = (|| {
                    let mut file = File::open(path)?;
                    let mut stored = vec![0; entry.stored as usize];
                    file.seek(SeekFrom::Start(entry.offset))?;
                    file.read_exact(&mut stored)?;
                    Ok::<_, std::io::Error>(stored)
                })()
                .map_err(|error| error.to_string())
                .and_then(|stored| unpack(entry, &stored));
                if let Err(error) = &read {
                    eprintln!("A record of {} does not read: {error}", path.display());
                }
                read.ok()
            }
            Source::Kept { path } => std::fs::read(path).ok(),
        };
        read.map(|bytes| (at, bytes))
    })
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

/// The pack at `path`, its directory read: workers open it again for each
/// record they read, so none holds it open.
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
        index,
    })
}
