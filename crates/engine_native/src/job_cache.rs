use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use engine_worker::{JobCache, PACK_HEADER, PackEntry, PackIndex, pack_directory, unpack};
use serde_json::json;

const INDEX: &str = "index.json";
pub const DEFAULT_BUDGET: u64 = engine::host::policy::JOB_CACHE_MIB * 1024 * 1024;

struct Pack {
    path: PathBuf,
    index: PackIndex,
}

#[derive(Clone, Debug)]
pub enum Source {
    Packed { path: PathBuf, entry: PackEntry },
    Kept { path: PathBuf },
}

pub struct JobStore {
    folder: PathBuf,
    index: JobCache,
    packs: Vec<Pack>,
    pack_hits: u64,
}

impl JobStore {
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
                let record = json!({"event": "record_store_off", "error": error.to_string()});
                eprintln!("{record}");
                None
            }
        }
    }

    pub fn verify(&self) -> bool {
        self.index.verify()
    }

    pub fn counters(&self) -> (u64, u64, u64, u64) {
        let (hits, misses, bytes, mismatches) = self.index.counters();
        (hits + self.pack_hits, misses, bytes, mismatches)
    }

    pub fn packed(&self) -> usize {
        self.packs.iter().map(|pack| pack.index.len()).sum()
    }

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

    pub fn keep(&mut self, key: &str, output: &[u8]) {
        if self.index.store(key, output.len() as u64) {
            let written = (|| {
                let partial = self.folder.join(format!("{key}.partial"));
                File::create(&partial)?.write_all(output)?;
                std::fs::rename(&partial, self.folder.join(key))
            })();
            if let Err(error) = written {
                self.index.failed();
                let record = json!({"event": "record_store_stopped", "error": error.to_string()});
                eprintln!("{record}");
            }
        }
        self.removals();
    }

    pub fn mismatch(&mut self, key: &str) {
        eprintln!("{}", json!({"event": "record_mismatch", "key": key}));
        self.index.mismatch(key);
        self.removals();
    }

    pub fn close_packs(&mut self) {
        self.packs.clear();
    }

    pub fn save(&mut self) {
        if let Some(index) = self.index.index()
            && let Err(error) = std::fs::write(self.folder.join(INDEX), index)
        {
            let record = json!({"event": "record_index_unwritten", "error": error.to_string()});
            eprintln!("{record}");
        }
    }

    fn removals(&mut self) {
        for key in self.index.take_removed() {
            let _ = std::fs::remove_file(self.folder.join(key));
        }
    }
}

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
                    let record = json!({
                        "event": "record_unreadable",
                        "pack": path.display().to_string(),
                        "error": error,
                    });
                    eprintln!("{record}");
                }
                read.ok()
            }
            Source::Kept { path } => std::fs::read(path).ok(),
        };
        read.map(|bytes| (at, bytes))
    })
}

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
                let record = json!({
                    "event": "pack_left_out",
                    "pack": path.display().to_string(),
                    "error": error,
                });
                eprintln!("{record}");
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
        index,
    })
}
