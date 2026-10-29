use std::collections::BTreeMap;
use std::path::Path;

use engine::collections::HashSet;
use engine::{Error, ErrorCode};

use crate::file_error;

const LEVEL: i32 = 19;

#[derive(Default)]
pub(crate) struct PackWriter {
    records: Vec<([u8; 32], &'static str, Vec<u8>)>,
    seen: HashSet<[u8; 32]>,
}

impl PackWriter {
    pub(crate) fn take(&mut self, key: &str, kind: &'static str, bytes: &[u8]) {
        let Some(key) = engine_worker::key_from_hex(key) else {
            return;
        };
        if self.seen.insert(key) {
            self.records.push((key, kind, bytes.to_vec()));
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.records.len()
    }

    pub(crate) fn kinds(&self) -> BTreeMap<&'static str, usize> {
        let mut kinds = BTreeMap::new();
        for &(_, kind, _) in &self.records {
            *kinds.entry(kind).or_default() += 1;
        }
        kinds
    }

    pub(crate) fn write(&self, path: &Path) -> Result<u64, Error> {
        let listing =
            engine_worker::listing_bytes(self.records.iter().map(|(key, kind, _)| (key, *kind)));
        let records: Vec<([u8; 32], &[u8])> =
            std::iter::once((engine_worker::listing_key(), listing.as_slice()))
                .chain(
                    self.records
                        .iter()
                        .map(|(key, _, bytes)| (*key, bytes.as_slice())),
                )
                .collect();
        let cores = std::thread::available_parallelism().map_or(4, usize::from);
        let chunk = records.len().div_ceil(cores).max(1);
        let compressed: Vec<Vec<u8>> = std::thread::scope(|scope| {
            let parts: Vec<_> = records
                .chunks(chunk)
                .map(|records| {
                    scope.spawn(move || {
                        records
                            .iter()
                            .map(|(_, bytes)| compress(bytes))
                            .collect::<Result<Vec<_>, Error>>()
                    })
                })
                .collect();
            parts
                .into_iter()
                .map(|part| {
                    part.join().unwrap_or_else(|_| {
                        Err(Error::new(ErrorCode::RuntimePanicked, "$.compress"))
                    })
                })
                .collect::<Result<Vec<Vec<Vec<u8>>>, Error>>()
                .map(|parts| parts.into_iter().flatten().collect())
        })?;
        let records: Vec<([u8; 32], &[u8], Vec<u8>)> = records
            .into_iter()
            .zip(compressed)
            .map(|((key, bytes), stored)| (key, bytes, stored))
            .collect();
        let bytes = engine_worker::pack_bytes(&records, |directory| {
            compress(directory).unwrap_or_default()
        });
        if let Some(folder) = path.parent() {
            std::fs::create_dir_all(folder).map_err(|error| file_error("$", folder, &error))?;
        }
        let partial = path.with_extension("pack.partial");
        std::fs::write(&partial, &bytes).map_err(|error| file_error("$", &partial, &error))?;
        std::fs::rename(&partial, path).map_err(|error| file_error("$", path, &error))?;
        Ok(bytes.len() as u64)
    }
}

fn compress(bytes: &[u8]) -> Result<Vec<u8>, Error> {
    zstd::bulk::compress(bytes, LEVEL)
        .map_err(|error| Error::new(ErrorCode::SourceIo, "$.compress").got(error.to_string()))
}
