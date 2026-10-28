use std::path::Path;

use engine::collections::HashSet;

const LEVEL: i32 = 19;

#[derive(Default)]
pub(crate) struct PackWriter {
    records: Vec<([u8; 32], Vec<u8>)>,
    seen: HashSet<[u8; 32]>,
}

impl PackWriter {
    pub(crate) fn take(&mut self, key: &str, bytes: &[u8]) {
        let Some(key) = engine_worker::key_from_hex(key) else {
            return;
        };
        if self.seen.insert(key) {
            self.records.push((key, bytes.to_vec()));
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.records.len()
    }

    pub(crate) fn write(&self, path: &Path) -> Result<u64, String> {
        let cores = std::thread::available_parallelism().map_or(4, usize::from);
        let chunk = self.records.len().div_ceil(cores).max(1);
        let compressed: Vec<Vec<u8>> = std::thread::scope(|scope| {
            let parts: Vec<_> = self
                .records
                .chunks(chunk)
                .map(|records| {
                    scope.spawn(move || {
                        records
                            .iter()
                            .map(|(_, bytes)| compress(bytes))
                            .collect::<Result<Vec<_>, String>>()
                    })
                })
                .collect();
            parts
                .into_iter()
                .map(|part| {
                    part.join()
                        .unwrap_or_else(|_| Err("a compressor panicked".into()))
                })
                .collect::<Result<Vec<Vec<Vec<u8>>>, String>>()
                .map(|parts| parts.into_iter().flatten().collect())
        })?;
        let records: Vec<([u8; 32], &[u8], Vec<u8>)> = self
            .records
            .iter()
            .zip(compressed)
            .map(|((key, bytes), stored)| (*key, bytes.as_slice(), stored))
            .collect();
        let bytes = engine_worker::pack_bytes(&records, |directory| {
            compress(directory).unwrap_or_default()
        });
        if let Some(folder) = path.parent() {
            std::fs::create_dir_all(folder).map_err(|error| error.to_string())?;
        }
        let partial = path.with_extension("pack.partial");
        std::fs::write(&partial, &bytes).map_err(|error| error.to_string())?;
        std::fs::rename(&partial, path).map_err(|error| error.to_string())?;
        Ok(bytes.len() as u64)
    }
}

fn compress(bytes: &[u8]) -> Result<Vec<u8>, String> {
    zstd::bulk::compress(bytes, LEVEL).map_err(|error| error.to_string())
}
