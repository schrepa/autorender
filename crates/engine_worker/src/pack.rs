//! Packs: the records the cook made, in one file a host reads by byte
//! range, on disk or from a server. A header, then a directory of every
//! record by key, then the records in the order the cook took them (the
//! coarsest and nearest first), each compressed on its own with zstd so a
//! byte range of the file is a record a server can cache whole. The
//! directory, compressed too, lists every record by key: its key's 32
//! bytes, its offset from the first record, and its bytes stored and
//! decompressed.
//!
//! Header, little-endian: the magic `EPAK`, the records' count, the
//! directory's offset, its bytes stored and its bytes decompressed.

/// The header's bytes, which name the directory's range.
pub const PACK_HEADER: usize = 32;
const MAGIC: &[u8; 4] = b"EPAK";
/// Bytes of one directory entry: a key, an offset and two lengths.
const ENTRY: usize = 48;

/// Where a record lies in its pack (its offset from the file's start) and
/// how large it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackEntry {
    pub key: [u8; 32],
    pub offset: u64,
    pub stored: u32,
    pub raw: u32,
}

/// A pack's directory: its records by key.
#[derive(Clone, Debug, Default)]
pub struct PackIndex {
    entries: Vec<PackEntry>,
}

/// The directory's range in a pack and its bytes decompressed, from the
/// pack's first [`PACK_HEADER`] bytes.
pub fn pack_directory(header: &[u8]) -> Result<(u64, u64, u64), String> {
    let header = header
        .get(..PACK_HEADER)
        .ok_or("a pack's header is 32 bytes")?;
    if &header[..4] != MAGIC {
        return Err("not a pack".into());
    }
    let word = |at: usize| u64::from_le_bytes(header[at..at + 8].try_into().unwrap_or_default());
    Ok((word(8), word(16), word(24)))
}

impl PackIndex {
    /// The directory of the pack whose `header` names it and whose
    /// directory's bytes, as stored, are `directory`.
    pub fn open(header: &[u8], directory: &[u8]) -> Result<Self, String> {
        let (at, stored, raw) = pack_directory(header)?;
        let count = u32::from_le_bytes(header[4..8].try_into().unwrap_or_default()) as usize;
        if directory.len() as u64 != stored || raw != (count * ENTRY) as u64 {
            return Err("the pack's directory has the wrong size".into());
        }
        let first = at + stored;
        let bytes = decompress(directory, count * ENTRY)?;
        let entries: Vec<PackEntry> = bytes
            .as_chunks::<ENTRY>()
            .0
            .iter()
            .map(|entry| {
                let mut key = [0; 32];
                key.copy_from_slice(&entry[..32]);
                PackEntry {
                    key,
                    offset: first
                        + u64::from_le_bytes(entry[32..40].try_into().unwrap_or_default()),
                    stored: u32::from_le_bytes(entry[40..44].try_into().unwrap_or_default()),
                    raw: u32::from_le_bytes(entry[44..48].try_into().unwrap_or_default()),
                }
            })
            .collect();
        if !entries.windows(2).all(|pair| pair[0].key < pair[1].key) {
            return Err("the pack's directory is out of order".into());
        }
        Ok(Self { entries })
    }

    /// The record kept as `key`, 64 hexadecimal digits.
    pub fn find(&self, key: &str) -> Option<PackEntry> {
        let key = key_from_hex(key)?;
        self.entries
            .binary_search_by(|entry| entry.key.cmp(&key))
            .ok()
            .map(|at| self.entries[at])
    }

    /// Its records.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The record `entry` names, from its `stored` bytes.
pub fn unpack(entry: &PackEntry, stored: &[u8]) -> Result<Vec<u8>, String> {
    if stored.len() != entry.stored as usize {
        return Err("a pack's record has the wrong size".into());
    }
    decompress(stored, entry.raw as usize)
}

fn decompress(stored: &[u8], raw: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(raw);
    ruzstd::decoding::FrameDecoder::new()
        .decode_all_to_vec(stored, &mut out)
        .map_err(|error| format!("a pack's record does not decompress: {error}"))?;
    if out.len() != raw {
        return Err("a pack's record has the wrong size".into());
    }
    Ok(out)
}

/// A pack's bytes, from its `records` (each key's 32 bytes, the record's
/// bytes and the record compressed) in the order they are stored, the
/// first of each key's alone, and `compress`, which compresses its
/// directory.
pub fn pack_bytes(
    records: &[([u8; 32], &[u8], Vec<u8>)],
    compress: impl Fn(&[u8]) -> Vec<u8>,
) -> Vec<u8> {
    let mut seen = std::collections::BTreeSet::new();
    let records: Vec<&([u8; 32], &[u8], Vec<u8>)> = records
        .iter()
        .filter(|(key, ..)| seen.insert(*key))
        .collect();
    let mut entries: Vec<PackEntry> = Vec::with_capacity(records.len());
    let mut offset = 0_u64;
    for (key, raw, stored) in &records {
        entries.push(PackEntry {
            key: *key,
            offset,
            stored: stored.len() as u32,
            raw: raw.len() as u32,
        });
        offset += stored.len() as u64;
    }
    entries.sort_by_key(|entry| entry.key);
    let mut directory = Vec::with_capacity(entries.len() * ENTRY);
    for entry in &entries {
        directory.extend_from_slice(&entry.key);
        directory.extend_from_slice(&entry.offset.to_le_bytes());
        directory.extend_from_slice(&entry.stored.to_le_bytes());
        directory.extend_from_slice(&entry.raw.to_le_bytes());
    }
    let packed = compress(&directory);
    let mut out = Vec::with_capacity(PACK_HEADER + packed.len() + offset as usize);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    out.extend_from_slice(&(PACK_HEADER as u64).to_le_bytes());
    out.extend_from_slice(&(packed.len() as u64).to_le_bytes());
    out.extend_from_slice(&(directory.len() as u64).to_le_bytes());
    out.extend_from_slice(&packed);
    for (_, _, stored) in records {
        out.extend_from_slice(stored);
    }
    out
}

/// The key `hex` names, 64 lowercase hexadecimal digits.
pub fn key_from_hex(hex: &str) -> Option<[u8; 32]> {
    let digits = hex.as_bytes();
    if digits.len() != 64 {
        return None;
    }
    let digit = |byte: u8| match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    };
    let mut key = [0; 32];
    for (at, pair) in digits.as_chunks::<2>().0.iter().enumerate() {
        key[at] = digit(pair[0])? << 4 | digit(pair[1])?;
    }
    Some(key)
}
