use std::ops::Range;
use std::sync::Arc;

use engine::{Error, ErrorCode, Expected};

const MAGIC: [u8; 4] = *b"ESB1";
const HEADER_LEN: usize = 8;
const ENTRY_HEADER_LEN: usize = 12;

pub fn is_ogg_vorbis(bytes: &[u8]) -> bool {
    if bytes.len() < 35 || &bytes[..4] != b"OggS" {
        return false;
    }
    let start = 27 + usize::from(bytes[26]);
    bytes.get(start..start + 7) == Some(b"\x01vorbis")
}

#[derive(Clone, Debug)]
pub struct SoundBank {
    bytes: Arc<[u8]>,
    entries: Vec<(String, Range<usize>)>,
}

fn word(bytes: &[u8], at: usize) -> Option<u32> {
    let slice: [u8; 4] = bytes.get(at..at + 4)?.try_into().ok()?;
    Some(u32::from_le_bytes(slice))
}

impl SoundBank {
    pub fn read(bytes: Arc<[u8]>, at: &str) -> Result<Self, Error> {
        let invalid = |field: &str| {
            Error::new(ErrorCode::InvalidRecord, format!("$.{field}")).within(at)
        };
        if bytes.len() < HEADER_LEN || bytes[..4] != MAGIC {
            return Err(Error::new(ErrorCode::InvalidRecord, "$")
                .expected(Expected::Matching)
                .within(at));
        }
        let count = word(&bytes, 4).unwrap_or(0) as usize;
        let mut entries = Vec::with_capacity(count);
        let mut cursor = HEADER_LEN;
        for index in 0..count {
            let (Some(path_len), Some(offset), Some(length)) = (
                word(&bytes, cursor),
                word(&bytes, cursor + 4),
                word(&bytes, cursor + 8),
            ) else {
                return Err(invalid(&format!("entries[{index}]")));
            };
            cursor += ENTRY_HEADER_LEN;
            let path_end = cursor
                .checked_add(path_len as usize)
                .filter(|&end| end <= bytes.len())
                .ok_or_else(|| invalid(&format!("entries[{index}].path")))?;
            let path = std::str::from_utf8(&bytes[cursor..path_end])
                .map_err(|_| invalid(&format!("entries[{index}].path")).expected(Expected::Utf8))?
                .to_owned();
            cursor = path_end;
            if let Some((previous, _)) = entries.last()
                && path <= *previous
            {
                return Err(invalid(&format!("entries[{index}].path")));
            }
            let start = offset as usize;
            let end = start
                .checked_add(length as usize)
                .filter(|&end| end <= bytes.len())
                .ok_or_else(|| invalid(&format!("entries[{index}].data")))?;
            entries.push((path, start..end));
        }
        Ok(Self { bytes, entries })
    }

    pub fn write(entries: &[(&str, &[u8])]) -> Result<Vec<u8>, String> {
        if entries.is_empty() {
            return Err("sound bank has no entries".to_owned());
        }
        let mut ordered: Vec<(&str, &[u8])> = entries.to_vec();
        ordered.sort_by_key(|&(path, _)| path);
        for window in ordered.windows(2) {
            if window[0].0 == window[1].0 {
                return Err(format!("duplicate sound bank entry `{}`", window[0].0));
            }
        }
        let mut index_len = HEADER_LEN;
        for &(path, data) in &ordered {
            if !is_ogg_vorbis(data) {
                return Err(format!("`{path}` is not Ogg Vorbis"));
            }
            index_len = index_len
                .checked_add(ENTRY_HEADER_LEN + path.len())
                .ok_or_else(|| "sound bank header overflow".to_owned())?;
        }
        let mut offsets = Vec::with_capacity(ordered.len());
        let mut data_at = index_len.next_multiple_of(4);
        for &(_, data) in &ordered {
            let offset = u32::try_from(data_at)
                .map_err(|_| "sound bank exceeds 4 GiB".to_owned())?;
            let length = u32::try_from(data.len())
                .map_err(|_| "sound bank entry exceeds 4 GiB".to_owned())?;
            offsets.push((offset, length));
            data_at = data_at
                .checked_add(data.len())
                .ok_or_else(|| "sound bank size overflow".to_owned())?
                .next_multiple_of(4);
        }
        let mut out = vec![0_u8; data_at];
        out[..4].copy_from_slice(&MAGIC);
        let count = u32::try_from(ordered.len())
            .map_err(|_| "sound bank entry count overflow".to_owned())?;
        out[4..8].copy_from_slice(&count.to_le_bytes());
        let mut cursor = HEADER_LEN;
        for (&(path, data), &(offset, length)) in ordered.iter().zip(&offsets) {
            let path_len = u32::try_from(path.len())
                .map_err(|_| "sound bank path too long".to_owned())?;
            out[cursor..cursor + 4].copy_from_slice(&path_len.to_le_bytes());
            out[cursor + 4..cursor + 8].copy_from_slice(&offset.to_le_bytes());
            out[cursor + 8..cursor + 12].copy_from_slice(&length.to_le_bytes());
            cursor += ENTRY_HEADER_LEN;
            out[cursor..cursor + path.len()].copy_from_slice(path.as_bytes());
            cursor += path.len();
            let start = offset as usize;
            out[start..start + data.len()].copy_from_slice(data);
        }
        Ok(out)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &[u8])> + '_ {
        self.entries
            .iter()
            .map(|(path, range)| (path.as_str(), &self.bytes[range.clone()]))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_ogg(payload: &[u8]) -> Vec<u8> {
        let mut out = vec![0_u8; 27];
        out[..4].copy_from_slice(b"OggS");
        out[26] = 1;
        out.push(0x1E);
        out.extend_from_slice(b"\x01vorbis");
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn round_trips_sorted_entries() {
        let a = fake_ogg(b"alpha");
        let b = fake_ogg(b"beta_payload");
        let encoded = SoundBank::write(&[("assets/sfx/b.ogg", &b), ("assets/sfx/a.ogg", &a)])
            .expect("write_sound_bank");
        let bank = SoundBank::read(Arc::from(encoded), "test.esb").expect("read_sound_bank");
        assert_eq!(bank.len(), 2, "bank_has_two_entries");
        let items: Vec<(&str, &[u8])> = bank.iter().collect();
        assert_eq!(items[0].0, "assets/sfx/a.ogg", "entries_sorted_first");
        assert_eq!(items[0].1, a.as_slice(), "first_payload_matches");
        assert_eq!(items[1].0, "assets/sfx/b.ogg", "entries_sorted_second");
        assert_eq!(items[1].1, b.as_slice(), "second_payload_matches");
    }

    #[test]
    fn rejects_non_ogg_and_duplicates() {
        let invalid = SoundBank::write(&[("assets/sfx/bad.ogg", b"not_an_ogg_file")]);
        assert!(invalid.is_err(), "non_ogg_rejected");
        let a = fake_ogg(b"one");
        let dup = SoundBank::write(&[("assets/sfx/a.ogg", &a), ("assets/sfx/a.ogg", &a)]);
        assert!(dup.is_err(), "duplicate_entry_rejected");
    }
}
