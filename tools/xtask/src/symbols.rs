//! Match Windows symbols by the linker's CodeView GUID and age. Reproducible
//! links can emit identical executables with differently ordered PDB streams.
//! Raw hashes still protect cache integrity; these IDs describe compatibility.
//! Format references: <https://llvm.org/docs/PDB/PdbStream.html> and
//! <https://learn.microsoft.com/en-us/windows/win32/debug/pe-format>.

use serde_json::{Value, json};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

struct Reader {
    file: File,
    size: u64,
}

impl Reader {
    fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let size = file.metadata().map_err(|e| e.to_string())?.len();
        Ok(Self { file, size })
    }

    fn at(&mut self, offset: u64, count: usize) -> Result<Vec<u8>, String> {
        if count > 4 * 1024 * 1024
            || offset
                .checked_add(count as u64)
                .is_none_or(|end| end > self.size)
        {
            return Err("invalid or oversized symbol metadata range".into());
        }
        let mut bytes = vec![0; count];
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|e| e.to_string())?;
        self.file
            .read_exact(&mut bytes)
            .map_err(|e| e.to_string())?;
        Ok(bytes)
    }
}

fn word(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let bytes = bytes
        .get(offset..offset + 4)
        .ok_or("truncated symbol metadata")?;
    Ok(u32::from_le_bytes(bytes.try_into().expect("four bytes")))
}

fn short(bytes: &[u8], offset: usize) -> Result<u16, String> {
    let bytes = bytes.get(offset..offset + 2).ok_or("truncated PE header")?;
    Ok(u16::from_le_bytes(bytes.try_into().expect("two bytes")))
}

#[derive(PartialEq)]
struct Id {
    guid: [u8; 16],
    age: u32,
}

fn pdb_id(path: &Path) -> Result<Id, String> {
    let mut reader = Reader::open(path)?;
    let header = reader.at(0, 56)?;
    if &header[..32] != b"Microsoft C/C++ MSF 7.00\r\n\x1aDS\0\0\0" {
        return Err("unsupported PDB container (expected MSF 7)".into());
    }
    let block = word(&header, 32)? as usize;
    let directory_size = word(&header, 44)? as usize;
    if !(512..=65536).contains(&block)
        || !block.is_power_of_two()
        || u64::from(word(&header, 40)?) * block as u64 != reader.size
        || directory_size > 4 * 1024 * 1024
    {
        return Err("invalid PDB block geometry".into());
    }
    let map = reader.at(
        u64::from(word(&header, 52)?) * block as u64,
        directory_size.div_ceil(block) * 4,
    )?;
    let mut directory = Vec::with_capacity(directory_size);
    for position in (0..map.len()).step_by(4) {
        let count = block.min(directory_size - directory.len());
        directory.extend(reader.at(u64::from(word(&map, position)?) * block as u64, count)?);
    }
    let streams = word(&directory, 0)? as usize;
    if streams < 2 || streams > directory.len() / 4 {
        return Err("PDB has no valid info stream".into());
    }
    let first_size = word(&directory, 4)?;
    let first_blocks = if first_size == u32::MAX {
        0
    } else {
        (first_size as usize).div_ceil(block)
    };
    let info_size = word(&directory, 8)?;
    if !(28..u32::MAX).contains(&info_size) {
        return Err("PDB info stream is missing or truncated".into());
    }
    let info_block = word(&directory, 4 + streams * 4 + first_blocks * 4)?;
    let info = reader.at(u64::from(info_block) * block as u64, 28)?;
    Ok(Id {
        guid: info[12..28].try_into().expect("sixteen bytes"),
        age: word(&info, 8)?,
    })
}

fn debug_offset(
    reader: &mut Reader,
    pe: u64,
    header: &[u8],
    rva: u32,
    size: u32,
) -> Result<u64, String> {
    let count = usize::from(short(header, 6)?);
    if count > 96 {
        return Err("invalid PE section count".into());
    }
    let sections = reader.at(pe + 24 + u64::from(short(header, 20)?), count * 40)?;
    for section in sections.as_chunks::<40>().0 {
        let start = word(section, 12)?;
        let length = word(section, 16)?;
        if let Some(relative) = rva.checked_sub(start)
            && relative.checked_add(size).is_some_and(|end| end <= length)
        {
            return Ok(u64::from(word(section, 20)?) + u64::from(relative));
        }
    }
    Err("PE debug directory is outside file-backed sections".into())
}

fn pe_id(path: &Path) -> Result<Id, String> {
    let mut reader = Reader::open(path)?;
    let dos = reader.at(0, 64)?;
    if &dos[..2] != b"MZ" {
        return Err("symbols require a PE executable".into());
    }
    let pe = u64::from(word(&dos, 60)?);
    let header = reader.at(pe, 24)?;
    if &header[..4] != b"PE\0\0" {
        return Err("invalid PE signature".into());
    }
    let optional = reader.at(pe + 24, usize::from(short(&header, 20)?))?;
    let directories = match short(&optional, 0)? {
        0x10b => 96,
        0x20b => 112,
        _ => return Err("unsupported PE optional header".into()),
    };
    if word(&optional, directories - 4)? <= 6 {
        return Err("PE has no debug directory".into());
    }
    let rva = word(&optional, directories + 6 * 8)?;
    let size = word(&optional, directories + 6 * 8 + 4)?;
    if size == 0 || size > 16384 || size % 28 != 0 {
        return Err("invalid PE debug directory size".into());
    }
    let offset = debug_offset(&mut reader, pe, &header, rva, size)?;
    let entries = reader.at(offset, size as usize)?;
    let mut found = None;
    for entry in entries.as_chunks::<28>().0 {
        if word(entry, 12)? != 2 {
            continue;
        }
        if word(entry, 16)? < 24 {
            return Err("truncated CodeView entry".into());
        }
        let cv = reader.at(u64::from(word(entry, 24)?), 24)?;
        if &cv[..4] != b"RSDS" || found.is_some() {
            return Err("unsupported or ambiguous CodeView identifier".into());
        }
        found = Some(Id {
            guid: cv[4..20].try_into().expect("sixteen bytes"),
            age: word(&cv, 20)?,
        });
    }
    found.ok_or_else(|| "PE has no PDB identifier".into())
}

/// Fail closed on mismatched or malformed symbols. MSF allocation and type
/// record order may differ while the executable and its symbol ID are equal.
pub fn matched_id(executable: &Path, pdb: &Path) -> Result<Value, String> {
    let expected = pe_id(executable)?;
    let actual = pdb_id(pdb)?;
    if actual != expected {
        return Err(format!(
            "{} does not match {}",
            pdb.display(),
            executable.display()
        ));
    }
    let guid: String = actual
        .guid
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok(json!({"format":"pdb70", "guid_bytes":guid, "age":actual.age}))
}
