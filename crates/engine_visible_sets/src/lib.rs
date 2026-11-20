#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

const MAGIC: &[u8; 4] = b"LVS1";

#[derive(Clone, Debug, PartialEq)]
pub struct VisibleSets {
    origin: [f32; 2],
    cell_m: f32,
    cells: [u32; 2],
    top_m: f32,
    ids: Vec<String>,
    floors: Vec<f32>,
    stride: usize,
    kept: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
pub struct Cell<'a> {
    pub cell: [u32; 2],
    row: &'a [u8],
}

impl Cell<'_> {
    pub fn kept(&self, chunk: u32) -> bool {
        self.row
            .get(chunk as usize / 8)
            .is_some_and(|byte| byte & (1 << (chunk % 8)) != 0)
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, count: usize) -> Result<&[u8], String> {
        let end = self
            .at
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len());
        let Some(end) = end else {
            return Err(format!(
                "ends at byte {} while {count} more were needed",
                self.bytes.len()
            ));
        };
        let taken = &self.bytes[self.at..end];
        self.at = end;
        Ok(taken)
    }

    fn word(&mut self) -> Result<[u8; 4], String> {
        let taken = self.take(4)?;
        Ok([taken[0], taken[1], taken[2], taken[3]])
    }

    fn count(&mut self) -> Result<u32, String> {
        self.word().map(u32::from_le_bytes)
    }

    fn real(&mut self) -> Result<f32, String> {
        self.word().map(f32::from_le_bytes)
    }
}

fn sized(cell_m: f32, top_m: f32) -> Result<(), String> {
    if !(cell_m.is_finite() && cell_m > 0.0 && top_m.is_finite() && top_m > 0.0) {
        return Err(format!("cell_m {cell_m} and top_m {top_m} must be above 0"));
    }
    Ok(())
}

impl VisibleSets {
    pub fn new(
        origin: [f32; 2],
        cell_m: f32,
        cells: [u32; 2],
        top_m: f32,
        ids: Vec<String>,
        floors: Vec<f32>,
        kept: &[Vec<bool>],
    ) -> Result<Self, String> {
        sized(cell_m, top_m)?;
        u32::try_from(ids.len()).map_err(|_| "too many chunks".to_owned())?;
        if let Some(long) = ids.iter().find(|id| u16::try_from(id.len()).is_err()) {
            return Err(format!("chunk id `{long}` is longer than 65535 bytes"));
        }
        let count = (cells[0] as usize).checked_mul(cells[1] as usize);
        if count != Some(floors.len()) || count != Some(kept.len()) {
            return Err(format!(
                "{} floors and {} kept rows for {} by {} cells",
                floors.len(),
                kept.len(),
                cells[0],
                cells[1]
            ));
        }
        if let Some(at) = kept.iter().position(|row| row.len() != ids.len()) {
            return Err(format!(
                "kept row {at} holds {} chunks, not {}",
                kept[at].len(),
                ids.len()
            ));
        }
        let stride = ids.len().div_ceil(8);
        let mut packed = Vec::with_capacity(kept.len() * stride);
        for row in kept {
            let start = packed.len();
            packed.resize(start + stride, 0_u8);
            for (at, held) in row.iter().enumerate() {
                if *held {
                    packed[start + at / 8] |= 1 << (at % 8);
                }
            }
        }
        Ok(Self {
            origin,
            cell_m,
            cells,
            top_m,
            ids,
            floors,
            stride,
            kept: packed,
        })
    }

    pub fn read(bytes: &[u8]) -> Result<Self, String> {
        let mut read = Reader { bytes, at: 0 };
        if read.take(4)? != MAGIC {
            return Err("does not start with LVS1".into());
        }
        let cells = [read.count()?, read.count()?];
        let origin = [read.real()?, read.real()?];
        let cell_m = read.real()?;
        let top_m = read.real()?;
        sized(cell_m, top_m)?;
        let chunks = read.count()? as usize;
        let mut ids = Vec::new();
        for _ in 0..chunks {
            let length = read.take(2)?;
            let length = u16::from_le_bytes([length[0], length[1]]) as usize;
            let id = std::str::from_utf8(read.take(length)?)
                .map_err(|_| format!("chunk id {} is not UTF-8", ids.len()))?;
            ids.push(id.to_owned());
        }
        let padding = (4 - read.at % 4) % 4;
        read.take(padding)?;
        let stride = chunks.div_ceil(8);
        let count = (cells[0] as usize).checked_mul(cells[1] as usize);
        let sizes =
            count.and_then(|count| Some((count.checked_mul(4)?, count.checked_mul(stride)?)));
        let left = bytes.len() - read.at;
        let Some((floor_bytes, kept_bytes)) = sizes
            .filter(|(floor_bytes, kept_bytes)| floor_bytes.checked_add(*kept_bytes) == Some(left))
        else {
            return Err(format!(
                "{} by {} cells of {chunks} chunks do not fill the {left} bytes after the ids",
                cells[0], cells[1]
            ));
        };
        let (words, _) = read.take(floor_bytes)?.as_chunks::<4>();
        let floors = words.iter().map(|word| f32::from_le_bytes(*word)).collect();
        let kept = read.take(kept_bytes)?.to_vec();
        Ok(Self {
            origin,
            cell_m,
            cells,
            top_m,
            ids,
            floors,
            stride,
            kept,
        })
    }

    pub fn write(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        for value in self.cells {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for value in [self.origin[0], self.origin[1], self.cell_m, self.top_m] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&(self.ids.len() as u32).to_le_bytes());
        for id in &self.ids {
            out.extend_from_slice(&(id.len() as u16).to_le_bytes());
            out.extend_from_slice(id.as_bytes());
        }
        out.resize(out.len().div_ceil(4) * 4, 0);
        for floor in &self.floors {
            out.extend_from_slice(&floor.to_le_bytes());
        }
        out.extend_from_slice(&self.kept);
        out
    }

    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    pub fn at(&self, eye: [f32; 3]) -> Option<Cell<'_>> {
        let x = ((eye[0] - self.origin[0]) / self.cell_m).floor();
        let z = ((eye[2] - self.origin[1]) / self.cell_m).floor();
        if !(x >= 0.0 && z >= 0.0 && x < self.cells[0] as f32 && z < self.cells[1] as f32) {
            return None;
        }
        let cell = [x as u32, z as u32];
        let row = cell[1] as usize * self.cells[0] as usize + cell[0] as usize;
        let floor = *self.floors.get(row)?;
        let height = eye[1] - floor;
        if !(floor.is_finite() && (0.0..=self.top_m).contains(&height)) {
            return None;
        }
        let span = row * self.stride..(row + 1) * self.stride;
        Some(Cell {
            cell,
            row: self.kept.get(span)?,
        })
    }
}
