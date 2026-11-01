use engine_codec::{Error, ErrorCode, Expected};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodLevel {
    pub first_index: u32,
    pub index_count: u32,
    pub error: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshLods {
    pub levels: Vec<LodLevel>,
    pub casters: Vec<(u32, u32)>,
    pub indices: Vec<u32>,
}

const LODS_MAGIC: &[u8; 4] = b"ELOD";
const INPUT_MAGIC: &[u8; 4] = b"ELDI";

impl MeshLods {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            16 + self.levels.len() * 12 + self.casters.len() * 8 + self.indices.len() * 4,
        );
        bytes.extend_from_slice(LODS_MAGIC);
        for count in [self.levels.len(), self.casters.len(), self.indices.len()] {
            bytes.extend_from_slice(&(count as u32).to_le_bytes());
        }
        for level in &self.levels {
            bytes.extend_from_slice(&level.first_index.to_le_bytes());
            bytes.extend_from_slice(&level.index_count.to_le_bytes());
            bytes.extend_from_slice(&level.error.to_le_bytes());
        }
        for &(first, count) in &self.casters {
            bytes.extend_from_slice(&first.to_le_bytes());
            bytes.extend_from_slice(&count.to_le_bytes());
        }
        for index in &self.indices {
            bytes.extend_from_slice(&index.to_le_bytes());
        }
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut words = Words::new(bytes, LODS_MAGIC)
            .ok_or_else(|| invalid("$.lods").expected(Expected::Matching))?;
        let counts = [words.next(), words.next(), words.next()];
        let [Some(levels), Some(casters), Some(indices)] = counts else {
            return Err(invalid("$.lods")
                .expected(Expected::Present)
                .got(bytes.len()));
        };
        let expected = (levels as usize)
            .checked_mul(12)
            .and_then(|levels| levels.checked_add(casters as usize * 8))
            .and_then(|size| size.checked_add(indices as usize * 4))
            .and_then(|size| size.checked_add(16));
        if expected != Some(bytes.len()) {
            return Err(sized("$.lods", expected, bytes.len()));
        }
        let mut lods = Self::default();
        for level in 0..levels {
            let (first_index, index_count, error) = (words.next(), words.next(), words.next());
            let (Some(first_index), Some(index_count), Some(error)) =
                (first_index, index_count, error)
            else {
                return Err(invalid(format!("$.lods.levels[{level}]")).expected(Expected::Present));
            };
            lods.levels.push(LodLevel {
                first_index,
                index_count,
                error: f32::from_bits(error),
            });
        }
        for caster in 0..casters {
            let (Some(first), Some(count)) = (words.next(), words.next()) else {
                return Err(
                    invalid(format!("$.lods.casters[{caster}]")).expected(Expected::Present)
                );
            };
            lods.casters.push((first, count));
        }
        lods.indices = (0..indices).filter_map(|_| words.next()).collect();
        Ok(lods)
    }

    pub fn check(&self, index_count: usize, vertices: usize) -> Result<(), Error> {
        let total = index_count + self.indices.len();
        let within = |first: u32, count: u32| {
            count.is_multiple_of(3)
                && (first as usize)
                    .checked_add(count as usize)
                    .is_some_and(|end| end <= total)
        };
        let outside = |path: String, first: u32, count: u32| {
            invalid(path)
                .expected(Expected::Triangles)
                .at_most(total as u64)
                .got([first, count])
        };
        let mut last = 0.0;
        for (index, level) in self.levels.iter().enumerate() {
            if !within(level.first_index, level.index_count) {
                return Err(outside(
                    format!("$.lods.levels[{index}]"),
                    level.first_index,
                    level.index_count,
                ));
            }
            if !(level.error.is_finite() && level.error > last) {
                return Err(invalid(format!("$.lods.levels[{index}].error"))
                    .expected(Expected::Above)
                    .range(f64::from(last), f64::MAX)
                    .got(level.error));
            }
            last = level.error;
        }
        if !(self.casters.is_empty() || self.casters.len() == self.levels.len() + 1) {
            return Err(invalid("$.lods.casters")
                .expected(Expected::Matching)
                .got(self.casters.len())
                .nearest(self.levels.len() + 1));
        }
        if let Some(index) = self
            .casters
            .iter()
            .position(|&(first, count)| !within(first, count))
        {
            let (first, count) = self.casters[index];
            return Err(outside(format!("$.lods.casters[{index}]"), first, count));
        }
        if let Some(index) = self
            .indices
            .iter()
            .position(|&index| index as usize >= vertices)
        {
            return Err(invalid(format!("$.lods.indices[{index}]"))
                .at_most(vertices.saturating_sub(1) as u64)
                .got(self.indices[index]));
        }
        Ok(())
    }

    pub fn byte_len(&self) -> usize {
        self.indices.len() * 4 + self.levels.len() * 12 + self.casters.len() * 8
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LodInput {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

pub fn input_bytes(vertices: &[MeshVertex], indices: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(12 + vertices.len() * 32 + indices.len() * 4);
    bytes.extend_from_slice(INPUT_MAGIC);
    bytes.extend_from_slice(&(vertices.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(indices.len() as u32).to_le_bytes());
    for vertex in vertices {
        for value in vertex
            .position
            .iter()
            .chain(&vertex.normal)
            .chain(&vertex.uv)
        {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    for index in indices {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    bytes
}

impl LodInput {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut words = Words::new(bytes, INPUT_MAGIC)
            .ok_or_else(|| invalid("$.input").expected(Expected::Matching))?;
        let (Some(vertices), Some(indices)) = (words.next(), words.next()) else {
            return Err(invalid("$.input")
                .expected(Expected::Present)
                .got(bytes.len()));
        };
        let expected = (vertices as usize)
            .checked_mul(32)
            .and_then(|size| size.checked_add(indices as usize * 4))
            .and_then(|size| size.checked_add(12));
        if expected != Some(bytes.len()) {
            return Err(sized("$.input", expected, bytes.len()));
        }
        let mut float = || words.next().map_or(0.0, f32::from_bits);
        let vertices = (0..vertices)
            .map(|_| MeshVertex {
                position: [float(), float(), float()],
                normal: [float(), float(), float()],
                uv: [float(), float()],
            })
            .collect();
        let indices = (0..indices).filter_map(|_| words.next()).collect();
        Ok(Self { vertices, indices })
    }
}

fn invalid(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidMesh, path)
}

fn sized(path: &str, expected: Option<usize>, got: usize) -> Error {
    let error = invalid(path).expected(Expected::Matching).got(got);
    match expected {
        Some(expected) => error.range(expected as f64, expected as f64),
        None => error,
    }
}

struct Words<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Words<'a> {
    fn new(bytes: &'a [u8], magic: &[u8; 4]) -> Option<Self> {
        (bytes.get(..4) == Some(magic.as_slice())).then_some(Self { bytes, at: 4 })
    }

    fn next(&mut self) -> Option<u32> {
        let word = self.bytes.get(self.at..self.at + 4)?;
        self.at += 4;
        Some(u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
    }
}
