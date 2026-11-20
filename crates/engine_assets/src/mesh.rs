use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use engine_codec::{Error, ErrorCode, Expected};
use glam::Vec3;

const MAGIC: [u8; 4] = *b"EMSH";
const FIXED: usize = 48;

fn invalid(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidMesh, path)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum VertexFormat {
    Float32x2,
    Float32x3,
    Snorm16x2,
    Unorm16x4,
    Uint32x4,
}

impl VertexFormat {
    const ALL: [Self; 5] = [
        Self::Float32x2,
        Self::Float32x3,
        Self::Snorm16x2,
        Self::Unorm16x4,
        Self::Uint32x4,
    ];

    pub const fn bytes(self) -> u32 {
        match self {
            Self::Snorm16x2 => 4,
            Self::Float32x2 | Self::Unorm16x4 => 8,
            Self::Float32x3 => 12,
            Self::Uint32x4 => 16,
        }
    }

    const fn floats(self) -> usize {
        match self {
            Self::Float32x2 => 2,
            Self::Float32x3 => 3,
            Self::Snorm16x2 | Self::Unorm16x4 | Self::Uint32x4 => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexInput {
    pub location: u32,
    pub format: VertexFormat,
    pub offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VertexStream {
    pub stride: u32,
    pub inputs: Cow<'static, [VertexInput]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

impl IndexFormat {
    const ALL: [Self; 2] = [Self::Uint16, Self::Uint32];

    pub const fn bytes(self) -> u32 {
        match self {
            Self::Uint16 => 2,
            Self::Uint32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshDescription {
    pub vertex_count: u32,
    pub index_format: IndexFormat,
    pub index_count: u32,
    pub bounds: Aabb,
    pub streams: Vec<VertexStream>,
}

impl MeshDescription {
    fn input_count(&self) -> usize {
        self.streams.iter().map(|stream| stream.inputs.len()).sum()
    }

    fn header_len(&self) -> usize {
        FIXED + 4 * self.streams.len() + 16 * self.input_count()
    }

    fn stream_len(&self, stream: &VertexStream) -> usize {
        self.vertex_count as usize * stream.stride as usize
    }

    fn index_len(&self) -> usize {
        self.index_count as usize * self.index_format.bytes() as usize
    }

    fn start(&self, stream: usize) -> usize {
        self.streams[..stream]
            .iter()
            .fold(self.header_len(), |start, held| {
                start + self.stream_len(held)
            })
    }

    fn file_len(&self) -> Option<usize> {
        let indices = (self.index_count as usize).checked_mul(self.index_format.bytes() as usize)?;
        self.streams
            .iter()
            .try_fold(self.header_len(), |at, stream| {
                (self.vertex_count as usize)
                    .checked_mul(stream.stride as usize)
                    .and_then(|length| at.checked_add(length))
            })?
            .checked_add(indices.next_multiple_of(4))
    }

    fn check(&self) -> Result<usize, Error> {
        if self.vertex_count == 0 {
            return Err(invalid("$.vertex_count")
                .expected(Expected::NonEmpty)
                .got(0));
        }
        if self.index_count == 0 || !self.index_count.is_multiple_of(3) {
            return Err(invalid("$.index_count")
                .expected(Expected::Triangles)
                .got(self.index_count));
        }
        if self.streams.is_empty() {
            return Err(invalid("$.streams").expected(Expected::NonEmpty).got(0));
        }
        let mut locations = Vec::with_capacity(self.input_count());
        for (at, stream) in self.streams.iter().enumerate() {
            let path = format!("$.streams[{at}]");
            if stream.stride == 0 || !stream.stride.is_multiple_of(4) {
                return Err(invalid(format!("{path}.stride"))
                    .expected(Expected::Multiple)
                    .got(stream.stride));
            }
            if stream.inputs.is_empty() {
                return Err(invalid(format!("{path}.inputs"))
                    .expected(Expected::NonEmpty)
                    .got(0));
            }
            for (place, input) in stream.inputs.iter().enumerate() {
                let path = format!("{path}.inputs[{place}]");
                let room = stream.stride.checked_sub(input.format.bytes());
                if !input.offset.is_multiple_of(4) || room.is_none_or(|room| input.offset > room) {
                    return Err(invalid(format!("{path}.offset"))
                        .expected(Expected::Multiple)
                        .at_most(u64::from(room.unwrap_or_default()))
                        .got(input.offset));
                }
                if locations.contains(&input.location) {
                    return Err(invalid(format!("{path}.location"))
                        .expected(Expected::Unique)
                        .got(input.location));
                }
                locations.push(input.location);
            }
        }
        let Aabb { min, max } = self.bounds;
        if !min.is_finite() || !max.is_finite() {
            return Err(invalid("$.bounds")
                .expected(Expected::Finite)
                .got([min.to_array(), max.to_array()]));
        }
        if !min.cmple(max).all() {
            return Err(invalid("$.bounds")
                .expected(Expected::Sorted)
                .got([min.to_array(), max.to_array()]));
        }
        self.file_len()
            .ok_or_else(|| invalid("$.vertex_count").got(self.vertex_count))
    }
}

#[derive(Clone)]
pub struct MeshFile {
    description: MeshDescription,
    bytes: Arc<[u8]>,
}

impl fmt::Debug for MeshFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.description.fmt(f)
    }
}

fn word(bytes: &[u8], at: usize) -> Result<u32, Error> {
    bytes
        .get(at..at.saturating_add(4))
        .map(|held| u32::from_le_bytes(held.try_into().expect("four_bytes")))
        .ok_or_else(|| {
            invalid("$")
                .at_least(at.saturating_add(4) as u64)
                .got(bytes.len())
        })
}

fn corner(bytes: &[u8], at: usize) -> Result<Vec3, Error> {
    let lane = |offset: usize| word(bytes, at + offset).map(f32::from_bits);
    Ok(Vec3::new(lane(0)?, lane(4)?, lane(8)?))
}

fn parse(bytes: &[u8]) -> Result<MeshDescription, Error> {
    if bytes.get(..4) != Some(&MAGIC[..]) {
        return Err(invalid("$.magic")
            .variants(["EMSH"])
            .got(String::from_utf8_lossy(&bytes[..bytes.len().min(4)])));
    }
    let code = word(bytes, 8)?;
    let index_format = *IndexFormat::ALL
        .get(code as usize)
        .ok_or_else(|| invalid("$.index_format").at_most(1).got(code))?;
    let (stream_count, input_count) = (word(bytes, 16)? as usize, word(bytes, 20)? as usize);
    let table_end = stream_count
        .checked_mul(4)
        .and_then(|strides| input_count.checked_mul(16)?.checked_add(strides))
        .and_then(|table| table.checked_add(FIXED));
    if table_end.is_none_or(|end| end > bytes.len()) {
        return Err(invalid("$.inputs").got([stream_count, input_count]));
    }
    let records = FIXED + 4 * stream_count;
    let mut streams = (0..stream_count)
        .map(|at| {
            Ok(VertexStream {
                stride: word(bytes, FIXED + 4 * at)?,
                inputs: Cow::Owned(Vec::new()),
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    for at in 0..input_count {
        let base = records + 16 * at;
        let (stream, code) = (word(bytes, base)?, word(bytes, base + 8)?);
        let format = *VertexFormat::ALL.get(code as usize).ok_or_else(|| {
            invalid(format!("$.inputs[{at}].format"))
                .at_most(VertexFormat::ALL.len() as u64 - 1)
                .got(code)
        })?;
        let held = streams.get_mut(stream as usize).ok_or_else(|| {
            invalid(format!("$.inputs[{at}].stream"))
                .at_most(stream_count.saturating_sub(1) as u64)
                .got(stream)
        })?;
        held.inputs.to_mut().push(VertexInput {
            location: word(bytes, base + 4)?,
            format,
            offset: word(bytes, base + 12)?,
        });
    }
    Ok(MeshDescription {
        vertex_count: word(bytes, 4)?,
        index_format,
        index_count: word(bytes, 12)?,
        bounds: Aabb {
            min: corner(bytes, 24)?,
            max: corner(bytes, 36)?,
        },
        streams,
    })
}

fn index_values(format: IndexFormat, bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes.chunks_exact(format.bytes() as usize).map(|value| {
        value
            .iter()
            .rev()
            .fold(0, |sum, &byte| (sum << 8) | u32::from(byte))
    })
}

fn check_indices(description: &MeshDescription, indices: &[u8]) -> Result<(), Error> {
    let count = description.vertex_count;
    match index_values(description.index_format, indices)
        .enumerate()
        .find(|&(_, value)| value >= count)
    {
        Some((at, value)) => Err(invalid(format!("$.indices[{at}]"))
            .at_most(u64::from(count) - 1)
            .got(value)),
        None => Ok(()),
    }
}

fn lanes<const N: usize>(
    file: &MeshFile,
    (location, format): (u32, VertexFormat),
    path: &str,
) -> Result<Vec<[u32; N]>, Error> {
    let found = file.input(location);
    let Some((input, stride, bytes)) = found.filter(|(input, ..)| input.format == format) else {
        return Err(invalid(format!("$.locations.{location}"))
            .expected(Expected::Present)
            .got(found.map(|(input, ..)| format!("{:?}", input.format)))
            .within(path));
    };
    let start = input.offset as usize;
    Ok(bytes
        .chunks_exact(stride as usize)
        .map(|vertex| {
            let lanes = vertex[start..start + 4 * N].as_chunks::<4>().0;
            std::array::from_fn(|lane| u32::from_le_bytes(lanes[lane]))
        })
        .collect())
}

fn check_floats(stream: &VertexStream, at: usize, bytes: &[u8]) -> Result<(), Error> {
    for input in stream
        .inputs
        .iter()
        .filter(|input| input.format.floats() > 0)
    {
        let floats = input.format.floats();
        for (vertex, held) in bytes.chunks_exact(stream.stride as usize).enumerate() {
            let start = input.offset as usize;
            let values = held[start..start + 4 * floats].as_chunks::<4>().0;
            if let Some(value) = values
                .iter()
                .map(|value| f32::from_le_bytes(*value))
                .find(|value| !value.is_finite())
            {
                return Err(invalid(format!(
                    "$.streams[{at}].locations.{}[{vertex}]",
                    input.location
                ))
                .expected(Expected::Finite)
                .got(value));
            }
        }
    }
    Ok(())
}

impl MeshFile {
    pub fn read(bytes: Arc<[u8]>, path: &str) -> Result<Self, Error> {
        let (description, expected) = parse(&bytes)
            .and_then(|description| description.check().map(|length| (description, length)))
            .map_err(|error| error.within(path))?;
        if bytes.len() != expected {
            return Err(invalid("$")
                .range(expected as f64, expected as f64)
                .got(bytes.len())
                .within(path));
        }
        let file = Self { description, bytes };
        check_indices(&file.description, file.indices()).map_err(|error| error.within(path))?;
        Ok(file)
    }

    pub fn write(
        description: &MeshDescription,
        streams: &[&[u8]],
        indices: &[u8],
    ) -> Result<Vec<u8>, Error> {
        description.check()?;
        let held = description.streams.len();
        if streams.len() != held {
            return Err(invalid("$.streams")
                .range(held as f64, held as f64)
                .got(streams.len()));
        }
        for (at, (stream, bytes)) in description.streams.iter().zip(streams).enumerate() {
            let expected = description.stream_len(stream);
            if bytes.len() != expected {
                return Err(invalid(format!("$.streams[{at}]"))
                    .range(expected as f64, expected as f64)
                    .got(bytes.len()));
            }
            check_floats(stream, at, bytes)?;
        }
        let expected = description.index_len();
        if indices.len() != expected {
            return Err(invalid("$.indices")
                .range(expected as f64, expected as f64)
                .got(indices.len()));
        }
        check_indices(description, indices)?;
        Ok(encode(description, streams, indices))
    }

    pub fn description(&self) -> &MeshDescription {
        &self.description
    }

    pub fn stream(&self, at: usize) -> &[u8] {
        let description = &self.description;
        let start = description.start(at);
        &self.bytes[start..start + description.stream_len(&description.streams[at])]
    }

    pub fn indices(&self) -> &[u8] {
        let description = &self.description;
        let start = description.start(description.streams.len());
        &self.bytes[start..start + description.index_len()]
    }

    pub fn input(&self, location: u32) -> Option<(VertexInput, u32, &[u8])> {
        self.description
            .streams
            .iter()
            .enumerate()
            .find_map(|(at, stream)| {
                let input = stream
                    .inputs
                    .iter()
                    .find(|input| input.location == location)?;
                Some((*input, stream.stride, self.stream(at)))
            })
    }

    pub fn floats3(&self, location: u32, path: &str) -> Result<Vec<[f32; 3]>, Error> {
        let lanes = lanes::<3>(self, (location, VertexFormat::Float32x3), path)?;
        Ok(lanes
            .into_iter()
            .map(|held| held.map(f32::from_bits))
            .collect())
    }

    pub fn uints4(&self, location: u32, path: &str) -> Result<Vec<[u32; 4]>, Error> {
        lanes::<4>(self, (location, VertexFormat::Uint32x4), path)
    }

    pub fn index_values(&self) -> Vec<u32> {
        index_values(self.description.index_format, self.indices()).collect()
    }

    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }
}

fn encode(description: &MeshDescription, streams: &[&[u8]], indices: &[u8]) -> Vec<u8> {
    let mut out = Vec::from(MAGIC);
    let Aabb { min, max } = description.bounds;
    let words = [
        description.vertex_count,
        description.index_format as u32,
        description.index_count,
        description.streams.len() as u32,
        description.input_count() as u32,
    ]
    .into_iter()
    .chain(
        min.to_array()
            .into_iter()
            .chain(max.to_array())
            .map(f32::to_bits),
    )
    .chain(description.streams.iter().map(|stream| stream.stride));
    for value in words {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for (at, stream) in description.streams.iter().enumerate() {
        for input in stream.inputs.iter() {
            for value in [at as u32, input.location, input.format as u32, input.offset] {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    for stream in streams {
        out.extend_from_slice(stream);
    }
    out.extend_from_slice(indices);
    out.resize(out.len().next_multiple_of(4), 0);
    out
}
