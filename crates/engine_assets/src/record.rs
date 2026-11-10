#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use engine_codec::{Error, ErrorCode, Expected};
use serde::Serialize;

use crate::{AttributeFormat, Mesh, MeshLods, MeshVertex, NamedStream};

pub const MESH_BUFFER_MAGIC: [u8; 4] = *b"EMB1";
pub const MESH_BUFFER_EXTRAS_MAGIC: [u8; 4] = *b"EMB2";

const COLORS: u32 = 1;
const UV1: u32 = 2;
const NAMED: u32 = 4;
const LODS: u32 = 8;
const FLAGS: u32 = COLORS | UV1 | NAMED | LODS;

#[derive(Clone, Copy, Debug, Default)]
pub struct MeshStreams<'a> {
    pub colors: Option<&'a [[f32; 4]]>,
    pub uv1: Option<&'a [[f32; 2]]>,
    pub named: &'a [NamedStream],
    pub lods: Option<&'a MeshLods>,
}

fn count(length: usize) -> [u8; 4] {
    u32::try_from(length)
        .expect("buffer_count_fits_u32")
        .to_le_bytes()
}

fn flag(held: bool, bit: u32) -> u32 {
    if held { bit } else { 0 }
}

fn width(format: AttributeFormat) -> usize {
    usize::try_from(format.bytes()).expect("format_width_fits_usize")
}

pub fn encode_mesh_buffer_with<V: Copy + Into<MeshVertex>>(
    vertices: &[V],
    indices: &[u32],
    streams: MeshStreams<'_>,
) -> Vec<u8> {
    let MeshStreams {
        colors,
        uv1,
        named,
        lods,
    } = streams;
    let flags = flag(colors.is_some(), COLORS)
        | flag(uv1.is_some(), UV1)
        | flag(!named.is_empty(), NAMED)
        | flag(lods.is_some(), LODS);
    let extras = flags != 0;
    let named_bytes: usize = named.iter().map(|stream| stream.bytes().len()).sum();
    let mut bytes = Vec::with_capacity(
        16 + vertices.len() * (32 + if extras { 24 } else { 0 })
            + named_bytes
            + indices.len() * 4,
    );
    bytes.extend_from_slice(if extras {
        &MESH_BUFFER_EXTRAS_MAGIC
    } else {
        &MESH_BUFFER_MAGIC
    });
    bytes.extend_from_slice(&count(vertices.len()));
    bytes.extend_from_slice(&count(indices.len()));
    if extras {
        bytes.extend_from_slice(&flags.to_le_bytes());
    }
    if !named.is_empty() {
        bytes.extend_from_slice(&count(named.len()));
        for stream in named {
            let name = stream.name().as_bytes();
            bytes.extend_from_slice(&count(name.len()));
            bytes.extend_from_slice(name);
            bytes.resize(bytes.len().next_multiple_of(4), 0);
            bytes.extend_from_slice(&stream.format().code().to_le_bytes());
        }
    }
    for &vertex in vertices {
        let vertex: MeshVertex = vertex.into();
        for value in vertex
            .position
            .iter()
            .chain(&vertex.normal)
            .chain(&vertex.uv)
        {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    for value in colors
        .into_iter()
        .flatten()
        .flatten()
        .chain(uv1.into_iter().flatten().flatten())
    {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for stream in named {
        bytes.extend_from_slice(stream.bytes());
    }
    for index in indices {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    if let Some(lods) = lods {
        bytes.extend_from_slice(&lods.to_bytes());
    }
    bytes
}

impl Mesh {
    pub fn to_record(&self) -> Vec<u8> {
        encode_mesh_buffer_with(
            self.vertices(),
            self.indices(),
            MeshStreams {
                colors: self.colors(),
                uv1: self.uv1(),
                named: self.streams(),
                lods: self.lods().map(AsRef::as_ref),
            },
        )
    }

    pub fn from_record(bytes: &[u8]) -> Result<Self, Error> {
        MeshBuffer::parse(bytes, "$")?.mesh("$")
    }
}

#[derive(Serialize)]
struct VertexPosition<'a> {
    vertex: usize,
    position: &'a [f32],
}

pub struct MeshBuffer {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub colors: Option<Vec<[f32; 4]>>,
    pub uv1: Option<Vec<[f32; 2]>>,
    pub named: Vec<NamedStream>,
    pub lods: Option<MeshLods>,
}

struct Layout {
    header: usize,
    vertex_count: usize,
    index_count: usize,
    flags: u32,
    named: Vec<(String, AttributeFormat)>,
}

impl MeshBuffer {
    pub fn mesh(self, path: &str) -> Result<Mesh, Error> {
        let invalid = |error: Error| error.coded(ErrorCode::InvalidMesh).within(path);
        let mut mesh = Mesh::new(self.vertices, self.indices).map_err(invalid)?;
        if let Some(colors) = self.colors {
            mesh = mesh.with_colors(colors).map_err(invalid)?;
        }
        if let Some(uv1) = self.uv1 {
            mesh = mesh.with_uv1(uv1).map_err(invalid)?;
        }
        for stream in self.named {
            mesh = mesh.with_stream(stream).map_err(invalid)?;
        }
        match self.lods {
            Some(lods) => mesh
                .with_lods(lods)
                .map_err(|error| error.coded(ErrorCode::InvalidMesh).within(path)),
            None => Ok(mesh),
        }
    }

    fn named_table(
        bytes: &[u8],
        mut header: usize,
        path: &str,
    ) -> Result<(usize, Vec<(String, AttributeFormat)>), Error> {
        let invalid = || Error::new(ErrorCode::InvalidMesh, path);
        let short = |needed: usize| {
            invalid()
                .at_least(u64::try_from(needed).unwrap_or(u64::MAX))
                .got(bytes.len())
        };
        let word = |offset: usize| {
            bytes
                .get(offset..offset + 4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("four_bytes")))
                .ok_or_else(|| short(offset + 4))
        };
        let streams = word(header)? as usize;
        header += 4;
        let mut named = Vec::new();
        for _ in 0..streams {
            let length = word(header)? as usize;
            header += 4;
            let end = header.saturating_add(length);
            let name = bytes.get(header..end).ok_or_else(|| short(end))?;
            let name = std::str::from_utf8(name)
                .ok()
                .filter(|name| !name.is_empty())
                .ok_or_else(|| invalid().got(String::from_utf8_lossy(name)))?
                .to_owned();
            header += length.next_multiple_of(4);
            let code = word(header)?;
            header += 4;
            let format = AttributeFormat::of_code(code).ok_or_else(|| {
                invalid()
                    .at_most(u64::from(AttributeFormat::Uint32x4.code()))
                    .got(code)
            })?;
            named.push((name, format));
        }
        Ok((header, named))
    }

    fn layout(bytes: &[u8], path: &str) -> Result<Layout, Error> {
        let invalid = || Error::new(ErrorCode::InvalidMesh, path);
        let word = |offset: usize| {
            bytes
                .get(offset..offset + 4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("four_bytes")))
        };
        let extras = match bytes.get(..4) {
            Some(magic) if magic == MESH_BUFFER_MAGIC => false,
            Some(magic) if magic == MESH_BUFFER_EXTRAS_MAGIC => true,
            Some(_) | None => {
                return Err(invalid()
                    .got(String::from_utf8_lossy(&bytes[..bytes.len().min(4)]))
                    .variants(["EMB1", "EMB2"]));
            }
        };
        let (Some(vertex_count), Some(index_count)) = (word(4), word(8)) else {
            return Err(invalid().at_least(12).got(bytes.len()));
        };
        let flags = if extras {
            match word(12) {
                Some(flags) if flags != 0 && flags & !FLAGS == 0 => flags,
                flags @ (Some(_) | None) => {
                    return Err(invalid().at_most(u64::from(FLAGS)).got(flags));
                }
            }
        } else {
            0
        };
        let header = if extras { 16 } else { 12 };
        let (header, named) = if flags & NAMED == 0 {
            (header, Vec::new())
        } else {
            Self::named_table(bytes, header, path)?
        };
        let (vertex_count, index_count) = (vertex_count as usize, index_count as usize);
        let floats = 8
            + if flags & COLORS != 0 { 4 } else { 0 }
            + if flags & UV1 != 0 { 2 } else { 0 };
        let named_width: usize = named.iter().map(|&(_, format)| width(format)).sum();
        let expected = vertex_count
            .checked_mul(floats * 4 + named_width)
            .and_then(|vertices| vertices.checked_add(index_count.checked_mul(4)?))
            .and_then(|payload| payload.checked_add(header));
        let fits = expected.is_some_and(|expected| {
            expected == bytes.len() || (flags & LODS != 0 && expected < bytes.len())
        });
        if !fits {
            let failure = invalid().got(bytes.len());
            return Err(match expected.map(u64::try_from) {
                Some(Ok(expected)) if flags & LODS != 0 => failure.at_least(expected),
                Some(Ok(expected)) => failure.range(expected as f64, expected as f64),
                Some(Err(_)) | None => failure,
            });
        }
        Ok(Layout {
            header,
            vertex_count,
            index_count,
            flags,
            named,
        })
    }

    pub fn parse(bytes: &[u8], path: &str) -> Result<Self, Error> {
        let invalid = || Error::new(ErrorCode::InvalidMesh, path);
        let Layout {
            header,
            vertex_count,
            index_count,
            flags,
            named,
        } = Self::layout(bytes, path)?;
        let float = |offset: usize| {
            f32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four_bytes"))
        };
        let mut vertices = Vec::with_capacity(vertex_count);
        for vertex in 0..vertex_count {
            let base = header + vertex * 32;
            let values: [f32; 8] = std::array::from_fn(|index| float(base + index * 4));
            if !values[..3]
                .iter()
                .all(|value| value.is_finite() && value.abs() <= 1_000_000.0)
            {
                return Err(invalid()
                    .expected(Expected::Finite)
                    .range(-1_000_000.0, 1_000_000.0)
                    .got(VertexPosition {
                        vertex,
                        position: &values[..3],
                    }));
            }
            vertices.push(MeshVertex {
                position: [values[0], values[1], values[2]],
                normal: [values[3], values[4], values[5]],
                uv: [values[6], values[7]],
            });
        }
        let mut offset = header + vertex_count * 32;
        let colors = (flags & COLORS != 0).then(|| {
            let colors: Vec<[f32; 4]> = (0..vertex_count)
                .map(|vertex| std::array::from_fn(|lane| float(offset + (vertex * 4 + lane) * 4)))
                .collect();
            offset += vertex_count * 16;
            colors
        });
        let uv1 = (flags & UV1 != 0).then(|| {
            let uv1: Vec<[f32; 2]> = (0..vertex_count)
                .map(|vertex| std::array::from_fn(|lane| float(offset + (vertex * 2 + lane) * 4)))
                .collect();
            offset += vertex_count * 8;
            uv1
        });
        let named = named
            .into_iter()
            .map(|(name, format)| {
                let length = vertex_count * width(format);
                let held = bytes[offset..offset + length].to_vec();
                let stream = NamedStream::new(name, format, held);
                offset += length;
                stream
            })
            .collect();
        let indices = (0..index_count)
            .map(|index| {
                u32::from_le_bytes(
                    bytes[offset + index * 4..offset + index * 4 + 4]
                        .try_into()
                        .expect("four_bytes"),
                )
            })
            .collect();
        let lods = (flags & LODS != 0)
            .then(|| {
                MeshLods::from_bytes(&bytes[offset + index_count * 4..])
                    .map_err(|error| error.coded(ErrorCode::InvalidMesh).within(path))
            })
            .transpose()?;
        Ok(Self {
            vertices,
            indices,
            colors,
            uv1,
            named,
            lods,
        })
    }
}
