#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use std::borrow::Cow;

use engine_assets::{
    Aabb, IndexFormat, MeshDescription, MeshFile, VertexFormat, VertexInput, VertexStream,
};
use engine_codec::{Error, ErrorCode, Expected};
use engine_math::clamp;
use glam::Vec3;

pub const WGSL: &str = include_str!("unpack.wgsl");

pub const POSITION: u32 = 0;

pub const JOINTS: u32 = 5;

pub const BASE: VertexStream = VertexStream {
    stride: 24,
    inputs: Cow::Borrowed(&[
        VertexInput {
            location: POSITION,
            format: VertexFormat::Float32x3,
            offset: 0,
        },
        VertexInput {
            location: 1,
            format: VertexFormat::Snorm16x2,
            offset: 12,
        },
        VertexInput {
            location: 2,
            format: VertexFormat::Float32x2,
            offset: 16,
        },
    ]),
};

pub const LIGHTMAP: VertexStream = VertexStream {
    stride: 8,
    inputs: Cow::Borrowed(&[VertexInput {
        location: 3,
        format: VertexFormat::Float32x2,
        offset: 0,
    }]),
};

pub const COLOR: VertexStream = VertexStream {
    stride: 8,
    inputs: Cow::Borrowed(&[VertexInput {
        location: 4,
        format: VertexFormat::Unorm16x4,
        offset: 0,
    }]),
};

pub const SKIN: VertexStream = VertexStream {
    stride: 24,
    inputs: Cow::Borrowed(&[
        VertexInput {
            location: JOINTS,
            format: VertexFormat::Uint32x4,
            offset: 0,
        },
        VertexInput {
            location: 6,
            format: VertexFormat::Unorm16x4,
            offset: 16,
        },
    ]),
};

pub const SHAPE: VertexStream = VertexStream {
    stride: 12,
    inputs: Cow::Borrowed(&[VertexInput {
        location: POSITION,
        format: VertexFormat::Float32x3,
        offset: 0,
    }]),
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[derive(Debug, Clone, Copy)]
pub struct Unpacked<'a> {
    pub vertices: &'a [MeshVertex],
    pub indices: &'a [u32],
    pub uv1: Option<&'a [[f32; 2]]>,
    pub colors: Option<&'a [[f32; 4]]>,
    pub skin: Option<&'a [([u32; 4], [u16; 4])]>,
}

fn invalid(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidMesh, path)
}

fn floats(bytes: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn octahedral(normal: [f32; 3]) -> [i16; 2] {
    let [x, y, z] = normal;
    let length = x.abs() + y.abs() + z.abs();
    let (u, v) = (x / length, y / length);
    let sign = |value: f32| if value >= 0.0 { 1.0 } else { -1.0 };
    let (u, v) = if z < 0.0 {
        ((1.0 - v.abs()) * sign(u), (1.0 - u.abs()) * sign(v))
    } else {
        (u, v)
    };
    [u, v].map(|value| (clamp(value, -1.0, 1.0) * 32767.0).round() as i16)
}

fn unit_normal(index: usize, normal: [f32; 3]) -> Result<[f32; 3], Error> {
    let held = Vec3::from_array(normal);
    let scale = held.abs().max_element();
    if !held.is_finite() || scale == 0.0 {
        return Err(invalid(format!("$.vertices[{index}].normal"))
            .expected(Expected::Nonzero)
            .got(normal));
    }
    Ok((held / scale).normalize().to_array())
}

fn bounds<'a>(positions: impl Iterator<Item = &'a [f32; 3]>) -> Aabb {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for position in positions {
        min = min.min(Vec3::from_array(*position));
        max = max.max(Vec3::from_array(*position));
    }
    Aabb { min, max }
}

fn written(
    streams: Vec<(VertexStream, Vec<u8>)>,
    vertex_count: usize,
    indices: &[u32],
    bounds: Aabb,
) -> Result<Vec<u8>, Error> {
    let count = |path: &str, length: usize| {
        u32::try_from(length).map_err(|_| invalid(path).at_most(u64::from(u32::MAX)).got(length))
    };
    let index_format = if vertex_count <= 1 << 16 {
        IndexFormat::Uint16
    } else {
        IndexFormat::Uint32
    };
    let mut index_bytes = Vec::with_capacity(indices.len() * index_format.bytes() as usize);
    for (at, &value) in indices.iter().enumerate() {
        match index_format {
            IndexFormat::Uint16 => {
                let narrow = u16::try_from(value).map_err(|_| {
                    invalid(format!("$.indices[{at}]"))
                        .at_most(vertex_count.saturating_sub(1) as u64)
                        .got(value)
                })?;
                index_bytes.extend_from_slice(&narrow.to_le_bytes());
            }
            IndexFormat::Uint32 => index_bytes.extend_from_slice(&value.to_le_bytes()),
        }
    }
    let description = MeshDescription {
        vertex_count: count("$.vertices", vertex_count)?,
        index_format,
        index_count: count("$.indices", indices.len())?,
        bounds,
        streams: streams.iter().map(|(stream, _)| stream.clone()).collect(),
    };
    let bytes: Vec<&[u8]> = streams.iter().map(|(_, bytes)| bytes.as_slice()).collect();
    MeshFile::write(&description, &bytes, &index_bytes)
}

pub fn pack(mesh: &Unpacked<'_>) -> Result<Vec<u8>, Error> {
    let mut base = Vec::with_capacity(mesh.vertices.len() * BASE.stride as usize);
    for (index, vertex) in mesh.vertices.iter().enumerate() {
        floats(&mut base, &vertex.position);
        for value in octahedral(unit_normal(index, vertex.normal)?) {
            base.extend_from_slice(&value.to_le_bytes());
        }
        floats(&mut base, &vertex.uv);
    }
    let mut streams = vec![(BASE, base)];
    if let Some(uv1) = mesh.uv1 {
        let mut bytes = Vec::with_capacity(uv1.len() * LIGHTMAP.stride as usize);
        floats(&mut bytes, uv1.as_flattened());
        streams.push((LIGHTMAP, bytes));
    }
    if let Some(colors) = mesh.colors {
        let mut bytes = Vec::with_capacity(colors.len() * COLOR.stride as usize);
        for (vertex, color) in colors.iter().enumerate() {
            if !color.iter().all(|value| (0.0..=1.0).contains(value)) {
                return Err(invalid(format!("$.colors[{vertex}]"))
                    .range(0.0, 1.0)
                    .got(color));
            }
            for value in color {
                bytes.extend_from_slice(&((value * 65535.0).round() as u16).to_le_bytes());
            }
        }
        streams.push((COLOR, bytes));
    }
    if let Some(skin) = mesh.skin {
        let mut bytes = Vec::with_capacity(skin.len() * SKIN.stride as usize);
        for (joints, weights) in skin {
            bytes.extend(joints.iter().flat_map(|joint| joint.to_le_bytes()));
            bytes.extend(weights.iter().flat_map(|weight| weight.to_le_bytes()));
        }
        streams.push((SKIN, bytes));
    }
    let bounds = bounds(mesh.vertices.iter().map(|vertex| &vertex.position));
    written(streams, mesh.vertices.len(), mesh.indices, bounds)
}

pub fn shape(positions: &[[f32; 3]], indices: &[u32]) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::with_capacity(positions.len() * SHAPE.stride as usize);
    floats(&mut bytes, positions.as_flattened());
    written(
        vec![(SHAPE, bytes)],
        positions.len(),
        indices,
        bounds(positions.iter()),
    )
}
