use std::sync::{Arc, OnceLock};

use engine_codec::{Error, ErrorCode, Expected};
use glam::Vec3;

use crate::{Attribute, AttributeFormat, Attributes, MeshLods, MeshVertex, StreamShape};

fn invalid(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidMesh, path)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    pub fn min(&self) -> Vec3 {
        self.min
    }

    pub fn max(&self) -> Vec3 {
        self.max
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamedStream {
    name: String,
    format: AttributeFormat,
    bytes: Box<[u8]>,
}

impl NamedStream {
    pub fn new(name: impl Into<String>, format: AttributeFormat, bytes: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            format,
            bytes: bytes.into_boxed_slice(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn format(&self) -> AttributeFormat {
        self.format
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn is(&self, shape: StreamShape) -> bool {
        self.name == shape.name && self.format == shape.format
    }

    pub fn words(&self) -> impl Iterator<Item = [u8; 4]> + '_ {
        self.bytes.as_chunks::<4>().0.iter().copied()
    }
}

#[derive(Debug)]
pub struct Mesh {
    vertices: Box<[MeshVertex]>,
    indices: Box<[u32]>,
    bounds: Aabb,
    colors: Option<Box<[[f32; 4]]>>,
    uv1: Option<Box<[[f32; 2]]>>,
    streams: Vec<NamedStream>,
    lods: Option<Arc<MeshLods>>,
    identity: OnceLock<u128>,
}

impl Mesh {
    pub fn new(mut vertices: Vec<MeshVertex>, indices: Vec<u32>) -> Result<Self, Error> {
        if vertices.is_empty() {
            return Err(invalid("$.vertices").expected(Expected::NonEmpty).got(0));
        }
        if indices.is_empty() || !indices.len().is_multiple_of(3) {
            return Err(invalid("$.indices")
                .expected(Expected::Triangles)
                .got(indices.len()));
        }
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for (index, vertex) in vertices.iter_mut().enumerate() {
            if !vertex.position.iter().all(|value| value.is_finite()) {
                return Err(invalid(format!("$.vertices[{index}].position"))
                    .expected(Expected::Finite)
                    .got(vertex.position));
            }
            let position = Vec3::from_array(vertex.position);
            min = min.min(position);
            max = max.max(position);
            if !vertex.uv.iter().all(|value| value.is_finite()) {
                return Err(invalid(format!("$.vertices[{index}].uv"))
                    .expected(Expected::Finite)
                    .got(vertex.uv));
            }
            let normal = Vec3::from_array(vertex.normal);
            let scale = normal.abs().max_element();
            if !normal.is_finite() || scale == 0.0 {
                return Err(invalid(format!("$.vertices[{index}].normal"))
                    .expected(Expected::Nonzero)
                    .got(vertex.normal));
            }
            vertex.normal = (normal / scale).normalize().to_array();
        }
        for (index, value) in indices.iter().copied().enumerate() {
            if value as usize >= vertices.len() {
                return Err(invalid(format!("$.indices[{index}]"))
                    .at_most(vertices.len() as u64 - 1)
                    .got(value));
            }
        }
        Ok(Self {
            vertices: vertices.into_boxed_slice(),
            indices: indices.into_boxed_slice(),
            bounds: Aabb { min, max },
            colors: None,
            uv1: None,
            streams: Vec::new(),
            lods: None,
            identity: OnceLock::new(),
        })
    }

    pub fn identity(&self, hash: impl FnOnce(&[u8]) -> u128) -> u128 {
        *self.identity.get_or_init(|| {
            let mut bytes = Vec::with_capacity(self.byte_len() + 16);
            bytes.extend_from_slice(&(self.vertices.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&(self.indices.len() as u32).to_le_bytes());
            for vertex in &self.vertices {
                for value in vertex
                    .position
                    .iter()
                    .chain(&vertex.normal)
                    .chain(&vertex.uv)
                {
                    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }
            for index in &self.indices {
                bytes.extend_from_slice(&index.to_le_bytes());
            }
            bytes.push(u8::from(self.colors.is_some()));
            for value in self.colors.iter().flatten().flatten() {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            bytes.push(u8::from(self.uv1.is_some()));
            for value in self.uv1.iter().flatten().flatten() {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            bytes.push(u8::from(!self.streams.is_empty()));
            for stream in &self.streams {
                bytes.extend_from_slice(&(stream.name.len() as u32).to_le_bytes());
                bytes.extend_from_slice(stream.name.as_bytes());
                bytes.extend_from_slice(&stream.format.code().to_le_bytes());
                bytes.extend_from_slice(&stream.bytes);
            }
            hash(&bytes)
        })
    }

    pub fn with_lods(mut self, lods: MeshLods) -> Result<Self, Error> {
        lods.check(self.indices.len(), self.vertices.len())?;
        self.lods = Some(Arc::new(lods));
        Ok(self)
    }

    pub fn lods(&self) -> Option<&Arc<MeshLods>> {
        self.lods.as_ref()
    }

    pub fn with_colors(mut self, colors: Vec<[f32; 4]>) -> Result<Self, Error> {
        self.check_count("colors", colors.len())?;
        if let Some(vertex) = colors.iter().position(|color| {
            !color
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        }) {
            return Err(invalid(format!("$.colors[{vertex}]"))
                .range(0.0, 1.0)
                .got(colors[vertex]));
        }
        self.colors = Some(colors.into_boxed_slice());
        self.identity = OnceLock::new();
        Ok(self)
    }

    pub fn with_uv1(mut self, uv1: Vec<[f32; 2]>) -> Result<Self, Error> {
        self.check_count("uv1", uv1.len())?;
        if let Some(vertex) = uv1
            .iter()
            .position(|uv| !uv.iter().all(|value| value.is_finite()))
        {
            return Err(invalid(format!("$.uv1[{vertex}]"))
                .expected(Expected::Finite)
                .got(uv1[vertex]));
        }
        self.uv1 = Some(uv1.into_boxed_slice());
        self.identity = OnceLock::new();
        Ok(self)
    }

    pub fn with_stream(mut self, stream: NamedStream) -> Result<Self, Error> {
        let path = format!("$.streams.{}", stream.name);
        if stream.name.is_empty() {
            return Err(invalid("$.streams").expected(Expected::NonEmpty).got(""));
        }
        let width = stream.format.bytes() as usize;
        let vertices = self.vertices.len();
        if stream.bytes.len() != vertices * width {
            return Err(invalid(path)
                .expected(Expected::Array)
                .range((vertices * width) as f64, (vertices * width) as f64)
                .got(stream.bytes.len()));
        }
        if stream.format.float()
            && let Some((at, value)) = stream
                .words()
                .map(f32::from_le_bytes)
                .enumerate()
                .find(|(_, value)| !value.is_finite())
        {
            return Err(invalid(format!("{path}[{}]", at * 4 / width))
                .expected(Expected::Finite)
                .got(value));
        }
        self.streams.retain(|held| held.name != stream.name);
        let at = self.streams.partition_point(|held| held.name < stream.name);
        self.streams.insert(at, stream);
        self.identity = OnceLock::new();
        Ok(self)
    }

    fn check_count(&self, attribute: &'static str, actual: usize) -> Result<(), Error> {
        let vertices = self.vertices.len();
        if actual == vertices {
            Ok(())
        } else {
            Err(invalid(format!("$.{attribute}"))
                .expected(Expected::Array)
                .range(vertices as f64, vertices as f64)
                .got(actual))
        }
    }

    pub fn vertices(&self) -> &[MeshVertex] {
        &self.vertices
    }

    pub fn colors(&self) -> Option<&[[f32; 4]]> {
        self.colors.as_deref()
    }

    pub fn uv1(&self) -> Option<&[[f32; 2]]> {
        self.uv1.as_deref()
    }

    pub fn streams(&self) -> &[NamedStream] {
        &self.streams
    }

    pub fn stream(&self, shape: StreamShape) -> Option<&NamedStream> {
        self.streams.iter().find(|stream| stream.is(shape))
    }

    pub fn attributes(&self) -> Attributes {
        let mut held = Attributes::BASE;
        if self.uv1.is_some() {
            held = held.with(Attribute::Uv1);
        }
        if self.colors.is_some() {
            held = held.with(Attribute::Color);
        }
        held
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn bounds(&self) -> Aabb {
        self.bounds
    }

    pub fn byte_len(&self) -> usize {
        std::mem::size_of_val(self.vertices())
            + std::mem::size_of_val(self.indices())
            + self.colors().map_or(0, std::mem::size_of_val)
            + self.uv1().map_or(0, std::mem::size_of_val)
            + self
                .streams
                .iter()
                .map(|stream| stream.bytes.len())
                .sum::<usize>()
            + self.lods().map_or(0, |lods| lods.byte_len())
    }

    pub fn cube() -> Self {
        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);
        for normal in [
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
        ] {
            let tangent = if normal.y.abs() == 1.0 {
                Vec3::X
            } else {
                Vec3::Y.cross(normal)
            };
            let bitangent = normal.cross(tangent);
            let offset = vertices.len() as u32;
            for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                vertices.push(MeshVertex {
                    position: ((normal + tangent * x + bitangent * y) * 0.5).to_array(),
                    normal: normal.to_array(),
                    uv: [(x + 1.0) * 0.5, (1.0 - y) * 0.5],
                });
            }
            indices.extend([
                offset,
                offset + 1,
                offset + 2,
                offset,
                offset + 2,
                offset + 3,
            ]);
        }
        Self::new(vertices, indices).expect("cube_valid")
    }

    pub fn plane() -> Self {
        let vertices = [
            [-0.5, 0.0, -0.5],
            [-0.5, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.0, -0.5],
        ]
        .into_iter()
        .map(|position| MeshVertex {
            position,
            normal: [0.0, 1.0, 0.0],
            uv: [position[0] + 0.5, position[2] + 0.5],
        })
        .collect();
        Self::new(vertices, vec![0, 1, 2, 0, 2, 3]).expect("plane_valid")
    }
}
