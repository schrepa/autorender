//! Levels of detail: coarser index lists over a mesh's own vertices, each
//! with the largest geometric deviation from the mesh it shows in mesh
//! units, the last possibly empty, and the triangles shadow maps draw of
//! each. The cook makes them offline for every mesh a scene draws (the
//! [`LOD_RECORD`] records, from [`lod_input`]) and the importer writes them
//! into the mesh files it makes; runtimes only load them. Renderers draw
//! the coarsest level whose error projects within a pixel budget: past the
//! empty level's, nothing.

use crate::{AssetError, Mesh, MeshVertex};

/// The kind of a mesh's levels' records, and of the job making one.
pub const LOD_RECORD: &str = "mesh_lods";

/// A level of detail: where its indices start in its mesh's indices
/// followed by [`MeshLods::indices`], how many there are (none for the
/// empty level), and the largest geometric deviation from the mesh in mesh
/// units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodLevel {
    pub first_index: u32,
    pub index_count: u32,
    pub error: f32,
}

/// A mesh's levels of detail, finest first, with errors that grow from
/// level to level; the triangles shadow maps draw of each level, the mesh
/// itself first (none where those are each level's own); and the indices
/// both name beyond the mesh's own, which follow them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshLods {
    pub levels: Vec<LodLevel>,
    pub casters: Vec<(u32, u32)>,
    pub indices: Vec<u32>,
}

/// The magic of a mesh's levels' bytes, and of a level job's input.
const LODS_MAGIC: &[u8; 4] = b"ELOD";
const INPUT_MAGIC: &[u8; 4] = b"ELDI";

impl MeshLods {
    /// The levels as bytes: `ELOD`, the counts of levels, of caster ranges
    /// and of indices (little-endian u32 each), then each level's first
    /// index, index count and error, each caster range's first index and
    /// count, and the indices.
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

    /// The levels [`Self::to_bytes`] wrote.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AssetError> {
        let invalid = |reason| AssetError::InvalidLevels { reason };
        let mut words = Words::new(bytes, LODS_MAGIC).ok_or(invalid("not levels of detail"))?;
        let counts = [words.next(), words.next(), words.next()];
        let [Some(levels), Some(casters), Some(indices)] = counts else {
            return Err(invalid("levels of detail start with their counts"));
        };
        let expected = (levels as usize)
            .checked_mul(12)
            .and_then(|levels| levels.checked_add(casters as usize * 8))
            .and_then(|size| size.checked_add(indices as usize * 4))
            .and_then(|size| size.checked_add(16));
        if expected != Some(bytes.len()) {
            return Err(invalid("the bytes must match their counts"));
        }
        let mut lods = Self::default();
        for _ in 0..levels {
            let (first_index, index_count, error) = (words.next(), words.next(), words.next());
            let (Some(first_index), Some(index_count), Some(error)) =
                (first_index, index_count, error)
            else {
                return Err(invalid("a level holds three words"));
            };
            lods.levels.push(LodLevel {
                first_index,
                index_count,
                error: f32::from_bits(error),
            });
        }
        for _ in 0..casters {
            let (Some(first), Some(count)) = (words.next(), words.next()) else {
                return Err(invalid("a caster range holds two words"));
            };
            lods.casters.push((first, count));
        }
        lods.indices = (0..indices).filter_map(|_| words.next()).collect();
        Ok(lods)
    }

    /// Whether they suit a mesh of `index_count` indices over `vertices`
    /// vertices: every range within its indices followed by theirs, whole
    /// triangles of its vertices, errors finite, positive and growing, and
    /// a caster range for the mesh and each level, or none.
    pub fn check(&self, index_count: usize, vertices: usize) -> Result<(), AssetError> {
        let invalid = |reason| Err(AssetError::InvalidLevels { reason });
        let total = index_count + self.indices.len();
        let within = |first: u32, count: u32| {
            count.is_multiple_of(3)
                && (first as usize)
                    .checked_add(count as usize)
                    .is_some_and(|end| end <= total)
        };
        let mut last = 0.0;
        for level in &self.levels {
            if !within(level.first_index, level.index_count) {
                return invalid("a level's indices lie within the mesh's and the levels'");
            }
            if !(level.error.is_finite() && level.error > last) {
                return invalid("levels' errors are finite, positive and grow");
            }
            last = level.error;
        }
        if !(self.casters.is_empty() || self.casters.len() == self.levels.len() + 1) {
            return invalid("caster ranges cover the mesh and every level, or none");
        }
        if !self.casters.iter().all(|&(first, count)| within(first, count)) {
            return invalid("a caster range lies within the mesh's and the levels' indices");
        }
        if self.indices.iter().any(|&index| index as usize >= vertices) {
            return invalid("the levels' indices name the mesh's vertices");
        }
        Ok(())
    }

    /// Bytes their indices and ranges hold.
    pub fn byte_len(&self) -> usize {
        self.indices.len() * 4 + self.levels.len() * 12 + self.casters.len() * 8
    }
}

/// What the levels of a mesh are made from: its vertices and indices.
#[derive(Clone, Debug, PartialEq)]
pub struct LodInput {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

/// Whether `mesh` takes the levels of detail the cook makes: it carries
/// none of its own and can have some, having neither joints, which move its
/// vertices, nor virtual geometry, whose clusters are its levels.
pub fn takes_levels(mesh: &Mesh) -> bool {
    mesh.lods().is_none() && mesh.joint_count() == 0 && mesh.virtual_geometry().is_none()
}

/// The input of `mesh`'s levels' job ([`LOD_RECORD`]): `ELDI`, the vertex
/// and index counts, each vertex's position, normal and texture coordinates
/// (eight f32), then the indices, little-endian. `None` where it takes no
/// levels from the cook ([`takes_levels`]).
pub fn lod_input(mesh: &Mesh) -> Option<Vec<u8>> {
    if !takes_levels(mesh) {
        return None;
    }
    let (vertices, indices) = (mesh.vertices(), mesh.indices());
    let mut bytes = Vec::with_capacity(12 + vertices.len() * 32 + indices.len() * 4);
    bytes.extend_from_slice(INPUT_MAGIC);
    bytes.extend_from_slice(&(vertices.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(indices.len() as u32).to_le_bytes());
    for vertex in vertices {
        for value in vertex.position.iter().chain(&vertex.normal).chain(&vertex.uv) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    for index in indices {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    Some(bytes)
}

impl LodInput {
    /// The input [`lod_input`] wrote.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AssetError> {
        let invalid = |reason| AssetError::InvalidLevels { reason };
        let mut words = Words::new(bytes, INPUT_MAGIC).ok_or(invalid("not a level job's input"))?;
        let (Some(vertices), Some(indices)) = (words.next(), words.next()) else {
            return Err(invalid("a level job's input starts with its counts"));
        };
        let expected = (vertices as usize)
            .checked_mul(32)
            .and_then(|size| size.checked_add(indices as usize * 4))
            .and_then(|size| size.checked_add(12));
        if expected != Some(bytes.len()) {
            return Err(invalid("the input must match its counts"));
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

/// Little-endian words after a magic.
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
