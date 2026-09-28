use std::sync::Arc;

use crate::virtual_mesh::VirtualMesh;
use glam::Vec3;

use crate::AssetError;

/// CPU vertex data, independent of any GPU upload layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// Texture coordinates with (0, 0) at the upper-left image corner.
    /// Finite values outside [0, 1] are valid and use the renderer's sampler.
    pub uv: [f32; 2],
}

/// Immutable local axis-aligned bounds of every mesh vertex, including unused
/// vertices. Finite endpoints avoid center/extent overflow for extreme meshes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    pub(crate) fn from_min_max(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn min(&self) -> Vec3 {
        self.min
    }

    pub fn max(&self) -> Vec3 {
        self.max
    }
}

/// Validated, immutable indexed triangles. Winding is counterclockwise.
/// Optional vertex colors and second texture coordinates (decision 034) and
/// skinning joints and weights (decision 036) follow the vertices one to one.
/// A mesh with virtual geometry (decision 044) is its hierarchy's fallback
/// cut, which renderers draw where they cannot draw its clusters.
#[derive(Debug)]
pub struct Mesh {
    vertices: Box<[MeshVertex]>,
    indices: Box<[u32]>,
    bounds: Aabb,
    colors: Option<Box<[[f32; 4]]>>,
    uv1: Option<Box<[[f32; 2]]>>,
    skin: Option<MeshSkin>,
    virtual_geometry: Option<Arc<VirtualMesh>>,
}

/// Up to four joints per vertex and their weights (glTF `JOINTS_0` and
/// `WEIGHTS_0`): indices into the joints of the skin an entity binds.
#[derive(Debug)]
struct MeshSkin {
    joints: Box<[[u16; 4]]>,
    weights: Box<[[f32; 4]]>,
    joint_count: u32,
}

impl Mesh {
    /// Takes ownership, validates finite positions/UVs, nonzero normals and indices,
    /// and discards spare vector capacity. Normals are normalized once here.
    pub fn new(mut vertices: Vec<MeshVertex>, indices: Vec<u32>) -> Result<Self, AssetError> {
        if vertices.is_empty() {
            return Err(AssetError::EmptyMesh);
        }
        if indices.is_empty() || !indices.len().is_multiple_of(3) {
            return Err(AssetError::InvalidTriangleCount {
                indices: indices.len(),
            });
        }
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for (index, vertex) in vertices.iter_mut().enumerate() {
            if !vertex.position.iter().all(|value| value.is_finite()) {
                return Err(AssetError::InvalidVertex { vertex: index });
            }
            let position = Vec3::from_array(vertex.position);
            min = min.min(position);
            max = max.max(position);
            if !vertex.uv.iter().all(|value| value.is_finite()) {
                return Err(AssetError::InvalidUv { vertex: index });
            }
            let normal = Vec3::from_array(vertex.normal);
            // Scale first so all finite nonzero normals, including very small
            // and very large values, can be normalized without overflow.
            let scale = normal.abs().max_element();
            if !normal.is_finite() || scale == 0.0 {
                return Err(AssetError::InvalidNormal { vertex: index });
            }
            vertex.normal = (normal / scale).normalize().to_array();
        }
        for (index, value) in indices.iter().copied().enumerate() {
            if value as usize >= vertices.len() {
                return Err(AssetError::IndexOutOfBounds {
                    index,
                    value,
                    vertices: vertices.len(),
                });
            }
        }
        Ok(Self {
            vertices: vertices.into_boxed_slice(),
            indices: indices.into_boxed_slice(),
            bounds: Aabb { min, max },
            colors: None,
            uv1: None,
            skin: None,
            virtual_geometry: None,
        })
    }

    /// A mesh drawn from `geometry`'s cluster hierarchy (decision 044) where
    /// renderers can, and as its fallback cut of at most `fallback_triangles`
    /// triangles elsewhere; its bounds hold every level.
    pub fn from_virtual(
        geometry: Arc<VirtualMesh>,
        fallback_triangles: u32,
    ) -> Result<Self, AssetError> {
        let mut mesh = geometry.fallback(fallback_triangles)?;
        let bounds = geometry.bounds();
        mesh.bounds = Aabb {
            min: mesh.bounds.min.min(bounds.min()),
            max: mesh.bounds.max.max(bounds.max()),
        };
        mesh.virtual_geometry = Some(geometry);
        Ok(mesh)
    }

    /// The cluster hierarchy of a mesh with virtual geometry.
    pub fn virtual_geometry(&self) -> Option<&Arc<VirtualMesh>> {
        self.virtual_geometry.as_ref()
    }

    /// This mesh deformed by joints: per vertex up to four joint indices
    /// into the skin an entity binds (decision 036) and their weights, which
    /// must be finite, not negative and not all zero; they are normalized to
    /// sum to one. Joints of zero weight are ignored and read as joint 0.
    pub fn with_skin(
        mut self,
        mut joints: Vec<[u16; 4]>,
        mut weights: Vec<[f32; 4]>,
    ) -> Result<Self, AssetError> {
        self.check_count("joints", joints.len())?;
        self.check_count("weights", weights.len())?;
        let mut joint_count = 0;
        for (vertex, (joints, weights)) in joints.iter_mut().zip(&mut weights).enumerate() {
            let sum: f32 = weights.iter().sum();
            if !weights
                .iter()
                .all(|weight| weight.is_finite() && *weight >= 0.0)
                || !(sum > 0.0 && sum.is_finite())
            {
                return Err(AssetError::InvalidAttribute {
                    attribute: "weights",
                    vertex,
                });
            }
            for (joint, weight) in joints.iter_mut().zip(weights.iter_mut()) {
                *weight /= sum;
                if *weight == 0.0 {
                    *joint = 0;
                }
                joint_count = joint_count.max(u32::from(*joint) + 1);
            }
        }
        self.skin = Some(MeshSkin {
            joints: joints.into_boxed_slice(),
            weights: weights.into_boxed_slice(),
            joint_count,
        });
        Ok(self)
    }

    /// This mesh with linear RGBA vertex colors in `[0, 1]`, one per vertex,
    /// multiplying its material's base color (glTF `COLOR_0`).
    pub fn with_colors(mut self, colors: Vec<[f32; 4]>) -> Result<Self, AssetError> {
        self.check_count("colors", colors.len())?;
        if let Some(vertex) = colors.iter().position(|color| {
            !color
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        }) {
            return Err(AssetError::InvalidAttribute {
                attribute: "colors",
                vertex,
            });
        }
        self.colors = Some(colors.into_boxed_slice());
        Ok(self)
    }

    /// This mesh with finite second texture coordinates, one per vertex,
    /// which texture maps may read (glTF `TEXCOORD_1`).
    pub fn with_uv1(mut self, uv1: Vec<[f32; 2]>) -> Result<Self, AssetError> {
        self.check_count("uv1", uv1.len())?;
        if let Some(vertex) = uv1
            .iter()
            .position(|uv| !uv.iter().all(|value| value.is_finite()))
        {
            return Err(AssetError::InvalidAttribute {
                attribute: "uv1",
                vertex,
            });
        }
        self.uv1 = Some(uv1.into_boxed_slice());
        Ok(self)
    }

    fn check_count(&self, attribute: &'static str, actual: usize) -> Result<(), AssetError> {
        if actual == self.vertices.len() {
            Ok(())
        } else {
            Err(AssetError::AttributeCount {
                attribute,
                expected: self.vertices.len(),
                actual,
            })
        }
    }

    pub fn vertices(&self) -> &[MeshVertex] {
        &self.vertices
    }

    /// Vertex colors, when the mesh has them.
    pub fn colors(&self) -> Option<&[[f32; 4]]> {
        self.colors.as_deref()
    }

    /// Second texture coordinates, when the mesh has them.
    pub fn uv1(&self) -> Option<&[[f32; 2]]> {
        self.uv1.as_deref()
    }

    /// Joint indices of skinned meshes, four per vertex.
    pub fn joints(&self) -> Option<&[[u16; 4]]> {
        self.skin.as_ref().map(|skin| &skin.joints[..])
    }

    /// Joint weights of skinned meshes, four per vertex summing to one.
    pub fn weights(&self) -> Option<&[[f32; 4]]> {
        self.skin.as_ref().map(|skin| &skin.weights[..])
    }

    /// The joints a skin must provide: one more than the largest weighted
    /// joint index, and 0 for meshes without joints.
    pub fn joint_count(&self) -> u32 {
        self.skin.as_ref().map_or(0, |skin| skin.joint_count)
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Bounds computed once at construction; reading them never scans vertices.
    pub fn bounds(&self) -> Aabb {
        self.bounds
    }

    /// Owned vertex/index payload bytes; excludes bounds, allocation and `Arc` metadata.
    pub fn byte_len(&self) -> usize {
        std::mem::size_of_val(self.vertices())
            + std::mem::size_of_val(self.indices())
            + self.colors().map_or(0, std::mem::size_of_val)
            + self.uv1().map_or(0, std::mem::size_of_val)
            + self.joints().map_or(0, std::mem::size_of_val)
            + self.weights().map_or(0, std::mem::size_of_val)
            + self
                .virtual_geometry()
                .map_or(0, |geometry| geometry.byte_len())
    }

    /// Unit cube centered at the origin, with flat face normals.
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
        Self::new(vertices, indices).expect("built-in cube is valid")
    }

    /// Unit square centered on the XZ plane; its front face points upward.
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
        Self::new(vertices, vec![0, 1, 2, 0, 2, 3]).expect("built-in plane is valid")
    }

    /// Centered octahedron spanning one unit along each axis, with flat normals.
    pub fn octahedron() -> Self {
        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(24);
        for y in [-1.0, 1.0] {
            for (a, b) in [
                (Vec3::X, Vec3::Z),
                (Vec3::Z, Vec3::NEG_X),
                (Vec3::NEG_X, Vec3::NEG_Z),
                (Vec3::NEG_Z, Vec3::X),
            ] {
                let tip = Vec3::Y * y;
                let (a, b) = if y > 0.0 { (b, a) } else { (a, b) };
                let normal = (a - tip).cross(b - tip).normalize().to_array();
                for (position, uv) in
                    [tip, a, b]
                        .into_iter()
                        .zip([[0.5, 0.0], [0.0, 1.0], [1.0, 1.0]])
                {
                    indices.push(vertices.len() as u32);
                    vertices.push(MeshVertex {
                        position: (position * 0.5).to_array(),
                        normal,
                        uv,
                    });
                }
            }
        }
        Self::new(vertices, indices).expect("built-in octahedron is valid")
    }
}
