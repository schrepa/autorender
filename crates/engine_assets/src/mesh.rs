use glam::Vec3;

use crate::{AssetError, TextureHandle};

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
    pub fn min(&self) -> Vec3 {
        self.min
    }

    pub fn max(&self) -> Vec3 {
        self.max
    }
}

/// Validated, immutable indexed triangles. Winding is counterclockwise.
#[derive(Debug)]
pub struct Mesh {
    vertices: Box<[MeshVertex]>,
    indices: Box<[u32]>,
    bounds: Aabb,
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
        })
    }

    pub fn vertices(&self) -> &[MeshVertex] {
        &self.vertices
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
        std::mem::size_of_val(self.vertices()) + std::mem::size_of_val(self.indices())
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

/// How a surface responds to light beyond its color (decision 019).
///
/// [`Material::with_surface`] validates the ranges. Renderers quantize each
/// value to 8 bits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Surface {
    /// Microsurface roughness in `[0, 1]`: 0 is a mirror, 1 spreads highlights
    /// and sky reflections widest.
    pub roughness: f32,
    /// In `[0, 1]`: metals reflect in their own color and diffuse no light.
    pub metalness: f32,
    /// Emitted light as a multiple of the surface color, in
    /// `[0, Surface::MAX_EMISSION]`. Texture alpha masks it per texel.
    pub emission: f32,
    /// Animated ripples of liquid surfaces in `[0, 1]`: wave normals that move
    /// with presentation time. 0 keeps the mesh normals.
    pub ripples: f32,
}

impl Surface {
    /// Broad, dim highlights and no emission: the default.
    pub const MATTE: Self = Self {
        roughness: 0.8,
        metalness: 0.0,
        emission: 0.0,
        ripples: 0.0,
    };

    /// The largest accepted emission multiple.
    pub const MAX_EMISSION: f32 = 16.0;

    fn validate(&self) -> Result<(), AssetError> {
        for (field, value, max) in [
            ("roughness", self.roughness, 1.0),
            ("metalness", self.metalness, 1.0),
            ("emission", self.emission, Self::MAX_EMISSION),
            ("ripples", self.ripples, 1.0),
        ] {
            if !value.is_finite() || !(0.0..=max).contains(&value) {
                return Err(AssetError::InvalidSurface { field });
            }
        }
        Ok(())
    }
}

impl Default for Surface {
    fn default() -> Self {
        Self::MATTE
    }
}

/// A linear RGBA factor with an optional weak sRGB texture reference and a
/// [`Surface`] response. Materials are opaque unless given an opacity below 1
/// ([`Material::with_opacity`], decision 026); the color's alpha is that
/// opacity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    color: [f32; 4],
    texture: Option<TextureHandle>,
    surface: Surface,
}

impl Material {
    pub fn new(color: [f32; 4]) -> Result<Self, AssetError> {
        for (channel, value) in color.iter().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(value) {
                return Err(AssetError::InvalidColor { channel });
            }
        }
        if color[3] != 1.0 {
            return Err(AssetError::UnsupportedAlpha);
        }
        Ok(Self {
            color,
            texture: None,
            surface: Surface::MATTE,
        })
    }

    /// This material drawn transparent with `opacity` in `(0, 1]` (decision
    /// 026): the fraction of light behind it that it covers. 1 keeps it
    /// opaque. Transparent surfaces blend in any order, write no depth and
    /// cast no shadow.
    pub fn with_opacity(mut self, opacity: f32) -> Result<Self, AssetError> {
        if !opacity.is_finite() || !(opacity > 0.0 && opacity <= 1.0) {
            return Err(AssetError::InvalidOpacity);
        }
        self.color[3] = opacity;
        Ok(self)
    }

    /// The fraction of light behind the surface it covers: 1 for opaque
    /// materials.
    pub fn opacity(&self) -> f32 {
        self.color[3]
    }

    /// Whether this material draws in the transparent pass.
    pub fn is_transparent(&self) -> bool {
        self.color[3] < 1.0
    }

    /// This material with `surface`, whose values must lie in their ranges.
    pub fn with_surface(mut self, surface: Surface) -> Result<Self, AssetError> {
        surface.validate()?;
        self.surface = surface;
        Ok(self)
    }

    /// Multiplies an opaque texture's decoded linear color by this color factor.
    /// The library checks the texture handle when this material is inserted.
    pub fn textured(color: [f32; 4], texture: TextureHandle) -> Result<Self, AssetError> {
        let mut material = Self::new(color)?;
        material.texture = Some(texture);
        Ok(material)
    }

    pub fn color(&self) -> [f32; 4] {
        self.color
    }

    pub fn texture(&self) -> Option<TextureHandle> {
        self.texture
    }

    pub fn surface(&self) -> Surface {
        self.surface
    }
}
