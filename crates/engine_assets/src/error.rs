use std::fmt;

/// Validation or admission failure. No library mutation occurs on failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetError {
    EmptyMesh,
    InvalidTriangleCount {
        indices: usize,
    },
    InvalidVertex {
        vertex: usize,
    },
    InvalidNormal {
        vertex: usize,
    },
    InvalidUv {
        vertex: usize,
    },
    /// An optional vertex attribute (decision 034) of the wrong length.
    AttributeCount {
        attribute: &'static str,
        expected: usize,
        actual: usize,
    },
    /// An optional vertex attribute value outside its range.
    InvalidAttribute {
        attribute: &'static str,
        vertex: usize,
    },
    IndexOutOfBounds {
        index: usize,
        value: u32,
        vertices: usize,
    },
    InvalidColor {
        channel: usize,
    },
    UnsupportedAlpha,
    InvalidOpacity,
    InvalidSurface {
        field: &'static str,
    },
    /// A texture map parameter outside its range (decision 031).
    InvalidMaterialMap {
        field: &'static str,
    },
    InvalidTextureDimensions {
        width: u32,
        height: u32,
        limit: u32,
    },
    InvalidTextureLength {
        expected: usize,
        actual: usize,
    },
    UnsupportedTextureAlpha {
        pixel: usize,
    },
    InvalidTextureHandle,
    InvalidMaterialHandle,
    /// An animation clip or channel breaking a rule (decision 035).
    InvalidAnimation {
        reason: &'static str,
    },
    /// Block-compressed texture levels breaking a rule (decision 037).
    InvalidCompressedTexture {
        reason: &'static str,
    },
    MeshCapacity {
        limit: usize,
    },
    MaterialCapacity {
        limit: usize,
    },
    TextureCapacity {
        limit: usize,
    },
    CpuBudget {
        requested: usize,
        available: usize,
    },
}

impl AssetError {
    /// Stable machine-readable error identifier; display wording may evolve.
    pub const fn code(&self) -> &'static str {
        match self {
            Self::EmptyMesh => "asset.empty_mesh",
            Self::InvalidTriangleCount { .. } => "asset.invalid_triangle_count",
            Self::InvalidVertex { .. } => "asset.invalid_vertex",
            Self::InvalidNormal { .. } => "asset.invalid_normal",
            Self::InvalidUv { .. } => "asset.invalid_uv",
            Self::AttributeCount { .. } => "asset.attribute_count",
            Self::InvalidAttribute { .. } => "asset.invalid_attribute",
            Self::IndexOutOfBounds { .. } => "asset.index_out_of_bounds",
            Self::InvalidColor { .. } => "asset.invalid_color",
            Self::UnsupportedAlpha => "asset.unsupported_alpha",
            Self::InvalidOpacity => "asset.invalid_opacity",
            Self::InvalidSurface { .. } => "asset.invalid_surface",
            Self::InvalidMaterialMap { .. } => "asset.invalid_material_map",
            Self::InvalidTextureDimensions { .. } => "asset.invalid_texture_dimensions",
            Self::InvalidTextureLength { .. } => "asset.invalid_texture_length",
            Self::UnsupportedTextureAlpha { .. } => "asset.unsupported_texture_alpha",
            Self::InvalidTextureHandle => "asset.invalid_texture_handle",
            Self::InvalidMaterialHandle => "asset.invalid_material_handle",
            Self::InvalidAnimation { .. } => "asset.invalid_animation",
            Self::InvalidCompressedTexture { .. } => "asset.invalid_compressed_texture",
            Self::MeshCapacity { .. } => "asset.mesh_capacity",
            Self::MaterialCapacity { .. } => "asset.material_capacity",
            Self::TextureCapacity { .. } => "asset.texture_capacity",
            Self::CpuBudget { .. } => "asset.cpu_budget",
        }
    }
}

impl fmt::Display for AssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMesh => write!(f, "mesh requires vertices and at least one triangle"),
            Self::InvalidTriangleCount { indices } => write!(
                f,
                "mesh index count {indices} must be a positive multiple of 3"
            ),
            Self::InvalidVertex { vertex } => {
                write!(f, "mesh vertex {vertex} position must be finite")
            }
            Self::InvalidNormal { vertex } => {
                write!(f, "mesh vertex {vertex} normal must be finite and nonzero")
            }
            Self::InvalidUv { vertex } => {
                write!(f, "mesh vertex {vertex} texture coordinates must be finite")
            }
            Self::AttributeCount {
                attribute,
                expected,
                actual,
            } => write!(
                f,
                "mesh {attribute} need one value per vertex: {expected}; received {actual}"
            ),
            Self::InvalidAttribute { attribute, vertex } => write!(
                f,
                "mesh vertex {vertex} {attribute} must be finite and within range"
            ),
            Self::IndexOutOfBounds {
                index,
                value,
                vertices,
            } => write!(
                f,
                "mesh index {index} is {value}, outside {vertices} vertices"
            ),
            Self::InvalidColor { channel } => write!(
                f,
                "material color channel {channel} must be finite and in [0, 1]"
            ),
            Self::UnsupportedAlpha => write!(
                f,
                "material alpha must equal 1; use Material::with_opacity for transparency"
            ),
            Self::InvalidOpacity => write!(f, "material opacity must be finite and in (0, 1]"),
            Self::InvalidSurface { field } => write!(
                f,
                "material surface {field} must be finite and within its documented range"
            ),
            Self::InvalidMaterialMap { field } => write!(
                f,
                "material {field} must be finite and within its documented range"
            ),
            Self::InvalidTextureDimensions {
                width,
                height,
                limit,
            } => write!(
                f,
                "texture dimensions {width} x {height} must each be in 1..={limit}"
            ),
            Self::InvalidTextureLength { expected, actual } => write!(
                f,
                "texture needs {expected} tightly packed RGBA8 bytes; received {actual}"
            ),
            Self::UnsupportedTextureAlpha { pixel } => {
                write!(f, "opaque texture pixel {pixel} alpha must equal 255")
            }
            Self::InvalidTextureHandle => {
                write!(f, "material texture must be live in the same asset library")
            }
            Self::InvalidMaterialHandle => {
                write!(f, "material must be live in the same asset library")
            }
            Self::InvalidAnimation { reason } => write!(f, "animation: {reason}"),
            Self::InvalidCompressedTexture { reason } => {
                write!(f, "compressed texture: {reason}")
            }
            Self::MeshCapacity { limit } => write!(f, "mesh capacity of {limit} reached"),
            Self::MaterialCapacity { limit } => write!(f, "material capacity of {limit} reached"),
            Self::TextureCapacity { limit } => write!(f, "texture capacity of {limit} reached"),
            Self::CpuBudget {
                requested,
                available,
            } => write!(
                f,
                "asset needs {requested} CPU payload bytes; {available} remain"
            ),
        }
    }
}

impl std::error::Error for AssetError {}
