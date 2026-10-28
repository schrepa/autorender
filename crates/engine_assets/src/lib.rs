#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod cards;
mod error;
mod library;
mod lod;
mod material;
mod mesh;
mod texture;

pub use cards::{AlphaImage, trim_cards};
pub use error::AssetError;
pub use library::{AssetBudget, AssetLibrary, MaterialHandle, MeshHandle, TextureHandle};
pub use lod::{LOD_RECORD, LodInput, LodLevel, MeshLods, lod_input, takes_levels};
pub use material::{
    AlphaMode, Filter, Grounding, MAX_GROUNDING_HEIGHT, MAX_WORLD_TILING, Material,
    MaterialBinding, Sampler, Surface, TextureRef, TopLayer, UvTransform, Wrap,
};
pub use mesh::{Aabb, Mesh, MeshVertex};
pub use texture::{
    MAX_TEXTURE_DIMENSION, Texture, TextureFormat, TextureSummary, full_chain, srgb_decoded,
};
