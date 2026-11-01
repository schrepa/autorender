#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod cards;
mod library;
mod lod;
mod material;
mod mesh;
mod texture;

pub use cards::{AlphaImage, trim_cards};
pub use engine_lodbuild::{LodInput, LodLevel, MeshLods, MeshVertex};
pub use library::{
    AssetBudget, AssetLibrary, MaterialHandle, MeshHandle, SceneAssets, TextureHandle,
};
pub use lod::{lod_input, takes_levels};
pub use material::{
    AlphaMode, Filter, Grounding, MAX_GROUNDING_HEIGHT, MAX_WORLD_TILING, Material,
    MaterialBinding, Sampler, Surface, TextureRef, TopLayer, UvTransform, Wrap,
};
pub use mesh::{Aabb, Mesh};
pub use texture::{
    MAX_TEXTURE_DIMENSION, Texture, TextureFormat, TextureSummary, full_chain, srgb_decoded,
};
