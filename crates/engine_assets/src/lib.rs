#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod library;
mod material;
mod mesh;
mod texture;

pub use library::{Asset, AssetId, AssetKind, AssetNames, Assets, GroupReport, GroupStage};
pub use material::{
    AlphaMode, Compare, Cull, DepthBias, Filter, LightingModel, Material, RenderState, Sampler,
    TextureRef, UvTransform, Wrap,
};
pub use mesh::{
    Aabb, IndexFormat, MeshDescription, MeshFile, VertexFormat, VertexInput, VertexStream,
};
pub use texture::{ColorSpace, Texture, TextureFormat};
