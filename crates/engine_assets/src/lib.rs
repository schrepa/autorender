#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod animation;
mod cards;
mod error;
mod library;
mod lod;
mod material;
mod mesh;
mod texture;
mod virtual_mesh;

pub use animation::{
    AnimatedProperty, AnimationChannel, AnimationClip, AnimationSample, KeyInterpolation, Keyframes,
};
pub use cards::{AlphaImage, trim_cards};
pub use error::AssetError;
pub use library::{
    AssetBudget, AssetLibrary, ClipHandle, MaterialHandle, MeshHandle, TextureHandle,
};
pub use lod::{LOD_RECORD, LodInput, LodLevel, MeshLods, lod_input, takes_levels};
pub use material::{
    AlphaMode, Filter, Grounding, MAX_GROUNDING_HEIGHT, MAX_WORLD_TILING, Material,
    MaterialBinding, Sampler, Surface, TextureRef, TopLayer, UvTransform, Wrap,
};
pub use mesh::{Aabb, Mesh, MeshVertex};
pub use texture::{
    MAX_TEXTURE_DIMENSION, Texture, TextureFormat, TextureSummary, full_chain, srgb_decoded,
};
pub use virtual_mesh::header as virtual_mesh_header;
pub use virtual_mesh::{
    CLUSTER_TRIANGLES, CLUSTER_VERTICES, CLUSTER_WORDS, ClusterInfo, DEFAULT_FALLBACK_TRIANGLES,
    ErrorSphere, FLAG_CLOSED, FLAG_UVS, GROUP_WORDS, HEADER_WORDS, NODE_WORDS,
    NONE as VIRTUAL_NONE, PAGE_WORDS, VERTEX_WORDS, VIRTUAL_MESH_MAGIC, VIRTUAL_MESH_VERSION,
    VirtualMesh, octahedral_encode, octahedral_normal,
};
