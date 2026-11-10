#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod attributes;
mod library;
mod lod;
mod material;
mod mesh;
mod record;
mod texture;

pub use attributes::{
    Attribute, AttributeFormat, Attributes, FIRST_NAMED_LOCATION, GeometryStream, InstanceInput,
    StreamShape, attributes_wgsl,
};
pub use engine_lodbuild::{LodInput, LodLevel, MeshLods, MeshVertex};
pub use library::{
    Asset, AssetBudget, AssetLibrary, GeometryHandle, MaterialHandle, SceneAssets, SceneTable,
    TextureHandle,
};
pub use lod::{lod_input, takes_levels};
pub use material::{
    AlphaMode, BASE_COLOR_MAP, Compare, Cull, DepthBias, EMISSIVE_MAP, Filter, LightingModel,
    MAPPED, MAPPED_MAPS, MAPPED_SLOTS, Material, MaterialBinding, NORMAL_MAP, RenderState, Sampler,
    SurfaceSlot, TextureRef, UvTransform, Wrap,
};
pub use mesh::{Aabb, Mesh, NamedStream};
pub use record::{
    MESH_BUFFER_EXTRAS_MAGIC, MESH_BUFFER_MAGIC, MeshBuffer, MeshStreams, encode_mesh_buffer_with,
};
pub use texture::{ColorSpace, Texture, TextureFormat, TextureSummary, full_chain, srgb_decoded};
