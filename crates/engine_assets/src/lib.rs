//! Immutable, validated CPU geometry, textures and materials with bounded ownership.
//!
//! [`AssetLibrary`] owns assets. Handles are weak runtime references and cannot
//! be persisted or used in another library. Meshes are shared with owned render
//! snapshots using `Arc`; removing a handle never invalidates an existing snapshot.

mod animation;
mod error;
mod library;
mod lod;
mod material;
mod mesh;
pub mod simplify;
mod texture;
mod virtual_mesh;

pub use animation::{
    AnimatedProperty, AnimationChannel, AnimationClip, AnimationSample, KeyInterpolation, Keyframes,
};
pub use error::AssetError;
pub use library::{
    AssetBudget, AssetLibrary, ClipHandle, MaterialHandle, MeshHandle, TextureHandle,
};
pub use lod::{LodLevel, MeshLods, mesh_lods};
pub use material::{
    AlphaMode, Filter, Grounding, MAX_GROUNDING_HEIGHT, MAX_WORLD_TILING, Material,
    MaterialBinding, Sampler, Surface, TextureRef, TopLayer, Transmission, UvTransform, Wrap,
};
pub use mesh::{Aabb, Mesh, MeshVertex};
pub use texture::{MAX_TEXTURE_DIMENSION, Texture, TextureFormat};
pub use virtual_mesh::header as virtual_mesh_header;
pub use virtual_mesh::{
    CLUSTER_TRIANGLES, CLUSTER_VERTICES, CLUSTER_WORDS, ClusterInfo, DEFAULT_FALLBACK_TRIANGLES,
    ErrorSphere, FLAG_CLOSED, FLAG_UVS, GROUP_WORDS, HEADER_WORDS, NODE_WORDS,
    NONE as VIRTUAL_NONE, PAGE_WORDS, VERTEX_WORDS, VIRTUAL_MESH_MAGIC, VIRTUAL_MESH_VERSION,
    VirtualMesh, octahedral_encode, octahedral_normal,
};
