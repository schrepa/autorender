//! Immutable, validated CPU geometry, textures and materials with bounded ownership.
//!
//! [`AssetLibrary`] owns assets. Handles are weak runtime references and cannot
//! be persisted or used in another library. Meshes are shared with owned render
//! snapshots using `Arc`; removing a handle never invalidates an existing snapshot.

mod animation;
mod error;
mod library;
mod material;
mod mesh;
mod texture;

pub use animation::{
    AnimatedProperty, AnimationChannel, AnimationClip, AnimationSample, KeyInterpolation, Keyframes,
};
pub use error::AssetError;
pub use library::{
    AssetBudget, AssetLibrary, ClipHandle, MaterialHandle, MeshHandle, TextureHandle,
};
pub use material::{
    AlphaMode, Filter, Material, MaterialBinding, Sampler, Surface, TextureRef, UvTransform, Wrap,
};
pub use mesh::{Aabb, Mesh, MeshVertex};
pub use texture::{MAX_TEXTURE_DIMENSION, Texture, TextureFormat};

#[cfg(test)]
mod tests;
