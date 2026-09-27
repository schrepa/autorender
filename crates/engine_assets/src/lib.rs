//! Immutable, validated CPU geometry, textures and materials with bounded ownership.
//!
//! [`AssetLibrary`] owns assets. Handles are weak runtime references and cannot
//! be persisted or used in another library. Meshes are shared with owned render
//! snapshots using `Arc`; removing a handle never invalidates an existing snapshot.

mod error;
mod library;
mod material;
mod mesh;
mod texture;

pub use error::AssetError;
pub use library::{AssetBudget, AssetLibrary, MaterialHandle, MeshHandle, TextureHandle};
pub use material::{AlphaMode, Filter, Material, Sampler, Surface, TextureRef, UvTransform, Wrap};
pub use mesh::{Aabb, Mesh, MeshVertex};
pub use texture::{MAX_TEXTURE_DIMENSION, Texture};

#[cfg(test)]
mod tests;
