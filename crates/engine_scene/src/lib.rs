#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod bricks;
mod components;
mod document;
mod environment;
mod hierarchy;
mod instance;
mod material;
mod spawning;
mod texture;
mod validation;
mod writer;

pub use components::SceneKinds;
pub use document::{
    MAX_SOURCE_BYTES, SceneDocument, SceneMesh, SceneObject, ScenePrefab, SceneTransform,
};
pub use engine_core::SceneEntity;
pub use environment::SceneEnvironment;
pub use instance::SceneInstance;
pub use material::{
    SceneAlphaMode, SceneCull, SceneDepthBias, SceneFilter, SceneMap, SceneMaterial,
    SceneMipmapFilter, SceneSlot, SceneSlotFill, SceneTextureMap, SceneWrap,
};
pub use spawning::{described, named, remove, spawn};
pub use texture::SceneTexture;
pub use validation::{identifier, unknown};
