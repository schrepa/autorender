//! Versioned authoring data and explicitly owned scene instances.
//!
//! Parse and validate at load/reload boundaries, then instantiate through an
//! exclusive [`engine_core::bevy_ecs::world::World`] borrow between frames.
//! Serialization and authored identifier lookup are never part of extraction.
//! Runtime entities and asset handles are deliberately absent from the format.

mod animation;
mod collider;
mod decal;
mod document;
mod environment;
mod error;
mod files;
mod hierarchy;
mod instance;
mod lighting;
mod material;
mod texture;
mod validation;

pub use animation::{
    SceneAnimation, SceneChannel, SceneInterpolation, ScenePlayer, SceneProperty, SceneSkin,
};
pub use collider::{SceneCollider, SceneColliderShape};
pub use decal::{SceneDecal, SceneDecalBlend};
pub use document::{
    MeshSource, SCENE_VERSION, SceneCamera, SceneDocument, SceneMesh, SceneObject, ScenePrefab,
    SceneSurface, SceneTransform, SceneVertex,
};
pub use environment::{
    AtmospherePreset, EnvironmentSettings, SceneAmbientOcclusion, SceneAtmosphere, SceneBloom,
    SceneBounds, SceneDecals, SceneEnvironment, SceneLocalLights, SceneParticles, SceneReflections,
    SceneRefraction, SceneShadows, SceneSunShafts,
};
pub use error::SceneError;
pub use files::{
    ANIMATION_BUFFER_MAGIC, MESH_BUFFER_EXTRAS_MAGIC, MESH_BUFFER_MAGIC, MeshStreams, SceneFile,
    SceneFileKind, SceneFiles, encode_animation_buffer, encode_mesh_buffer,
    encode_mesh_buffer_with,
};
pub use instance::{SceneEntity, SceneInstance};
pub use lighting::{SceneEmitter, SceneLight, SceneParticleBlend, SceneSpot};
pub use material::{
    SceneAlphaMode, SceneFilter, SceneMap, SceneMaterial, SceneMipmapFilter, SceneTextureMap,
    SceneWrap,
};
pub use texture::SceneTexture;

#[cfg(test)]
mod animation_tests;
#[cfg(test)]
mod collider_tests;
#[cfg(test)]
mod hierarchy_tests;
#[cfg(test)]
mod tests;
