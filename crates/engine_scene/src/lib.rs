//! Versioned authoring data and explicitly owned scene instances.
//!
//! Parse and validate at load/reload boundaries, then instantiate through an
//! exclusive [`engine_core::bevy_ecs::world::World`] borrow between frames.
//! Serialization and authored identifier lookup are never part of extraction.
//! Runtime entities and asset handles are deliberately absent from the format.

mod collider;
mod document;
mod environment;
mod error;
mod hierarchy;
mod instance;
mod lighting;
mod texture;
mod validation;

pub use collider::{SceneCollider, SceneColliderShape};
pub use document::{
    MeshSource, SCENE_VERSION, SceneCamera, SceneDocument, SceneMaterial, SceneMesh, SceneObject,
    ScenePrefab, SceneSurface, SceneTransform, SceneVertex,
};
pub use environment::{
    AtmospherePreset, EnvironmentSettings, SceneAmbientOcclusion, SceneAtmosphere, SceneBloom,
    SceneBounds, SceneEnvironment, SceneLocalLights, SceneParticles, SceneReflections,
    SceneShadows, SceneSunShafts,
};
pub use error::SceneError;
pub use instance::{SceneEntity, SceneInstance};
pub use lighting::{SceneEmitter, SceneLight, SceneParticleBlend, SceneSpot};
pub use texture::SceneTexture;

#[cfg(test)]
mod collider_tests;
#[cfg(test)]
mod hierarchy_tests;
#[cfg(test)]
mod tests;
