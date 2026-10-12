//! Authoring data and explicitly owned scene instances.
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
mod features;
mod files;
mod hierarchy;
mod instance;
mod json;
mod lighting;
mod material;
mod probe;
mod spline;
mod terrain;
mod texture;
mod validation;

pub use animation::{
    SceneAnimation, SceneChannel, SceneInterpolation, ScenePlayer, SceneProperty, SceneSkin,
};
pub use collider::{SceneCollider, SceneColliderShape};
pub use decal::{SceneDecal, SceneDecalBlend};
pub use document::{
    MAX_SOURCE_BYTES, MeshSource, SceneCamera, SceneDocument, SceneMesh, SceneObject, ScenePrefab,
    SceneSurface, SceneTransform, SceneVertex,
};
pub use engine_core::EnvironmentSection;
pub use environment::{AtmospherePreset, SceneAtmosphere, SceneEnvironment, SectionVisitor};
pub use error::SceneError;
pub use files::{
    ANIMATION_BUFFER_MAGIC, MESH_BUFFER_EXTRAS_MAGIC, MESH_BUFFER_MAGIC, MeshStreams, SceneFile,
    SceneFileKind, SceneFiles, encode_animation_buffer, encode_mesh_buffer,
    encode_mesh_buffer_with,
};
pub use instance::{ClipOrigin, MeshOrigin, SceneEntity, SceneInstance};
pub use json::parse_json;
pub use lighting::{SceneEmitter, SceneLight, SceneParticleBlend, SceneSpot};
pub use material::{
    SceneAlphaMode, SceneFilter, SceneGrounding, SceneMap, SceneMaterial, SceneMipmapFilter,
    SceneTextureMap, SceneTopLayer, SceneWrap,
};
pub use probe::SceneReflectionProbe;
pub use spline::SceneSpline;
pub use terrain::{
    SceneHeightmap, SceneSurroundings, SceneTerrain, SceneTerrainAsset, SceneTerrainLayer,
    SceneTerrainLod, SceneTerrainMaterial, SceneTerrainSurface,
};
pub use texture::SceneTexture;
