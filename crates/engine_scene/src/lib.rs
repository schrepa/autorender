#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod authored;
mod collider;
mod components;
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

pub use authored::active_scene;
pub use collider::{SceneCollider, SceneColliderShape};
pub use components::COMPONENT_SHOWS;
pub use document::{
    MAX_SOURCE_BYTES, MeshSource, SceneCamera, SceneDocument, SceneMesh, SceneObject, ScenePrefab,
    SceneSurface, SceneTransform, SceneVertex,
};
pub use engine_core::EnvironmentSection;
pub use environment::{AtmospherePreset, SceneAtmosphere, SceneEnvironment, SectionVisitor};
pub use error::SceneError;
pub use files::{
    MESH_BUFFER_EXTRAS_MAGIC, MESH_BUFFER_MAGIC, MeshStreams, SceneFile, SceneFileKind, SceneFiles,
    decode_mesh_buffer, encode_mesh_buffer, encode_mesh_buffer_with,
};
pub use instance::{MeshOrigin, SceneEntity, SceneInstance};
pub use json::{parse_json, parse_value};
pub use lighting::{SceneEmitter, SceneLight, SceneParticleBlend, SceneSpot};
pub use material::{
    SceneAlphaMode, SceneFilter, SceneGrounding, SceneMap, SceneMaterial, SceneMipmapFilter,
    SceneTextureMap, SceneTopLayer, SceneWrap,
};
pub use probe::SceneReflectionProbe;
pub use spline::SceneSpline;
pub use terrain::{
    SceneHeightmap, SceneSurroundings, SceneTerrain, SceneTerrainAsset, SceneTerrainLayer,
    SceneTerrainLod, SceneTerrainMaterial, SceneTerrainPart, SceneTerrainSurface,
};
pub use texture::SceneTexture;
