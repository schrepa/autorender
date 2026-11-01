#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod authored;
mod bricks;
mod components;
mod document;
mod environment;
mod files;
mod grounded;
mod hierarchy;
mod instance;
mod lighting;
mod material;
mod probe;
pub mod rules;
mod spline;
mod texture;
mod validation;
mod views;
mod writer;

pub use bricks::added;
pub use components::{
    CORE_COMPONENTS, ComponentContext, ComponentKind, ComponentPath, GROUNDED, Placement, Prepared,
    PreparedComponent, Reach, SceneComponents, SceneComponentsList, admitted, component_kind,
    component_kinds, component_kinds_in, component_names, no_files, no_paths, object_example,
    parsed, register_components, ruled, valid, valid_world,
};
pub use document::{
    MAX_SOURCE_BYTES, MeshSource, SceneCamera, SceneDocument, SceneMesh, SceneObject, ScenePrefab,
    SceneSurface, SceneTransform, SceneVertex,
};
pub use engine_core::{parse_json, parse_value};
pub use environment::SceneEnvironment;
#[cfg(feature = "schema")]
pub use environment::section_schemas;
pub use files::{
    MESH_BUFFER_EXTRAS_MAGIC, MESH_BUFFER_MAGIC, MeshStreams, SceneFile, SceneFileKind, SceneFiles,
    decode_mesh_buffer, encode_mesh_buffer, encode_mesh_buffer_with, valid_uri,
};
pub use grounded::Grounded;
pub use instance::{Lookups, MeshOrigin, SceneEntity, SceneInstance, TextureOrigin, authored_id};
pub use lighting::{SceneEmitter, SceneLight, SceneParticleBlend, SceneSpot};
pub use material::{
    SceneAlphaMode, SceneFilter, SceneGrounding, SceneMap, SceneMaterial, SceneMipmapFilter,
    SceneTextureMap, SceneTopLayer, SceneWrap,
};
pub use probe::SceneReflectionProbe;
pub use rules::{SceneAlong, SceneAlongStreamed};
pub use spline::SceneSpline;
pub use texture::SceneTexture;
pub use validation::{identifier, unknown};
pub use views::{SCENE_LOOK, SceneLook, SceneView, SceneViews, show_look, show_view};
