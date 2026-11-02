#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod bricks;
mod components;
mod document;
mod environment;
mod files;
mod hierarchy;
mod instance;
mod loading;
mod material;
mod texture;
mod validation;
mod views;
mod writer;

pub use components::SceneKinds;
pub use document::{
    MAX_SOURCE_BYTES, MeshSource, SceneCamera, SceneDocument, SceneMesh, SceneObject, ScenePrefab,
    SceneSurface, SceneTransform, SceneVertex,
};
pub use engine_assets::{MeshStreams, encode_mesh_buffer_with};
pub use engine_registry::SceneEntity;
pub use engine_settings::{parse_json, parse_value};
pub use environment::{STREAMED, SceneEnvironment};
#[cfg(feature = "schema")]
pub use environment::section_schemas;
pub use files::{SceneFile, SceneFileKind, SceneFiles};
pub use instance::{MeshOrigin, SceneInstance, TextureOrigin};
pub use material::{
    SceneAlphaMode, SceneFilter, SceneGrounding, SceneMap, SceneMaterial, SceneMipmapFilter,
    SceneTextureMap, SceneTopLayer, SceneWrap,
};
pub use texture::SceneTexture;
pub use validation::{identifier, unknown};
pub use views::{
    SCENE_LOOK, SceneLook, SceneLookControl, SceneView, SceneViewControl, SceneViews, show_look,
    show_view,
};
