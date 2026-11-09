use engine_heightfield::layer::LayerKind;
use engine_place::{Error, List};
use serde_json::Value;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScenePath {
    pub id: String,
    pub origin: [f32; 3],
    pub points: Vec<[f32; 2]>,
    pub at: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScenePaths(pub Vec<ScenePath>);

pub struct ScenePathSources;

impl List for ScenePathSources {
    const NAME: &'static str = "scene_path_sources";
    type Member = ScenePathSource;
}

#[derive(Clone, Copy)]
pub struct ScenePathSource {
    pub key: &'static str,
    pub paths: fn(&Value, &[&'static LayerKind], &str) -> Result<ScenePaths, Error>,
}
