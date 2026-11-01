use engine::Engine;
use engine_core::{BrickList, Error, ErrorCode};
use engine_registry::Registry;

pub fn standard_engine() -> Result<Engine, Error> {
    engine::engine(engine_standard::standard::<BrickList>)
}

pub fn scene_schema(engine: &Engine) -> Result<String, Error> {
    engine
        .app
        .world()
        .get_resource::<Registry>()
        .map(engine_scene::SceneDocument::schema_json)
        .ok_or_else(|| Error::new(ErrorCode::MissingResource, "$.registry").got("Registry"))
}
