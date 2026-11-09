use engine::Engine;
use engine_core::{BrickList, Error, ErrorCode};
use engine_registry::Registry;
use engine_scene::{SceneDocument, SceneKinds};

pub fn standard_engine() -> Result<Engine, Error> {
    engine::engine(engine_standard::everything::<BrickList>)
}

pub fn standard_kinds() -> Result<SceneKinds, Error> {
    Ok(SceneKinds::of(standard_engine()?.app.world()))
}

pub fn scene_schema(engine: &Engine) -> Result<String, Error> {
    let world = engine.app.world();
    world
        .get_resource::<Registry>()
        .map(|registry| SceneDocument::schema_json(registry, &SceneKinds::of(world)))
        .ok_or_else(|| Error::new(ErrorCode::MissingResource, "$.registry").got("Registry"))
}
