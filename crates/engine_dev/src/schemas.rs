use bevy_ecs::prelude::World;
use engine_registry::Registry;

pub fn game(world: &World) -> Result<Vec<(&'static str, String)>, String> {
    let registry = world
        .get_resource::<Registry>()
        .ok_or_else(|| "the schemas' engine lists no bricks: list the game's bricks".to_owned())?;
    let scene =
        engine_scene::SceneDocument::schema_json(registry, &engine_scene::SceneKinds::of(world));
    Ok(vec![(engine_core::SCENE_SCHEMA, scene)])
}
