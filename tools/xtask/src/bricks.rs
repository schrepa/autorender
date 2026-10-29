use engine::bevy_ecs::prelude::World;
use engine::bricks::{AddBrick, BrickRegistry};

pub fn standard_engine() -> engine::Engine {
    let mut engine = engine::Engine::new(engine::EngineConfig::default());
    engine.app.add_bricks(engine::bricks::standard());
    engine
}

pub fn environment_sections(world: &World) -> Vec<(&'static str, &'static str)> {
    world
        .get_resource::<BrickRegistry>()
        .into_iter()
        .flat_map(BrickRegistry::sections)
        .map(|section| (section.key, section.name))
        .collect()
}
