use bevy_ecs::prelude::World;
use engine_core::Error;

pub fn show_view(world: &mut World, name: &str, path: &str) -> Result<(), Error> {
    engine_scene::show_view(world, name, path)?;
    Ok(())
}

pub fn show_look(world: &mut World, name: &str, path: &str) -> Result<(), Error> {
    engine_scene::show_look(world, name, path)
}
