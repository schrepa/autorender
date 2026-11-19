use bevy_ecs::component::ComponentId;
use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;

use crate::{ComponentKind, Error, scene_components};

pub fn admit(
    world: &mut World,
    entity: Entity,
    parts: Option<&[ComponentId]>,
    path: &str,
) -> Result<(), Error> {
    let kinds: Vec<ComponentKind> = scene_components(world)
        .filter(|kind| kind.admits())
        .cloned()
        .collect();
    for kind in kinds {
        if parts.is_none_or(|parts| parts.contains(&kind.register(world))) {
            let at = format!("{path}.{}", kind.key());
            kind.admit_held(world, entity, &at)?;
        }
    }
    Ok(())
}
