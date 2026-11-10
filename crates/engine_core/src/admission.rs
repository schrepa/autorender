use bevy_ecs::component::ComponentId;
use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;

use crate::{ComponentKind, Error, MAX_HIERARCHY_DEPTH, Parent, Placed, Transform, scene_components};

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
    let parented = world.get::<Parent>(entity).is_some();
    let placed = placed(world, entity);
    for kind in kinds {
        if parts.is_none_or(|parts| parts.contains(&kind.register(world))) {
            let at = format!("{path}.{}", kind.key());
            kind.admit_held(world, entity, (parented, placed), &at)?;
        }
    }
    Ok(())
}

fn placed(world: &World, entity: Entity) -> Option<Placed> {
    let mut matrix = world.get::<Transform>(entity)?.matrix();
    let mut above = world.get::<Parent>(entity).map(Parent::parent);
    for _ in 0..MAX_HIERARCHY_DEPTH {
        let Some(node) = above else {
            return Some(Placed { world: matrix });
        };
        matrix = world.get::<Transform>(node)?.matrix() * matrix;
        above = world.get::<Parent>(node).map(Parent::parent);
    }
    None
}
