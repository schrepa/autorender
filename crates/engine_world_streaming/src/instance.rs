use std::sync::Arc;

use engine_core::{
    bevy_ecs::prelude::{Entity, World},
    set_parents,
};

use crate::Streamed;

struct StreamedScene {
    lookups: Lookups,
    prepared: PreparedStreamed,
}

struct StreamedTree {
    scene: Arc<StreamedScene>,
    item: usize,
}

impl Streamed for StreamedTree {
    fn entities(&self) -> u32 {
        self.scene.prepared.items[self.item].entities.len() as u32
    }

    fn spawn(&self, world: &mut World, spawned: &mut Vec<Entity>) {
        let scene = &*self.scene;
        let prepared = &scene.prepared;
        let item = &prepared.items[self.item];
        let mine: Vec<Entity> = item
            .entities
            .iter()
            .map(|&index| {
                spawn_entity(
                    world,
                    &scene.lookups,
                    &prepared.entities[index],
                    prepared.transforms[index],
                    &prepared.components[index],
                )
            })
            .collect();
        let entity_of = |index: usize| {
            mine[item
                .entities
                .iter()
                .position(|&own| own == index)
                .expect("reference_within_item")]
        };
        let parents: Vec<_> = item
            .entities
            .iter()
            .zip(&mine)
            .filter_map(|(&index, &entity)| {
                prepared.parents[index].map(|parent| (entity, Some(entity_of(parent))))
            })
            .collect();
        if set_parents(world, &parents).is_err() {
            for entity in mine {
                let _ = world.try_despawn(entity);
            }
            return;
        }
        spawned.extend(mine);
    }
}
