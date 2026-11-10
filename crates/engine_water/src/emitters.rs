use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{Cursor, RecordKey, Since, Spawner};
use engine_heightfield::{Fall, TileKey};
use glam::Vec3;

use crate::WaterBrick;
use crate::state::{WaterState, WaterTerrain};
use crate::waterfalls::spawn_falls;
use crate::white_water::spawn_tile;

struct Falls {
    origin: Vec3,
    falls: Arc<[Fall]>,
    entities: Vec<Entity>,
}

struct White {
    origin: Vec3,
    record: RecordKey,
    entities: Vec<Entity>,
}

#[derive(Default)]
pub(crate) struct Emitters {
    falls: Option<Falls>,
    white: BTreeMap<TileKey, White>,
    cursor: Cursor,
}

impl Emitters {
    pub(crate) fn white_water(&self) -> u32 {
        self.white
            .values()
            .map(|white| white.entities.len() as u32)
            .sum()
    }

    fn despawn(self, spawner: &mut Spawner<WaterBrick>) {
        let falls = self.falls.into_iter().flat_map(|falls| falls.entities);
        let white = self.white.into_values().flat_map(|white| white.entities);
        for entity in falls.chain(white) {
            spawner.despawn(entity);
        }
    }

    fn follow(&mut self, terrain: &WaterTerrain, spawner: &mut Spawner<WaterBrick>) {
        let origin = terrain.origin;
        let current = self
            .falls
            .as_ref()
            .is_some_and(|held| held.origin == origin && held.falls[..] == terrain.falls[..]);
        if !current {
            for entity in self.falls.take().into_iter().flat_map(|held| held.entities) {
                spawner.despawn(entity);
            }
            self.falls = Some(Falls {
                origin,
                falls: Arc::clone(&terrain.falls),
                entities: spawn_falls(&terrain.falls, origin, spawner),
            });
        }
        let keys: Vec<TileKey> = match terrain.touched.since(&mut self.cursor) {
            Since::Same => return,
            Since::Listed(keys) => keys.iter().copied().filter(|key| key.level == 0).collect(),
            Since::All => self
                .white
                .keys()
                .chain(terrain.settled.held.keys())
                .copied()
                .collect(),
        };
        for key in keys {
            self.follow_tile(terrain, key, spawner);
        }
    }

    fn follow_tile(
        &mut self,
        terrain: &WaterTerrain,
        key: TileKey,
        spawner: &mut Spawner<WaterBrick>,
    ) {
        let origin = terrain.origin;
        let wanted = terrain.settled.held.get(&key);
        let keeps = self.white.get(&key).is_some_and(|white| {
            white.origin == origin && wanted.is_some_and(|(_, made)| made.record == white.record)
        });
        if keeps {
            return;
        }
        for entity in self
            .white
            .remove(&key)
            .into_iter()
            .flat_map(|white| white.entities)
        {
            spawner.despawn(entity);
        }
        if let Some((water, made)) = wanted {
            let entities = spawn_tile(water, &terrain.falls, (key, origin), spawner);
            self.white.insert(
                key,
                White {
                    origin,
                    record: made.record,
                    entities,
                },
            );
        }
    }
}

pub(crate) fn sync(mut state: ResMut<WaterState>, mut spawner: Spawner<WaterBrick>) {
    let state = state.bypass_change_detection();
    let gone: Vec<Entity> = state
        .emitters
        .keys()
        .copied()
        .filter(|&entity| {
            !state
                .terrains
                .iter()
                .any(|terrain| terrain.entity == entity)
        })
        .collect();
    for entity in gone {
        if let Some(emitters) = state.emitters.remove(&entity) {
            emitters.despawn(&mut spawner);
        }
    }
    for terrain in &state.terrains {
        state
            .emitters
            .entry(terrain.entity)
            .or_default()
            .follow(terrain, &mut spawner);
    }
}
