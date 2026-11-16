use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{Cursor, Since, Spawner};
use engine_heightfield::{Fall, Pyramid, TileKey};
use glam::{Vec2, Vec3};

use crate::jobs::tile_water::point;
use crate::state::{WaterState, WaterTerrain};
use crate::waterfalls::{FallLook, drowned, falling, judged_at, spawn_fall};
use crate::{Water, WaterBrick, WaterPoint};

struct Falls {
    origin: Vec3,
    baked: Arc<[Fall]>,
    drowned: Vec<bool>,
    falling: Arc<[Fall]>,
    entities: Vec<Vec<Entity>>,
}

#[derive(Default)]
pub(crate) struct Emitters {
    falls: Option<Falls>,
    cursor: Cursor,
    look: Option<FallLook>,
    pyramid: Option<Pyramid>,
}

fn drawn_water(terrain: &WaterTerrain, at: Vec2) -> Option<WaterPoint> {
    let pyramid = &terrain.pyramid;
    (0..pyramid.levels())
        .find_map(|level| {
            let key = pyramid.tile_at(level, at)?;
            let held = terrain.resident(&key)?;
            point(
                &held.face,
                held.water.as_deref(),
                at - pyramid.tile_origin(key),
            )
        })
        .flatten()
}

fn touches_falls(terrain: &WaterTerrain, keys: &[TileKey]) -> bool {
    if keys.is_empty() {
        return false;
    }
    let pyramid = &terrain.pyramid;
    let points = judged_at(&terrain.falls);
    keys.iter().any(|&key| {
        points
            .iter()
            .any(|&at| pyramid.tile_at(u32::from(key.level), at) == Some(key))
    })
}

impl Emitters {
    pub(crate) fn falling(&self) -> Option<&Arc<[Fall]>> {
        self.falls.as_ref().map(|falls| &falls.falling)
    }

    fn despawn(self, spawner: &mut Spawner<WaterBrick>) {
        let falls = self
            .falls
            .into_iter()
            .flat_map(|falls| falls.entities)
            .flatten();
        for entity in falls {
            spawner.despawn(entity);
        }
    }

    fn follow(
        &mut self,
        terrain: &WaterTerrain,
        look: &FallLook,
        spawner: &mut Spawner<WaterBrick>,
    ) {
        if self.look.as_ref() != Some(look) {
            let falls = self.falls.take().into_iter();
            for entity in falls.flat_map(|falls| falls.entities).flatten() {
                spawner.despawn(entity);
            }
            self.look = Some(*look);
        }
        let tiled = self.pyramid.replace(terrain.pyramid) != Some(terrain.pyramid);
        let touched = match terrain.touched.since(&mut self.cursor) {
            Since::Same => tiled,
            Since::Listed(keys) => tiled || touches_falls(terrain, keys),
            Since::All => true,
        };
        self.follow_falls(terrain, touched, look, spawner);
    }

    fn follow_falls(
        &mut self,
        terrain: &WaterTerrain,
        touched: bool,
        look: &FallLook,
        spawner: &mut Spawner<WaterBrick>,
    ) {
        let origin = terrain.origin;
        let baked = self.falls.as_ref().is_some_and(|held| {
            Arc::ptr_eq(&held.baked, &terrain.falls) || held.baked[..] == terrain.falls[..]
        });
        let placed = self
            .falls
            .as_ref()
            .is_some_and(|held| held.origin == origin);
        if baked && placed && !touched {
            return;
        }
        let slowest = look.0.slowest;
        let sunk = drowned(&terrain.falls, slowest, &|at| drawn_water(terrain, at));
        let judged = baked
            && self
                .falls
                .as_ref()
                .is_some_and(|held| held.drowned == sunk);
        if judged && placed {
            return;
        }
        let before = self.falls.take();
        let falling: Arc<[Fall]> = match &before {
            Some(held) if judged => Arc::clone(&held.falling),
            Some(_) | None => falling(&terrain.falls, &sunk).into(),
        };
        let (was, mut kept): (Arc<[Fall]>, Vec<Option<Vec<Entity>>>) = match before {
            Some(held) => (
                held.falling,
                held.entities.into_iter().map(Some).collect(),
            ),
            None => (Arc::from(Vec::new()), Vec::new()),
        };
        if !placed {
            for entity in kept.iter_mut().filter_map(Option::take).flatten() {
                spawner.despawn(entity);
            }
        }
        let entities = falling
            .iter()
            .map(|fall| {
                was.iter()
                    .zip(&mut kept)
                    .find(|(held, entities)| entities.is_some() && *held == fall)
                    .and_then(|(_, entities)| entities.take())
                    .unwrap_or_else(|| spawn_fall(fall, origin, look, spawner))
            })
            .collect();
        for entity in kept.into_iter().flatten().flatten() {
            spawner.despawn(entity);
        }
        self.falls = Some(Falls {
            origin,
            baked: Arc::clone(&terrain.falls),
            drowned: sunk,
            falling,
            entities,
        });
    }
}

pub(crate) fn sync(
    mut state: ResMut<WaterState>,
    water: Res<Water>,
    mut spawner: Spawner<WaterBrick>,
) {
    let look = (water.fall_sheets, water.fall_spray, water.mist);
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
            .follow(terrain, &look, &mut spawner);
    }
}
