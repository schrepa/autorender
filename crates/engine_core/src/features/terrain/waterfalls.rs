//! Waterfalls: where terrains' hydrology bakes find rivers leaving their beds,
//! the engine spawns their spray and mist and despawns them as terrains'
//! bakes finish and terrains go. The water pass draws the jets themselves
//! from the terrains' falls (the renderer's `water_falls`). Fine spray
//! leaves each lip at the jet's speed and falls under gravity along it,
//! droplets splash up where the jet lands, and mist billows up there, as
//! much as the falling water's power stirs (particles). Water that does not
//! draw spawns none.

use crate::collections::HashMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use glam::Vec3;

use super::Terrains;
use crate::math::{clamp, max};
use crate::terrain_data::Fall;
use crate::{ParticleBlend, ParticleEmitter, Transform, Water};

/// The gravity spray falls under.
const GRAVITY: f32 = 9.8;

/// What each terrain's falls spawned.
#[derive(Resource, Default)]
pub(crate) struct TerrainFalls {
    spawned: HashMap<Entity, Spawned>,
}

/// A terrain's falls and the entities spawned for them.
struct Spawned {
    set: Arc<[Fall]>,
    entities: Vec<Entity>,
}

/// Water's density (kilograms a cubic meter): with a fall's discharge,
/// drop and gravity, the power its jet brings down.
const WATER_DENSITY: f32 = 1000.0;
/// Watts of falling water each splashing droplet and each puff of mist a
/// second stands for, and the fewest and most a fall's splash and mist
/// emit.
const SPLASH_WATTS: f32 = 2.0e3;
const SPLASH_RATE: [f32; 2] = [20.0, 600.0];
const MIST_WATTS: f32 = 1.5e5;
const MIST_RATE: [f32; 2] = [1.0, 12.0];

/// Each jet of fall `fall`: where it leaves (the index of its path's point),
/// where it lands, and the speed it lands at (meters a second).
fn jets(fall: &Fall) -> Vec<(usize, usize, f32)> {
    let mut found = Vec::new();
    let mut leaving = None;
    for index in 0..fall.path.len() {
        let aloft = fall.aloft.get(index).copied().unwrap_or(false);
        match (leaving, aloft) {
            (None, true) => leaving = Some(index),
            (Some(lip), false) => {
                let (a, b) = (
                    Vec3::from_array(fall.path[index - 1]),
                    Vec3::from_array(fall.path[index]),
                );
                let seconds = fall.times.get(index).zip(fall.times.get(index - 1));
                let speed = seconds.map_or(fall.speed, |(after, before)| {
                    a.distance(b) / max(after - before, 1.0e-3)
                });
                found.push((lip, index, speed));
                leaving = None;
            }
            _ => {}
        }
    }
    found
}

/// A fall's emitters at `origin`, seeded by `seed`: fine spray leaving its
/// lip at the jet's speed, falling under gravity along the jet's parabola;
/// and where each of its jets lands, droplets splashing up and out, as fast
/// as a share of its landing speed throws them, and mist billowing up, as
/// much as the jet's power stirs, its puffs as big as the fall is wide.
fn emitters(fall: &Fall, origin: Vec3, seed: u32) -> Vec<(Transform, ParticleEmitter)> {
    let width = fall.width;
    let at = |point: Vec3| Transform {
        translation: origin + point,
        ..Transform::default()
    };
    let point = |index: usize| Vec3::from_array(fall.path[index]);
    let jets = jets(fall);
    let Some(&(lip, landing, _)) = jets.first() else {
        return Vec::new();
    };
    let (top, bottom) = (point(lip), point(landing));
    let flight = Vec3::new(bottom.x - top.x, 0.0, bottom.z - top.z);
    let (reach, drop) = (flight.length(), max(top.y - bottom.y, 0.1));
    // The jet leaves the lip level at the speed that lands it `reach` away
    // `drop` below: drop = g reach^2 / (2 v^2).
    let speed = max(reach * (GRAVITY / (2.0 * drop)).sqrt(), 0.5);
    let seconds = (2.0 * drop / GRAVITY).sqrt();
    let mut spawned = vec![(
        at(top),
        ParticleEmitter {
            rate: 30.0 + 10.0 * width,
            lifetime: seconds,
            speed,
            direction: flight.normalize_or(Vec3::X),
            spread: 0.2,
            acceleration: Vec3::new(0.0, -GRAVITY, 0.0),
            size: [0.05, 0.12],
            color: [[0.9, 0.93, 0.97, 0.6], [0.94, 0.96, 1.0, 0.15]],
            emission: 1.0,
            blend: ParticleBlend::Alpha,
            stretch: 0.04,
            offset: Vec3::ZERO,
            radius: 0.45 * width,
            seed: seed.wrapping_mul(97),
        },
    )];
    let puffs = clamp(width / 10.0, 0.6, 2.5);
    for (number, &(lip, landing, landing_speed)) in jets.iter().enumerate() {
        let landed = point(landing);
        let drop = max(point(lip).y - landed.y, 0.1);
        let power = WATER_DENSITY * GRAVITY * fall.discharge * drop;
        let key = seed.wrapping_mul(131).wrapping_add(number as u32);
        spawned.push((
            at(landed),
            ParticleEmitter {
                rate: clamp(power / SPLASH_WATTS, SPLASH_RATE[0], SPLASH_RATE[1]),
                lifetime: 1.4,
                speed: clamp(landing_speed * 0.4, 2.0, 9.0),
                direction: Vec3::Y,
                spread: 0.9,
                acceleration: Vec3::new(0.0, -GRAVITY, 0.0),
                size: [0.06, 0.16],
                color: [[0.92, 0.95, 0.98, 0.65], [0.95, 0.97, 1.0, 0.0]],
                emission: 1.0,
                blend: ParticleBlend::Alpha,
                stretch: 0.05,
                offset: Vec3::ZERO,
                radius: 0.35 * width,
                seed: key.wrapping_mul(31) ^ 0x5bd1_e995,
            },
        ));
        spawned.push((
            at(landed + Vec3::Y * 1.5),
            ParticleEmitter {
                rate: clamp(power / MIST_WATTS, MIST_RATE[0], MIST_RATE[1]),
                lifetime: 6.0,
                speed: clamp(landing_speed * 0.12, 0.5, 2.5),
                direction: Vec3::Y,
                spread: 1.3,
                acceleration: Vec3::new(0.0, 0.3, 0.0),
                size: [4.0 * puffs, 11.0 * puffs],
                color: [[0.94, 0.96, 1.0, 0.4], [1.0, 1.0, 1.0, 0.0]],
                emission: 1.0,
                blend: ParticleBlend::Mist,
                stretch: 0.0,
                offset: Vec3::ZERO,
                radius: 0.5 * width,
                seed: key ^ 0x9e37_79b9,
            },
        ));
    }
    spawned
}

/// Spawns and despawns waterfall emitters to match terrains' falls.
pub(crate) fn sync(world: &mut World) {
    let drawn = world
        .get_resource::<Water>()
        .is_some_and(|water| water.enabled);
    let wanted: Vec<(Entity, Arc<[Fall]>, Vec3)> = match world.get_resource::<Terrains>() {
        Some(terrains) if drawn => terrains.fall_sets().collect(),
        _ => Vec::new(),
    };
    let mut falls = world.remove_resource::<TerrainFalls>().unwrap_or_default();
    let mut stale = Vec::new();
    falls.spawned.retain(|terrain, spawned| {
        let keep = wanted
            .iter()
            .any(|(entity, wanted, _)| entity == terrain && Arc::ptr_eq(wanted, &spawned.set));
        if !keep {
            stale.append(&mut spawned.entities);
        }
        keep
    });
    for entity in stale {
        if let Ok(entity) = world.get_entity_mut(entity) {
            entity.despawn();
        }
    }
    for (terrain, set, origin) in wanted {
        if set.is_empty() || falls.spawned.contains_key(&terrain) {
            continue;
        }
        let mut entities = Vec::with_capacity(3 * set.len());
        for (index, fall) in set.iter().enumerate() {
            for (transform, emitter) in emitters(fall, origin, index as u32) {
                entities.push(world.spawn((transform, emitter)).id());
            }
        }
        falls.spawned.insert(terrain, Spawned { set, entities });
    }
    world.insert_resource(falls);
}
