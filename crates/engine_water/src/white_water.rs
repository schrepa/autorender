use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::math::{GRAVITY, clamp, max};
use engine_core::{RecordKey, Spawner, Transform};
use engine_heightfield::{Fall, TileKey};
use engine_places::particles::{ParticleBlend, ParticleEmitter};
use glam::{Vec2, Vec3};

use crate::state::WaterState;
use crate::{SteadyCell, SteadyWater, WaterBrick, WaterStats};

const SPRAY_ENERGY: f32 = 200.0;
const SPRAY_LIP: f32 = 0.2;
const SITE_APART: f32 = 8.0;
const SITES_MOST: usize = 32;
const SPRAY_WATTS: f32 = 400.0;
const SPRAY_RATE: [f32; 2] = [8.0, 200.0];
const SPRAY_THROW: f32 = 0.8;
const SPRAY_SPEED: [f32; 2] = [1.5, 5.0];
const SPRAY_RISE: f32 = 1.5;
const MIST_POWER: f32 = 4.0e4;
const MIST_WATTS: f32 = 2.0e4;
const MIST_RATE: [f32; 2] = [1.0, 6.0];

#[derive(Resource, Default)]
pub(crate) struct TerrainWhiteWater {
    spawned: Vec<Spawned>,
}

struct Spawned {
    terrain: Entity,
    origin: Vec3,
    key: TileKey,
    record: RecordKey,
    entities: Vec<Entity>,
}

struct Held {
    terrain: Entity,
    origin: Vec3,
    falls: Arc<[Fall]>,
    key: TileKey,
    record: RecordKey,
    water: Arc<SteadyWater>,
}

struct Breaking {
    strength: f32,
    index: u32,
    middle: Vec2,
    surface: f32,
    flow: Vec2,
    power: f32,
}

struct Site {
    at: Vec3,
    flow: Vec2,
    power: f32,
    reach: f32,
}

fn held(state: &WaterState) -> Vec<Held> {
    let mut found = Vec::new();
    for terrain in &state.terrains {
        for (key, (water, record)) in terrain.settled.held.iter() {
            found.push(Held {
                terrain: terrain.entity,
                origin: terrain.origin,
                falls: terrain.falls.clone(),
                key,
                record: *record,
                water: Arc::clone(water),
            });
        }
    }
    found
}

fn sites(water: &SteadyWater, falls: &[Fall]) -> Vec<Site> {
    let cell = water.cell;
    let [columns, rows] = water.cells.map(i64::from);
    let middle = |i: i64, j: i64| Vec2::new((i as f32 + 0.5) * cell, (j as f32 + 0.5) * cell);
    let mut breaking = Vec::new();
    for z in 0..rows {
        for x in 0..columns {
            let (i, j) = (water.first[0] + x, water.first[1] + z);
            let Some(SteadyCell::Flowing {
                surface,
                velocity,
                energy,
                lip,
                ..
            }) = water.at(i, j)
            else {
                continue;
            };
            let energy = if energy.is_finite() {
                max(energy, 0.0)
            } else {
                0.0
            };
            let lip = if lip.is_finite() { lip } else { 0.0 };
            if energy < SPRAY_ENERGY && lip < SPRAY_LIP {
                continue;
            }
            breaking.push(Breaking {
                strength: max(energy / SPRAY_ENERGY, lip / SPRAY_LIP),
                index: (z * columns + x) as u32,
                middle: middle(i, j),
                surface,
                flow: Vec2::from(velocity),
                power: energy * cell * cell,
            });
        }
    }
    breaking.sort_unstable_by(|a, b| {
        b.strength
            .total_cmp(&a.strength)
            .then(a.index.cmp(&b.index))
    });
    let apart = SITE_APART * SITE_APART;
    let low = middle(water.first[0], water.first[1]) - Vec2::splat(SITE_APART);
    let high = middle(water.first[0] + columns, water.first[1] + rows) + Vec2::splat(SITE_APART);
    let falling: Vec<Vec2> = falls
        .iter()
        .flat_map(|fall| fall.path.iter())
        .map(|point| Vec2::new(point[0], point[2]))
        .filter(|point| point.cmpge(low).all() && point.cmple(high).all())
        .collect();
    let mut found: Vec<(Breaking, f32, u32)> = Vec::new();
    for candidate in breaking {
        let near = found
            .iter_mut()
            .find(|(first, ..)| first.middle.distance_squared(candidate.middle) < apart);
        if let Some((_, power, cells)) = near {
            *power += candidate.power;
            *cells += 1;
        } else if found.len() < SITES_MOST
            && falling
                .iter()
                .all(|point| point.distance_squared(candidate.middle) >= apart)
        {
            let power = candidate.power;
            found.push((candidate, power, 1));
        }
    }
    found
        .into_iter()
        .map(|(first, power, cells)| Site {
            at: Vec3::new(first.middle.x, first.surface, first.middle.y),
            flow: first.flow,
            power,
            reach: clamp(
                cell * (cells as f32 / std::f32::consts::PI).sqrt(),
                0.5 * cell,
                0.25 * SITE_APART,
            ),
        })
        .collect()
}

fn emitters(site: &Site, origin: Vec3, seed: u32) -> Vec<(Transform, ParticleEmitter)> {
    let at = |rise: f32| Transform {
        translation: origin + site.at + Vec3::Y * rise,
        ..Transform::default()
    };
    let along = site.flow.normalize_or_zero();
    let direction = Vec3::new(along.x, SPRAY_RISE, along.y).normalize();
    let speed = clamp(
        SPRAY_THROW * site.flow.length(),
        SPRAY_SPEED[0],
        SPRAY_SPEED[1],
    );
    let flight = 2.0 * speed * direction.y / GRAVITY;
    let mut spawned = vec![(
        at(0.05),
        ParticleEmitter {
            rate: clamp(site.power / SPRAY_WATTS, SPRAY_RATE[0], SPRAY_RATE[1]),
            lifetime: flight,
            speed,
            direction,
            spread: 0.6,
            acceleration: Vec3::new(0.0, -GRAVITY, 0.0),
            size: [0.03, 0.08],
            color: [[0.9, 0.93, 0.97, 0.55], [0.94, 0.96, 1.0, 0.0]],
            emission: 1.0,
            blend: ParticleBlend::Alpha,
            stretch: 0.04,
            offset: Vec3::ZERO,
            radius: site.reach,
            seed: seed ^ 0x27d4_eb2f,
        },
    )];
    if site.power >= MIST_POWER {
        let puffs = clamp(0.5 * site.reach, 0.5, 1.5);
        spawned.push((
            at(0.4),
            ParticleEmitter {
                rate: clamp(site.power / MIST_WATTS, MIST_RATE[0], MIST_RATE[1]),
                lifetime: 4.0,
                speed: clamp(0.15 * site.flow.length(), 0.3, 1.0),
                direction: Vec3::Y,
                spread: 1.2,
                acceleration: Vec3::new(0.0, 0.2, 0.0),
                size: [puffs, 3.0 * puffs],
                color: [[0.94, 0.96, 1.0, 0.16], [1.0, 1.0, 1.0, 0.0]],
                emission: 1.0,
                blend: ParticleBlend::Mist,
                stretch: 0.0,
                offset: Vec3::ZERO,
                radius: site.reach,
                seed: seed ^ 0x1656_67b1,
            },
        ));
    }
    spawned
}

fn seed_of(key: TileKey, number: usize) -> u32 {
    let tile =
        u32::from(key.x).wrapping_mul(0x9e37_79b9) ^ u32::from(key.z).wrapping_mul(0x85eb_ca6b);
    tile.wrapping_add((number as u32).wrapping_mul(0xc2b2_ae35))
}

impl TerrainWhiteWater {
    pub(crate) fn take(&mut self) -> Vec<Entity> {
        self.spawned
            .drain(..)
            .flat_map(|spawned| spawned.entities)
            .collect()
    }
}

pub(crate) fn sync(
    state: Res<WaterState>,
    mut white: ResMut<TerrainWhiteWater>,
    mut stats: ResMut<WaterStats>,
    mut spawner: Spawner<WaterBrick>,
) {
    let wanted = held(&state);
    let white = white.bypass_change_detection();
    let mut stale = Vec::new();
    white.spawned.retain_mut(|spawned| {
        let keep = wanted.iter().any(|tile| {
            tile.terrain == spawned.terrain
                && tile.key == spawned.key
                && tile.record == spawned.record
                && tile.origin == spawned.origin
        });
        if !keep {
            stale.append(&mut spawned.entities);
        }
        keep
    });
    for entity in stale {
        spawner.despawn(entity);
    }
    for tile in wanted {
        let known = white
            .spawned
            .iter()
            .any(|spawned| spawned.terrain == tile.terrain && spawned.key == tile.key);
        if known {
            continue;
        }
        let mut entities = Vec::new();
        for (number, site) in sites(&tile.water, &tile.falls).iter().enumerate() {
            for (transform, emitter) in emitters(site, tile.origin, seed_of(tile.key, number)) {
                entities.push(spawner.spawn((transform, emitter)));
            }
        }
        white.spawned.push(Spawned {
            terrain: tile.terrain,
            origin: tile.origin,
            key: tile.key,
            record: tile.record,
            entities,
        });
    }
    let count: u32 = white
        .spawned
        .iter()
        .map(|spawned| spawned.entities.len() as u32)
        .sum();
    if stats.white_water != count {
        stats.white_water = count;
    }
}
