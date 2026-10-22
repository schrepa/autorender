//! White water: where terrains' settled water loses energy fastest, at
//! their rivers' rapids, steps and ledges, the engine spawns spray thrown up
//! and along the flow, with mist rising over the strongest, and despawns
//! them as that water goes. It reads the records of settled water terrains
//! hold whole, those of the level-0 tiles around the camera (`settled`):
//! a record's sites are its cells breaking hardest, the strongest within
//! `SITE_APART` meters of each other and clear of falls, which spray their
//! own (`waterfalls`), each spraying as much as the power its water loses
//! there throws (particles). A tile's emitters change only as its record
//! does. Water that does not draw spawns none.

use std::sync::Arc;

use bevy_ecs::prelude::*;
use glam::{Vec2, Vec3};

use super::Terrains;
use crate::math::{clamp, max};
use crate::terrain_data::hydraulics::GRAVITY;
use crate::terrain_data::{Fall, SteadyCell, SteadyWater, TileKey};
use crate::{ParticleBlend, ParticleEmitter, RecordKey, Transform, Water};

/// The watts a square meter of bed water loses from which it throws spray:
/// about what half a square meter a second loses running down a step
/// pool's four percent (ρ·g·q·S).
const SPRAY_ENERGY: f32 = 200.0;
/// The share of the time water leaves its bed from which it throws spray
/// off a ledge, however little it loses.
const SPRAY_LIP: f32 = 0.2;
/// Meters apart white water's sites stand at least: about as far apart as
/// a steep stream's steps, one to four channel widths (Montgomery and
/// Buffington 1997); and the most a tile's water sprays from, the strongest
/// first, which bounds the emitters a tile spawns.
const SITE_APART: f32 = 8.0;
const SITES_MOST: usize = 32;
/// Watts of breaking water each droplet of spray a second stands for, and
/// the fewest and most a site throws a second.
const SPRAY_WATTS: f32 = 400.0;
const SPRAY_RATE: [f32; 2] = [8.0, 200.0];
/// The share of its water's speed spray is thrown at, as falls' splashes
/// are at a share of the speed they land at, and the slowest and fastest
/// it is thrown (meters a second); and the meters it rises for each it
/// flies along the flow.
const SPRAY_THROW: f32 = 0.8;
const SPRAY_SPEED: [f32; 2] = [1.5, 5.0];
const SPRAY_RISE: f32 = 1.5;
/// The watts a site's water loses from which mist rises over it, the watts
/// each puff a second stands for, and the fewest and most it puffs.
const MIST_POWER: f32 = 4.0e4;
const MIST_WATTS: f32 = 2.0e4;
const MIST_RATE: [f32; 2] = [1.0, 6.0];

/// What white water spawned over the settled water terrains hold whole.
#[derive(Resource, Default)]
pub(crate) struct TerrainWhiteWater {
    spawned: Vec<Spawned>,
}

/// The emitters spawned over a level-0 tile's settled water: its terrain,
/// where the terrain stood, the tile and its record's key, and the
/// emitters.
struct Spawned {
    terrain: Entity,
    origin: Vec3,
    key: TileKey,
    record: RecordKey,
    entities: Vec<Entity>,
}

/// A level-0 tile's settled water a terrain holds whole: the terrain, its
/// origin and its falls, the tile, its record's key and the record.
struct Held {
    terrain: Entity,
    origin: Vec3,
    falls: Arc<[Fall]>,
    key: TileKey,
    record: RecordKey,
    water: Arc<SteadyWater>,
}

/// A cell breaking hard enough to spray: how many times the least it
/// breaks by, its index among its tile's cells, its middle (terrain-local x
/// and z), the surface its water stands at, the velocity it runs at (meters
/// a second along x and z) and the watts it loses.
struct Breaking {
    strength: f32,
    index: u32,
    middle: Vec2,
    surface: f32,
    flow: Vec2,
    power: f32,
}

/// Where white water sprays from: the middle of its strongest cell
/// (terrain-local, at its water's surface), the velocity its water runs at
/// there, the watts the water breaking around it loses, and the meters
/// about its middle that water spans, at most a quarter of the least
/// spacing of sites.
struct Site {
    at: Vec3,
    flow: Vec2,
    power: f32,
    reach: f32,
}

/// The settled water every terrain holds whole, terrain by terrain and its
/// tiles in key order.
fn held(terrains: &Terrains) -> Vec<Held> {
    let mut found = Vec::new();
    for state in terrains.entries.iter().filter(|state| state.around.is_none()) {
        for (key, (water, record)) in state.settled.held.iter() {
            found.push(Held {
                terrain: state.entity,
                origin: state.terrain.origin,
                falls: state.falls.clone(),
                key,
                record: *record,
                water: Arc::clone(water),
            });
        }
    }
    found
}

/// Where white water sprays over `water`, a level-0 tile's settled water,
/// clear of `falls` (terrain-local): its cells losing SPRAY_ENERGY or more
/// or leaving their bed SPRAY_LIP of the time or more, the hardest breaking
/// first (in their order where alike), each a site of its own where none
/// stands within SITE_APART, fewer than SITES_MOST do and no fall's path
/// passes that near, else adding the watts it loses to the first site
/// standing that near.
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
            let energy = if energy.is_finite() { max(energy, 0.0) } else { 0.0 };
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
    // Each site's strongest cell, the watts its cells lose and its cells.
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

/// White water's emitters at `site`, its terrain at `origin`, seeded by
/// `seed`: droplets thrown up and along the flow at a share of its water's
/// speed, falling back to the water as they fly, as many as the power its
/// water loses throws; and over a site losing MIST_POWER or more, mist
/// rising, as much as that power stirs, its puffs as big as the breaking
/// water is wide.
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
    // A droplet lands back on the water when gravity has turned its rise
    // round and brought it down again.
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

/// The seed of site `number` of tile `key`'s white water: the same wherever
/// and whenever its record is held.
fn seed_of(key: TileKey, number: usize) -> u32 {
    let tile =
        u32::from(key.x).wrapping_mul(0x9e37_79b9) ^ u32::from(key.z).wrapping_mul(0x85eb_ca6b);
    tile.wrapping_add((number as u32).wrapping_mul(0xc2b2_ae35))
}

/// Spawns and despawns white water's emitters to match the settled water
/// terrains hold whole, where water draws: a tile's once its record is
/// held, anew when another record replaces it or its terrain moves, and
/// none once it goes, in the order terrains hold them. Frames draw them
/// within the particle budget as every emitter, the nearest first. The
/// terrains' stats count them.
pub(crate) fn sync(world: &mut World) {
    let drawn = world
        .get_resource::<Water>()
        .is_some_and(|water| water.enabled);
    let wanted = match world.get_resource::<Terrains>() {
        Some(terrains) if drawn => held(terrains),
        _ => Vec::new(),
    };
    let mut white = world
        .remove_resource::<TerrainWhiteWater>()
        .unwrap_or_default();
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
        if let Ok(entity) = world.get_entity_mut(entity) {
            entity.despawn();
        }
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
                entities.push(world.spawn((transform, emitter)).id());
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
    if let Some(mut terrains) = world.get_resource_mut::<Terrains>()
        && terrains.stats.white_water != count
    {
        terrains.stats.white_water = count;
    }
    world.insert_resource(white);
}
