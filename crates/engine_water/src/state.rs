use std::collections::VecDeque;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{Cut, Readback, Spawner, Ticket};
use engine_heightfield::{Fall, HydrologyBake, Pyramid, TileKey};
use engine_places::ground::{TerrainTiles, TileFace};
use engine_places::water::{Hydrology, WaterBody, WaterQuery, WaterSample, Waterline};
use glam::{Vec2, Vec3};

use crate::gpu::{
    PARTICLE_FOAM_BORN, PARTICLE_FOAM_DROPPED, PARTICLE_FOAM_LIVE, PARTICLE_WAVES_BORN,
    PARTICLE_WAVES_DROPPED, PARTICLE_WAVES_LIVE,
};
use crate::settled::Settled;
use crate::{RiverFlow, TileWater, Water, WaterBrick, WaterStats, WaterWindow};

#[derive(Component)]
pub(crate) struct RiverBody;

#[derive(Component)]
pub(crate) struct FallSet;

pub(crate) struct HeldWater {
    pub(crate) face: TileFace,
    pub(crate) water: Arc<TileWater>,
    pub(crate) rewatered: bool,
}

pub(crate) struct WaterTerrain {
    pub(crate) entity: Entity,
    pub(crate) origin: Vec3,
    pub(crate) pyramid: Pyramid,
    pub(crate) cutout: Option<(Vec2, Vec2)>,
    pub(crate) bake: Arc<HydrologyBake>,
    pub(crate) falls: Arc<[Fall]>,
    pub(crate) flow: Option<RiverFlow>,
    pub(crate) window: Option<Arc<WaterWindow>>,
    pub(crate) settled: Settled,
    pub(crate) faces: Arc<[TileFace]>,
    pub(crate) held: Vec<HeldWater>,
}

#[derive(Resource, Default)]
pub(crate) struct WaterState {
    pub(crate) terrains: Vec<WaterTerrain>,
    pub(crate) watered: Vec<Entity>,
    pub(crate) rebuilt: bool,
    pub(crate) kept: Vec<Ticket>,
}

impl WaterTerrain {
    pub(crate) fn rewatered(&self, key: TileKey) -> bool {
        self.held
            .iter()
            .any(|held| held.face.key == key && held.rewatered)
    }
}

impl WaterState {
    pub(crate) fn clear(&mut self) {
        self.terrains.clear();
        self.watered.clear();
        self.rebuilt = true;
    }

    fn follow(&mut self, tiles: &TerrainTiles, hydrology: &Hydrology) {
        let mut before = std::mem::take(&mut self.terrains);
        for entry in tiles.terrains.iter().filter(|entry| !entry.surroundings) {
            let Some((_, bake)) = hydrology
                .bakes
                .iter()
                .find(|(entity, _)| *entity == entry.terrain)
            else {
                continue;
            };
            let kept = before
                .iter()
                .position(|terrain| terrain.entity == entry.terrain)
                .map(|at| before.swap_remove(at));
            let mut terrain = kept.unwrap_or_else(|| WaterTerrain {
                entity: entry.terrain,
                origin: entry.origin,
                pyramid: entry.pyramid,
                cutout: entry.cutout,
                bake: Arc::clone(bake),
                falls: Arc::clone(&bake.falls),
                flow: None,
                window: None,
                settled: Settled::default(),
                faces: Arc::clone(&entry.tiles),
                held: Vec::new(),
            });
            if !Arc::ptr_eq(&terrain.bake, bake) {
                terrain.bake = Arc::clone(bake);
                terrain.falls = Arc::clone(&bake.falls);
                terrain.flow = None;
            }
            terrain.origin = entry.origin;
            terrain.pyramid = entry.pyramid;
            terrain.cutout = entry.cutout;
            terrain.faces = Arc::clone(&entry.tiles);
            terrain.held.retain(|held| {
                entry
                    .tiles
                    .iter()
                    .any(|face| face.key == held.face.key && face.stamp == held.face.stamp)
            });
            self.terrains.push(terrain);
        }
        self.watered.clear();
        self.watered
            .extend(self.terrains.iter().map(|terrain| terrain.entity));
        self.rebuilt = true;
    }

    fn surface(&self) -> WaterSurface {
        WaterSurface {
            terrains: self
                .terrains
                .iter()
                .map(|terrain| SurfaceTerrain {
                    origin: terrain.origin,
                    pyramid: terrain.pyramid,
                    cutout: terrain.cutout,
                    falls: Arc::clone(&terrain.falls),
                    faces: Arc::clone(&terrain.faces),
                    held: terrain
                        .held
                        .iter()
                        .map(|held| (held.face.key, Arc::clone(&held.water)))
                        .collect(),
                    rivers: (!terrain.bake.rivers.is_empty())
                        .then(|| terrain.bake.rivers.bounds()),
                    window: terrain.window.as_deref().map(WaterWindow::square),
                })
                .collect(),
        }
    }
}

pub(crate) struct SurfaceTerrain {
    pub(crate) origin: Vec3,
    pub(crate) pyramid: Pyramid,
    pub(crate) cutout: Option<(Vec2, Vec2)>,
    pub(crate) falls: Arc<[Fall]>,
    pub(crate) faces: Arc<[TileFace]>,
    pub(crate) held: Vec<(TileKey, Arc<TileWater>)>,
    rivers: Option<(Vec2, Vec2)>,
    window: Option<(Vec2, Vec2)>,
}

pub(crate) struct WaterSurface {
    pub(crate) terrains: Vec<SurfaceTerrain>,
}

impl WaterQuery for WaterSurface {
    fn water(&self, point: Vec3) -> Option<WaterSample> {
        crate::water_source::water_at(&self.terrains, point)
    }

    fn reaches(&self) -> Vec<(Vec3, Vec3)> {
        let rivers = self.terrains.iter().filter_map(|terrain| {
            let (low, high) = terrain.rivers?;
            Some((
                terrain.origin + Vec3::new(low.x, -1.0e5, low.y),
                terrain.origin + Vec3::new(high.x, 1.0e5, high.y),
            ))
        });
        let live = self.terrains.iter().filter_map(|terrain| {
            let (low, high) = terrain.window?;
            Some((
                Vec3::new(low.x, -1.0e5, low.y),
                Vec3::new(high.x, 1.0e5, high.y),
            ))
        });
        rivers.chain(live).collect()
    }
}

pub(crate) const MEANS: u32 = 0;
pub(crate) const GROUND: u32 = 1;
pub(crate) const TALLIES: u32 = 2;

#[derive(Resource, Default)]
pub struct SettleFeedback {
    pub(crate) asked: VecDeque<Vec<u8>>,
    pub(crate) means: VecDeque<(u64, Vec<u8>)>,
    pub(crate) ground: VecDeque<(u64, Vec<u8>)>,
}

impl SettleFeedback {
    pub(crate) fn joined(&mut self) -> Option<(Vec<u8>, Vec<u8>)> {
        let (means, ground) = (self.means.front()?.0, self.ground.front()?.0);
        if means != ground {
            return None;
        }
        let (_, means) = self.means.pop_front()?;
        let (_, ground) = self.ground.pop_front()?;
        let input = self.asked.pop_front()?;
        let mut bytes = means;
        bytes.extend(ground);
        Some((input, bytes))
    }
}

impl Readback for SettleFeedback {
    fn arrived(&mut self, frame: u64, tag: u32, bytes: &[u8]) {
        if tag == MEANS {
            self.means.push_back((frame, bytes.to_vec()));
        } else if tag == GROUND {
            self.ground.push_back((frame, bytes.to_vec()));
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct ParticleTallies {
    pub(crate) counts: Vec<u32>,
    pub(crate) frame: u64,
}

impl Readback for ParticleTallies {
    fn arrived(&mut self, frame: u64, _: u32, bytes: &[u8]) {
        self.frame = frame;
        self.counts.clear();
        self.counts
            .extend(bytes.chunks_exact(4).map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]])));
    }
}

pub(crate) fn rewater(
    tiles: Res<TerrainTiles>,
    hydrology: Res<Hydrology>,
    mut state: ResMut<WaterState>,
) {
    if !(tiles.is_changed() || hydrology.is_changed()) {
        return;
    }
    state.follow(&tiles, &hydrology);
}

pub(crate) fn publish_waterline(
    water: Res<Water>,
    state: Res<WaterState>,
    mut line: ResMut<Waterline>,
) {
    let same = line.subdivision_error == water.subdivision_error
        && line.subdivision_leaves == water.subdivision_leaves
        && line.watered == state.watered;
    if same {
        return;
    }
    let line = &mut *line;
    line.subdivision_error = water.subdivision_error;
    line.subdivision_leaves = water.subdivision_leaves;
    line.watered.clear();
    line.watered.extend_from_slice(&state.watered);
}

pub(crate) fn publish_bodies(
    mut state: ResMut<WaterState>,
    rivers: Query<Entity, With<RiverBody>>,
    mut spawner: Spawner<WaterBrick>,
) {
    if !state.rebuilt {
        return;
    }
    state.rebuilt = false;
    let body = WaterBody {
        precedence: 0,
        shape: Arc::new(state.surface()),
    };
    match rivers.single() {
        Ok(river) => spawner.insert(river, body),
        Err(_) => {
            spawner.spawn((RiverBody, body));
        }
    }
}

pub(crate) fn water_stats(
    water: Res<Water>,
    state: Res<WaterState>,
    tallies: Res<ParticleTallies>,
    mut stats: ResMut<WaterStats>,
) {
    let settled = state
        .terrains
        .iter()
        .map(|terrain| terrain.settled.counts())
        .fold([0_u32; 3], |sum, counts| {
            [
                sum[0].saturating_add(counts[0]),
                sum[1].saturating_add(counts[1]),
                sum[2].saturating_add(counts[2]),
            ]
        });
    let tally = |word: u32| tallies.counts.get(word as usize).copied().unwrap_or(0);
    let next = WaterStats {
        looks_left: settled[2],
        settled_records: settled[0],
        settling_records: settled[1],
        falls: state
            .terrains
            .iter()
            .map(|terrain| terrain.falls.len() as u32)
            .sum(),
        foam_particles: tally(PARTICLE_FOAM_LIVE),
        wave_particles: tally(PARTICLE_WAVES_LIVE),
        particles_born: tally(PARTICLE_FOAM_BORN) + tally(PARTICLE_WAVES_BORN),
        dropped_particles: tally(PARTICLE_FOAM_DROPPED) + tally(PARTICLE_WAVES_DROPPED),
        leaf_cut: Cut::of(water.subdivision_leaves, water.subdivision_leaves),
        ..stats.clone()
    };
    stats.set_if_neq(next);
}
