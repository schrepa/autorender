use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{
    FrameTime, JobFailures, Jobs, Records, Spawner, Ticket, VIEW_JOBS, View, WorkClock, job_priority,
};
use engine_heightfield::tile::{Solid, TileJob};
use engine_heightfield::{
    ErrorPyramid, Fall, HydrologyBake, Layers, Pyramid, Rivers, Stamp, TileKey, reaches,
};
use engine_places::ground::{GroundWork, TerrainTiles, TileFace};
use engine_places::water::{WaterBody, WaterQuery, WaterSample, WaterTileErrors, Waterline};
use glam::{Vec2, Vec3};

use crate::jobs::{TILE_WATER, tile_water_output};
use crate::settled::Settled;
use crate::water_blocks::Blocks;
use crate::water_particles::Particles;
use crate::water_sim::LiveSim;
use crate::work::Work;
use crate::{RiverFlow, TileWater, Water, WaterBrick, WaterMesh, WaterWindow};

pub(crate) const RETRY_AFTER: u64 = 120;

pub(crate) type Watered = Option<Arc<Rivers>>;

#[derive(Component)]
pub(crate) struct RiverBody;

pub(crate) enum Base {
    Waiting,
    Asking(Ticket),
    Failed(u64),
    Made,
}

pub(crate) struct HeldWater {
    pub(crate) face: TileFace,
    pub(crate) base: Base,
    pub(crate) water: Option<Arc<TileWater>>,
    pub(crate) mesh: Option<Arc<WaterMesh>>,
    pub(crate) errors: Option<Arc<[u16]>>,
    pub(crate) watered: Watered,
}

impl HeldWater {
    fn new(face: TileFace) -> Self {
        Self {
            face,
            base: Base::Waiting,
            water: None,
            mesh: None,
            errors: None,
            watered: None,
        }
    }

    pub(crate) fn resident(&self) -> bool {
        matches!(self.base, Base::Made)
    }

    pub(crate) fn hold(&mut self, water: Option<TileWater>) {
        let mesh = water.as_ref().map(|water| water.mesh(&self.face.heights));
        self.errors = mesh.as_ref().map(|mesh| {
            let holding: Vec<bool> = (0..mesh.levels().len()).map(|at| mesh.holds(at)).collect();
            ErrorPyramid::over(mesh.levels(), Some(&holding[..])).shared_halves()
        });
        self.mesh = mesh.map(Arc::new);
        self.water = water.map(Arc::new);
    }
}

pub(crate) struct WaterTerrain {
    pub(crate) entity: Entity,
    pub(crate) origin: Vec3,
    pub(crate) pyramid: Pyramid,
    pub(crate) cutout: Option<(Vec2, Vec2)>,
    pub(crate) surroundings: bool,
    pub(crate) recipe: Option<[u8; 32]>,
    pub(crate) baked: Arc<Layers>,
    pub(crate) edits: Arc<[Stamp]>,
    pub(crate) solids: Arc<[Option<Solid>]>,
    pub(crate) relief: u128,
    pub(crate) falls: Arc<[Fall]>,
    pub(crate) flow: Option<RiverFlow>,
    pub(crate) window: Option<Arc<WaterWindow>>,
    pub(crate) settled: Settled,
    pub(crate) faces: Arc<[TileFace]>,
    pub(crate) held: BTreeMap<TileKey, HeldWater>,
    pub(crate) left: BTreeMap<TileKey, HeldWater>,
    pub(crate) revision: u64,
}

impl WaterTerrain {
    pub(crate) fn hydrology(&self) -> Option<&HydrologyBake> {
        self.baked.hydrology()
    }

    pub(crate) fn rivers(&self) -> Option<&Arc<Rivers>> {
        self.hydrology().map(|hydrology| &hydrology.rivers)
    }

    pub(crate) fn resident(&self, key: &TileKey) -> Option<&HeldWater> {
        self.held.get(key).filter(|held| held.resident())
    }

    pub(crate) fn resident_keys(&self, wanted: impl Fn(&TileKey) -> bool) -> Vec<TileKey> {
        self.held
            .iter()
            .filter(|(key, held)| held.resident() && wanted(key))
            .map(|(&key, _)| key)
            .collect()
    }

    pub(crate) fn rewatered(&self, key: TileKey) -> bool {
        self.resident(&key)
            .is_some_and(|held| held.watered.is_some())
    }

    pub(crate) fn tile_job(&self, key: TileKey) -> Option<TileJob> {
        let pyramid = self.pyramid;
        Some(TileJob {
            recipe: self.recipe?,
            key,
            stamps: self
                .edits
                .iter()
                .copied()
                .filter(|stamp| reaches(stamp, &pyramid, key))
                .collect(),
        })
    }

    fn drop_asks(&mut self, jobs: &mut Jobs) {
        for (_, held) in std::mem::take(&mut self.held) {
            if let Base::Asking(ticket) = held.base {
                jobs.cancel(ticket);
            }
        }
        self.left.clear();
        self.settled.cancel_all(jobs);
    }

    fn take_faces(&mut self, jobs: &mut Jobs, kept: &[(TileKey, engine_core::Stamp)]) {
        let faces = Arc::clone(&self.faces);
        let mut changed = false;
        let mut following = BTreeMap::new();
        for face in faces.iter() {
            let held = match self.held.remove(&face.key) {
                Some(mut held) if held.face.stamp == face.stamp => {
                    held.face = face.clone();
                    held
                }
                Some(held) => {
                    if let Base::Asking(ticket) = held.base {
                        jobs.cancel(ticket);
                    }
                    changed = true;
                    HeldWater::new(face.clone())
                }
                None => {
                    changed = true;
                    match self.left.remove(&face.key) {
                        Some(mut held) if held.face.stamp == face.stamp => {
                            held.face = face.clone();
                            held
                        }
                        Some(_) | None => HeldWater::new(face.clone()),
                    }
                }
            };
            following.insert(face.key, held);
        }
        for (key, held) in std::mem::replace(&mut self.held, following) {
            changed = true;
            if let Base::Asking(ticket) = held.base {
                jobs.cancel(ticket);
            } else if held.resident() && kept.contains(&(key, held.face.stamp)) {
                self.left.insert(key, held);
            }
        }
        if changed {
            self.revision += 1;
        }
    }

    fn follow_faces(
        &mut self,
        jobs: &mut Jobs,
        kept: &[(TileKey, engine_core::Stamp)],
        (eye, frame): (Vec2, u64),
    ) {
        self.left
            .retain(|&key, held| kept.contains(&(key, held.face.stamp)));
        let pyramid = self.pyramid;
        let size = pyramid.tile_size(0);
        let mut asking: Vec<(u32, TileKey)> = self
            .held
            .iter()
            .filter(|(_, held)| match held.base {
                Base::Waiting => true,
                Base::Failed(at) => frame >= at + RETRY_AFTER,
                Base::Asking(_) | Base::Made => false,
            })
            .map(|(&key, _)| {
                let span = pyramid.tile_size(u32::from(key.level));
                let middle = pyramid.tile_origin(key) + Vec2::splat(0.5 * span);
                ((middle.distance(eye) / size) as u32, key)
            })
            .collect();
        asking.sort_unstable();
        for (place, key) in asking {
            let Some(job) = self.tile_job(key) else {
                return;
            };
            let ticket = jobs.fetch(
                &TILE_WATER,
                &engine_codec::encode(&job),
                job_priority(VIEW_JOBS, place),
            );
            if let Some(held) = self.held.get_mut(&key) {
                held.base = Base::Asking(ticket);
            }
        }
    }

    fn take_bases(
        &mut self,
        (records, jobs, work): (&Records, &mut Jobs, &Work),
        failures: &mut JobFailures,
        frame: u64,
    ) {
        let mut taken = false;
        let asked: Vec<TileKey> = self
            .held
            .iter()
            .filter(|(_, held)| matches!(held.base, Base::Asking(_)))
            .map(|(&key, _)| key)
            .collect();
        for key in asked {
            if taken && work.expired() {
                break;
            }
            let Some(held) = self.held.get_mut(&key) else {
                continue;
            };
            let Base::Asking(ticket) = &held.base else {
                continue;
            };
            let Some(output) = records.get(ticket) else {
                continue;
            };
            taken = true;
            let Base::Asking(ticket) = std::mem::replace(&mut held.base, Base::Made) else {
                continue;
            };
            jobs.cancel(ticket);
            let made = output.map_err(|error| error.to_string()).and_then(|bytes| {
                tile_water_output(&bytes).map_err(|error| {
                    format!("{}: its output does not decode: {error}", TILE_WATER.name())
                })
            });
            match made {
                Ok(water) => {
                    held.hold(water);
                    held.watered = None;
                    self.revision += 1;
                }
                Err(error) => {
                    held.base = Base::Failed(frame);
                    failures.record(frame, error);
                }
            }
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct WaterState {
    pub(crate) terrains: Vec<WaterTerrain>,
    pub(crate) watered: Vec<Entity>,
    pub(crate) published: Vec<(Entity, u64, usize)>,
    pub(crate) failures: JobFailures,
    pub(crate) settling: SettleQueue,
    pub(crate) blocks: Blocks,
    pub(crate) live: LiveSim,
    pub(crate) particles: Particles,
    pub(crate) rewatered_tiles: u32,
    pub(crate) stale_water_tiles: u32,
    pub(crate) settled_records: u32,
    pub(crate) settling_records: u32,
    pub(crate) settles_left: u64,
    pub(crate) looks_left: u64,
    pub(crate) work_end: f64,
    dry: Option<Arc<[u16]>>,
}

#[derive(Default)]
pub(crate) struct SettleQueue {
    pub(crate) asked: Vec<crate::water_settle::SettleAsk>,
    pub(crate) settled: Vec<(Vec<u8>, Result<Vec<u8>, String>)>,
}

impl SettleQueue {
    pub(crate) fn take(&mut self, input: &[u8]) -> Option<Result<Vec<u8>, String>> {
        let at = self
            .settled
            .iter()
            .position(|(held, _)| held[..] == *input)?;
        Some(self.settled.swap_remove(at).1)
    }
}

impl WaterState {
    pub(crate) fn clear(&mut self, jobs: &mut Jobs) {
        for mut terrain in std::mem::take(&mut self.terrains) {
            terrain.drop_asks(jobs);
        }
        self.watered.clear();
        self.published.clear();
        self.settling = SettleQueue::default();
        self.blocks = Blocks::default();
        self.live = LiveSim::default();
        self.particles = Particles::default();
        self.rewatered_tiles = 0;
        self.stale_water_tiles = 0;
        self.settled_records = 0;
        self.settling_records = 0;
        self.settles_left = 0;
        self.looks_left = 0;
    }

    fn surface(&self) -> WaterSurface {
        let waters = self.terrains.iter().filter(|terrain| !terrain.surroundings);
        WaterSurface {
            terrains: waters
                .clone()
                .map(|terrain| SurfaceTerrain {
                    origin: terrain.origin,
                    pyramid: terrain.pyramid,
                    falls: Arc::clone(&terrain.falls),
                    held: terrain
                        .held
                        .values()
                        .filter(|held| held.resident())
                        .map(|held| (held.face.clone(), held.water.clone()))
                        .collect(),
                })
                .collect(),
            rivers: waters
                .filter_map(|terrain| {
                    let rivers = terrain.rivers()?;
                    (!rivers.is_empty()).then(|| {
                        let (low, high) = rivers.bounds();
                        (
                            terrain.origin + Vec3::new(low.x, -1.0e5, low.y),
                            terrain.origin + Vec3::new(high.x, 1.0e5, high.y),
                        )
                    })
                })
                .collect(),
            live: self
                .terrains
                .iter()
                .filter_map(|terrain| terrain.window.as_deref().map(WaterWindow::square))
                .map(|(low, high)| {
                    (
                        Vec3::new(low.x, -1.0e5, low.y),
                        Vec3::new(high.x, 1.0e5, high.y),
                    )
                })
                .collect(),
        }
    }

    fn signature(&self) -> Vec<(Entity, u64, usize)> {
        self.terrains
            .iter()
            .map(|terrain| {
                let window = terrain
                    .window
                    .as_ref()
                    .map_or(0, |window| Arc::as_ptr(window) as usize);
                let falls = Arc::as_ptr(&terrain.falls).cast::<u8>() as usize;
                (
                    terrain.entity,
                    terrain.revision,
                    window ^ falls.rotate_left(17),
                )
            })
            .collect()
    }
}

pub(crate) struct SurfaceTerrain {
    pub(crate) origin: Vec3,
    pub(crate) pyramid: Pyramid,
    pub(crate) falls: Arc<[Fall]>,
    pub(crate) held: Vec<(TileFace, Option<Arc<TileWater>>)>,
}

pub(crate) struct WaterSurface {
    pub(crate) terrains: Vec<SurfaceTerrain>,
    rivers: Vec<(Vec3, Vec3)>,
    live: Vec<(Vec3, Vec3)>,
}

impl WaterQuery for WaterSurface {
    fn water(&self, point: Vec3) -> Option<WaterSample> {
        crate::water_source::water_at(&self.terrains, point)
    }

    fn reaches(&self) -> Vec<(Vec3, Vec3)> {
        self.rivers.iter().chain(&self.live).copied().collect()
    }
}

pub(crate) fn follow(
    tiles: Res<TerrainTiles>,
    records: Res<Records>,
    view: Res<View>,
    time: Res<FrameTime>,
    (clock, ground): (Res<WorkClock>, Res<GroundWork>),
    mut jobs: Jobs,
    mut state: ResMut<WaterState>,
) {
    let eye = view.detail_eye();
    let frame = time.frame;
    let state = &mut *state;
    let work = Work::of(&clock, &ground);
    state.work_end = work.end();
    let mut before = std::mem::take(&mut state.terrains);
    for entry in tiles.terrains.iter() {
        let Some(baked) = entry
            .baked
            .clone()
            .filter(|baked| baked.hydrology().is_some())
        else {
            continue;
        };
        let kept = before
            .iter()
            .position(|terrain| terrain.entity == entry.terrain)
            .map(|at| before.swap_remove(at));
        let fresh = kept.is_none();
        let mut terrain = kept.unwrap_or_else(|| WaterTerrain {
            entity: entry.terrain,
            origin: entry.origin,
            pyramid: entry.pyramid,
            cutout: entry.cutout,
            surroundings: entry.surroundings,
            recipe: entry.recipe,
            baked: Arc::clone(&baked),
            edits: Arc::clone(&entry.edits),
            solids: Arc::clone(&entry.solids),
            relief: entry.relief,
            falls: Arc::from(Vec::new()),
            flow: None,
            window: None,
            settled: Settled::default(),
            faces: Arc::from(Vec::new()),
            held: BTreeMap::new(),
            left: BTreeMap::new(),
            revision: 1,
        });
        if (fresh || !Arc::ptr_eq(&terrain.baked, &baked)) && baked.unbaked().is_none() {
            terrain.falls = baked
                .hydrology()
                .map_or_else(|| Arc::from(Vec::new()), |water| water.falls.clone());
        }
        terrain.baked = baked;
        terrain.origin = entry.origin;
        terrain.pyramid = entry.pyramid;
        terrain.cutout = entry.cutout;
        terrain.surroundings = entry.surroundings;
        terrain.recipe = entry.recipe;
        terrain.edits = Arc::clone(&entry.edits);
        terrain.solids = Arc::clone(&entry.solids);
        terrain.relief = entry.relief;
        let local = Vec2::new(eye.x - entry.origin.x, eye.z - entry.origin.z);
        if fresh || !Arc::ptr_eq(&terrain.faces, &entry.tiles) {
            terrain.faces = Arc::clone(&entry.tiles);
            terrain.take_faces(&mut jobs, &entry.kept);
        }
        terrain.follow_faces(&mut jobs, &entry.kept, (local, frame));
        terrain.take_bases((&records, &mut jobs, &work), &mut state.failures, frame);
        state.terrains.push(terrain);
    }
    for mut gone in before {
        gone.drop_asks(&mut jobs);
    }
}

pub(crate) fn publish_waterline(
    water: Res<Water>,
    mut state: ResMut<WaterState>,
    mut line: ResMut<Waterline>,
) {
    let state = &mut *state;
    let dry = state
        .dry
        .get_or_insert_with(|| ErrorPyramid::dry().shared_halves())
        .clone();
    let errors: Vec<WaterTileErrors> = state
        .terrains
        .iter()
        .flat_map(|terrain| {
            terrain
                .held
                .values()
                .filter(|held| held.resident())
                .map(|held| WaterTileErrors {
                    terrain: terrain.entity,
                    key: held.face.key,
                    stamp: held.face.stamp,
                    errors: held.errors.clone().unwrap_or_else(|| Arc::clone(&dry)),
                })
        })
        .collect();
    let same = line.subdivision_error == water.subdivision_error
        && line.subdivision_leaves == water.subdivision_leaves
        && line.watered == state.watered
        && line.errors.len() == errors.len()
        && line.errors.iter().zip(&errors).all(|(held, found)| {
            held.terrain == found.terrain
                && held.key == found.key
                && held.stamp == found.stamp
                && Arc::ptr_eq(&held.errors, &found.errors)
        });
    if same {
        return;
    }
    let line = &mut *line;
    line.subdivision_error = water.subdivision_error;
    line.subdivision_leaves = water.subdivision_leaves;
    line.watered.clear();
    line.watered.extend_from_slice(&state.watered);
    line.errors = errors;
}

pub(crate) fn publish_bodies(
    mut state: ResMut<WaterState>,
    rivers: Query<Entity, With<RiverBody>>,
    mut spawner: Spawner<WaterBrick>,
) {
    let signature = state.signature();
    if signature == state.published && rivers.single().is_ok() {
        return;
    }
    state.published = signature;
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
