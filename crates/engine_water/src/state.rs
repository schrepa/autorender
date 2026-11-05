use std::collections::BTreeMap;
use std::sync::{Arc, PoisonError, RwLock};

use bevy_ecs::prelude::*;
use engine_core::{
    FrameTime, JobFailures, JobId, Jobs, RecordKey, Records, Spawner, Ticket, VIEW_JOBS, View,
    WorkClock, job_priority,
};
use engine_heightfield::tile::{Solid, TileJob};
use engine_heightfield::{
    ErrorPyramid, Fall, HydrologyBake, Layers, Pyramid, Rivers, Stamp, TileKey, reaches,
};
use engine_places::ground::{
    GroundRead, GroundWork, TerrainEntry, TerrainTiles, TileFace, TilesSince,
};
use engine_places::water::{WaterBody, WaterQuery, WaterSample, WaterTileErrors, Waterline};
use glam::{Vec2, Vec3};

use crate::drawn::{DrawAsk, DrawnWater, is_dry};
use crate::flow::Running;
use crate::jobs::TILE_WATER;
use crate::settled::Settled;
use crate::water_blocks::Blocks;
use crate::water_particles::Particles;
use crate::water_sim::LiveSim;
use crate::white_water::Spawned;
use crate::work::Work;
use crate::{RiverFlow, TileWater, Water, WaterBrick, WaterWindow};

pub(crate) const RETRY_AFTER: u64 = 120;

pub(crate) type Watered = Option<Arc<Running>>;

#[derive(Component)]
pub(crate) struct RiverBody;

pub(crate) enum Base {
    Waiting,
    Asking(Ticket),
    Drawing,
    Failed(u64),
    Made,
}

pub(crate) enum Drawing {
    Base,
    Given(RecordKey),
    Flood(Watered),
}

pub(crate) struct HeldWater {
    pub(crate) face: TileFace,
    pub(crate) base: Base,
    pub(crate) water: Option<Arc<TileWater>>,
    pub(crate) drawn: Option<Arc<DrawnWater>>,
    pub(crate) watered: Watered,
    pub(crate) given: Option<RecordKey>,
    pub(crate) drawing: Option<(JobId, Drawing)>,
}

impl HeldWater {
    fn new(face: TileFace) -> Self {
        Self {
            face,
            base: Base::Waiting,
            water: None,
            drawn: None,
            watered: None,
            given: None,
            drawing: None,
        }
    }

    pub(crate) fn resident(&self) -> bool {
        matches!(self.base, Base::Made)
    }

    pub(crate) fn hold(&mut self, made: Option<(TileWater, DrawnWater)>) {
        match made {
            Some((water, drawn)) => {
                self.water = Some(Arc::new(water));
                self.drawn = Some(Arc::new(drawn));
            }
            None => {
                self.water = None;
                self.drawn = None;
            }
        }
    }
}

pub(crate) struct WaterTerrain {
    pub(crate) entity: Entity,
    pub(crate) origin: Vec3,
    pub(crate) pyramid: Pyramid,
    pub(crate) surroundings: bool,
    pub(crate) recipe: Option<[u8; 32]>,
    pub(crate) baked: Arc<Layers>,
    pub(crate) edits: Arc<[Stamp]>,
    pub(crate) solids: Arc<[Option<Solid>]>,
    pub(crate) relief: u128,
    pub(crate) falls: Arc<[Fall]>,
    pub(crate) falls_fresh: bool,
    pub(crate) fall_entities: Vec<Entity>,
    pub(crate) white: BTreeMap<TileKey, Spawned>,
    pub(crate) flow: Option<RiverFlow>,
    pub(crate) flow_moved: bool,
    pub(crate) window: Option<WaterWindow>,
    pub(crate) window_moved: bool,
    pub(crate) window_changed: Vec<[Vec2; 2]>,
    pub(crate) settled: Settled,
    pub(crate) held: BTreeMap<TileKey, HeldWater>,
    pub(crate) left: BTreeMap<TileKey, HeldWater>,
    pub(crate) touched: Vec<TileKey>,
    pub(crate) touched_all: bool,
    cursor: u64,
    kept: Arc<[(TileKey, engine_core::Stamp)]>,
}

impl WaterTerrain {
    fn new(entry: &TerrainEntry, baked: &Arc<Layers>) -> Self {
        Self {
            entity: entry.terrain,
            origin: entry.origin,
            pyramid: entry.pyramid,
            surroundings: entry.surroundings,
            recipe: entry.recipe,
            baked: Arc::clone(baked),
            edits: Arc::clone(&entry.edits),
            solids: Arc::clone(&entry.solids),
            relief: entry.relief,
            falls: Arc::from(Vec::new()),
            falls_fresh: false,
            fall_entities: Vec::new(),
            white: BTreeMap::new(),
            flow: None,
            flow_moved: false,
            window: None,
            window_moved: false,
            window_changed: Vec::new(),
            settled: Settled::default(),
            held: BTreeMap::new(),
            left: BTreeMap::new(),
            touched: Vec::new(),
            touched_all: false,
            cursor: 0,
            kept: Arc::from(Vec::new()),
        }
    }

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

    pub(crate) fn touch(&mut self, key: TileKey) {
        self.touched.push(key);
        self.settled.stir(key.level == 0);
    }

    pub(crate) fn touch_all(&mut self) {
        self.touched_all = true;
        self.settled.stir(true);
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

    fn emitters(&mut self) -> impl Iterator<Item = Entity> + '_ {
        let white = std::mem::take(&mut self.white);
        self.fall_entities
            .drain(..)
            .chain(white.into_values().flat_map(|spawned| spawned.entities))
    }

    fn take_face(&mut self, jobs: &mut Jobs, key: TileKey, face: Option<&TileFace>) {
        let held = self.held.remove(&key);
        let Some(face) = face else {
            if let Some(mut held) = held {
                if let Base::Asking(ticket) = held.base {
                    jobs.cancel(ticket);
                } else if held.resident() && self.kept.contains(&(key, held.face.stamp)) {
                    held.drawing = None;
                    self.left.insert(key, held);
                }
            }
            self.touch(key);
            return;
        };
        let held = match held {
            Some(mut held) if held.face.stamp == face.stamp => {
                held.face = face.clone();
                held
            }
            Some(held) => {
                if let Base::Asking(ticket) = held.base {
                    jobs.cancel(ticket);
                }
                HeldWater::new(face.clone())
            }
            None => match self.left.remove(&key) {
                Some(mut held) if held.face.stamp == face.stamp => {
                    held.face = face.clone();
                    held
                }
                Some(_) | None => HeldWater::new(face.clone()),
            },
        };
        self.held.insert(key, held);
        self.touch(key);
    }

    fn take_faces(&mut self, jobs: &mut Jobs, entry: &TerrainEntry) {
        if !Arc::ptr_eq(&self.kept, &entry.kept) {
            self.kept = Arc::clone(&entry.kept);
            let kept = &self.kept;
            self.left
                .retain(|&key, held| kept.contains(&(key, held.face.stamp)));
        }
        let mut keys: Vec<TileKey> = match entry.changes.since(&mut self.cursor) {
            TilesSince::Same => return,
            TilesSince::Listed(changes) => changes.iter().map(|change| change.key).collect(),
            TilesSince::All => entry
                .tiles
                .iter()
                .map(|face| face.key)
                .chain(self.held.keys().copied())
                .collect(),
        };
        keys.sort_unstable();
        keys.dedup();
        for key in keys {
            let face = entry
                .tiles
                .binary_search_by_key(&key, |face| face.key)
                .ok()
                .map(|at| &entry.tiles[at]);
            self.take_face(jobs, key, face);
        }
    }

    fn follow_faces(&mut self, jobs: &mut Jobs, (eye, frame): (Vec2, u64)) {
        let pyramid = self.pyramid;
        let size = pyramid.tile_size(0);
        let mut asking: Vec<(u32, TileKey)> = self
            .held
            .iter()
            .filter(|(_, held)| match held.base {
                Base::Waiting => true,
                Base::Failed(at) => frame >= at + RETRY_AFTER,
                Base::Asking(_) | Base::Drawing | Base::Made => false,
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
        (failures, draws): (&mut JobFailures, &mut Vec<DrawAsk>),
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
            let Base::Asking(ticket) = std::mem::replace(&mut held.base, Base::Drawing) else {
                continue;
            };
            jobs.cancel(ticket);
            let made = output.map_err(|error| error.to_string()).and_then(|bytes| {
                let dry = is_dry(&bytes).map_err(|error| {
                    format!("{}: its output does not decode: {error}", TILE_WATER.name())
                })?;
                Ok((bytes, dry))
            });
            match made {
                Ok((_, true)) => {
                    held.base = Base::Made;
                    held.hold(None);
                    held.watered = None;
                    self.touch(key);
                }
                Ok((record, false)) => draws.push(DrawAsk {
                    terrain: self.entity,
                    key,
                    stamp: held.face.stamp,
                    record,
                    drawing: Drawing::Base,
                }),
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
    pub(crate) baking: Vec<(Vec3, Pyramid)>,
    pub(crate) watered: Vec<Entity>,
    pub(crate) surface: Arc<WaterSurface>,
    pub(crate) draw_asks: Vec<DrawAsk>,
    pub(crate) flights: Vec<(JobId, Entity, TileKey)>,
    pub(crate) gone: Vec<Entity>,
    pub(crate) dropped: bool,
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
}

#[derive(Default)]
pub(crate) struct SettleQueue {
    pub(crate) asked: Vec<crate::water_settle::SettleAsk>,
    pub(crate) settled: Vec<crate::water_settle::SettleMade>,
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
    pub(crate) fn take_emitters(&mut self) -> Vec<Entity> {
        let mut emitters = std::mem::take(&mut self.gone);
        for terrain in &mut self.terrains {
            emitters.extend(terrain.emitters());
        }
        emitters
    }

    pub(crate) fn clear(&mut self, jobs: &mut Jobs) {
        for mut terrain in std::mem::take(&mut self.terrains) {
            terrain.drop_asks(jobs);
        }
        self.baking.clear();
        self.watered.clear();
        self.surface = Arc::default();
        self.draw_asks.clear();
        self.dropped = false;
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
}

pub(crate) struct SurfaceTerrain {
    entity: Entity,
    pub(crate) origin: Vec3,
    pub(crate) pyramid: Pyramid,
    pub(crate) falls: Arc<[Fall]>,
    pub(crate) held: BTreeMap<TileKey, (TileFace, Option<Arc<TileWater>>)>,
}

impl SurfaceTerrain {
    fn of(terrain: &WaterTerrain) -> Self {
        Self {
            entity: terrain.entity,
            origin: terrain.origin,
            pyramid: terrain.pyramid,
            falls: Arc::clone(&terrain.falls),
            held: BTreeMap::new(),
        }
    }

    fn follow(&mut self, terrain: &WaterTerrain, whole: bool) {
        self.origin = terrain.origin;
        self.pyramid = terrain.pyramid;
        self.falls = Arc::clone(&terrain.falls);
        if whole || terrain.touched_all {
            self.held = terrain
                .held
                .iter()
                .filter(|(_, held)| held.resident())
                .map(|(&key, held)| (key, (held.face.clone(), held.water.clone())))
                .collect();
            return;
        }
        for key in &terrain.touched {
            match terrain.resident(key) {
                Some(held) => {
                    self.held
                        .insert(*key, (held.face.clone(), held.water.clone()));
                }
                None => {
                    self.held.remove(key);
                }
            }
        }
    }
}

#[derive(Default)]
struct Served {
    terrains: Vec<SurfaceTerrain>,
    baking: Vec<(Vec3, Pyramid)>,
    rivers: Vec<(Vec3, Vec3)>,
    live: Vec<(Vec3, Vec3)>,
}

#[derive(Default)]
pub(crate) struct WaterSurface(RwLock<Served>);

impl WaterSurface {
    fn follow(&self, terrains: &[WaterTerrain], baking: &[(Vec3, Pyramid)]) {
        let mut served = self.0.write().unwrap_or_else(PoisonError::into_inner);
        let served = &mut *served;
        served.baking.clear();
        served.baking.extend_from_slice(baking);
        let waters = terrains.iter().filter(|terrain| !terrain.surroundings);
        let mut before = std::mem::take(&mut served.terrains);
        for terrain in waters.clone() {
            let found = before.iter().position(|held| held.entity == terrain.entity);
            let mut held =
                found.map_or_else(|| SurfaceTerrain::of(terrain), |at| before.swap_remove(at));
            held.follow(terrain, found.is_none());
            served.terrains.push(held);
        }
        served.rivers.clear();
        served.rivers.extend(waters.filter_map(|terrain| {
            let rivers = terrain.rivers()?;
            (!rivers.is_empty()).then(|| {
                let (low, high) = rivers.bounds();
                (
                    terrain.origin + Vec3::new(low.x, -1.0e5, low.y),
                    terrain.origin + Vec3::new(high.x, 1.0e5, high.y),
                )
            })
        }));
        served.live.clear();
        served.live.extend(
            terrains
                .iter()
                .filter_map(|terrain| terrain.window.as_ref().map(WaterWindow::square))
                .map(|(low, high)| {
                    (
                        Vec3::new(low.x, -1.0e5, low.y),
                        Vec3::new(high.x, 1.0e5, high.y),
                    )
                }),
        );
    }
}

impl WaterQuery for WaterSurface {
    fn water(&self, point: Vec3) -> GroundRead<Option<WaterSample>> {
        let served = self.0.read().unwrap_or_else(PoisonError::into_inner);
        crate::water_source::water_at(&served.terrains, &served.baking, point)
    }

    fn reaches(&self) -> Vec<(Vec3, Vec3)> {
        let served = self.0.read().unwrap_or_else(PoisonError::into_inner);
        served.rivers.iter().chain(&served.live).copied().collect()
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
    state.dropped = false;
    let mut before = std::mem::take(&mut state.terrains);
    state.baking.clear();
    for entry in tiles.terrains.iter() {
        let Some(baked) = entry
            .baked
            .clone()
            .filter(|baked| baked.hydrology().is_some())
        else {
            if entry.baking {
                state.baking.push((entry.origin, entry.pyramid));
            }
            continue;
        };
        let kept = before
            .iter()
            .position(|terrain| terrain.entity == entry.terrain)
            .map(|at| before.swap_remove(at));
        let fresh = kept.is_none();
        let mut terrain = kept.unwrap_or_else(|| WaterTerrain::new(entry, &baked));
        terrain.touched.clear();
        terrain.touched_all = fresh;
        if terrain.origin != entry.origin {
            terrain.touch_all();
        }
        if (fresh || !Arc::ptr_eq(&terrain.baked, &baked)) && baked.unbaked().is_none() {
            let falls: Arc<[Fall]> = baked
                .hydrology()
                .map_or_else(|| Arc::from(Vec::new()), |water| water.falls.clone());
            terrain.falls_fresh |= !Arc::ptr_eq(&terrain.falls, &falls);
            terrain.falls = falls;
        }
        terrain.baked = baked;
        terrain.origin = entry.origin;
        terrain.pyramid = entry.pyramid;
        terrain.surroundings = entry.surroundings;
        terrain.recipe = entry.recipe;
        terrain.edits = Arc::clone(&entry.edits);
        terrain.solids = Arc::clone(&entry.solids);
        terrain.relief = entry.relief;
        let local = Vec2::new(eye.x - entry.origin.x, eye.z - entry.origin.z);
        terrain.take_faces(&mut jobs, entry);
        terrain.follow_faces(&mut jobs, (local, frame));
        terrain.take_bases(
            (&records, &mut jobs, &work),
            (&mut state.failures, &mut state.draw_asks),
            frame,
        );
        state.terrains.push(terrain);
    }
    for mut gone in before {
        state.dropped = true;
        gone.drop_asks(&mut jobs);
        state.gone.extend(gone.emitters());
    }
}

pub(crate) fn publish_waterline(
    water: Res<Water>,
    state: Res<WaterState>,
    mut line: ResMut<Waterline>,
) {
    let touched = state
        .terrains
        .iter()
        .any(|terrain| terrain.touched_all || !terrain.touched.is_empty());
    if !(touched || state.dropped || water.is_changed() || line.watered != state.watered) {
        return;
    }
    let dry = ErrorPyramid::dry().shared_halves();
    let errors: Vec<WaterTileErrors> = state
        .terrains
        .iter()
        .flat_map(|terrain| {
            let dry = &dry;
            terrain
                .held
                .values()
                .filter(|held| held.resident())
                .map(move |held| WaterTileErrors {
                    terrain: terrain.entity,
                    key: held.face.key,
                    stamp: held.face.stamp,
                    errors: held
                        .drawn
                        .as_ref()
                        .map_or_else(|| Arc::clone(dry), |drawn| Arc::clone(&drawn.errors)),
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
    state: Res<WaterState>,
    rivers: Query<Entity, With<RiverBody>>,
    mut spawner: Spawner<WaterBrick>,
) {
    if rivers.is_empty() {
        spawner.spawn((
            RiverBody,
            WaterBody {
                precedence: 0,
                shape: state.surface.clone(),
            },
        ));
    }
    state.surface.follow(&state.terrains, &state.baking);
}
