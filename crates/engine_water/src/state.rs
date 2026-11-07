use std::collections::BTreeMap;
use std::sync::{Arc, PoisonError, RwLock};

use bevy_ecs::prelude::*;
use engine_core::{
    Cursor, Feed, FrameBudget, FrameTime, Interest, JobFailures, Jobs, RecordKey,
    Records, Since, Spawner, VIEW_JOBS, PrimaryView, job_priority,
};
use engine_heightfield::tile::{Solid, TileJob};
use engine_heightfield::{
    ErrorPyramid, Fall, HydrologyBake, Layers, Pyramid, Rivers, Stamp, TileKey, reaches,
};
use engine_places_simulation::ground::GroundRead;
use engine_places_simulation::water::{WaterBody, WaterQuery, WaterSample};
use engine_places_world::ground::{TerrainEntry, TerrainTiles, TileFace};
use engine_places_world::water::{WaterTileErrors, Waterline};
use glam::{Vec2, Vec3};

use crate::drawn::{DrawnWater, draw_input, is_dry};
use crate::emitters::Emitters;
use crate::flow::Running;
use crate::jobs::{DRAWN, TILE_WATER};
use crate::settled::Settled;
use crate::water_blocks::Blocks;
use crate::water_particles::Particles;
use crate::water_sim::LiveSim;
use crate::{RiverFlow, TileWater, Water, WaterBrick, WaterWindow};

pub(crate) type Watered = Option<Arc<Running>>;

pub(crate) enum Base {
    Waiting,
    Asking(Interest<RecordKey>),
    Drawing(Interest<RecordKey>),
    Failed,
    Made,
    Redrawing(Interest<RecordKey>, Drawing),
    Left,
}

pub(crate) enum Drawing {
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
        }
    }

    pub(crate) fn resident(&self) -> bool {
        matches!(self.base, Base::Made | Base::Redrawing(..))
    }

    pub(crate) fn on_face(&self) -> bool {
        !matches!(self.base, Base::Left)
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
    pub(crate) flow: Option<RiverFlow>,
    pub(crate) window: Option<WaterWindow>,
    pub(crate) window_moved: bool,
    pub(crate) window_changed: Vec<[Vec2; 2]>,
    pub(crate) settled: Settled,
    pub(crate) held: BTreeMap<TileKey, HeldWater>,
    pub(crate) touched: Feed<TileKey>,
    pub(crate) window_cursor: Cursor,
    cursor: Cursor,
    kept: Arc<[(TileKey, engine_core::Stamp)]>,
}

impl WaterTerrain {
    fn new(entry: &TerrainEntry, baked: &Arc<Layers>, feed: usize) -> Self {
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
            flow: None,
            window: None,
            window_moved: false,
            window_changed: Vec::new(),
            settled: Settled::default(),
            held: BTreeMap::new(),
            touched: Feed::new(feed),
            window_cursor: Cursor::default(),
            cursor: Cursor::default(),
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
        self.touched.replace();
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

    fn take_face(&mut self, key: TileKey, face: Option<&TileFace>) {
        let held = self.held.remove(&key);
        let Some(face) = face else {
            if let Some(mut held) = held
                && (held.resident() || !held.on_face())
                && self.kept.contains(&(key, held.face.stamp))
            {
                held.base = Base::Left;
                self.held.insert(key, held);
            }
            self.touch(key);
            return;
        };
        let held = match held {
            Some(mut held) if held.face.stamp == face.stamp => {
                held.face = face.clone();
                if !held.on_face() {
                    held.base = Base::Made;
                }
                held
            }
            Some(_) | None => HeldWater::new(face.clone()),
        };
        self.held.insert(key, held);
        self.touch(key);
    }

    fn take_faces(&mut self, entry: &TerrainEntry) {
        if !Arc::ptr_eq(&self.kept, &entry.kept) {
            self.kept = Arc::clone(&entry.kept);
            let kept = &self.kept;
            self.held
                .retain(|&key, held| held.on_face() || kept.contains(&(key, held.face.stamp)));
        }
        let mut keys: Vec<TileKey> = match entry.changes.since(&mut self.cursor) {
            Since::Same => return,
            Since::Listed(changes) => changes.iter().map(|change| change.key).collect(),
            Since::All => entry
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
            self.take_face(key, face);
        }
    }

    fn follow_faces(&mut self, jobs: &mut Jobs, eye: Vec2) {
        let pyramid = self.pyramid;
        let size = pyramid.tile_size(0);
        let mut asking: Vec<(u32, TileKey)> = self
            .held
            .iter()
            .filter(|(_, held)| match held.base {
                Base::Waiting => true,
                Base::Asking(_)
                | Base::Drawing(_)
                | Base::Failed
                | Base::Made
                | Base::Redrawing(..)
                | Base::Left => false,
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
        (records, budget): (&Records, &FrameBudget),
        (failures, jobs): (&mut JobFailures, &mut Jobs),
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
            if taken && budget.expired() {
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
            held.base = Base::Failed;
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
                Ok((record, false)) => {
                    let input = draw_input(held, &record);
                    let priority = job_priority(VIEW_JOBS, 0);
                    held.base = Base::Drawing(jobs.start(&DRAWN, input, priority));
                }
                Err(error) => {
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
    pub(crate) body: Option<Entity>,
    pub(crate) emitters: BTreeMap<Entity, Emitters>,
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
    pub(crate) squares_left: u32,
}

#[derive(Default)]
pub(crate) struct SettleQueue {
    pub(crate) asked: Vec<crate::water_settle::SettleAsk>,
    pub(crate) settled: Vec<(Vec<u8>, SettleMade)>,
}

pub(crate) type SettleMade = Result<Arc<[u8]>, String>;

impl SettleQueue {
    pub(crate) fn made(&self, input: &[u8]) -> Option<SettleMade> {
        self.settled
            .iter()
            .find(|(held, _)| held[..] == *input)
            .map(|(_, made)| made.clone())
    }
}

pub(crate) struct SurfaceTerrain {
    entity: Entity,
    pub(crate) origin: Vec3,
    pub(crate) pyramid: Pyramid,
    pub(crate) falls: Arc<[Fall]>,
    pub(crate) held: BTreeMap<TileKey, (TileFace, Option<Arc<TileWater>>)>,
    cursor: Cursor,
}

impl SurfaceTerrain {
    fn of(terrain: &WaterTerrain) -> Self {
        Self {
            entity: terrain.entity,
            origin: terrain.origin,
            pyramid: terrain.pyramid,
            falls: Arc::clone(&terrain.falls),
            held: BTreeMap::new(),
            cursor: Cursor::default(),
        }
    }

    fn follow(&mut self, terrain: &WaterTerrain) {
        self.origin = terrain.origin;
        self.pyramid = terrain.pyramid;
        self.falls = Arc::clone(&terrain.falls);
        let keys = match terrain.touched.since(&mut self.cursor) {
            Since::Same => return,
            Since::Listed(keys) => keys,
            Since::All => {
                self.held = terrain
                    .held
                    .iter()
                    .filter(|(_, held)| held.resident())
                    .map(|(&key, held)| (key, (held.face.clone(), held.water.clone())))
                    .collect();
                return;
            }
        };
        for key in keys {
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
            held.follow(terrain);
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
    view: Res<PrimaryView>,
    time: Res<FrameTime>,
    (budget, settings): (Res<FrameBudget>, Res<Water>),
    mut jobs: Jobs,
    mut state: ResMut<WaterState>,
) {
    let eye = view.detail_eye();
    let frame = time.frame;
    let feed = settings.tile_feed as usize;
    let state = &mut *state;
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
        let mut terrain = kept.unwrap_or_else(|| WaterTerrain::new(entry, &baked, feed));
        terrain.touched.resize(feed);
        if terrain.origin != entry.origin {
            terrain.touch_all();
        }
        if (fresh || !Arc::ptr_eq(&terrain.baked, &baked)) && baked.unbaked().is_none() {
            terrain.falls = baked
                .hydrology()
                .map_or_else(|| Arc::from(Vec::new()), |water| water.falls.clone());
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
        terrain.take_faces(entry);
        terrain.follow_faces(&mut jobs, local);
        terrain.take_bases(
            (&records, &budget),
            (&mut state.failures, &mut jobs),
            frame,
        );
        state.terrains.push(terrain);
    }
    state.dropped = !before.is_empty();
}

pub(crate) fn publish_waterline(
    water: Res<Water>,
    state: Res<WaterState>,
    mut line: ResMut<Waterline>,
    mut cursors: Local<Vec<(Entity, Cursor)>>,
) {
    let before = std::mem::take(&mut *cursors);
    let mut touched = false;
    for terrain in &state.terrains {
        let mut cursor = before
            .iter()
            .find(|(entity, _)| *entity == terrain.entity)
            .map_or_else(Cursor::default, |&(_, cursor)| cursor);
        touched |= !matches!(terrain.touched.since(&mut cursor), Since::Same);
        cursors.push((terrain.entity, cursor));
    }
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

pub(crate) fn publish_bodies(mut state: ResMut<WaterState>, mut spawner: Spawner<WaterBrick>) {
    let state = state.bypass_change_detection();
    if state.body.is_none() {
        state.body = Some(spawner.spawn(WaterBody {
            precedence: 0,
            shape: state.surface.clone(),
        }));
    }
    state.surface.follow(&state.terrains, &state.baking);
}
