use std::collections::BTreeMap;
use std::sync::{Arc, PoisonError, RwLock};

use bevy_ecs::prelude::*;
use engine_core::{
    Cursor, Error, Feed, JobFailures, Jobs, RecordKey, Since, Spawner, VIEW_JOBS, Views,
    job_priority,
};
use engine_heightfield::tile::TileJob;
use engine_heightfield::{
    ErrorPyramid, Fall, HydrologyBake, Layers, Pyramid, Rivers, Stamp, TileKey, reaches,
};
use engine_open_world_kit::ground::{TerrainEntry, TerrainTiles, TileFace};
use engine_open_world_kit::water::{WaterTileErrors, Waterline};
use engine_places_simulation::ground::GroundRead;
use engine_places_simulation::water::{WaterBody, WaterQuery, WaterSample};
use glam::{Vec2, Vec3};

use crate::drawn::{DrawnWork, draw_input, start_drawing};
use crate::emitters::Emitters;
use crate::flow::Running;
use crate::jobs::DRAWN;
use crate::jobs::drawn::DrawnWater;
use crate::water_blocks::Blocks;
use crate::waves::{Swell, Waves};
use crate::{RiverFlow, TileWater, Water, WaterBrick};

pub(crate) type Watered = Option<Arc<Running>>;

pub(crate) enum Work {
    Draw(DrawnWork),
    Redraw(DrawnWork, Watered),
}

pub(crate) struct HeldWater {
    pub(crate) face: TileFace,
    pub(crate) work: Option<Work>,
    pub(crate) made: Option<Result<(), Error>>,
    pub(crate) left: bool,
    pub(crate) water: Option<Arc<TileWater>>,
    pub(crate) drawn: Option<Arc<DrawnWater>>,
    pub(crate) watered: Watered,
}

impl HeldWater {
    fn new(face: TileFace) -> Self {
        Self {
            face,
            work: None,
            made: None,
            left: false,
            water: None,
            drawn: None,
            watered: None,
        }
    }

    pub(crate) fn resident(&self) -> bool {
        !self.left && matches!(self.made, Some(Ok(())))
    }

    pub(crate) fn on_face(&self) -> bool {
        !self.left
    }

    fn unasked(&self) -> bool {
        self.work.is_none() && self.made.is_none()
    }

    pub(crate) fn redrawing(&self) -> bool {
        matches!(self.work, Some(Work::Redraw(..)))
    }

    pub(crate) fn keep_made(&mut self) {
        self.work = None;
        self.made = Some(Ok(()));
        self.left = false;
    }

    pub(crate) fn shown(&self) -> Option<RecordKey> {
        self.drawn.as_ref().map(|drawn| drawn.record)
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
    pub(crate) falls: Arc<[Fall]>,
    pub(crate) flow: Option<RiverFlow>,
    pub(crate) held: BTreeMap<TileKey, HeldWater>,
    pub(crate) touched: Feed<TileKey>,
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
            falls: Arc::from(Vec::new()),
            flow: None,
            held: BTreeMap::new(),
            touched: Feed::new(feed),
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
    }

    pub(crate) fn touch_all(&mut self) {
        self.touched.replace();
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
                held.work = None;
                held.left = true;
                self.held.insert(key, held);
            }
            self.touch(key);
            return;
        };
        let held = match held {
            Some(mut held) if held.face.stamp == face.stamp => {
                held.face = face.clone();
                if !held.on_face() {
                    held.keep_made();
                }
                held
            }
            Some(_) | None => HeldWater::new(face.clone()),
        };
        self.held.insert(key, held);
        self.touch(key);
    }

    fn take_faces(&mut self, entry: &TerrainEntry) {
        if self.kept[..] != entry.kept[..] {
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
            .filter(|(_, held)| held.unasked())
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
            if let Some(held) = self.held.get_mut(&key) {
                let input = draw_input(held, &job);
                let priority = job_priority(VIEW_JOBS, place);
                held.work = Some(Work::Draw(start_drawing(jobs, &DRAWN, input, priority)));
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
    pub(crate) blocks: Blocks,
    pub(crate) rewatered_tiles: u32,
    pub(crate) stale_water_tiles: u32,
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
    swell: Arc<Swell>,
}

#[derive(Default)]
pub(crate) struct WaterSurface(RwLock<Served>);

impl WaterSurface {
    fn follow(&self, terrains: &[WaterTerrain], baking: &[(Vec3, Pyramid)], swell: &Arc<Swell>) {
        let mut served = self.0.write().unwrap_or_else(PoisonError::into_inner);
        let served = &mut *served;
        served.swell = Arc::clone(swell);
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
    }
}

impl WaterQuery for WaterSurface {
    fn water(&self, point: Vec3) -> GroundRead<Option<WaterSample>> {
        let served = self.0.read().unwrap_or_else(PoisonError::into_inner);
        crate::water_source::water_at(&served.terrains, &served.baking, &served.swell, point)
    }

    fn reaches(&self) -> Vec<(Vec3, Vec3)> {
        let served = self.0.read().unwrap_or_else(PoisonError::into_inner);
        served.rivers.clone()
    }
}

pub(crate) fn follow(
    tiles: Res<TerrainTiles>,
    views: Res<Views>,
    settings: Res<Water>,
    mut jobs: Jobs,
    mut state: ResMut<WaterState>,
) {
    let eye = views.main().map_or(Vec3::ZERO, |main| main.detail_eye);
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
        let mut terrain = kept.unwrap_or_else(|| WaterTerrain::new(entry, &baked, feed));
        terrain.touched.resize(feed);
        if terrain.origin != entry.origin {
            terrain.touch_all();
        }
        if baked.unbaked().is_none() {
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
        let local = Vec2::new(eye.x - entry.origin.x, eye.z - entry.origin.z);
        terrain.take_faces(entry);
        terrain.follow_faces(&mut jobs, local);
        state.terrains.push(terrain);
    }
    state.dropped = !before.is_empty();
}

type Waved = BTreeMap<(Entity, TileKey), (Arc<[u16]>, Arc<[u16]>)>;

pub(crate) fn publish_waterline(
    (water, waves): (Res<Water>, Res<Waves>),
    state: Res<WaterState>,
    mut line: ResMut<Waterline>,
    mut cursors: Local<Vec<(Entity, Cursor)>>,
    mut waved: Local<(u64, Waved)>,
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
    let (made, kept) = &mut *waved;
    let remade = *made != waves.makes();
    let changed = touched || state.dropped || water.is_changed() || remade;
    if !(changed || line.watered != state.watered) {
        return;
    }
    let before = std::mem::take(kept);
    *made = waves.makes();
    let swell = waves.swell();
    let dry = ErrorPyramid::dry().shared_halves();
    let mut errors = Vec::new();
    for terrain in &state.terrains {
        for held in terrain.held.values().filter(|held| held.resident()) {
            let at = (terrain.entity, held.face.key);
            let waved = match (&held.drawn, &held.water) {
                (Some(drawn), Some(water)) => {
                    let still = &drawn.errors;
                    let waved = before
                        .get(&at)
                        .filter(|(from, _)| !remade && Arc::ptr_eq(from, still))
                        .map_or_else(
                            || swell.errors(still, water, held.face.spacing),
                            |(_, waved)| Arc::clone(waved),
                        );
                    kept.insert(at, (Arc::clone(still), Arc::clone(&waved)));
                    waved
                }
                (Some(drawn), None) => Arc::clone(&drawn.errors),
                (None, _) => Arc::clone(&dry),
            };
            errors.push(WaterTileErrors {
                terrain: terrain.entity,
                key: held.face.key,
                stamp: held.face.stamp,
                errors: waved,
            });
        }
    }
    let same = line.subdivision_error == water.subdivision_error
        && line.watered == state.watered
        && line.errors.len() == errors.len()
        && line.errors.iter().zip(&errors).all(|(held, found)| {
            held.terrain == found.terrain
                && held.key == found.key
                && held.stamp == found.stamp
                && (Arc::ptr_eq(&held.errors, &found.errors)
                    || held.errors[..] == found.errors[..])
        });
    if same {
        return;
    }
    let line = &mut *line;
    line.subdivision_error = water.subdivision_error;
    line.watered.clear();
    line.watered.extend_from_slice(&state.watered);
    line.errors = errors;
}

pub(crate) fn publish_bodies(
    waves: Res<Waves>,
    mut state: ResMut<WaterState>,
    mut spawner: Spawner<WaterBrick>,
) {
    let state = state.bypass_change_detection();
    if state.body.is_none() {
        state.body = Some(spawner.spawn(WaterBody {
            precedence: 0,
            shape: state.surface.clone(),
        }));
    }
    state
        .surface
        .follow(&state.terrains, &state.baking, waves.swell());
}
