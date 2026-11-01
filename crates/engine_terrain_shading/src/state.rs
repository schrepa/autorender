use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::collections::HashMap;
use engine_core::{
    AssetLibrary, DeviceLimits, FrameTime, JobFailures, JobKind, Jobs, MakeSources, Readback,
    RecordExchange, Records, Spawner, TextureHandle, TextureSet, TextureUsage, Ticket, View, codec,
    content_hash,
};
use engine_heightfield::tile::shaded_elsewhere;
use engine_places::ground::{
    GroundTiles, HeldTile, Heightfield, Pyramid, TerrainEntry, TerrainTiles, TileFace, TileKey,
};
use engine_places::shadows::ShadowChanges;
use glam::{Vec2, Vec3};

use crate::bounce::TerrainBounce;
use crate::gpu::{HORIZON_RECORDS, HorizonsPiece};
use crate::jobs::{BOUNCE, BounceJob, LIGHTING, LightingJob, MACRO_COLORS, MacroJob};
use crate::lighting::TerrainLighting;
use crate::settings::{TerrainShading, TerrainShadingStats};
use crate::shading::TileShading;
use crate::{TerrainShadingBrick, terrain_bounce, terrain_horizons};

const RETRY_AFTER: u64 = 120;
const VIEW_PRIORITY: u32 = 0;
const SKIRT_SPACINGS: f32 = 4.0;
const MOST_PROBES: u32 = 512;
const FEWEST_PROBES: u32 = 16;
const ERROR_CELLS: usize = engine_heightfield::TILE_QUADS as usize / 2;
const PEAK_MIPS: u32 = crate::gpu::SURFACE_ERROR_MIPS;

pub(crate) type TileId = (Entity, TileKey);

pub(crate) struct Bake<T> {
    asked: Option<(Ticket, Vec<u8>)>,
    pub(crate) made: Option<Arc<T>>,
    from: Option<Vec<u8>>,
    failed: Option<u64>,
}

impl<T> Default for Bake<T> {
    fn default() -> Self {
        Self {
            asked: None,
            made: None,
            from: None,
            failed: None,
        }
    }
}

impl<T: engine_codec::Codec> Bake<T> {
    fn drop_asked(&mut self, jobs: &mut Jobs<'_, '_>) {
        if let Some((ticket, _)) = self.asked.take() {
            jobs.cancel(ticket);
        }
    }

    fn take(&mut self, records: &Records, kind: &JobKind, (failures, frame): (&mut JobFailures, u64)) {
        let Some(arrived) = self.asked.as_ref().and_then(|(ticket, _)| records.get(ticket)) else {
            return;
        };
        let Some((_, input)) = self.asked.take() else {
            return;
        };
        let made = match arrived {
            Ok(bytes) => codec::decode::<T>(&bytes)
                .map_err(|error| format!("{}: its output does not decode: {error}", kind.name())),
            Err(error) => Err(format!("{}: {error}", kind.name())),
        };
        match made {
            Ok(made) => {
                self.made = Some(Arc::new(made));
                self.from = Some(input);
                self.failed = None;
            }
            Err(error) => {
                failures.record(frame, error);
                self.failed = Some(frame);
            }
        }
    }

    fn ask(&mut self, jobs: &mut Jobs<'_, '_>, kind: &'static JobKind, input: Vec<u8>, frame: u64) {
        if self.failed.is_some_and(|at| frame < at + RETRY_AFTER) {
            return;
        }
        if self.from.as_ref() == Some(&input) {
            self.drop_asked(jobs);
            return;
        }
        if self.asked.as_ref().is_some_and(|(_, asked)| *asked == input) {
            return;
        }
        self.drop_asked(jobs);
        let ticket = jobs.fetch(kind, &input, VIEW_PRIORITY);
        self.asked = Some((ticket, input));
    }
}

#[derive(Default)]
pub(crate) struct Colors {
    wanted: Option<MacroJob>,
    asked: Option<(MacroJob, Option<TextureHandle>)>,
    pub(crate) shown: Option<(MacroJob, TextureHandle)>,
}

impl Colors {
    fn asking(&self) -> bool {
        self.asked.as_ref().is_some_and(|(job, held)| {
            held.is_some() && self.shown.as_ref().is_none_or(|(shown, _)| shown != job)
        })
    }

    fn handles(&self) -> impl Iterator<Item = TextureHandle> + '_ {
        self.asked
            .iter()
            .filter_map(|(_, held)| *held)
            .chain(self.shown.iter().map(|&(_, handle)| handle))
    }
}

#[derive(Default)]
pub(crate) struct TerrainBakes {
    pub(crate) lighting: Bake<TerrainLighting>,
    pub(crate) bounce: Bake<TerrainBounce>,
    pub(crate) shadings: HashMap<TileKey, Arc<TileShading>>,
    pub(crate) colors: Colors,
}

#[derive(Resource, Default)]
pub(crate) struct Bakes {
    pub(crate) terrains: BTreeMap<Entity, TerrainBakes>,
    failures: JobFailures,
    colors_gone: Vec<TextureHandle>,
}

impl Bakes {
    pub(crate) fn of(&self, terrain: Entity) -> Option<&TerrainBakes> {
        self.terrains.get(&terrain)
    }

    pub(crate) fn macro_map(&self, terrain: Entity) -> Option<TextureHandle> {
        self.of(terrain)?.colors.shown.as_ref().map(|&(_, handle)| handle)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Upload {
    Arrived,
    Replaced,
    Errors,
}

#[derive(Resource, Default)]
pub(crate) struct Atlas {
    pub(crate) slots: HashMap<TileId, u32>,
    pub(crate) held: HashMap<TileId, TileFace>,
    pub(crate) peaks: HashMap<TileId, Arc<[f32]>>,
    pub(crate) arrivals: HashMap<TileId, f32>,
    pub(crate) surfaced: HashMap<TileId, f32>,
    pub(crate) holed: BTreeSet<TileId>,
    pub(crate) unplaced: BTreeSet<TileId>,
    pub(crate) free: Vec<u32>,
    pub(crate) next_layer: u32,
    pub(crate) capacity: u32,
    pub(crate) dropped: u32,
    pub(crate) now: f32,
    pub(crate) uploads: Vec<(TileId, u32, Upload)>,
    pub(crate) vacated: Vec<u32>,
    pub(crate) changed: Vec<TileId>,
    pub(crate) ground_changes: Vec<(Vec3, Vec3, f32)>,
    programs: HashMap<Entity, bool>,
    synced: HashMap<Entity, Arc<[TileFace]>>,
}

impl Atlas {
    pub(crate) fn layer_of(&self, terrain: Entity, key: TileKey) -> Option<u32> {
        self.slots.get(&(terrain, key)).copied()
    }

    pub(crate) fn tile(&self, terrain: Entity, key: TileKey) -> Option<&TileFace> {
        self.held.get(&(terrain, key))
    }

    fn release(&mut self, id: TileId) {
        self.unplaced.remove(&id);
        let Some(layer) = self.slots.remove(&id) else {
            return;
        };
        self.free.push(layer);
        self.vacated.push(layer);
        self.changed.push(id);
        self.held.remove(&id);
        self.peaks.remove(&id);
        self.arrivals.remove(&id);
        self.surfaced.remove(&id);
        self.holed.remove(&id);
    }

    fn place(
        &mut self,
        id: TileId,
        tile: &TileFace,
        replaced: Option<u32>,
        (shades, program, shading): (bool, bool, bool),
    ) -> bool {
        let arriving = replaced.is_none();
        let layer = match replaced.or_else(|| self.free.pop()) {
            Some(layer) => layer,
            None if self.next_layer < self.capacity => {
                self.next_layer += 1;
                self.next_layer - 1
            }
            None => {
                self.dropped += 1;
                self.unplaced.insert(id);
                return false;
            }
        };
        self.unplaced.remove(&id);
        let before = self.held.get(&id).filter(|_| !arriving);
        let heights = before.is_none_or(|held| held.heights != tile.heights);
        let rivers = before.is_none_or(|held| {
            held.river_edges != tile.river_edges || held.river_beds != tile.river_beds
        });
        let had_control = before.map(|held| held.control.is_some());
        if !shades {
            match (had_control, tile.control.is_some()) {
                (_, false) if !program => {
                    self.surfaced.remove(&id);
                }
                (Some(false), true) => {
                    self.surfaced.insert(id, self.now);
                }
                _ => {}
            }
        }
        if arriving {
            self.arrivals.insert(id, self.now);
            if tile.control.is_some() && !shades {
                self.surfaced.insert(id, self.now);
            }
        }
        if shades {
            if shading {
                self.surfaced.insert(id, self.now);
            } else {
                self.surfaced.remove(&id);
            }
        } else if program && tile.control.is_none() && (heights || rivers) {
            self.surfaced.insert(id, self.now);
        }
        if heights || !self.peaks.contains_key(&id) {
            self.peaks.insert(id, peaks(&tile.heights));
        }
        self.slots.insert(id, layer);
        self.held.insert(id, tile.clone());
        self.changed.push(id);
        let upload = if arriving { Upload::Arrived } else { Upload::Replaced };
        self.uploads.push((id, layer, upload));
        if tile.holes.is_some() {
            self.holed.insert(id);
        } else {
            self.holed.remove(&id);
        }
        true
    }

    pub(crate) fn top(&self, id: TileId) -> f32 {
        self.peaks.get(&id).map_or(-f32::MAX, |peaks| peaks_top(peaks))
    }
}

pub(crate) fn peaks(heights: &[f32]) -> Arc<[f32]> {
    let stride = crate::gpu::TILE_STRIDE as usize;
    let height = |x: usize, z: usize| heights.get((z + 1) * stride + x + 1).copied();
    let mut side = ERROR_CELLS;
    let mut level: Vec<f32> = Vec::with_capacity(side * side);
    for cell_z in 0..side {
        for cell_x in 0..side {
            let mut peak = -f32::MAX;
            for z in 2 * cell_z..=2 * cell_z + 2 {
                for x in 2 * cell_x..=2 * cell_x + 2 {
                    peak = engine_math::max(peak, height(x, z).unwrap_or(peak));
                }
            }
            level.push(peak);
        }
    }
    let texels = ((4_usize.pow(PEAK_MIPS)) - 1) / 3;
    let mut all = Vec::with_capacity(texels);
    all.extend_from_slice(&level);
    while side > 1 {
        let half = side / 2;
        let peak = |x: usize, z: usize| level[z * side + x];
        let coarser: Vec<f32> = (0..half * half)
            .map(|at| {
                let (x, z) = (2 * (at % half), 2 * (at / half));
                engine_math::max(
                    engine_math::max(peak(x, z), peak(x + 1, z)),
                    engine_math::max(peak(x, z + 1), peak(x + 1, z + 1)),
                )
            })
            .collect();
        all.extend_from_slice(&coarser);
        level = coarser;
        side = half;
    }
    Arc::from(all)
}

fn peaks_top(peaks: &[f32]) -> f32 {
    peaks.last().copied().unwrap_or(-f32::MAX)
}

pub(crate) fn place_tiles(
    tiles: Res<TerrainTiles>,
    limits: Res<DeviceLimits>,
    time: Res<FrameTime>,
    bakes: Res<Bakes>,
    mut atlas: ResMut<Atlas>,
    mut ground: ResMut<GroundTiles>,
) {
    let atlas = &mut *atlas;
    atlas.now = time.elapsed_seconds as f32;
    atlas.uploads.clear();
    atlas.vacated.clear();
    atlas.changed.clear();
    atlas.ground_changes.clear();
    let shaded = shaded_changes(atlas, &tiles, &bakes);
    if !tiles.is_changed() && atlas.unplaced.is_empty() && shaded.is_empty() {
        return;
    }
    atlas.capacity = limits.texture_layers;
    atlas.dropped = 0;
    let listed = |terrain: &Entity| tiles.terrains.iter().any(|entry| entry.terrain == *terrain);
    let mut left: Vec<TileId> = atlas
        .held
        .keys()
        .filter(|(terrain, key)| {
            tiles
                .terrains
                .iter()
                .find(|entry| entry.terrain == *terrain)
                .is_none_or(|entry| listed_tile(&entry.tiles, *key).is_none())
        })
        .copied()
        .collect();
    left.sort_unstable();
    for &id in &left {
        if let Some(entry) = atlas.held.get(&id) {
            let entry_terrain = tiles.terrains.iter().find(|held| held.terrain == id.0);
            if let Some(terrain) = entry_terrain {
                atlas.ground_changes.push(ground_box(terrain, entry));
            }
        }
        atlas.release(id);
    }
    let mut missing: Vec<(TileId, Option<u32>)> = atlas.unplaced.iter().map(|&id| (id, None)).collect();
    for entry in tiles.terrains.iter() {
        for tile in entry.tiles.iter() {
            let id = (entry.terrain, tile.key);
            match atlas.held.get(&id) {
                None => missing.push((id, None)),
                Some(held) if held.stamp == tile.stamp && !shaded.contains(&id) => {}
                Some(held) => {
                    let same_ground = held.heights == tile.heights
                        && held.holes == tile.holes
                        && held.control == tile.control
                        && held.river_edges == tile.river_edges
                        && held.river_beds == tile.river_beds
                        && !shaded.contains(&id);
                    let layer = atlas.slots.get(&id).copied();
                    let (stood, stands) = (ground_box(entry, held), ground_box(entry, tile));
                    if held.heights != tile.heights || stood.0 != stands.0 || stood.1 != stands.1 {
                        atlas.ground_changes.push((stood.0.min(stands.0), stood.1.max(stands.1), stands.2));
                    }
                    if same_ground {
                        atlas.held.insert(id, tile.clone());
                        if let Some(layer) = layer {
                            atlas.uploads.push((id, layer, Upload::Errors));
                        }
                        continue;
                    }
                    atlas.slots.remove(&id);
                    missing.push((id, layer));
                }
            }
        }
    }
    let stopped: Vec<Entity> = tiles
        .terrains
        .iter()
        .filter(|entry| !entry.surface_program && atlas.programs.get(&entry.terrain) == Some(&true))
        .map(|entry| entry.terrain)
        .collect();
    for terrain in stopped {
        let Some(entry) = tiles.terrains.iter().find(|entry| entry.terrain == terrain) else {
            continue;
        };
        for tile in entry.tiles.iter() {
            let id = (terrain, tile.key);
            let unsurfaced =
                tile.control.is_none() && !shaded_elsewhere(tile.key.level, entry.surroundings);
            if unsurfaced && atlas.surfaced.remove(&id).is_some() {
                atlas.changed.push(id);
            }
        }
    }
    atlas.programs.clear();
    atlas
        .programs
        .extend(tiles.terrains.iter().map(|entry| (entry.terrain, entry.surface_program)));
    missing.sort_unstable_by_key(|((terrain, key), _)| (std::cmp::Reverse(key.level), *terrain, key.x, key.z));
    missing.dedup_by_key(|(id, _)| *id);
    for (id, replaced) in missing {
        let Some(entry) = tiles.terrains.iter().find(|entry| entry.terrain == id.0) else {
            continue;
        };
        let Some(tile) = listed_tile(&entry.tiles, id.1) else {
            continue;
        };
        if replaced.is_none() {
            atlas.ground_changes.push(ground_box(entry, tile));
        }
        let shades = shaded_elsewhere(id.1.level, entry.surroundings);
        let shading = bakes.of(id.0).is_some_and(|held| held.shadings.contains_key(&id.1));
        atlas.place(id, tile, replaced, (shades, entry.surface_program, shading));
    }
    atlas.synced.retain(|terrain, _| listed(terrain));
    for entry in tiles.terrains.iter() {
        atlas.synced.insert(entry.terrain, Arc::clone(&entry.tiles));
    }
    let held: Vec<HeldTile> = {
        let mut held: Vec<HeldTile> = atlas
            .slots
            .iter()
            .filter_map(|(&(terrain, key), &layer)| {
                let tile = atlas.held.get(&(terrain, key))?;
                Some(HeldTile {
                    terrain,
                    key,
                    stamp: tile.stamp,
                    layer,
                    top: atlas.top((terrain, key)),
                })
            })
            .collect();
        held.sort_unstable_by_key(|tile| (tile.terrain, tile.key.level, tile.key.x, tile.key.z));
        held
    };
    let tables: Vec<(Entity, u32)> = tiles
        .terrains
        .iter()
        .enumerate()
        .map(|(index, entry)| (entry.terrain, index as u32))
        .collect();
    let published = GroundTiles {
        held,
        layers: atlas.capacity,
        tables,
    };
    ground.set_if_neq(published);
}

fn shaded_changes(atlas: &Atlas, tiles: &TerrainTiles, bakes: &Bakes) -> BTreeSet<TileId> {
    let mut changed = BTreeSet::new();
    for entry in tiles.terrains.iter() {
        for tile in entry.tiles.iter() {
            let id = (entry.terrain, tile.key);
            if !shaded_elsewhere(tile.key.level, entry.surroundings) || !atlas.slots.contains_key(&id)
            {
                continue;
            }
            let shading = bakes.of(entry.terrain).is_some_and(|held| held.shadings.contains_key(&tile.key));
            if shading != atlas.surfaced.contains_key(&id) {
                changed.insert(id);
            }
        }
    }
    changed
}

fn listed_tile(tiles: &[TileFace], key: TileKey) -> Option<&TileFace> {
    tiles.iter().find(|tile| tile.key == key)
}

fn ground_box(terrain: &TerrainEntry, tile: &TileFace) -> (Vec3, Vec3, f32) {
    let pyramid = terrain.pyramid;
    let level = u32::from(tile.key.level);
    let corner = pyramid.tile_origin(tile.key);
    let size = pyramid.tile_size(level);
    let spacing = pyramid.spacing(level);
    let skirt = SKIRT_SPACINGS * spacing;
    let (low, high) = (tile.bounds.0.y, tile.bounds.1.y);
    let reach = if level + 1 >= pyramid.levels() {
        f32::INFINITY
    } else {
        spacing
    };
    (
        terrain.origin + Vec3::new(corner.x, low - skirt, corner.y),
        terrain.origin + Vec3::new(corner.x + size, high, corner.y + size),
        reach,
    )
}

#[derive(Component)]
pub(crate) struct GroundChanges;

pub(crate) fn publish_changes(
    atlas: Res<Atlas>,
    own: Query<Entity, With<GroundChanges>>,
    mut spawner: Spawner<TerrainShadingBrick>,
) {
    if atlas.ground_changes.is_empty() {
        return;
    }
    let change = ShadowChanges {
        boxes: atlas.ground_changes.clone(),
        all: false,
        casters: false,
    };
    match own.single() {
        Ok(entity) => spawner.insert(entity, change),
        Err(_) => {
            spawner.spawn((GroundChanges, change));
        }
    }
}

pub(crate) fn lighting_level(pyramid: &Pyramid, pinned: u32) -> u32 {
    pyramid.levels().saturating_sub(pinned.max(1))
}

pub(crate) fn probe_columns(pyramid: &Pyramid, settings: &TerrainShading) -> u32 {
    let extent = pyramid.extent();
    let longer = engine_math::max(extent.x, extent.y);
    let wanted = (longer / settings.bounce_spacing).ceil();
    let wanted = if wanted.is_finite() && wanted < MOST_PROBES as f32 {
        wanted as u32
    } else {
        MOST_PROBES
    };
    wanted.next_power_of_two().clamp(FEWEST_PROBES, MOST_PROBES)
}

fn tile_stamps(entry: &TerrainEntry, key: TileKey) -> Vec<engine_heightfield::Stamp> {
    entry
        .edits
        .iter()
        .filter(|stamp| engine_heightfield::reaches(stamp, &entry.pyramid, key))
        .copied()
        .collect()
}

fn ground_tiles(entry: &TerrainEntry, level: u32) -> Vec<(TileKey, Vec<engine_heightfield::Stamp>)> {
    let pyramid = entry.pyramid;
    let cut = |key: TileKey| {
        let size = pyramid.tile_size(u32::from(key.level));
        let low = Vec2::new(f32::from(key.x), f32::from(key.z)) * size;
        entry.surroundings
            && entry.cutout.is_some_and(|(first, last)| {
                low.x >= first.x && low.y >= first.y && low.x + size <= last.x && low.y + size <= last.y
            })
    };
    pyramid
        .level_tiles(level)
        .map(|key| {
            let mut held = key;
            while cut(held) {
                let Some(parent) = pyramid.parent(held) else {
                    break;
                };
                held = parent;
            }
            (held, tile_stamps(entry, held))
        })
        .collect()
}

pub(crate) fn bake(
    settings: Res<TerrainShading>,
    tiles: Res<TerrainTiles>,
    records: Res<Records>,
    time: Res<FrameTime>,
    mut jobs: Jobs,
    mut bakes: ResMut<Bakes>,
) {
    let frame = time.frame;
    let bakes = &mut *bakes;
    let listed: Vec<Entity> = tiles.terrains.iter().map(|entry| entry.terrain).collect();
    let gone: Vec<Entity> = bakes.terrains.keys().filter(|terrain| !listed.contains(terrain)).copied().collect();
    for terrain in gone {
        if let Some(mut held) = bakes.terrains.remove(&terrain) {
            held.lighting.drop_asked(&mut jobs);
            held.bounce.drop_asked(&mut jobs);
            bakes.colors_gone.extend(held.colors.handles());
        }
    }
    for entry in tiles.terrains.iter() {
        let held = bakes.terrains.entry(entry.terrain).or_default();
        let failures = &mut bakes.failures;
        held.colors.wanted = entry.recipe.map(|recipe| MacroJob {
            recipe,
            level: lighting_level(&entry.pyramid, entry.pinned_levels),
            size: macro_texels(&settings),
        });
        let Some(recipe) = entry.recipe else {
            continue;
        };
        if !settings.lit() {
            held.lighting.drop_asked(&mut jobs);
            held.lighting = Bake::default();
        } else {
            held.lighting.take(&records, &LIGHTING, (&mut *failures, frame));
            let level = lighting_level(&entry.pyramid, entry.pinned_levels);
            let job = LightingJob {
                recipe,
                level,
                tiles: ground_tiles(entry, level),
                sky: settings.sky_resolution,
                horizons: settings.shadow_resolution,
            };
            held.lighting.ask(&mut jobs, &LIGHTING, codec::encode(&job), frame);
        }
        if !settings.bounces() || entry.surroundings {
            held.bounce.drop_asked(&mut jobs);
            held.bounce = Bake::default();
            continue;
        }
        held.bounce.take(&records, &BOUNCE, (&mut *failures, frame));
        if held.lighting.made.is_none() {
            continue;
        }
        let level = lighting_level(&entry.pyramid, entry.pinned_levels);
        let job = BounceJob {
            recipe,
            level,
            tiles: ground_tiles(entry, level),
            probes: probe_columns(&entry.pyramid, &settings),
        };
        held.bounce.ask(&mut jobs, &BOUNCE, codec::encode(&job), frame);
    }
}

fn macro_texels(settings: &TerrainShading) -> u32 {
    let block = crate::macro_colors::MACRO_BLOCK;
    settings
        .macro_resolution
        .next_multiple_of(block)
        .min(crate::macro_colors::MAX_MACRO_TEXELS)
}

fn colors_set(job: &MacroJob) -> TextureSet {
    let identity = codec::encode(job);
    TextureSet {
        source: crate::SOURCE.hash(),
        name: format!("terrain/macro/{:032x}", content_hash(&identity)),
        textures: vec![("colors".to_owned(), TextureUsage::Data)],
        make: MakeSources::Job {
            kind: &MACRO_COLORS,
            input: identity,
            size: [job.size, job.size],
        },
    }
}

pub(crate) fn color(
    time: Res<FrameTime>,
    mut bakes: ResMut<Bakes>,
    mut records: ResMut<RecordExchange>,
    mut assets: ResMut<AssetLibrary>,
) {
    let frame = time.frame;
    let bakes = &mut *bakes;
    for held in bakes.terrains.values_mut() {
        let colors = &mut held.colors;
        if let Some((job, Some(handle))) = &colors.asked
            && assets.texture(*handle).is_some()
            && colors.shown.as_ref().is_none_or(|(_, shown)| shown != handle)
            && let Some((_, replaced)) = colors.shown.replace((job.clone(), *handle))
        {
            bakes.colors_gone.push(replaced);
        }
        let Some(job) = colors.wanted.clone() else {
            continue;
        };
        if colors.asked.as_ref().is_some_and(|(asked, _)| *asked == job) {
            continue;
        }
        if let Some((_, Some(handle))) = colors.asked.take()
            && colors.shown.as_ref().is_none_or(|(_, shown)| *shown != handle)
        {
            bakes.colors_gone.push(handle);
        }
        let asked = match records.textures(&mut *assets, colors_set(&job)) {
            Ok(handles) => handles.first().copied(),
            Err(error) => {
                bakes.failures.record(frame, format!("macro colors: {error}"));
                None
            }
        };
        colors.asked = Some((job, asked));
    }
    bakes
        .colors_gone
        .retain(|&handle| assets.texture(handle).is_none() || !assets.remove_texture(handle));
}

pub(crate) fn publish_piece(
    settings: Res<TerrainShading>,
    tiles: Res<TerrainTiles>,
    bakes: Res<Bakes>,
    view: Res<View>,
    limits: Res<DeviceLimits>,
    mut piece: ResMut<HorizonsPiece>,
    mut published: Local<Vec<[f32; 4]>>,
) {
    let records = terrain_horizons::records(
        &tiles.terrains,
        &bakes,
        &settings,
        terrain_bounce::windows(
            &tiles.terrains,
            &bakes,
            (view.detail_eye(), &settings),
            (limits.texture_layers, limits.storage_room),
        ),
        limits.texture_layers,
    );
    if *published == records && !piece.is_changed() {
        return;
    }
    let mut next = *piece;
    next.records = [[0.0; 4]; HORIZON_RECORDS as usize + 1];
    next.records[..records.len()].copy_from_slice(&records);
    let count = records.len() as u32 / crate::gpu::HORIZON_RECORD_VECTORS;
    let bounces = settings.bounces() && bakes.terrains.values().any(|held| held.bounce.made.is_some());
    next.records[HORIZON_RECORDS as usize] = [count as f32, if bounces { 1.0 } else { 0.0 }, 0.0, 0.0];
    piece.set_if_neq(next);
    *published = records;
}

pub(crate) const SURFACE_TOTALS: u32 = 0;

#[derive(Resource, Default)]
pub(crate) struct SurfaceTotals {
    pub(crate) grounds: [u32; 2],
    pub(crate) waters: [u32; 2],
    pub(crate) rests: u32,
    resting: bool,
}

impl SurfaceTotals {
    pub(crate) fn epoch(&self) -> u32 {
        self.rests.wrapping_add(1).max(1)
    }
}

impl Readback for SurfaceTotals {
    fn arrived(&mut self, _: u64, _: u32, bytes: &[u8]) {
        let words: &[u32] = bytemuck::cast_slice(bytes);
        let args = crate::gpu::SUBDIVISION_ARGS as usize;
        let mut sums = [0_u32; 4];
        for counters in words.chunks_exact(args) {
            let epoch = counters[crate::gpu::SUBDIVISION_EPOCH as usize];
            if epoch == 0 {
                continue;
            }
            if epoch != self.epoch() {
                return;
            }
            let sum = if counters[crate::gpu::SUBDIVISION_MEASURE as usize] == 0 { 0 } else { 2 };
            sums[sum] = sums[sum].saturating_add(counters[crate::gpu::SUBDIVISION_LEAVES as usize]);
            sums[sum + 1] = sums[sum + 1].saturating_add(counters[crate::gpu::SUBDIVISION_REFUSED as usize]);
        }
        self.grounds = [sums[0], sums[1]];
        self.waters = [sums[2], sums[3]];
    }
}

pub(crate) const PART_STATS: u32 = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct DrawnStats {
    pub(crate) shallow_subdivisions: u32,
    pub(crate) shadow_patches: u32,
    pub(crate) gpu_controls: u32,
    pub(crate) horizons: u32,
    pub(crate) sky_maps: u32,
    pub(crate) lighting_uploads: u32,
    pub(crate) bounce_relit: u32,
    pub(crate) bounce_probes: u32,
    pub(crate) bounce_levels: u32,
    pub(crate) bounce_dropped: u32,
    pub(crate) macro_maps: u32,
    pub(crate) dropped_materials: u32,
    pub(crate) surface_wanted: u32,
    pub(crate) surface_kept: u32,
}

#[derive(Resource, Default)]
pub(crate) struct PartStats {
    drawn: DrawnStats,
}

impl Readback for PartStats {
    fn arrived(&mut self, _: u64, _: u32, bytes: &[u8]) {
        if let Ok(drawn) = bytemuck::try_pod_read_unaligned::<DrawnStats>(bytes) {
            self.drawn = drawn;
        }
    }
}

pub(crate) fn rest_surface(tiles: Res<TerrainTiles>, atlas: Res<Atlas>, mut totals: ResMut<SurfaceTotals>) {
    let held = tiles.terrains.iter().any(|entry| atlas.slots.keys().any(|(terrain, _)| *terrain == entry.terrain));
    if held {
        if totals.resting {
            totals.resting = false;
        }
        return;
    }
    if !totals.resting {
        let totals = &mut *totals;
        totals.resting = true;
        totals.rests = totals.rests.wrapping_add(1);
        totals.grounds = [0; 2];
        totals.waters = [0; 2];
    }
}

pub(crate) fn publish_heightfield(
    settings: Res<TerrainShading>,
    totals: Res<SurfaceTotals>,
    mut heightfield: ResMut<Heightfield>,
) {
    let leaf_indices = if heightfield.leaf_indices.is_empty() {
        crate::terrain::meshlet_indices()
            .into_iter()
            .map(u32::from)
            .collect()
    } else {
        Arc::clone(&heightfield.leaf_indices)
    };
    heightfield.set_if_neq(Heightfield {
        leaf_indices,
        subdivision_error: settings.subdivision_error,
        water_leaves: totals.waters,
    });
}

fn records_left(settings: &TerrainShading, tiles: &TerrainTiles, bakes: &Bakes) -> u32 {
    let mut left = 0;
    for entry in tiles.terrains.iter() {
        let Some(held) = bakes.of(entry.terrain) else {
            continue;
        };
        if settings.lit() {
            left += u32::from(held.lighting.asked.is_some() || held.lighting.made.is_none());
        }
        if !entry.surroundings && settings.bounces() {
            left += u32::from(held.bounce.asked.is_some() || held.bounce.made.is_none());
        }
        let colors = &held.colors;
        left += u32::from(colors.wanted.as_ref().is_some_and(|wanted| {
            colors.shown.as_ref().is_none_or(|(shown, _)| shown != wanted)
        }));
    }
    left
}

pub(crate) fn count(
    settings: Res<TerrainShading>,
    tiles: Res<TerrainTiles>,
    atlas: Res<Atlas>,
    bakes: Res<Bakes>,
    totals: Res<SurfaceTotals>,
    part: Res<PartStats>,
    mut stats: ResMut<TerrainShadingStats>,
) {
    let drawn = part.drawn;
    let [leaves, refused] = totals.grounds;
    let shaded_tiles = bakes.terrains.values().map(|held| held.shadings.len() as u32).sum();
    let lit = bakes.terrains.values().filter(|held| held.lighting.made.is_some()).count() as u32;
    let lighting = bakes.terrains.values().filter(|held| held.lighting.asked.is_some()).count() as u32;
    let bounced = bakes.terrains.values().filter(|held| held.bounce.made.is_some()).count() as u32;
    let bouncing = bakes.terrains.values().filter(|held| held.bounce.asked.is_some()).count() as u32;
    let colored = bakes.terrains.values().filter(|held| held.colors.shown.is_some()).count() as u32;
    let coloring = bakes.terrains.values().filter(|held| held.colors.asking()).count() as u32;
    let next = TerrainShadingStats {
        pending: records_left(&settings, &tiles, &bakes),
        triangles: leaves.saturating_mul(crate::gpu::SURFACE_MESHLET_INDICES / 3),
        leaves: engine_core::Cut::of(leaves.saturating_add(refused), leaves),
        shallow_subdivisions: drawn.shallow_subdivisions,
        shadow_patches: drawn.shadow_patches,
        tiles: atlas.slots.len() as u32,
        tiles_dropped: engine_core::Cut::of(atlas.slots.len() as u32 + atlas.dropped, atlas.slots.len() as u32),
        gpu_controls: drawn.gpu_controls,
        horizons: drawn.horizons,
        sky_maps: drawn.sky_maps,
        lighting_uploads: drawn.lighting_uploads,
        bounce_relit: drawn.bounce_relit,
        bounce_probes: drawn.bounce_probes,
        bounce_levels: drawn.bounce_levels,
        bounce_dropped: drawn.bounce_dropped,
        macro_maps: drawn.macro_maps,
        dropped_materials: drawn.dropped_materials,
        shaded_tiles,
        surface_slots: engine_core::Cut::of(drawn.surface_wanted, drawn.surface_kept),
        lit,
        lighting,
        bounced,
        bouncing,
        colored,
        coloring,
        failed_jobs: bakes.failures.count,
        latest_job_failure: bakes.failures.last.as_ref().map(|(_, error)| error.clone()),
        ..stats.clone()
    };
    stats.set_if_neq(next);
}

pub(crate) fn forget(
    own: Query<Entity, With<GroundChanges>>,
    mut spawner: Spawner<TerrainShadingBrick>,
    mut atlas: ResMut<Atlas>,
    mut bakes: ResMut<Bakes>,
    mut jobs: Jobs,
) {
    for entity in &own {
        spawner.despawn(entity);
    }
    *atlas = Atlas::default();
    let bakes = &mut *bakes;
    for held in bakes.terrains.values_mut() {
        held.lighting.drop_asked(&mut jobs);
        held.bounce.drop_asked(&mut jobs);
        bakes.colors_gone.extend(held.colors.handles());
    }
    bakes.terrains.clear();
}
