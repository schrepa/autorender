use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::collections::HashMap;
use engine_core::{
    AssetLibrary, Background, Cursor, DeviceLimits, Error, Feed, FrameId,
    FrameTime, JobFailures, JobKind, Jobs, Landed, Progress, Readable, Readback, RecordExchange,
    RecordKey, Records, Settling, Since, Stamp, TextureHandle, TextureSet, TextureSource,
    TextureUsage, Views, codec, content_hash,
};
use engine_heightfield::tile::shaded_elsewhere;
use engine_open_world_kit::ground::{
    GroundBakes, GroundBaking, GroundLayerAsks, GroundLayersHeld, GroundTiles, Heightfield,
    HeldTile, PALETTE_ENTRIES, TerrainEntry, TerrainTiles, TileChange, TileFace, TileMaterials,
};
use engine_render::DebugShown;
use engine_tiles::{Pyramid, TileKey};
use glam::Vec2;

use crate::gpu::{HORIZON_RECORDS, HorizonsPiece};
use crate::jobs::bounce::TerrainBounce;
use crate::jobs::lighting::TerrainLighting;
use crate::jobs::shading::TileShading;
use crate::jobs::{BOUNCE, BounceJob, LIGHTING, LightingJob, MACRO_COLORS, MacroJob};
use crate::settings::{TerrainShading, TerrainShadingStats};
use crate::shadings::Shadings;
use crate::terrain_macro::Shape;
use crate::world_trace::traced;
use crate::{terrain_bounce, terrain_horizons};

const VIEW_PRIORITY: u32 = 0;
const MOST_PROBES: u32 = 512;
const FEWEST_PROBES: u32 = 16;

pub(crate) type TileId = (Entity, TileKey);

struct Baking<T> {
    input: Vec<u8>,
    work: Background<Arc<T>>,
}

pub(crate) struct Bake<T> {
    pub(crate) made: Option<Arc<T>>,
    baking: Option<Baking<T>>,
}

impl<T> Default for Bake<T> {
    fn default() -> Self {
        Self {
            made: None,
            baking: None,
        }
    }
}

impl<T: engine_codec::Codec + Send + Sync + 'static> Bake<T> {
    fn waiting(&self) -> bool {
        self.baking
            .as_ref()
            .is_none_or(|baking| baking.work.waiting())
    }

    fn asked(&self) -> bool {
        self.baking
            .as_ref()
            .is_some_and(|baking| baking.work.waiting())
    }

    fn progress(&self) -> Option<Progress> {
        self.baking
            .as_ref()
            .map(|baking| Progress::of(u64::from(!baking.work.waiting()), 1))
    }

    fn step(
        &mut self,
        (records, jobs): (&Records, &mut Jobs<'_, '_>),
        (kind, wanted): (&'static JobKind, Option<Vec<u8>>),
        (failures, frame): (&mut JobFailures, u64),
    ) {
        let Some(wanted) = wanted else {
            *self = Self::default();
            return;
        };
        if self
            .baking
            .as_ref()
            .is_none_or(|baking| baking.input != wanted)
        {
            let work = jobs.fetch(kind, &wanted, VIEW_PRIORITY, move |bytes| {
                codec::decode::<T>(&bytes)
                    .map(Arc::new)
                    .map_err(|error| kind.undecoded(error))
            });
            self.baking = Some(Baking {
                input: wanted,
                work,
            });
            return;
        }
        let Some(baking) = self.baking.as_mut().filter(|baking| baking.work.waiting()) else {
            return;
        };
        match baking.work.poll(records) {
            Landed::Waiting => {}
            Landed::Ready(made) => self.made = Some(Arc::clone(made)),
            Landed::Failed(error) => failures.record(frame, format!("{}: {error}", kind.name())),
        }
    }
}

enum ColorStep {
    Bytes(Background<Arc<[u8]>>),
    Texture(TextureHandle),
}

struct Coloring {
    job: MacroJob,
    shown: Option<TextureHandle>,
    step: Option<ColorStep>,
}

impl Coloring {
    fn settled(&self) -> Option<&MacroJob> {
        self.step.is_none().then_some(&self.job)
    }

    fn handles(&self) -> Vec<TextureHandle> {
        match self.step {
            Some(ColorStep::Texture(handle)) => std::iter::once(handle).chain(self.shown).collect(),
            Some(ColorStep::Bytes(_)) | None => self.shown.into_iter().collect(),
        }
    }

    fn advance(
        &mut self,
        (exchange, assets): (&mut RecordExchange, &mut AssetLibrary),
        records: &Records,
        gone: &mut Vec<TextureHandle>,
    ) -> Option<Error> {
        let mut failed = None;
        self.step = match self.step.take() {
            Some(ColorStep::Bytes(mut work)) => {
                let landed = match work.poll(records) {
                    Landed::Waiting => None,
                    Landed::Ready(pixels) => Some(Ok(Arc::clone(pixels))),
                    Landed::Failed(error) => Some(Err(error.clone())),
                };
                match landed.map(|made| {
                    made.and_then(|pixels| exchange.textures(assets, colors_set(&self.job, pixels)))
                }) {
                    None => Some(ColorStep::Bytes(work)),
                    Some(Ok(handles)) => handles.first().map(|&handle| ColorStep::Texture(handle)),
                    Some(Err(error)) => {
                        failed = Some(error);
                        None
                    }
                }
            }
            Some(ColorStep::Texture(handle)) if assets.texture(handle).is_some() => {
                gone.extend(self.shown.replace(handle));
                None
            }
            Some(ColorStep::Texture(handle)) => match exchange.failed(handle) {
                Some(error) => {
                    failed = Some(error.clone());
                    assets.remove_texture(handle);
                    None
                }
                None => Some(ColorStep::Texture(handle)),
            },
            None => None,
        };
        failed
    }
}

#[derive(Default)]
pub(crate) struct Colors {
    wanted: Option<MacroJob>,
    coloring: Option<Coloring>,
}

impl Colors {
    pub(crate) fn shown(&self) -> Option<TextureHandle> {
        self.coloring.as_ref().and_then(|coloring| coloring.shown)
    }

    fn asking(&self) -> bool {
        self.coloring
            .as_ref()
            .is_some_and(|coloring| coloring.step.is_some())
    }

    fn left(&self) -> bool {
        let settled = self.coloring.as_ref().and_then(Coloring::settled);
        self.wanted
            .as_ref()
            .is_some_and(|wanted| settled != Some(wanted))
    }

    fn handles(&self) -> Vec<TextureHandle> {
        self.coloring
            .as_ref()
            .map_or_else(Vec::new, Coloring::handles)
    }

    fn step(
        &mut self,
        (exchange, assets): (&mut RecordExchange, &mut AssetLibrary),
        (records, jobs): (&Records, &mut Jobs<'_, '_>),
        gone: &mut Vec<TextureHandle>,
    ) -> Option<Error> {
        let failed = self
            .coloring
            .as_mut()
            .and_then(|coloring| coloring.advance((exchange, assets), records, gone));
        let Some(wanted) = self
            .wanted
            .clone()
            .filter(|wanted| self.coloring.as_ref().map(|coloring| &coloring.job) != Some(wanted))
        else {
            return failed;
        };
        let shown = self.coloring.take().and_then(|coloring| {
            if let Some(ColorStep::Texture(handle)) = coloring.step {
                gone.push(handle);
            }
            coloring.shown
        });
        let work = jobs.fetch(&MACRO_COLORS, &codec::encode(&wanted), VIEW_PRIORITY, Ok);
        self.coloring = Some(Coloring {
            job: wanted,
            shown,
            step: Some(ColorStep::Bytes(work)),
        });
        failed
    }
}

#[derive(Default)]
pub(crate) struct TerrainBakes {
    pub(crate) lighting: Bake<TerrainLighting>,
    pub(crate) bounce: Bake<TerrainBounce>,
    pub(crate) flow: Shadings,
    pub(crate) colors: Colors,
}

impl TerrainBakes {
    pub(crate) fn shading(&self, tile: &TileFace) -> Option<&Arc<TileShading>> {
        self.flow.shown(tile)
    }

    fn lighting_left(&self, settings: &TerrainShading, entry: &TerrainEntry) -> bool {
        settings.lit() && entry.recipe.is_some() && self.lighting.waiting()
    }

    fn bounce_left(&self, settings: &TerrainShading, entry: &TerrainEntry) -> bool {
        settings.bounces()
            && entry.recipe.is_some()
            && !entry.surroundings
            && (self.lighting.made.is_some() || self.lighting.waiting())
            && self.bounce.waiting()
    }

    fn release(self, colors_gone: &mut Vec<TextureHandle>) -> BTreeSet<RecordKey> {
        colors_gone.extend(self.colors.handles());
        self.flow.release()
    }
}

#[derive(Resource, Default)]
pub(crate) struct Bakes {
    pub(crate) terrains: BTreeMap<Entity, TerrainBakes>,
    pub(crate) failures: JobFailures,
    colors_gone: Vec<TextureHandle>,
    lent: Vec<Option<TextureHandle>>,
}

impl Bakes {
    pub(crate) fn lent(&self) -> &[Option<TextureHandle>] {
        &self.lent
    }

    pub(crate) fn of(&self, terrain: Entity) -> Option<&TerrainBakes> {
        self.terrains.get(&terrain)
    }

    pub(crate) fn macro_map(&self, terrain: Entity) -> Option<TextureHandle> {
        self.of(terrain)?.colors.shown()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Upload {
    Whole,
    Shaded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AtlasChange {
    Placed(TileId, u32, Upload),
    Vacated(u32),
    Moved(TileId),
}

#[derive(Default)]
pub(crate) struct AtlasMoves {
    pub(crate) all: bool,
    pub(crate) uploads: Vec<(TileId, u32, Upload)>,
    pub(crate) vacated: Vec<u32>,
    pub(crate) changed: Vec<TileId>,
}

impl AtlasMoves {
    pub(crate) fn since(atlas: &Atlas, cursor: &mut Cursor) -> Self {
        let changes = match atlas.changes.since(cursor) {
            Since::Same => return Self::default(),
            Since::All => {
                return Self {
                    all: true,
                    uploads: all_uploads(atlas),
                    ..Self::default()
                };
            }
            Since::Listed(changes) => changes,
        };
        let mut moves = Self::default();
        let mut listed: HashMap<TileId, usize> = HashMap::default();
        for &change in changes {
            match change {
                AtlasChange::Placed(id, layer, upload) => {
                    if atlas.slots.get(&id) != Some(&layer) {
                        continue;
                    }
                    match listed.get(&id) {
                        Some(&at) if upload == Upload::Whole => moves.uploads[at].2 = upload,
                        Some(_) => {}
                        None => {
                            listed.insert(id, moves.uploads.len());
                            moves.uploads.push((id, layer, upload));
                        }
                    }
                }
                AtlasChange::Vacated(layer) => moves.vacated.push(layer),
                AtlasChange::Moved(id) => moves.changed.push(id),
            }
        }
        moves.changed.sort_unstable();
        moves.changed.dedup();
        moves
    }
}

pub(crate) fn all_uploads(atlas: &Atlas) -> Vec<(TileId, u32, Upload)> {
    let mut all: Vec<(TileId, u32, Upload)> = atlas
        .slots
        .iter()
        .map(|(&id, &layer)| (id, layer, Upload::Whole))
        .collect();
    all.sort_unstable_by_key(|((terrain, key), _, _)| {
        (std::cmp::Reverse(key.level), *terrain, key.x, key.z)
    });
    all
}

pub(crate) fn palette_layout(terrains: &[TerrainEntry]) -> Vec<(Entity, Option<u32>)> {
    let mut used = 0_u32;
    terrains
        .iter()
        .map(|terrain| {
            let count = u32::try_from(terrain.materials.len()).unwrap_or(u32::MAX);
            let base = (count <= PALETTE_ENTRIES - used).then_some(used);
            if base.is_some() {
                used += count;
            }
            (terrain.terrain, base)
        })
        .collect()
}

#[derive(Clone, Copy)]
struct PlacedTile {
    stamp: Stamp,
    low: f32,
    high: f32,
    control: bool,
}

#[derive(Resource)]
pub(crate) struct Atlas {
    pub(crate) slots: HashMap<TileId, u32>,
    placed: HashMap<TileId, PlacedTile>,
    cursors: HashMap<Entity, Cursor>,
    pub(crate) arrivals: HashMap<TileId, f32>,
    pub(crate) surfaced: HashMap<TileId, f32>,
    pub(crate) unplaced: BTreeSet<TileId>,
    pub(crate) free: Vec<u32>,
    pub(crate) next_layer: u32,
    pub(crate) capacity: u32,
    pub(crate) now: f32,
    changes: Feed<AtlasChange>,
}

impl Default for Atlas {
    fn default() -> Self {
        Self {
            slots: HashMap::default(),
            placed: HashMap::default(),
            cursors: HashMap::default(),
            arrivals: HashMap::default(),
            surfaced: HashMap::default(),
            unplaced: BTreeSet::new(),
            free: Vec::new(),
            next_layer: 0,
            capacity: 0,
            now: 0.0,
            changes: Feed::new(TerrainShading::default().tile_feed as usize),
        }
    }
}

impl Atlas {
    pub(crate) fn layer_of(&self, terrain: Entity, key: TileKey) -> Option<u32> {
        self.slots.get(&(terrain, key)).copied()
    }

    pub(crate) fn stamp_of(&self, id: TileId) -> Option<Stamp> {
        self.placed.get(&id).map(|placed| placed.stamp)
    }

    pub(crate) fn heights_of(&self, terrain: Entity) -> (f32, f32) {
        self.placed
            .iter()
            .filter(|((held, _), _)| *held == terrain)
            .fold(
                (f32::INFINITY, f32::NEG_INFINITY),
                |(low, high), (_, placed)| {
                    (
                        engine_math::min(low, placed.low),
                        engine_math::max(high, placed.high),
                    )
                },
            )
    }

    pub(crate) fn tile<'a>(&self, entry: &'a TerrainEntry, key: TileKey) -> Option<&'a TileFace> {
        listed_tile(&entry.tiles, key).filter(|_| !self.unplaced.contains(&(entry.terrain, key)))
    }

    fn release(&mut self, id: TileId) {
        self.unplaced.remove(&id);
        let Some(layer) = self.slots.remove(&id) else {
            return;
        };
        self.free.push(layer);
        self.changes.push(AtlasChange::Vacated(layer));
        self.changes.push(AtlasChange::Moved(id));
        self.placed.remove(&id);
        self.arrivals.remove(&id);
        self.surfaced.remove(&id);
    }

    fn has_room(&self) -> bool {
        !self.free.is_empty() || self.next_layer < self.capacity
    }

    fn follow_tile(
        &mut self,
        entry: &TerrainEntry,
        key: TileKey,
        missing: &mut Vec<(TileId, Option<u32>)>,
    ) {
        let id = (entry.terrain, key);
        match (
            listed_tile(&entry.tiles, key),
            self.placed.get(&id).copied(),
        ) {
            (None, None) => {
                self.unplaced.remove(&id);
            }
            (None, Some(_)) => self.release(id),
            (Some(_), None) => missing.push((id, None)),
            (Some(tile), Some(held)) if held.stamp == tile.stamp => {
                let unsurfaced = !programmed(entry, tile)
                    && tile.control.is_none()
                    && !shaded_elsewhere(key.level, entry.surroundings);
                if unsurfaced && self.surfaced.remove(&id).is_some() {
                    self.changes.push(AtlasChange::Moved(id));
                }
            }
            (Some(_), Some(_)) => missing.push((id, self.slots.get(&id).copied())),
        }
    }

    fn place(
        &mut self,
        id: TileId,
        tile: &TileFace,
        replaced: Option<u32>,
        (shades, program, shading): (bool, bool, bool),
    ) {
        let arriving = replaced.is_none();
        let layer = match replaced.or_else(|| self.free.pop()) {
            Some(layer) => layer,
            None if self.next_layer < self.capacity => {
                self.next_layer += 1;
                self.next_layer - 1
            }
            None => {
                self.unplaced.insert(id);
                return;
            }
        };
        self.unplaced.remove(&id);
        let had_control = self
            .placed
            .get(&id)
            .filter(|_| !arriving)
            .map(|held| held.control);
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
        } else if program && tile.control.is_none() {
            self.surfaced.insert(id, self.now);
        }
        self.slots.insert(id, layer);
        self.placed.insert(
            id,
            PlacedTile {
                stamp: tile.stamp,
                low: tile.bounds.0.y,
                high: tile.bounds.1.y,
                control: tile.control.is_some(),
            },
        );
        self.changes.push(AtlasChange::Moved(id));
        self.changes
            .push(AtlasChange::Placed(id, layer, Upload::Whole));
    }

    fn show_shading(&mut self, id: TileId, layer: u32) {
        self.surfaced.insert(id, self.now);
        self.changes.push(AtlasChange::Moved(id));
        self.changes
            .push(AtlasChange::Placed(id, layer, Upload::Shaded));
    }
}

pub(crate) fn programmed(entry: &TerrainEntry, tile: &TileFace) -> bool {
    entry.surface_program && tile.materials == TileMaterials::Program
}

fn tile_keys(atlas: &Atlas, entry: &TerrainEntry, since: Since<'_, TileChange>) -> Vec<TileKey> {
    let mut keys: Vec<TileKey> = match since {
        Since::Same => Vec::new(),
        Since::Listed(changes) => changes.iter().map(|change| change.key).collect(),
        Since::All => entry
            .tiles
            .iter()
            .map(|tile| tile.key)
            .chain(
                atlas
                    .placed
                    .keys()
                    .chain(&atlas.unplaced)
                    .filter(|(terrain, _)| *terrain == entry.terrain)
                    .map(|&(_, key)| key),
            )
            .collect(),
    };
    keys.sort_unstable();
    keys.dedup();
    keys
}

impl Atlas {
    fn keep_listed(&mut self, listed: impl Fn(&Entity) -> bool) {
        let mut gone: Vec<TileId> = self
            .placed
            .keys()
            .filter(|(terrain, _)| !listed(terrain))
            .copied()
            .collect();
        gone.sort_unstable();
        for id in gone {
            self.release(id);
        }
        self.unplaced.retain(|(terrain, _)| listed(terrain));
        self.cursors.retain(|terrain, _| listed(terrain));
    }
}

pub(crate) fn place_tiles(
    (tiles, limits, time, settings): (
        Res<TerrainTiles>,
        Res<DeviceLimits>,
        Res<FrameTime>,
        Res<TerrainShading>,
    ),
    bakes: Res<Bakes>,
    mut atlas: ResMut<Atlas>,
    (mut ground, mut baking): (ResMut<GroundTiles>, ResMut<GroundBaking>),
) {
    let atlas = &mut *atlas;
    atlas.now = time.real_elapsed as f32;
    atlas.changes.resize(settings.tile_feed as usize);
    let attached = |entry: &TerrainEntry| {
        bakes
            .of(entry.terrain)
            .map(|held| held.flow.attached())
            .unwrap_or_default()
    };
    let capacity = limits.texture_layers / crate::gpu::TILE_PLANES;
    let whole = macro_texels(&settings);
    if !tiles.is_changed()
        && capacity == atlas.capacity
        && whole == ground.whole
        && (atlas.unplaced.is_empty() || !atlas.has_room())
        && tiles
            .terrains
            .iter()
            .all(|entry| attached(entry).is_empty())
    {
        return;
    }
    let tables: Vec<(Entity, u32)> = tiles
        .terrains
        .iter()
        .enumerate()
        .map(|(index, entry)| (entry.terrain, index as u32))
        .collect();
    let palette = palette_layout(&tiles.terrains);
    if capacity == 0 {
        let now = atlas.now;
        *atlas = Atlas {
            now,
            ..Atlas::default()
        };
        ground.set_if_neq(GroundTiles {
            held: Vec::new(),
            layers: 0,
            tables,
            palette,
            whole,
        });
        if !baking.shaded.is_empty() {
            baking.shaded.clear();
        }
        return;
    }
    atlas.capacity = capacity;
    let listed = |terrain: &Entity| tiles.terrains.iter().any(|entry| entry.terrain == *terrain);
    atlas.keep_listed(listed);
    let mut missing: Vec<(TileId, Option<u32>)> = Vec::new();
    for entry in tiles.terrains.iter() {
        let mut cursor = atlas
            .cursors
            .get(&entry.terrain)
            .copied()
            .unwrap_or_default();
        let keys = tile_keys(atlas, entry, entry.changes.since(&mut cursor));
        atlas.cursors.insert(entry.terrain, cursor);
        for key in keys {
            atlas.follow_tile(entry, key, &mut missing);
        }
    }
    if atlas.has_room() {
        missing.extend(atlas.unplaced.iter().map(|&id| (id, None)));
    }
    missing.sort_unstable_by_key(|((terrain, key), _)| {
        (std::cmp::Reverse(key.level), *terrain, key.x, key.z)
    });
    missing.dedup_by_key(|(id, _)| *id);
    for &(id, replaced) in &missing {
        let Some(entry) = tiles.terrains.iter().find(|entry| entry.terrain == id.0) else {
            continue;
        };
        let Some(tile) = listed_tile(&entry.tiles, id.1) else {
            continue;
        };
        let shades = shaded_elsewhere(id.1.level, entry.surroundings);
        let shading = bakes.of(id.0).and_then(|held| held.shading(tile)).is_some();
        let program = programmed(entry, tile);
        atlas.place(id, tile, replaced, (shades, program, shading));
    }
    for entry in tiles.terrains.iter() {
        for &key in attached(entry) {
            let id = (entry.terrain, key);
            if let Some(layer) = atlas.layer_of(entry.terrain, key)
                && !missing.iter().any(|&(placing, _)| placing == id)
            {
                atlas.show_shading(id, layer);
            }
        }
    }
    let mut held: Vec<HeldTile> = atlas
        .placed
        .iter()
        .filter_map(|(&(terrain, key), placed)| {
            Some(HeldTile {
                terrain,
                key,
                stamp: placed.stamp,
                layer: atlas.layer_of(terrain, key)?,
                top: placed.high,
            })
        })
        .collect();
    held.sort_unstable_by_key(|tile| (tile.terrain, tile.key.level, tile.key.x, tile.key.z));
    let shaded = shaded_tiles(&tiles, &bakes, &held);
    if baking.shaded != shaded {
        baking.shaded = shaded;
    }
    let published = GroundTiles {
        held,
        layers: atlas.capacity,
        tables,
        palette,
        whole,
    };
    ground.set_if_neq(published);
}

fn shaded_tiles(tiles: &TerrainTiles, bakes: &Bakes, held: &[HeldTile]) -> Vec<(Entity, TileKey)> {
    held.iter()
        .filter(|tile| {
            let Some(entry) = tiles.terrains.iter().find(|entry| entry.terrain == tile.terrain)
            else {
                return false;
            };
            listed_tile(&entry.tiles, tile.key).is_some_and(|face| {
                bakes
                    .of(tile.terrain)
                    .and_then(|baked| baked.shading(face))
                    .is_some()
            })
        })
        .map(|tile| (tile.terrain, tile.key))
        .collect()
}

pub(crate) fn listed_tile(tiles: &[TileFace], key: TileKey) -> Option<&TileFace> {
    tiles
        .binary_search_by_key(&key, |tile| tile.key)
        .ok()
        .map(|at| &tiles[at])
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

fn ground_tiles(
    entry: &TerrainEntry,
    level: u32,
) -> Vec<(TileKey, Vec<engine_heightfield::Stamp>)> {
    let pyramid = entry.pyramid;
    let cut = |key: TileKey| {
        let size = pyramid.tile_size(u32::from(key.level));
        let low = Vec2::new(f32::from(key.x), f32::from(key.z)) * size;
        entry.surroundings
            && entry.cutout.is_some_and(|(first, last)| {
                low.x >= first.x
                    && low.y >= first.y
                    && low.x + size <= last.x
                    && low.y + size <= last.y
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
    mut baking: ResMut<GroundBaking>,
) {
    let frame = time.frame;
    let bakes = &mut *bakes;
    let listed: Vec<Entity> = tiles.terrains.iter().map(|entry| entry.terrain).collect();
    let gone: Vec<Entity> = bakes
        .terrains
        .keys()
        .filter(|terrain| !listed.contains(terrain))
        .copied()
        .collect();
    let mut released = BTreeSet::new();
    for terrain in gone {
        if let Some(held) = bakes.terrains.remove(&terrain) {
            released.extend(held.release(&mut bakes.colors_gone));
        }
    }
    for key in released {
        if !bakes
            .terrains
            .values()
            .any(|held| held.flow.holds_share(&key))
        {
            jobs.unshare(key);
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
        let level = lighting_level(&entry.pyramid, entry.pinned_levels);
        let lit = settings.lit().then(|| {
            codec::encode(&LightingJob {
                recipe,
                level,
                tiles: ground_tiles(entry, level),
                sky: settings.sky_resolution,
                horizons: settings.shadow_resolution,
            })
        });
        held.lighting.step(
            (&records, &mut jobs),
            (&LIGHTING, lit),
            (&mut *failures, frame),
        );
        let bounced = (settings.bounces() && !entry.surroundings && held.lighting.made.is_some())
            .then(|| {
                codec::encode(&BounceJob {
                    recipe,
                    level,
                    tiles: ground_tiles(entry, level),
                    probes: probe_columns(&entry.pyramid, &settings),
                })
            });
        held.bounce.step(
            (&records, &mut jobs),
            (&BOUNCE, bounced),
            (&mut *failures, frame),
        );
    }
    let terrains = tiles
        .terrains
        .iter()
        .filter_map(|entry| {
            let held = bakes.terrains.get(&entry.terrain)?;
            Some(GroundBakes {
                terrain: entry.terrain,
                lighting: held.lighting.progress(),
                bounce: held.bounce.progress(),
            })
        })
        .collect();
    if baking.terrains != terrains {
        baking.terrains = terrains;
    }
}

fn macro_texels(settings: &TerrainShading) -> u32 {
    let block = crate::jobs::macro_colors::MACRO_BLOCK;
    settings
        .macro_resolution
        .next_multiple_of(block)
        .min(crate::jobs::macro_colors::MAX_MACRO_TEXELS)
}

fn colors_set(job: &MacroJob, pixels: Arc<[u8]>) -> TextureSet {
    let identity = codec::encode(job);
    let size = job.size;
    TextureSet {
        source: crate::jobs::SOURCE.hash(),
        name: format!("terrain/macro/{:032x}", content_hash(&identity)),
        textures: vec![("colors".to_owned(), TextureUsage::Data)],
        make: Arc::new(move || {
            Ok::<_, String>(vec![TextureSource {
                width: size,
                height: size,
                pixels: pixels.to_vec(),
                usage: TextureUsage::Data,
            }])
            .map(Some)
        }),
    }
}

pub(crate) fn color(
    time: Res<FrameTime>,
    mut bakes: ResMut<Bakes>,
    (records, mut jobs): (Res<Records>, Jobs),
    mut exchange: ResMut<RecordExchange>,
    mut assets: ResMut<AssetLibrary>,
) {
    let frame = time.frame;
    let bakes = &mut *bakes;
    for held in bakes.terrains.values_mut() {
        if let Some(error) = held.colors.step(
            (&mut exchange, &mut assets),
            (&records, &mut jobs),
            &mut bakes.colors_gone,
        ) {
            bakes
                .failures
                .record(frame, format!("macro colors: {error}"));
        }
    }
    bakes
        .colors_gone
        .retain(|&handle| assets.texture(handle).is_none() || !assets.remove_texture(handle));
}

pub(crate) fn lend_layers(
    (tiles, limits, assets): (Res<TerrainTiles>, Res<DeviceLimits>, Res<AssetLibrary>),
    ground: Res<GroundTiles>,
    mut bakes: ResMut<Bakes>,
    mut asked: Query<(&GroundLayerAsks, &mut GroundLayersHeld)>,
) {
    let wanted: Vec<TextureHandle> = asked
        .iter()
        .flat_map(|(asks, _)| asks.textures.iter().copied())
        .collect();
    if wanted.is_empty() && bakes.lent.is_empty() {
        return;
    }
    let bakes = &mut *bakes;
    for slot in &mut bakes.lent {
        if slot.is_some_and(|held| !wanted.contains(&held)) {
            *slot = None;
        }
    }
    let room = (limits.texture_layers as usize).saturating_sub(tiles.terrains.len());
    for &texture in &wanted {
        if bakes.lent.contains(&Some(texture)) {
            continue;
        }
        match bakes.lent.iter().position(Option::is_none) {
            Some(free) => bakes.lent[free] = Some(texture),
            None if bakes.lent.len() < room => bakes.lent.push(Some(texture)),
            None => {}
        }
    }
    while bakes.lent.last() == Some(&None) {
        bakes.lent.pop();
    }
    let shape = Shape::first(&tiles, bakes, &assets);
    for (asks, mut held) in &mut asked {
        let layers = asks
            .textures
            .iter()
            .filter_map(|&texture| {
                let resident = assets.texture(texture)?;
                let slot = bakes.lent.iter().position(|lent| *lent == Some(texture));
                let whole = resident.width() == ground.whole && resident.height() == ground.whole;
                match slot {
                    Some(slot) if whole => {
                        (Some(Shape::of(resident)) == shape).then_some((texture, slot as u32 + 1))
                    }
                    Some(_) | None => Some((texture, 0)),
                }
            })
            .collect();
        held.set_if_neq(GroundLayersHeld { layers });
    }
}

pub(crate) fn publish_piece(
    settings: Res<TerrainShading>,
    tiles: Res<TerrainTiles>,
    ground: Res<GroundTiles>,
    bakes: Res<Bakes>,
    (views, limits): (Res<Views>, Res<DeviceLimits>),
    shown: Res<DebugShown>,
    mut piece: ResMut<HorizonsPiece>,
) {
    let slots = terrain_bounce::slots(&tiles.terrains, &settings, limits.texture_layers);
    let records = terrain_horizons::records(
        &tiles.terrains,
        &bakes,
        &settings,
        terrain_bounce::windows(
            &tiles.terrains,
            &bakes,
            &slots,
            (
                views
                    .main()
                    .map_or(glam::Vec3::ZERO, |main| main.detail_eye),
                &settings,
            ),
            limits.storage_room,
        ),
        limits.texture_layers,
    );
    let mut next = *piece;
    next.debug = [crate::debug::shown(&shown), 0, 0, 0];
    next.trace = [traced(&tiles, &ground).count() as u32, 0, 0, 0];
    next.records = [[0.0; 4]; HORIZON_RECORDS as usize + 1];
    next.records[..records.len()].copy_from_slice(&records);
    let count = records.len() as u32 / crate::gpu::HORIZON_RECORD_VECTORS;
    let bounces = settings.bounces()
        && bakes
            .terrains
            .values()
            .any(|held| held.bounce.made.is_some());
    next.records[HORIZON_RECORDS as usize] =
        [count as f32, if bounces { 1.0 } else { 0.0 }, 0.0, 0.0];
    piece.set_if_neq(next);
}

pub(crate) const SURFACE_TOTALS: u32 = 0;

#[derive(Resource, Default)]
pub(crate) struct SurfaceTotals {
    pub(crate) grounds: [u32; 2],
    pub(crate) watered: u32,
    written: FrameId,
}

impl Readable for SurfaceTotals {
    fn arrived(&mut self, _: u32, read: Readback<&[u8]>) {
        if read.written < self.written {
            return;
        }
        self.written = read.written;
        let words: Vec<u32> = read
            .value
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        use crate::gpu::{
            SUBDIVISION_ARGS, SUBDIVISION_LEAVES, SUBDIVISION_REFUSED, SUBDIVISION_WATERED,
        };
        let mut sums = [0_u32; 3];
        for counters in words.chunks_exact(SUBDIVISION_ARGS as usize) {
            let read = [SUBDIVISION_LEAVES, SUBDIVISION_REFUSED, SUBDIVISION_WATERED];
            for (sum, word) in sums.iter_mut().zip(read) {
                *sum = sum.saturating_add(counters[word as usize]);
            }
        }
        self.grounds = [sums[0], sums[1]];
        self.watered = sums[2];
    }
}

pub(crate) const PART_STATS: u32 = 1;
pub(crate) const TRACE_STATS: u32 = 2;

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
    pool: engine_core::Cut,
    turns: engine_core::Cut,
}

impl Readable for PartStats {
    fn arrived(&mut self, tag: u32, read: Readback<&[u8]>) {
        let bytes = read.value;
        if tag == TRACE_STATS {
            if let Ok([asked, kept, turns, limit]) =
                bytemuck::try_pod_read_unaligned::<[u32; 4]>(bytes)
            {
                self.pool = engine_core::Cut::of(asked, kept);
                self.turns = engine_core::Cut::of(turns, turns.min(limit));
            }
        } else if let Ok(drawn) = bytemuck::try_pod_read_unaligned::<DrawnStats>(bytes) {
            self.drawn = drawn;
        }
    }
}

pub(crate) fn publish_heightfield(
    totals: Res<SurfaceTotals>,
    mut heightfield: ResMut<Heightfield>,
) {
    let leaf_indices = if heightfield.leaf_indices.is_empty() {
        Arc::from(crate::terrain::meshlet_indices())
    } else {
        Arc::clone(&heightfield.leaf_indices)
    };
    heightfield.set_if_neq(Heightfield {
        leaf_indices,
        water_leaves: totals.watered,
    });
}

#[derive(Resource, Default)]
pub(crate) struct RecordsSettling(Settling);

fn records_left(settings: &TerrainShading, tiles: &TerrainTiles, bakes: &Bakes) -> u32 {
    let mut left = 0;
    for entry in tiles.terrains.iter() {
        let Some(held) = bakes.of(entry.terrain) else {
            continue;
        };
        left += u32::from(held.lighting_left(settings, entry));
        left += u32::from(held.bounce_left(settings, entry));
        left += u32::from(held.colors.left());
    }
    left
}

pub(crate) fn count(
    settings: Res<TerrainShading>,
    tiles: Res<TerrainTiles>,
    atlas: Res<Atlas>,
    bakes: Res<Bakes>,
    totals: Res<SurfaceTotals>,
    (part, sun): (Res<PartStats>, Res<crate::terrain_sun::SunPlan>),
    (mut stats, settling): (ResMut<TerrainShadingStats>, Res<RecordsSettling>),
) {
    settling.0.left(records_left(&settings, &tiles, &bakes));
    let drawn = part.drawn;
    let [leaves, refused] = totals.grounds;
    let (shaded_tiles, shading_records) = bakes
        .terrains
        .values()
        .map(|held| held.flow.counts())
        .fold((0, 0), |(shaded, asked), (more, asking)| {
            (shaded + more, asked + asking)
        });
    let lit = bakes
        .terrains
        .values()
        .filter(|held| held.lighting.made.is_some())
        .count() as u32;
    let lighting = bakes
        .terrains
        .values()
        .filter(|held| held.lighting.asked())
        .count() as u32;
    let bounced = bakes
        .terrains
        .values()
        .filter(|held| held.bounce.made.is_some())
        .count() as u32;
    let bouncing = bakes
        .terrains
        .values()
        .filter(|held| held.bounce.asked())
        .count() as u32;
    let colored = bakes
        .terrains
        .values()
        .filter(|held| held.colors.shown().is_some())
        .count() as u32;
    let coloring = bakes
        .terrains
        .values()
        .filter(|held| held.colors.asking())
        .count() as u32;
    let next = TerrainShadingStats {
        triangles: leaves.saturating_mul(crate::gpu::SURFACE_MESHLET_INDICES / 3),
        leaves: engine_core::Cut::of(leaves.saturating_add(refused), leaves),
        shallow_subdivisions: drawn.shallow_subdivisions,
        shadow_patches: drawn.shadow_patches,
        tiles: atlas.slots.len() as u32,
        tiles_dropped: engine_core::Cut::of(
            (atlas.slots.len() + atlas.unplaced.len()) as u32,
            atlas.slots.len() as u32,
        ),
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
        shading_records,
        surface_slots: engine_core::Cut::of(drawn.surface_wanted, drawn.surface_kept),
        trace_pool: part.pool,
        trace_turns: part.turns,
        sun_walks: engine_core::Cut::of(sun.walked, sun.walked.saturating_sub(sun.short)),
        sun_lost: sun.lost,
        lit,
        lighting,
        bounced,
        bouncing,
        colored,
        coloring,
        failed_jobs: bakes.failures.count,
        latest_job_failure: bakes.failures.last.as_ref().map(|(_, error)| error.clone()),
    };
    stats.set_if_neq(next);
}
