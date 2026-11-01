use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::collections::HashMap;
use engine_core::{DeviceLimits, FrameTime, Jobs, Readback, Records, Spawner, Ticket, View, codec};
use engine_places::ground::{
    GroundTiles, HeldTile, Pyramid, TerrainEntry, TerrainTiles, TileFace, TileKey,
};
use engine_places::shadows::ShadowChanges;
use glam::{Vec2, Vec3};

use crate::bounce::TerrainBounce;
use crate::gpu::{HORIZON_RECORDS, HorizonsPiece};
use crate::jobs::{BOUNCE, BounceJob, LIGHTING, LightingJob};
use crate::lighting::TerrainLighting;
use crate::settings::{TerrainShading, TerrainShadingStats};
use crate::shading::TileShading;
use crate::{TerrainShadingBrick, terrain_bounce, terrain_horizons};

const RETRY_AFTER: u64 = 120;
const VIEW_PRIORITY: u32 = 0;
const SKIRT_SPACINGS: f32 = 4.0;
const MOST_PROBES: u32 = 512;
const FEWEST_PROBES: u32 = 16;

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
    fn left(&self) -> bool {
        self.asked.is_some() || self.made.is_none()
    }

    fn drop_asked(&mut self, jobs: &mut Jobs<'_, '_>) {
        if let Some((ticket, _)) = self.asked.take() {
            jobs.cancel(ticket);
        }
    }

    fn take(&mut self, records: &Records, frame: u64) {
        let Some(arrived) = self.asked.as_ref().and_then(|(ticket, _)| records.get(ticket)) else {
            return;
        };
        let Some((_, input)) = self.asked.take() else {
            return;
        };
        match arrived.and_then(|bytes| codec::decode::<T>(&bytes)) {
            Ok(made) => {
                self.made = Some(Arc::new(made));
                self.from = Some(input);
                self.failed = None;
            }
            Err(_) => self.failed = Some(frame),
        }
    }

    fn ask(&mut self, jobs: &mut Jobs<'_, '_>, kind: &'static engine_core::JobKind, input: Vec<u8>, frame: u64) {
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
pub(crate) struct TerrainBakes {
    pub(crate) lighting: Bake<TerrainLighting>,
    pub(crate) bounce: Bake<TerrainBounce>,
    pub(crate) shadings: HashMap<TileKey, Arc<TileShading>>,
}

#[derive(Resource, Default)]
pub(crate) struct Bakes {
    pub(crate) terrains: BTreeMap<Entity, TerrainBakes>,
}

impl Bakes {
    pub(crate) fn of(&self, terrain: Entity) -> Option<&TerrainBakes> {
        self.terrains.get(&terrain)
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
        self.arrivals.remove(&id);
        self.surfaced.remove(&id);
        self.holed.remove(&id);
    }

    fn place(&mut self, id: TileId, tile: &TileFace, replaced: Option<u32>, surfaces: bool) -> bool {
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
        if arriving {
            self.arrivals.insert(id, self.now);
            if tile.control.is_some() || surfaces {
                self.surfaced.insert(id, self.now);
            }
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
}

pub(crate) fn place_tiles(
    tiles: Res<TerrainTiles>,
    limits: Res<DeviceLimits>,
    time: Res<FrameTime>,
    mut atlas: ResMut<Atlas>,
    mut ground: ResMut<GroundTiles>,
) {
    let atlas = &mut *atlas;
    atlas.now = time.elapsed_seconds as f32;
    atlas.uploads.clear();
    atlas.vacated.clear();
    atlas.changed.clear();
    atlas.ground_changes.clear();
    if !tiles.is_changed() && atlas.unplaced.is_empty() {
        return;
    }
    atlas.capacity = limits.texture_layers;
    atlas.dropped = 0;
    let listed = |terrain: &Entity| tiles.terrains.iter().any(|entry| entry.terrain == *terrain);
    let mut left: Vec<TileId> = atlas
        .held
        .keys()
        .filter(|(terrain, key)| {
            !listed(terrain)
                || tiles
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
                Some(held) if held.stamp == tile.stamp => {}
                Some(held) => {
                    let same_ground = held.heights == tile.heights && held.holes == tile.holes;
                    let layer = atlas.slots.get(&id).copied();
                    if same_ground {
                        let watered = held.errors != tile.errors;
                        atlas.held.insert(id, tile.clone());
                        if let (true, Some(layer)) = (watered, layer) {
                            atlas.uploads.push((id, layer, Upload::Errors));
                        }
                        continue;
                    }
                    atlas.ground_changes.push(ground_box(entry, held));
                    atlas.ground_changes.push(ground_box(entry, tile));
                    atlas.slots.remove(&id);
                    missing.push((id, layer));
                }
            }
        }
    }
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
        let surfaces = entry.surface_program && !TileShading::shades(id.1.level, entry.surroundings);
        atlas.place(id, tile, replaced, surfaces);
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
    (
        terrain.origin + Vec3::new(corner.x, low - skirt, corner.y),
        terrain.origin + Vec3::new(corner.x + size, high, corner.y + size),
        spacing,
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

fn ground_tiles(entry: &TerrainEntry, level: u32) -> Vec<(TileKey, Vec<engine_heightfield::Stamp>)> {
    let pyramid = entry.pyramid;
    let cut = |key: TileKey| {
        let size = pyramid.tile_size(u32::from(key.level));
        let low = Vec2::new(f32::from(key.x), f32::from(key.z)) * size;
        entry.cutout.is_some_and(|(first, last)| {
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
            let stamps = listed_tile(&entry.tiles, held).map_or_else(Vec::new, |tile| tile.stamps.to_vec());
            (held, stamps)
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
        }
    }
    for entry in tiles.terrains.iter() {
        let held = bakes.terrains.entry(entry.terrain).or_default();
        let Some(recipe) = entry.recipe else {
            continue;
        };
        if !settings.lit() {
            held.lighting.drop_asked(&mut jobs);
            held.lighting = Bake::default();
        } else {
            held.lighting.take(&records, frame);
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
        held.bounce.take(&records, frame);
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
            (limits.texture_layers, limits.storage_bytes),
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

#[derive(Resource, Default)]
pub(crate) struct SurfaceTotals {
    pub(crate) grounds: [u32; 2],
    pub(crate) waters: [u32; 2],
}

impl Readback for SurfaceTotals {
    fn arrived(&mut self, _: u64, waters: u32, bytes: &[u8]) {
        let words: &[u32] = bytemuck::cast_slice(bytes);
        let mut sums = [0_u32; 4];
        let args = crate::gpu::SUBDIVISION_ARGS as usize;
        for (slot, counters) in words.chunks_exact(args).enumerate() {
            let sum = if (waters >> slot) & 1 == 0 { 0 } else { 2 };
            sums[sum] = sums[sum].saturating_add(counters[crate::gpu::SUBDIVISION_LEAVES as usize]);
            sums[sum + 1] = sums[sum + 1].saturating_add(counters[crate::gpu::SUBDIVISION_REFUSED as usize]);
        }
        self.grounds = [sums[0], sums[1]];
        self.waters = [sums[2], sums[3]];
    }
}

pub(crate) fn count(
    atlas: Res<Atlas>,
    bakes: Res<Bakes>,
    totals: Res<SurfaceTotals>,
    mut stats: ResMut<TerrainShadingStats>,
) {
    let [leaves, refused] = totals.grounds;
    let lit = bakes.terrains.values().filter(|held| held.lighting.made.is_some()).count() as u32;
    let lighting = bakes.terrains.values().filter(|held| held.lighting.left()).count() as u32;
    let bounced = bakes.terrains.values().filter(|held| held.bounce.made.is_some()).count() as u32;
    let bouncing = bakes.terrains.values().filter(|held| held.bounce.asked.is_some()).count() as u32;
    let next = TerrainShadingStats {
        pending: lighting + bouncing,
        triangles: leaves.saturating_mul(crate::gpu::SURFACE_MESHLET_INDICES / 3),
        leaves: engine_core::Cut::of(leaves.saturating_add(refused), leaves),
        tiles: atlas.slots.len() as u32,
        tiles_dropped: engine_core::Cut::of(atlas.slots.len() as u32 + atlas.dropped, atlas.slots.len() as u32),
        lit,
        lighting,
        bounced,
        bouncing,
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
    for held in bakes.terrains.values_mut() {
        held.lighting.drop_asked(&mut jobs);
        held.bounce.drop_asked(&mut jobs);
    }
    bakes.terrains.clear();
}
