use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, AssetResidency, Cut, Detail, FrameTime, Joined, Spawner, View,
};
use engine_math::collections::HashMap;
use engine_places::ground::{
    Foliage, Occupancy, PlacedTiles, PooledTile, TerrainTiles, TileFace, TileKey,
};
use engine_places::shadows::ShadowChanges;
use engine_render::{DebugShown, ViewOutputs};
use glam::Vec3;

use crate::ScatterBrick;
use crate::far_cover::FarCoverTable;
use crate::groups::{Groups, Kept, Resolved, ScatterKey, resolve, scatter_key};
use crate::layouts::{CanopyPiece, GroundData, PlacedRecord, PoseData, SlotColor};
use crate::occupancy::{OccupancyWindows, canopy, texel_share};
use crate::pool::ScatterPool;
use crate::settings::{FoliageSettings, OccupancySettings, ScatterSettings, ScatterStats};

const SLOT_STRIDES: [u64; 4] = [
    size_of::<PoseData>() as u64,
    size_of::<SlotColor>() as u64,
    size_of::<PlacedRecord>() as u64,
    size_of::<GroundData>() as u64,
];

const SLOT_FIRST: [u64; 4] = [size_of::<PoseData>() as u64, 16, 32, 16];

#[derive(Resource)]
pub(crate) struct ScatterState {
    key: Option<ScatterKey>,
    pub(crate) groups: Groups,
    pub(crate) pool: ScatterPool,
    pub(crate) objects: u32,
    pub(crate) resident: u32,
    pub(crate) groups_revision: u64,
    pub(crate) slot_bytes: [u64; 4],
    pub(crate) dirty: Option<Range<u32>>,
    pub(crate) slots_revision: u64,
    pub(crate) moved: Vec<(Vec3, Vec3)>,
    pub(crate) restyled: bool,
    pub(crate) fade: f32,
    presented: f32,
    kept: Kept,
    changed: bool,
}

impl Default for ScatterState {
    fn default() -> Self {
        Self {
            key: None,
            groups: Groups::default(),
            pool: ScatterPool::default(),
            objects: 0,
            resident: 0,
            groups_revision: 0,
            slot_bytes: SLOT_FIRST,
            dirty: None,
            slots_revision: 0,
            moved: Vec::new(),
            restyled: false,
            fade: 1.0,
            presented: 0.0,
            kept: Kept::default(),
            changed: false,
        }
    }
}

#[derive(Component)]
pub(crate) struct ChangeMarker;

fn grow(held: &mut u64, bytes: u64) -> bool {
    let size = bytes.next_power_of_two();
    if bytes <= *held || size <= *held {
        return false;
    }
    *held = size;
    true
}

pub(crate) fn publish_foliage(settings: Res<FoliageSettings>, mut foliage: ResMut<Foliage>) {
    foliage.set_if_neq(Foliage {
        ground_blend: settings.ground_blend,
        crown_depth: settings.crown_depth,
    });
}

pub(crate) fn place_windows(
    (settings, foliage, view, outputs, shown): (
        Res<OccupancySettings>,
        Res<FoliageSettings>,
        Res<View>,
        Joined<ViewOutputs>,
        Res<DebugShown>,
    ),
    state: Res<ScatterState>,
    mut windows: ResMut<OccupancyWindows>,
    mut piece: ResMut<CanopyPiece>,
    mut occupancy: ResMut<Occupancy>,
) {
    let lit = windows.place(
        &settings,
        view.detail_eye(),
        (state.groups.covers.as_slice(), state.objects),
    );
    let temporal =
        shown.owner.is_none() && outputs.iter().any(|output| output.name == "motion");
    piece.set_if_neq(canopy(lit, &foliage, temporal));
    occupancy.set_if_neq(Occupancy {
        texel_share: texel_share(&settings),
    });
}

type ResolvedTerrains<'a> = Vec<(Entity, Vec<Option<Resolved<'a>>>)>;

fn resolve_all<'a>(
    tiles: &TerrainTiles,
    settings: &ScatterSettings,
    library: &'a AssetLibrary,
) -> (ResolvedTerrains<'a>, Kept) {
    let mut kept = Kept::default();
    let resolved = tiles
        .terrains
        .iter()
        .map(|terrain| {
            let assets = terrain
                .placement
                .iter()
                .map(|asset| resolve(asset, settings.look(asset.mesh), library, settings, &mut kept))
                .collect();
            (terrain.terrain, assets)
        })
        .collect();
    (resolved, kept)
}

pub(crate) fn place_pool(
    (tiles, settings, library, residency): (
        Res<TerrainTiles>,
        Res<ScatterSettings>,
        Res<AssetLibrary>,
        Res<AssetResidency>,
    ),
    (detail, time): (Res<Detail>, Res<FrameTime>),
    mut far: ResMut<FarCoverTable>,
    mut state: ResMut<ScatterState>,
) {
    let state = &mut *state;
    if let Some(slots) = state.dirty.take() {
        state.pool.uploaded(slots);
    }
    let (resolved, kept) = resolve_all(&tiles, &settings, &library);
    state.kept = kept;
    if tiles.is_changed() || settings.is_changed() || library.is_changed() {
        far.fill(&tiles.terrains, &resolved, &library);
    }
    let key = scatter_key(&residency, &resolved);
    let rebuilt = state.key.as_ref() != Some(&key) || state.pool.fragmented();
    if rebuilt {
        state.groups = Groups::build(&residency, &resolved);
        state.pool.clear(state.groups.groups.len());
        state.key = Some(key);
        state.changed = true;
    }
    if rebuilt || tiles.is_changed() {
        for terrain in tiles.terrains.iter() {
            let current: HashMap<TileKey, &TileFace> = terrain
                .tiles
                .iter()
                .filter(|tile| !tile.placed.is_empty())
                .map(|tile| (tile.key, tile))
                .collect();
            state.pool.retain(terrain.terrain, |key, held| {
                current.get(&key).is_some_and(|tile| {
                    tile.stamp == held.stamp || Arc::ptr_eq(&tile.placed, &held.placed)
                })
            });
            for tile in terrain.tiles.iter() {
                if tile.placed.is_empty() || state.pool.contains(terrain.terrain, tile.key) {
                    continue;
                }
                let objects = state.groups.tile_objects(terrain, tile);
                state.pool.insert(
                    (terrain.terrain, tile.key),
                    (Arc::clone(&tile.placed), tile.stamp),
                    objects,
                );
            }
        }
    }
    let (dirty, counts_changed) = state.pool.take_changes();
    state.moved.clear();
    state.moved.extend(state.pool.take_moved());
    let records = state.pool.records.len();
    let mut grown = false;
    for (held, stride) in state.slot_bytes.iter_mut().zip(SLOT_STRIDES) {
        grown |= grow(held, records as u64 * stride);
    }
    let range = if grown {
        Some(0..records as u32)
    } else {
        dirty.map(|range| range.start.min(records as u32)..range.end.min(records as u32))
    };
    state.dirty = range.filter(|range| !range.is_empty());
    if state.dirty.is_some() || grown {
        state.slots_revision += 1;
    }
    state.objects = records as u32;
    if counts_changed || std::mem::take(&mut state.changed) {
        let counts: Vec<u32> = state
            .groups
            .groups
            .iter()
            .map(|group| state.pool.counts[group.objects as usize])
            .collect();
        for (group, &count) in state.groups.groups.iter_mut().zip(&counts) {
            group.count = count;
        }
        state.resident = counts
            .iter()
            .zip(&state.groups.layouts)
            .filter(|(_, layout)| !layout.part)
            .map(|(&count, _)| count)
            .sum();
        state.groups_revision += 1;
    }
    let seconds = time.elapsed_seconds as f32;
    let since = seconds - std::mem::replace(&mut state.presented, seconds);
    state.fade = if detail.fade_seconds > 0.0 && since < 1.0 {
        engine_math::clamp(since / detail.fade_seconds, 0.0, 1.0)
    } else {
        1.0
    };
}

pub(crate) fn mark_moved(state: Res<ScatterState>, mut windows: ResMut<OccupancyWindows>) {
    windows.moved(&state.moved);
}

pub(crate) fn publish_tiles(
    state: Res<ScatterState>,
    mut placed: ResMut<PlacedTiles>,
    mut published: Local<Option<u64>>,
) {
    let revision = state.pool.revision();
    if *published == Some(revision) && !placed.is_changed() {
        return;
    }
    *published = Some(revision);
    let placed = &mut *placed;
    placed.tiles.clear();
    placed
        .tiles
        .extend(state.pool.tiles().into_iter().map(|((terrain, key), pooled)| {
            PooledTile {
                terrain,
                key,
                stamp: pooled.stamp,
                slots: pooled.slots.clone(),
                identity: pooled.placed.as_ptr() as usize as u64,
                impostors: Vec::new(),
            }
        }));
}

pub(crate) fn publish_changes(
    state: Res<ScatterState>,
    own: Query<Entity, With<ChangeMarker>>,
    mut spawner: Spawner<ScatterBrick>,
) {
    if state.moved.is_empty() && !state.restyled {
        return;
    }
    let change = ShadowChanges {
        boxes: state
            .moved
            .iter()
            .map(|&(low, high)| (low, high, f32::INFINITY))
            .collect(),
        all: state.restyled,
    };
    match own.single() {
        Ok(entity) => spawner.insert(entity, change),
        Err(_) => {
            spawner.spawn((ChangeMarker, change));
        }
    }
}

pub(crate) fn scatter_stats(
    state: Res<ScatterState>,
    windows: Res<OccupancyWindows>,
    mut stats: ResMut<ScatterStats>,
) {
    stats.set_if_neq(ScatterStats {
        placed_objects: state.resident,
        placed_far_objects: state.pool.far,
        placed_impostors: 0,
        dropped_objects: 0,
        impostors: Cut::of(0, 0),
        parts: Cut::of(state.kept.parts.0, state.kept.parts.1),
        lods: Cut::of(state.kept.lods.0, state.kept.lods.1),
        occupancy_redraws: windows.redraws,
        occupancy_averages: windows.averages,
    });
}
