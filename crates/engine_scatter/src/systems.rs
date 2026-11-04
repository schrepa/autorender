use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, AssetResidency, AssetUses, Cut, Detail, DeviceLimits, FrameTime, ImpostorAsks,
    ImpostorPolicy, ImpostorRecord, ImpostorRecords, MaterialHandle, MeshHandle, Readback,
    Spawner, TextureHandle, View,
};
use engine_math::collections::HashMap;
use engine_places::ground::{
    Foliage, Occupancy, PlacedTiles, PooledTile, TerrainTiles, TileFace, TileKey,
};
use engine_places::shadows::{CasterMap, Refresh, ShadowChanges, ShadowMaps};
use glam::Vec3;

use crate::ScatterBrick;
use crate::far_cover::FarCoverTable;
use crate::groups::{Groups, Kept, Resolved, resolve};
use crate::layouts::{
    CanopyPiece, CullFrame, CullGroup, CullView, DrawArgs, GroundData, LEVELS, PlacedRecord,
    PoseData, SlotColor, draw_args,
};
use crate::occupancy::{OccupancyWindows, canopy, texel_share};
use crate::pool::{Pooled, ScatterPool};
use crate::settings::{FoliageSettings, OccupancySettings, ScatterSettings, ScatterStats};

const FADE_MARGIN: f32 = 0.1;

const SLOT_STRIDES: [u64; 4] = [
    size_of::<PoseData>() as u64,
    size_of::<SlotColor>() as u64,
    size_of::<PlacedRecord>() as u64,
    size_of::<GroundData>() as u64,
];

const SLOT_FIRST: [u64; 4] = [size_of::<PoseData>() as u64, 16, 32, 16];

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sizes {
    pub(crate) groups: u64,
    pub(crate) views: u64,
    pub(crate) ids: u64,
    pub(crate) draws: u64,
    pub(crate) cards: u64,
    pub(crate) card_args: u64,
}

impl Default for Sizes {
    fn default() -> Self {
        Self {
            groups: 32,
            views: size_of::<CullView>() as u64,
            ids: 16,
            draws: 20,
            cards: 8,
            card_args: 16,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct GroupLists {
    pub(crate) ids: u32,
    pub(crate) count: u32,
}

#[derive(Default)]
pub(crate) struct CullPlan {
    pub(crate) prepared: bool,
    pub(crate) views: Vec<CullView>,
    pub(crate) maps: Vec<u32>,
    pub(crate) fading_view: usize,
    pub(crate) frame: Option<CullFrame>,
    pub(crate) card_args: Vec<[u32; 4]>,
    pub(crate) args: Vec<DrawArgs>,
    pub(crate) lists_left: bool,
    pub(crate) casts: bool,
    pub(crate) cards: bool,
    pub(crate) atlas: bool,
}

#[derive(Default)]
pub(crate) struct GroupData {
    pub(crate) data: Vec<CullGroup>,
    pub(crate) template: Vec<DrawArgs>,
    pub(crate) casters: Vec<DrawArgs>,
    pub(crate) lists: Vec<GroupLists>,
    pub(crate) ids_per_view: u32,
}

#[derive(Resource)]
pub(crate) struct ScatterState {
    pub(crate) groups: Groups,
    pub(crate) pool: ScatterPool,
    pub(crate) objects: u32,
    pub(crate) resident: u32,
    pub(crate) group_data: GroupData,
    pub(crate) slot_bytes: [u64; 4],
    pub(crate) sizes: Sizes,
    pub(crate) room: u64,
    pub(crate) generation: u64,
    pub(crate) slot_sequence: u64,
    pub(crate) whole: bool,
    pub(crate) left_objects: u32,
    pub(crate) moved: Vec<(Vec3, Vec3)>,
    pub(crate) restyled: bool,
    pub(crate) impostors: Vec<ImpostorRecord>,
    pub(crate) impostor_live: u32,
    pub(crate) impostor_refused: u32,
    pub(crate) uses: AssetUses,
    pub(crate) fade: f32,
    pub(crate) plan: CullPlan,
    presented: f32,
    kept: Kept,
    retake: bool,
    refilled: bool,
    covers_changed: bool,
}

impl Default for ScatterState {
    fn default() -> Self {
        Self {
            groups: Groups::default(),
            pool: ScatterPool::default(),
            objects: 0,
            resident: 0,
            group_data: GroupData::default(),
            slot_bytes: SLOT_FIRST,
            sizes: Sizes::default(),
            room: u64::MAX,
            generation: 0,
            slot_sequence: 0,
            whole: false,
            left_objects: 0,
            moved: Vec::new(),
            restyled: false,
            impostors: Vec::new(),
            impostor_live: 0,
            impostor_refused: 0,
            uses: AssetUses::default(),
            fade: 1.0,
            plan: CullPlan::default(),
            presented: 0.0,
            kept: Kept::default(),
            retake: false,
            refilled: false,
            covers_changed: false,
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct ScatterFeedback {
    entries: u32,
}

impl Readback for ScatterFeedback {
    fn arrived(&mut self, _frame: u64, _tag: u32, bytes: &[u8]) {
        self.entries = bytes
            .get(4..8)
            .map_or(0, |word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]));
    }
}

#[derive(Component)]
pub(crate) struct Placements;

fn grow(held: &mut u64, bytes: u64, room: u64) -> bool {
    let size = bytes.next_power_of_two().min(room);
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

pub(crate) fn take_impostors(
    records: Query<Ref<ImpostorRecords>, With<Placements>>,
    mut state: ResMut<ScatterState>,
) {
    let state = &mut *state;
    let held = records.single().ok();
    let arrived = held.as_ref().is_some_and(|held| held.is_changed());
    state.restyled = false;
    if std::mem::take(&mut state.retake) || arrived {
        let fresh = held
            .as_ref()
            .map_or(&[][..], |held| held.records.as_slice());
        let records: Vec<ImpostorRecord> = (0..state.groups.groups.len())
            .map(|index| fresh.get(index).copied().unwrap_or_default())
            .collect();
        state.groups.take_impostors(&records);
        state.restyled = records != state.impostors;
        state.impostors = records;
    }
    state.impostor_live = held.as_ref().map_or(0, |held| held.live);
    state.impostor_refused = held.as_ref().map_or(0, |held| held.refused);
}

pub(crate) fn place_windows(
    (settings, foliage, view): (Res<OccupancySettings>, Res<FoliageSettings>, Res<View>),
    state: Res<ScatterState>,
    mut windows: ResMut<OccupancyWindows>,
    mut piece: ResMut<CanopyPiece>,
    mut occupancy: ResMut<Occupancy>,
) {
    if state.covers_changed {
        windows.redraw_all();
    }
    let lit = windows.place(
        &settings,
        view.detail_eye(),
        (state.groups.covers.as_slice(), state.objects),
    );
    piece.set_if_neq(canopy(lit, &foliage, view.temporal()));
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
                .map(|asset| {
                    resolve(asset, settings.look(asset.mesh), library, settings, &mut kept)
                })
                .collect();
            (terrain.terrain, assets)
        })
        .collect();
    (resolved, kept)
}

fn asset_uses(resolved: &ResolvedTerrains<'_>) -> AssetUses {
    let mut meshes: Vec<MeshHandle> = Vec::new();
    let mut materials: Vec<MaterialHandle> = Vec::new();
    let mut textures: Vec<TextureHandle> = Vec::new();
    for asset in resolved.iter().flat_map(|(_, assets)| assets.iter().flatten()) {
        for mesh in asset.meshes() {
            if !meshes.contains(&mesh.handle) {
                meshes.push(mesh.handle);
            }
        }
        for material in asset.materials() {
            if !materials.contains(&material) {
                materials.push(material);
            }
        }
        for &texture in &asset.textures {
            if !textures.contains(&texture) {
                textures.push(texture);
            }
        }
    }
    AssetUses {
        meshes,
        materials,
        textures,
    }
}

fn set_groups(state: &mut ScatterState) {
    let layouts = &state.groups.layouts;
    let held = &mut state.group_data;
    held.data.clear();
    held.template.clear();
    held.casters.clear();
    held.lists.clear();
    let mut ids = 0;
    for (layout, group) in layouts.iter().zip(&state.groups.groups) {
        let count = group.count;
        held.data.push(CullGroup {
            ids,
            count,
            levels: layout.levels.len() as u32,
            camera: layout.camera,
            errors: layout.errors,
            impostor: layout.impostor.0,
            record: layout.impostor.1,
            casters: layout.casters.len() as u32,
            parts: layout.parts,
        });
        held.lists.push(GroupLists { ids, count });
        for level in 0..LEVELS as usize {
            held.template.push(draw_args(
                layout.levels.get(level).copied().unwrap_or((0, 0, 0)),
            ));
            held.casters.push(draw_args(
                layout.casters.get(level).copied().unwrap_or((0, 0, 0)),
            ));
        }
        ids += count * LEVELS;
    }
    held.ids_per_view = ids;
    if grow(
        &mut state.sizes.groups,
        (held.data.len() * size_of::<CullGroup>()) as u64,
        state.room,
    ) {
        state.generation += 1;
    }
    state.resident = state
        .groups
        .groups
        .iter()
        .zip(&state.groups.layouts)
        .filter(|(_, layout)| !layout.part)
        .map(|(group, _)| group.count)
        .sum();
}

fn write_slots(state: &mut ScatterState) {
    let end = state.pool.end;
    let room = SLOT_STRIDES
        .iter()
        .map(|stride| state.room / stride)
        .min()
        .unwrap_or(0);
    let held = u64::from(end).min(room) as u32;
    state.left_objects = end - held;
    let mut grown = false;
    for (bytes, stride) in state.slot_bytes.iter_mut().zip(SLOT_STRIDES) {
        grown |= grow(bytes, u64::from(held) * stride, state.room);
    }
    if grown {
        state.generation += 1;
    }
    state.whole = grown;
    state.slot_sequence += 1;
    state.objects = held;
}

pub(crate) fn place_pool(
    (tiles, settings, library, residency, limits): (
        Res<TerrainTiles>,
        Res<ScatterSettings>,
        Res<AssetLibrary>,
        Res<AssetResidency>,
        Res<DeviceLimits>,
    ),
    (detail, time): (Res<Detail>, Res<FrameTime>),
    mut far: ResMut<FarCoverTable>,
    mut state: ResMut<ScatterState>,
) {
    let state = &mut *state;
    state.room = match limits.storage_room {
        0 => u64::MAX,
        room => room,
    } / 16
        * 16;
    state.pool.settle();
    let placing = tiles
        .terrains
        .iter()
        .any(|terrain| !terrain.placement.is_empty());
    if !placing && (state.slot_bytes != SLOT_FIRST || state.sizes != Sizes::default()) {
        state.slot_bytes = SLOT_FIRST;
        state.sizes = Sizes::default();
        state.generation += 1;
    }
    let (resolved, kept) = resolve_all(&tiles, &settings, &library);
    state.kept = kept;
    if tiles.is_changed() || settings.is_changed() || library.is_changed() {
        far.fill(&tiles.terrains, &resolved, &library);
    }
    state.uses = asset_uses(&resolved);
    let mut groups = Groups::build(&residency, &resolved);
    let refill = !state.groups.same_objects(&groups);
    if !refill {
        for (layout, before) in groups.layouts.iter_mut().zip(&state.groups.layouts) {
            layout.impostor = before.impostor;
        }
    }
    state.covers_changed = state.groups.covers != groups.covers;
    state.groups = groups;
    state.refilled = refill;
    if refill {
        state.impostors = vec![ImpostorRecord::default(); state.groups.groups.len()];
        state.retake = true;
        state.pool.clear(state.groups.groups.len());
    }
    if refill || tiles.is_changed() {
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
                let objects =
                    state
                        .groups
                        .tile_objects(terrain.terrain, terrain.origin, &tile.placed);
                state.pool.insert(
                    (terrain.terrain, tile.key),
                    (Arc::clone(&tile.placed), tile.stamp, terrain.origin),
                    objects,
                );
            }
        }
    }
    state.pool.compact(&state.groups);
    state.moved.clear();
    state.moved.extend(state.pool.take_moved());
    write_slots(state);
    for group in &mut state.groups.groups {
        group.count = state.pool.counts[group.objects as usize];
    }
    set_groups(state);
    let seconds = time.elapsed_seconds as f32;
    let since = seconds - std::mem::replace(&mut state.presented, seconds);
    state.fade = if detail.fade_seconds > 0.0 && since < 1.0 {
        engine_math::clamp(since / detail.fade_seconds, 0.0, 1.0)
    } else {
        1.0
    };
}

pub(crate) fn plan_cull(
    (view, detail, maps, policy): (Res<View>, Res<Detail>, Res<ShadowMaps>, Res<ImpostorPolicy>),
    mut state: ResMut<ScatterState>,
    mut feedback: ResMut<ScatterFeedback>,
) {
    let state = &mut *state;
    let plan = &mut state.plan;
    plan.maps.clear();
    plan.prepared = false;
    plan.atlas = state
        .impostors
        .iter()
        .any(|record| record.sphere[3] > 0.0);
    plan.cards = plan.atlas;
    plan.casts = maps.on;
    if state.resident == 0 {
        return;
    }
    let view_projection = view.view_projection();
    let eye = view.detail_eye();
    let height = view.display_size().y as f32;
    let pixel_error = detail.enabled.then_some(detail.pixel_error);
    let impostors = plan.cards.then_some(policy.pixels);
    let y_row = view_projection.row(1).truncate().length();
    let spanned = y_row * height * 0.5;
    let pixels = pixel_error.map_or(0.0, |error| spanned / error);
    let cards = impostors.map_or(0.0, |size| y_row * height / size);
    plan.views.clear();
    plan.views.push(CullView {
        planes: engine_render::planes(view_projection),
        limits: [0.0; 4],
    });
    for (index, region) in maps.regions.iter().enumerate() {
        let (CasterMap::Region, Refresh::Kept(kept)) = (region.map, region.refresh) else {
            continue;
        };
        plan.views.push(CullView {
            planes: kept.planes,
            limits: [0.5 * kept.texel, kept.texel, 0.0, 0.0],
        });
        plan.maps.push(index as u32);
    }
    let view_count = plan.views.len() as u32;
    let group_count = state.groups.layouts.len() as u32;
    let draws_per_view = group_count * LEVELS;
    plan.fading_view = plan.views.len();
    let lists = view_count + 1;
    let list_count = u64::from(lists);
    let ids = u64::from(state.group_data.ids_per_view) * list_count * 4;
    let draws = u64::from(draws_per_view) * list_count * size_of::<DrawArgs>() as u64;
    let cards_bytes = u64::from(state.objects) * u64::from(view_count) * 8;
    let sizes = &mut state.sizes;
    let room = state.room;
    let mut replaced = grow(
        &mut sizes.views,
        u64::from(view_count) * size_of::<CullView>() as u64,
        room,
    );
    replaced |= grow(&mut sizes.ids, ids, room);
    replaced |= grow(&mut sizes.draws, draws, room);
    replaced |= grow(&mut sizes.cards, cards_bytes, room);
    replaced |= grow(&mut sizes.card_args, list_count * 16, room);
    if replaced {
        state.generation += 1;
    }
    plan.lists_left = ids > room || cards_bytes > room;
    if !plan.lists_left {
        feedback.entries = 0;
    }
    plan.frame = Some(CullFrame {
        eye: eye.extend(pixels).to_array(),
        counts: [state.objects, group_count, LEVELS, view_count],
        strides: [
            state.group_data.ids_per_view,
            draws_per_view,
            u32::from(impostors.is_some()),
            state.objects,
        ],
        fade: [state.fade, FADE_MARGIN, cards, spanned],
    });
    plan.card_args.clear();
    plan.card_args
        .resize(view_count as usize, [engine_render::IMPOSTOR_VERTICES, 0, 0, 0]);
    plan.card_args.push([0; 4]);
    plan.args.clear();
    for list in 0..lists {
        let shadow = list > 0 && list < view_count;
        plan.args.extend_from_slice(if shadow {
            &state.group_data.casters
        } else {
            &state.group_data.template
        });
    }
    plan.prepared = true;
}

pub(crate) fn publish_assets(
    state: Res<ScatterState>,
    own: Query<(Entity, &AssetUses, &ImpostorAsks), With<Placements>>,
    mut spawner: Spawner<ScatterBrick>,
) {
    let asks = ImpostorAsks {
        hemisphere: true,
        asks: state.groups.asks.clone(),
    };
    match own.single() {
        Ok((entity, uses, held)) => {
            if *uses != state.uses {
                spawner.insert(entity, state.uses.clone());
            }
            if *held != asks {
                spawner.insert(entity, asks);
            }
        }
        Err(_) => {
            spawner.spawn((Placements, state.uses.clone(), asks));
        }
    }
}

pub(crate) fn mark_moved(state: Res<ScatterState>, mut windows: ResMut<OccupancyWindows>) {
    windows.moved(&state.moved);
}

fn same_tile(tile: &PooledTile, pooled: &Pooled) -> bool {
    tile.stamp == pooled.stamp && tile.slots == pooled.slots && tile.identity == pooled.identity()
}

pub(crate) fn publish_tiles(state: Res<ScatterState>, mut placed: ResMut<PlacedTiles>) {
    let boxes = state
        .impostors
        .iter()
        .map(|record| (record.sphere, record.extent));
    if !placed.groups.iter().copied().eq(boxes.clone()) {
        placed.groups = boxes.collect();
    }
    let tiles = state.pool.tiles();
    let stale = state.restyled || state.refilled;
    let current = !stale
        && placed.tiles.len() == tiles.len()
        && placed
            .tiles
            .iter()
            .zip(&tiles)
            .all(|(tile, &((terrain, key), pooled))| {
                tile.terrain == terrain && tile.key == key && same_tile(tile, pooled)
            });
    if current {
        return;
    }
    let placed = &mut *placed;
    let mut before = std::mem::take(&mut placed.tiles);
    placed
        .tiles
        .extend(tiles.into_iter().map(|((terrain, key), pooled)| {
            let kept = before
                .binary_search_by_key(&(terrain, key), |tile| (tile.terrain, tile.key))
                .ok()
                .map(|at| &mut before[at])
                .filter(|tile| !stale && same_tile(tile, pooled))
                .map(|tile| std::mem::take(&mut tile.impostors));
            PooledTile {
                terrain,
                key,
                stamp: pooled.stamp,
                slots: pooled.slots.clone(),
                identity: pooled.identity(),
                impostors: kept.unwrap_or_else(|| {
                    state.groups.tile_boxes(
                        terrain,
                        pooled.origin,
                        &pooled.placed,
                        &state.impostors,
                    )
                }),
            }
        }));
}

pub(crate) fn publish_changes(
    state: Res<ScatterState>,
    own: Query<Entity, With<Placements>>,
    mut spawner: Spawner<ScatterBrick>,
) {
    if state.moved.is_empty() && !state.restyled {
        return;
    }
    let Ok(entity) = own.single() else {
        return;
    };
    spawner.insert(
        entity,
        ShadowChanges {
            boxes: state
                .moved
                .iter()
                .map(|&(low, high)| (low, high, f32::INFINITY))
                .collect(),
            all: state.restyled,
            casters: false,
        },
    );
}

pub(crate) fn scatter_stats(
    (state, windows, feedback): (
        Res<ScatterState>,
        Res<OccupancyWindows>,
        Res<ScatterFeedback>,
    ),
    mut stats: ResMut<ScatterStats>,
) {
    let entries = if state.plan.lists_left {
        feedback.entries
    } else {
        0
    };
    let placed = state.impostor_live;
    stats.set_if_neq(ScatterStats {
        placed_objects: state.resident,
        placed_far_objects: state.pool.far,
        placed_impostors: placed,
        dropped_objects: state.left_objects.saturating_add(entries),
        impostors: Cut::of(placed + state.impostor_refused, placed),
        parts: Cut::of(state.kept.parts.0, state.kept.parts.1),
        lods: Cut::of(state.kept.lods.0, state.kept.lods.1),
        occupancy_redraws: windows.redraws,
        occupancy_averages: windows.averages,
    });
}
