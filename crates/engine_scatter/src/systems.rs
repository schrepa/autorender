use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{
    AUTHORED_LEVELS, Asset, AssetLibrary, AssetResidency, AssetUses, Background, Cut, Detail,
    DeviceLimits, Drawn, DrawnArea, DrawnAreas, GeometryHandle, Jobs, Readable, Readback, Records,
    Spawner, Transform, VIEW_JOBS, Views, authored_input, job_priority,
};
use engine_math::collections::HashMap;
use engine_open_world_kit::ground::{
    Foliage, GroundTiles, PlacedAsset, PlacedTiles, PooledTile, TerrainEntry, TerrainTiles,
    TileFace,
};
use engine_places_rendering::impostors::{ImpostorAsks, ImpostorRecord, ImpostorRecords};
use engine_tiles::{Pyramid, TileKey};
use glam::Vec3;

use crate::ScatterBrick;
use crate::pool::Reach;
use crate::snapping::widened;
use crate::far_cover::{CanopyMaps, wanted};
use crate::gpu::PLACED;
use crate::groups::{Groups, Kept, Resolved, resolve};
use crate::layouts::{
    CanopyPiece, CullFrame, CullGroup, CullView, DrawArgs, LEVELS, PlacedRecord, PlacedValues,
    PoseData, SlotColor, draw_args,
};
use crate::pool::{Pooled, ScatterPool};
use crate::settings::{
    CanopySettings, FoliageSettings, PlacedLook, ScatterSettings, ScatterStats,
};

const SLOT_STRIDES: [u64; 4] = [
    size_of::<PoseData>() as u64,
    size_of::<SlotColor>() as u64,
    size_of::<PlacedRecord>() as u64,
    size_of::<PlacedValues>() as u64,
];

const SLOT_FIRST: [u64; 4] = [size_of::<PoseData>() as u64, 16, 32, 32];

const GROUPS_FIRST: u64 = 32;

#[derive(Clone, Copy, Default)]
pub(crate) struct GroupLists {
    pub(crate) ids: u32,
    pub(crate) count: u32,
}

#[derive(Default)]
pub(crate) struct CullPlan {
    pub(crate) prepared: bool,
    pub(crate) main: Option<CullView>,
    pub(crate) frame: Option<CullFrame>,
    pub(crate) cards: bool,
    pub(crate) atlas: bool,
}

pub(crate) struct Authored {
    meshes: Vec<GeometryHandle>,
    work: Background<Vec<f32>>,
}

#[derive(Default)]
pub(crate) struct GroupData {
    pub(crate) data: Vec<CullGroup>,
    pub(crate) template: Vec<DrawArgs>,
    pub(crate) lists: Vec<GroupLists>,
    pub(crate) ids_per_view: u32,
    pub(crate) first_instance: bool,
}

#[derive(Resource)]
pub(crate) struct ScatterState {
    pub(crate) groups: Groups,
    pub(crate) pool: ScatterPool,
    pub(crate) objects: u32,
    pub(crate) resident: u32,
    pub(crate) group_data: GroupData,
    pub(crate) slot_bytes: [u64; 4],
    pub(crate) group_bytes: u64,
    pub(crate) room: u64,
    pub(crate) generation: u64,
    pub(crate) slot_sequence: u64,
    pub(crate) library: (u64, u64),
    pub(crate) left_objects: u32,
    pub(crate) moved: Vec<(Entity, Reach)>,
    pub(crate) restyled: bool,
    pub(crate) impostors: Vec<ImpostorRecord>,
    pub(crate) impostor_generation: u64,
    pub(crate) impostor_live: u32,
    pub(crate) impostor_refused: u32,
    pub(crate) uses: AssetUses,
    pub(crate) plan: CullPlan,
    authored: Vec<Authored>,
    kept: Kept,
    retake: bool,
    refilled: bool,
    revisions: Option<((u64, u64), u64)>,
    placing: Vec<Placing>,
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
            group_bytes: GROUPS_FIRST,
            room: u64::MAX,
            generation: 0,
            slot_sequence: 0,
            library: (0, 0),
            left_objects: 0,
            moved: Vec::new(),
            restyled: false,
            impostors: Vec::new(),
            impostor_generation: 0,
            impostor_live: 0,
            impostor_refused: 0,
            uses: AssetUses::default(),
            plan: CullPlan::default(),
            authored: Vec::new(),
            kept: Kept::default(),
            retake: false,
            refilled: false,
            revisions: None,
            placing: Vec::new(),
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct ScatterFeedback {
    entries: u32,
}

impl Readable for ScatterFeedback {
    fn arrived(&mut self, _tag: u32, read: Readback<&[u8]>) {
        self.entries = read.value.get(4..8).map_or(0, |word| {
            u32::from_le_bytes([word[0], word[1], word[2], word[3]])
        });
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
    });
}

pub(crate) fn take_impostors(
    records: Query<Ref<ImpostorRecords>, With<Placements>>,
    mut state: ResMut<ScatterState>,
) {
    let state = &mut *state;
    let held = records
        .single()
        .ok()
        .filter(|held| held.generation == state.impostor_generation);
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

pub(crate) fn publish_canopy(
    (foliage, canopy, views): (Res<FoliageSettings>, Res<CanopySettings>, Res<Views>),
    maps: Res<CanopyMaps>,
    mut piece: ResMut<CanopyPiece>,
) {
    let Some(main) = views.main() else {
        return;
    };
    let cast = views
        .drawn()
        .iter()
        .filter(|view| view.owner.is_some() && view.detail > 0.0)
        .filter(|view| view.view_projection.row(3).truncate() == Vec3::ZERO)
        .map(|view| 2.0 * view.detail)
        .fold(f32::INFINITY, engine_math::min);
    piece.set_if_neq(CanopyPiece {
        handover: [
            canopy.handover_pixels,
            if cast.is_finite() { cast } else { 0.0 },
            maps.count() as f32,
            canopy.relief,
        ],
        light: [
            canopy.sky_through,
            canopy.sun_through,
            canopy.crown_light,
            canopy.roughness,
        ],
        foliage: [
            foliage.transmission,
            foliage.variation,
            if foliage.hashed_cutoff && main.jitter.is_some() {
                1.0
            } else {
                0.0
            },
            foliage.ground_blend,
        ],
    });
}

type ResolvedTerrains<'a> = Vec<(Entity, Vec<Option<Resolved<'a>>>)>;

#[derive(PartialEq)]
struct Placing {
    terrain: Entity,
    origin: Vec3,
    pyramid: Pyramid,
    recipe: Option<[u8; 32]>,
    surroundings: bool,
    placement: Arc<[PlacedAsset]>,
}

impl Placing {
    fn of(terrain: &TerrainEntry) -> Self {
        Self {
            terrain: terrain.terrain,
            origin: terrain.origin,
            pyramid: terrain.pyramid,
            recipe: terrain.recipe,
            surroundings: terrain.surroundings,
            placement: Arc::clone(&terrain.placement),
        }
    }
}

fn same_placing(held: &[Placing], tiles: &TerrainTiles) -> bool {
    held.len() == tiles.terrains.len()
        && held
            .iter()
            .zip(tiles.terrains.iter())
            .all(|(held, terrain)| *held == Placing::of(terrain))
}

fn look_meshes(look: &PlacedLook) -> Vec<GeometryHandle> {
    std::iter::once(look.mesh)
        .chain(look.lods.iter().copied())
        .collect()
}

fn measured<'a>(authored: &'a [Authored], look: &PlacedLook) -> &'a [f32] {
    let meshes = look_meshes(look);
    authored
        .iter()
        .find(|held| held.meshes == meshes)
        .and_then(|held| held.work.ready())
        .map_or(&[], Vec::as_slice)
}

fn measure_looks(
    authored: &mut Vec<Authored>,
    (settings, library): (&ScatterSettings, &AssetLibrary),
    (jobs, records): (&mut Jobs<'_, '_>, &Records),
) -> bool {
    authored.retain(|held| {
        settings
            .looks
            .iter()
            .any(|look| look_meshes(look) == held.meshes)
    });
    let mut landed = false;
    for held in authored.iter_mut().filter(|held| held.work.waiting()) {
        held.work.poll(records);
        landed |= !held.work.waiting();
    }
    for look in settings.looks.iter().filter(|look| !look.lods.is_empty()) {
        let meshes = look_meshes(look);
        if authored.iter().any(|held| held.meshes == meshes) {
            continue;
        }
        let Some(found) = meshes
            .iter()
            .map(|&handle| library.mesh(handle).map(|mesh| &**mesh))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let work = jobs.fetch(
            &AUTHORED_LEVELS,
            &authored_input(&found),
            job_priority(VIEW_JOBS, 0),
            |bytes| {
                Ok(bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&word| f32::from_le_bytes(word))
                    .collect())
            },
        );
        authored.push(Authored { meshes, work });
    }
    landed
}

fn resolve_all<'a>(
    tiles: &TerrainTiles,
    (settings, authored): (&ScatterSettings, &[Authored]),
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
                    let look = settings.look(asset.mesh);
                    let errors = look.map_or(&[][..], |look| measured(authored, look));
                    resolve(
                        asset,
                        (look, errors),
                        library,
                        settings,
                        &mut kept,
                    )
                })
                .collect();
            (terrain.terrain, assets)
        })
        .collect();
    (resolved, kept)
}

fn asset_uses(resolved: &ResolvedTerrains<'_>) -> Vec<Asset> {
    let mut wanted: Vec<Asset> = Vec::new();
    for asset in resolved
        .iter()
        .flat_map(|(_, assets)| assets.iter().flatten())
    {
        wanted.extend(asset.meshes().map(|mesh| Asset::Mesh(mesh.handle)));
        wanted.extend(asset.materials().map(Asset::Material));
        wanted.extend(asset.textures.iter().copied().map(Asset::Texture));
    }
    wanted.sort_unstable();
    wanted.dedup();
    wanted
}

fn set_groups(state: &mut ScatterState) {
    let layouts = &state.groups.layouts;
    let held = &mut state.group_data;
    held.data.clear();
    held.template.clear();
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
            parts: layout.parts,
            pad: [0; 2],
            middle: group.bounds[0],
            half: group.bounds[1],
        });
        held.lists.push(GroupLists { ids, count });
        for level in 0..LEVELS as usize {
            held.template.push(draw_args(
                layout.levels.get(level).copied().unwrap_or((0, 0, 0)),
            ));
        }
        ids += count * layout.levels.len() as u32;
    }
    held.ids_per_view = ids;
    if grow(
        &mut state.group_bytes,
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
    for (bytes, stride) in state.slot_bytes.iter_mut().zip(SLOT_STRIDES) {
        grow(bytes, u64::from(held) * stride, state.room);
    }
    state.slot_sequence += 1;
    state.objects = held;
}

type PlaceReads<'w> = (
    Res<'w, TerrainTiles>,
    Res<'w, ScatterSettings>,
    Res<'w, AssetLibrary>,
    Res<'w, AssetResidency>,
    Res<'w, DeviceLimits>,
);

fn derive_groups(
    state: &mut ScatterState,
    residency: &AssetResidency,
    resolved: &ResolvedTerrains<'_>,
) {
    let wanted = asset_uses(resolved);
    if state.uses.assets() != wanted.as_slice() {
        state.uses = AssetUses::new(residency, wanted);
    }
    let mut groups = Groups::build(residency, resolved);
    let refill = !state.groups.same_objects(&groups);
    if !refill {
        for (layout, before) in groups.layouts.iter_mut().zip(&state.groups.layouts) {
            layout.impostor = before.impostor;
        }
    }
    state.groups = groups;
    state.refilled = refill;
    if refill {
        state.impostor_generation += 1;
        state.impostors = vec![ImpostorRecord::default(); state.groups.groups.len()];
        state.retake = true;
        state.pool.clear(state.groups.groups.len());
    }
}

pub(crate) fn place_pool(
    (tiles, settings, library, residency, limits): PlaceReads<'_>,
    (foliage, ground, mut maps): (
        Res<FoliageSettings>,
        Res<GroundTiles>,
        ResMut<CanopyMaps>,
    ),
    (mut jobs, records): (Jobs, Res<Records>),
    mut state: ResMut<ScatterState>,
) {
    let state = &mut *state;
    let landed = measure_looks(
        &mut state.authored,
        (&settings, &library),
        (&mut jobs, &records),
    );
    state.room = match limits.storage_room {
        0 => u64::MAX,
        room => room,
    } / 16
        * 16;
    state.group_data.first_instance = limits.indirect_first_instance;
    state.pool.settle();
    let placing = tiles
        .terrains
        .iter()
        .any(|terrain| !terrain.placement.is_empty());
    if !placing && (state.slot_bytes != SLOT_FIRST || state.group_bytes != GROUPS_FIRST) {
        state.slot_bytes = SLOT_FIRST;
        state.group_bytes = GROUPS_FIRST;
        state.generation += 1;
    }
    let revisions = (library.revision(), residency.revision());
    state.library = revisions.0;
    let placed = tiles.is_changed() && !same_placing(&state.placing, &tiles);
    if placed {
        state.placing = tiles.terrains.iter().map(Placing::of).collect();
    }
    let derived =
        landed || placed || settings.is_changed() || state.revisions != Some(revisions);
    state.refilled = false;
    if derived || foliage.is_changed() || maps.whole() != ground.whole {
        let (resolved, kept) = resolve_all(&tiles, (&settings, &state.authored), &library);
        state.kept = kept;
        maps.want(
            wanted(
                &tiles.terrains,
                &resolved,
                &library,
                (foliage.variation, ground.whole),
            ),
            ground.whole,
        );
        if derived {
            state.revisions = Some(revisions);
            derive_groups(state, &residency, &resolved);
        }
    }
    let refill = state.refilled;
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
}

pub(crate) fn plan_cull(
    (views, detail, canopy): (Res<Views>, Res<Detail>, Res<CanopySettings>),
    mut state: ResMut<ScatterState>,
) {
    let state = &mut *state;
    let plan = &mut state.plan;
    plan.prepared = false;
    plan.atlas = state.impostors.iter().any(|record| record.sphere[3] > 0.0);
    plan.cards = plan.atlas;
    if state.resident == 0 {
        return;
    }
    let Some(main) = views.main() else {
        return;
    };
    let view_projection = main.view_projection;
    let eye = main.detail_eye;
    let height = main.size().y as f32;
    let spanned = view_projection.row(1).truncate().length() * height * 0.5;
    let pixels = detail.chosen_error().map_or(0.0, |error| spanned / error);
    plan.main = Some(CullView {
        eye: eye.extend(pixels).to_array(),
        shape: [1, 1, 0, 0],
    });
    let group_count = state.groups.layouts.len() as u32;
    plan.frame = Some(CullFrame {
        eye: eye.extend(spanned / canopy.handover_pixels).to_array(),
        counts: [state.objects, group_count, LEVELS, 0],
        strides: [
            state.group_data.ids_per_view,
            group_count * LEVELS,
            u32::from(plan.cards),
            state.objects,
        ],
        drawn: [0; 4],
    });
    plan.prepared = true;
}

pub(crate) fn publish_assets(
    state: Res<ScatterState>,
    own: Query<(Entity, &AssetUses, &ImpostorAsks), With<Placements>>,
    mut spawner: Spawner<ScatterBrick>,
) {
    let asks = || ImpostorAsks {
        generation: state.impostor_generation,
        hemisphere: true,
        asks: state.groups.asks.clone(),
    };
    match own.single() {
        Ok((entity, uses, held)) => {
            if *uses != state.uses {
                spawner.insert(entity, state.uses.clone());
            }
            if held.generation != state.impostor_generation
                || !held.hemisphere
                || held.asks != state.groups.asks
            {
                spawner.insert(entity, asks());
            }
        }
        Err(_) => {
            spawner.spawn((
                Placements,
                Transform::default(),
                state.uses.clone(),
                asks(),
                ImpostorRecords::default(),
                Drawn::procedural(PLACED),
            ));
        }
    }
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
    let stale = state.restyled || state.refilled;
    let held = &placed.tiles;
    let mut pool = state.pool.pooled();
    let current = !stale
        && held.len() == pool.len()
        && pool.all(|(&(terrain, key), pooled)| {
            held.binary_search_by_key(&(terrain, key), |tile| (tile.terrain, tile.key))
                .is_ok_and(|at| same_tile(&held[at], pooled))
        });
    if current {
        return;
    }
    let tiles = state.pool.tiles();
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

#[derive(Resource, Default)]
pub(crate) struct Resnapped {
    full: bool,
    boxes: Vec<Reach>,
}

impl Readable for Resnapped {
    fn arrived(&mut self, _tag: u32, read: Readback<&[u8]>) {
        let Some((full, boxes)) = read.value.split_first_chunk::<4>() else {
            return;
        };
        self.full |= u32::from_le_bytes(*full) != 0;
        let words: Vec<f32> = boxes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        self.boxes.extend(words.as_chunks::<7>().0.iter().map(|at| {
            (Vec3::new(at[0], at[1], at[2]), Vec3::new(at[3], at[4], at[5]), at[6])
        }));
    }
}

pub(crate) fn resnapped_bytes((full, boxes): (bool, Vec<Reach>)) -> Vec<u8> {
    let mut bytes = u32::from(full).to_le_bytes().to_vec();
    for (low, high, widest) in boxes {
        for word in [low.x, low.y, low.z, high.x, high.y, high.z, widest] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    bytes
}

pub(crate) fn publish_changes(
    (state, tiles): (Res<ScatterState>, Res<TerrainTiles>),
    mut resnapped: ResMut<Resnapped>,
    mut areas: ResMut<DrawnAreas>,
) {
    let moved = state.moved.iter().map(|&moved| widened(&tiles, moved));
    for (min, max, widest) in moved.chain(resnapped.boxes.drain(..)) {
        areas.push(DrawnArea::Box {
            min,
            max,
            spacing: widest,
        });
    }
    if state.restyled || std::mem::take(&mut resnapped.full) {
        areas.push(DrawnArea::All);
    }
}

pub(crate) fn scatter_stats(
    (state, feedback, maps): (Res<ScatterState>, Res<ScatterFeedback>, Res<CanopyMaps>),
    mut stats: ResMut<ScatterStats>,
) {
    let entries = if state.objects > 0 {
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
        canopy_maps: maps.shown(),
        canopy_waiting: maps.waiting(),
        canopy_failure: maps.failure().cloned(),
        impostors: Cut::of(placed + state.impostor_refused, placed),
        parts: Cut::of(state.kept.parts.0, state.kept.parts.1),
        lods: Cut::of(state.kept.lods.0, state.kept.lods.1),
        measured_levels: state
            .authored
            .iter()
            .map(|held| held.work.ready().cloned().unwrap_or_default())
            .collect(),
    });
}
