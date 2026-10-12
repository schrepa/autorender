//! World streaming: content that exists only near the
//! camera. Items — an entity and its descendants, spawned and despawned as
//! one — register with [`WorldCells`] at a world position with the meters
//! their content reaches from there. The engine bins them into cells of
//! three grids by that reach ([`STREAMING_GRIDS`]): small items into
//! 256 m cells loading within 512 m, those reaching over 4 m into 1 km
//! cells loading within 2 km, and those reaching over 16 m into 4 km cells
//! loading within 8 km, so what shows from farther away loads from farther
//! away. Cells are columns: distances are horizontal.
//!
//! Each extracted frame, before transforms refresh, cells come within
//! their range of a streaming source (the camera, every [`StreamingFocus`],
//! and where the camera's motion takes it [`WorldStreaming::prefetch`]
//! seconds ahead) load: their items spawn, the cells nearest their range's
//! middle first, until the frame has spawned
//! [`WorldStreaming::spawns_per_frame`] entities (a cell spawns whole, and
//! at least one cell loads a frame); the rest wait. A cell unloads,
//! despawning what its items spawned, once every source is farther than
//! [`WorldStreaming::unload`] times its range.
//!
//! What gameplay removes stays removed: an item whose root
//! (the first entity it spawned) gameplay despawned is gone — streaming
//! despawns the rest of it and records its ID in [`WorldChanges`], which
//! games save, and its cell never spawns it again. Removing or restoring
//! an ID there takes effect at the next extraction, in loaded cells too.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use bevy_ecs::prelude::*;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::math::{max, min};
use crate::{Camera, FrameTime, GlobalTransforms, StreamingFocus};
use engine_settings::settings;

/// One grid items stream in: its cells' side and the meters from a source
/// within which they load (at [`WorldStreaming::distance`] 1), and the
/// most meters its items reach; items reaching farther go to the next
/// grid.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct StreamingGrid {
    pub cell: f32,
    pub range: f32,
    pub reach: f32,
}

/// The grids, smallest first. An item reaching `r` meters
/// shows about as large from its grid's range as one reaching the grid's
/// most does there: a 128th of the range.
pub const STREAMING_GRIDS: [StreamingGrid; 3] = [
    StreamingGrid {
        cell: 256.0,
        range: 512.0,
        reach: 4.0,
    },
    StreamingGrid {
        cell: 1024.0,
        range: 2048.0,
        reach: 16.0,
    },
    StreamingGrid {
        cell: 4096.0,
        range: 8192.0,
        reach: f32::INFINITY,
    },
];

/// Meters a second past which the camera's motion between frames counts as
/// a jump, which prefetch ignores.
const JUMP_SPEED: f32 = 1000.0;

settings! {
    /// World streaming. Requires version 9.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneWorldStreaming")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct WorldStreaming {
        /// Whether streamed items load at all: off unloads every cell.
        pub enabled: bool = true => any,
        /// How far cells load, as a multiple of each grid's range.
        pub distance: f32 = 1.0 => range(0.05, 20.0),
        /// How much farther than that every source must be before a loaded
        /// cell unloads, as a multiple.
        pub unload: f32 = 1.25 => range(1.0, 4.0),
        /// The seconds ahead along the camera's motion cells load early for.
        pub prefetch: f32 = 1.5 => range(0.0, 10.0),
        /// The entities a frame spawns at most, pacing loads: cells spawn
        /// whole, and at least one a frame.
        pub spawns_per_frame: u32 = 2048 => at_least(1),
        /// Whether cells stay as they are, loading and unloading nothing, for
        /// captures and inspection.
        pub frozen: bool = false => any,
    }
}

impl WorldStreaming {
    /// What a world without these settings reads as: nothing streamed.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

/// What a streamed item spawns.
pub trait Streamed: Send + Sync + 'static {
    /// How many entities [`Self::spawn`] spawns, which spawn budgets count.
    fn entities(&self) -> u32;
    /// Spawns the item's entities into `world`, appending each to
    /// `spawned`; streaming despawns them when the item's cell unloads.
    fn spawn(&self, world: &mut World, spawned: &mut Vec<Entity>);
}

/// An item to stream: its ID, which [`WorldChanges`] records its removal
/// by (unique among items; scenes' are `scene/root`, and items without
/// one are never removed for good); where it stands (world space); the
/// meters its content reaches from there, which choose its grid; and what
/// it spawns.
#[derive(Clone)]
pub struct StreamedItem {
    pub id: Option<Arc<str>>,
    pub position: Vec3,
    pub reach: f32,
    pub content: Arc<dyn Streamed>,
}

/// The ID of the streamed item whose root this entity is:
/// despawning it removes the item for good ([`WorldChanges`]).
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct StreamedId(pub Arc<str>);

/// What gameplay changed of the streamed world, which games
/// save and load: the IDs of the items removed for good, which their cells
/// no longer spawn. Inserted by default; streaming adds an item's ID when
/// gameplay despawns its root. Removing or restoring an ID takes effect at
/// the next extraction.
#[derive(Resource, Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldChanges {
    #[serde(default)]
    removed: BTreeSet<String>,
    #[serde(skip, default = "fresh_revision")]
    revision: u64,
}

/// Revisions unique in the process, so a loaded save never matches a
/// revision loaded cells already reflect.
static REVISIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn fresh_revision() -> u64 {
    REVISIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl Default for WorldChanges {
    fn default() -> Self {
        Self {
            removed: BTreeSet::new(),
            revision: fresh_revision(),
        }
    }
}

impl PartialEq for WorldChanges {
    fn eq(&self, other: &Self) -> bool {
        self.removed == other.removed
    }
}

impl WorldChanges {
    /// Removes item `id` for good; whether it was not removed already.
    pub fn remove(&mut self, id: &str) -> bool {
        let added = !self.removed.contains(id) && self.removed.insert(id.to_owned());
        if added {
            self.revision = fresh_revision();
        }
        added
    }

    /// Lets item `id` spawn again; whether it was removed.
    pub fn restore(&mut self, id: &str) -> bool {
        let restored = self.removed.remove(id);
        if restored {
            self.revision = fresh_revision();
        }
        restored
    }

    /// Restores every item.
    pub fn clear(&mut self) {
        if !self.removed.is_empty() {
            self.removed.clear();
            self.revision = fresh_revision();
        }
    }

    pub fn is_removed(&self, id: &str) -> bool {
        self.removed.contains(id)
    }

    /// The removed items' IDs, in order.
    pub fn removed(&self) -> impl ExactSizeIterator<Item = &str> {
        self.removed.iter().map(String::as_str)
    }

    /// Changes whenever the removed IDs do; unique in the process, so a
    /// loaded save never reads as one already applied.
    pub fn revision(&self) -> u64 {
        self.revision
    }
}

/// Items registered together ([`WorldCells::add`]), removed together
/// ([`remove_streamed`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StreamSet(u64);

/// World streaming's state after the last extraction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CellStats {
    /// Cells holding items.
    pub cells: u32,
    /// Cells loaded.
    pub loaded: u32,
    /// Cells within range still waiting to load.
    pub pending: u32,
    /// Items registered.
    pub items: u32,
    /// Entities streamed items hold spawned.
    pub entities: u32,
    /// Entities the frame spawned.
    pub spawned: u32,
    /// Entities the frame despawned.
    pub despawned: u32,
}

/// One cell of a grid: its items, whether they are spawned and the
/// entities they hold spawned, and its distance from the nearest source at
/// the last extraction.
struct Cell {
    grid: u8,
    key: [i32; 2],
    /// The most meters an item reaches past the cell's side.
    reach: f32,
    items: Vec<Item>,
    loaded: bool,
    spawned: u32,
    distance: f32,
}

struct Item {
    set: u64,
    id: Option<Arc<str>>,
    content: Arc<dyn Streamed>,
    spawned: Vec<Entity>,
}

impl Item {
    /// Spawns it, marking its root with its ID; returns how many entities
    /// it spawned.
    fn spawn(&mut self, world: &mut World) -> u32 {
        self.content.spawn(world, &mut self.spawned);
        if let (Some(id), Some(&root)) = (&self.id, self.spawned.first())
            && let Ok(mut root) = world.get_entity_mut(root)
        {
            root.insert(StreamedId(id.clone()));
        }
        self.spawned.len() as u32
    }

    /// Despawns what it spawned; returns how many entities were left.
    fn despawn(&mut self, world: &mut World) -> u32 {
        let mut count = 0;
        for entity in self.spawned.drain(..) {
            // Gameplay may have despawned some already.
            if world.try_despawn(entity).is_ok() {
                count += 1;
            }
        }
        count
    }

    fn removed(&self, changes: &WorldChanges) -> bool {
        self.id.as_ref().is_some_and(|id| changes.is_removed(id))
    }
}

impl Cell {
    /// Horizontal meters from `point` to the cell's side grown by its
    /// items' reach.
    fn distance_to(&self, point: Vec2) -> f32 {
        let side = STREAMING_GRIDS[self.grid as usize].cell;
        let low = Vec2::new(self.key[0] as f32, self.key[1] as f32) * side - self.reach;
        let high = low + side + 2.0 * self.reach;
        let outside = (low - point).max(point - high).max(Vec2::ZERO);
        outside.length()
    }

    /// Despawns what its items spawned; returns how many entities that was.
    fn unload(&mut self, world: &mut World) -> u32 {
        let count = self.items.iter_mut().map(|item| item.despawn(world)).sum();
        self.loaded = false;
        self.spawned = 0;
        count
    }

    /// Spawns its items but those `changes` removed; returns how many
    /// entities they spawned.
    fn load(&mut self, world: &mut World, changes: &WorldChanges) -> u32 {
        let mut count = 0;
        for item in &mut self.items {
            if !item.removed(changes) {
                count += item.spawn(world);
            }
        }
        self.loaded = true;
        self.spawned = count;
        count
    }

    /// In a loaded cell: removes the items whose roots gameplay despawned,
    /// recording them in `changes`, despawns those `changes` removed since
    /// `revision` and spawns those it restored. Returns the entities
    /// despawned and spawned.
    fn reconcile(
        &mut self,
        world: &mut World,
        changes: &mut WorldChanges,
        revision: u64,
    ) -> [u32; 2] {
        let mut counts = [0, 0];
        for item in &mut self.items {
            let gone = item
                .spawned
                .first()
                .is_some_and(|&root| world.get_entity(root).is_err());
            if gone {
                counts[0] += item.despawn(world);
                if let Some(id) = &item.id {
                    changes.remove(id);
                }
            } else if changes.revision() != revision {
                if item.removed(changes) {
                    counts[0] += item.despawn(world);
                } else if item.spawned.is_empty() {
                    counts[1] += item.spawn(world);
                }
            }
        }
        self.spawned = self
            .items
            .iter()
            .map(|item| item.spawned.len() as u32)
            .sum();
        counts
    }

    fn entities(&self) -> u32 {
        self.items.iter().map(|item| item.content.entities()).sum()
    }
}

/// The camera's last position and frame, and its velocity then.
#[derive(Default)]
struct Motion {
    last: Option<(u64, f64, Vec3)>,
    velocity: Vec3,
}

/// Every streamed item, binned into cells of [`STREAMING_GRIDS`],
/// and what loaded at the last extraction. Inserted by
/// default; register items with [`Self::add`].
#[derive(Resource, Default)]
pub struct WorldCells {
    cells: Vec<Cell>,
    index: HashMap<(u8, [i32; 2]), usize>,
    next_set: u64,
    motion: Motion,
    stats: CellStats,
    /// The [`WorldChanges`] revision loaded cells match.
    changes: u64,
    // Scratch.
    sources: Vec<Vec2>,
    wanted: Vec<(f32, usize)>,
}

/// A loaded or unloaded cell as inspection reports it: its grid (0 small,
/// 1 medium, 2 large), its key along x and z (its corner over the grid's
/// side), its items, their entities, whether it is loaded and its meters
/// from the nearest streaming source at the last extraction.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CellView {
    pub grid: u8,
    pub key: [i32; 2],
    pub items: u32,
    pub entities: u32,
    pub loaded: bool,
    pub distance: f32,
}

impl WorldCells {
    /// Registers `items` as one set, binned into cells by their reach and
    /// position; they load from the next extraction on. Items whose
    /// position or reach is not finite are left out.
    pub fn add(&mut self, items: impl IntoIterator<Item = StreamedItem>) -> StreamSet {
        let set = self.next_set;
        self.next_set += 1;
        for item in items {
            if !item.position.is_finite() || !item.reach.is_finite() {
                continue;
            }
            let reach = max(item.reach, 0.0);
            let grid = STREAMING_GRIDS
                .iter()
                .position(|grid| reach <= grid.reach)
                .unwrap_or(STREAMING_GRIDS.len() - 1);
            let side = STREAMING_GRIDS[grid].cell;
            let key = [
                (item.position.x / side).floor() as i32,
                (item.position.z / side).floor() as i32,
            ];
            let cells = &mut self.cells;
            let index = *self.index.entry((grid as u8, key)).or_insert_with(|| {
                cells.push(Cell {
                    grid: grid as u8,
                    key,
                    reach: 0.0,
                    items: Vec::new(),
                    loaded: false,
                    spawned: 0,
                    distance: f32::INFINITY,
                });
                cells.len() - 1
            });
            let cell = &mut self.cells[index];
            // How far the item reaches past the cell's side.
            let corner = Vec2::new(key[0] as f32, key[1] as f32) * side;
            let local = Vec2::new(item.position.x, item.position.z) - corner;
            let past = max(
                max(reach - local.x, reach - local.y),
                max(local.x + reach - side, local.y + reach - side),
            );
            cell.reach = max(cell.reach, past);
            cell.items.push(Item {
                set,
                id: item.id,
                content: item.content,
                spawned: Vec::new(),
            });
        }
        StreamSet(set)
    }

    /// World streaming's state after the last extraction.
    pub fn stats(&self) -> CellStats {
        self.stats
    }

    /// Items registered and not removed.
    pub fn items(&self) -> usize {
        self.cells.iter().map(|cell| cell.items.len()).sum()
    }

    /// The camera's velocity prefetch followed at the last extraction.
    pub fn velocity(&self) -> Vec3 {
        self.motion.velocity
    }

    /// Up to `count` cells nearest a streaming source at the last
    /// extraction, nearest first.
    pub fn nearest(&self, count: usize) -> Vec<CellView> {
        let mut cells: Vec<CellView> = self
            .cells
            .iter()
            .map(|cell| CellView {
                grid: cell.grid,
                key: cell.key,
                items: cell.items.len() as u32,
                entities: cell.entities(),
                loaded: cell.loaded,
                distance: cell.distance,
            })
            .collect();
        cells.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        cells.truncate(count);
        cells
    }

    /// Cells of each grid, and those of them loaded.
    pub fn grids(&self) -> [[u32; 2]; 3] {
        let mut grids = [[0; 2]; 3];
        for cell in &self.cells {
            grids[cell.grid as usize][0] += 1;
            grids[cell.grid as usize][1] += u32::from(cell.loaded);
        }
        grids
    }
}

/// Removes the items of `set`, despawning what loaded ones spawned.
pub fn remove_streamed(world: &mut World, set: StreamSet) {
    if !world.contains_resource::<WorldCells>() {
        return;
    }
    world.resource_scope(|world, mut cells: Mut<WorldCells>| {
        let cells = &mut *cells;
        for cell in &mut cells.cells {
            for item in cell.items.iter_mut().filter(|item| item.set == set.0) {
                cell.spawned -= item.spawned.len() as u32;
                for entity in item.spawned.drain(..) {
                    let _ = world.try_despawn(entity);
                }
            }
            cell.items.retain(|item| item.set != set.0);
        }
        cells.cells.retain(|cell| !cell.items.is_empty());
        cells.index = cells
            .cells
            .iter()
            .enumerate()
            .map(|(index, cell)| ((cell.grid, cell.key), index))
            .collect();
    });
}

/// The terrain focus entities, which are streaming sources.
pub(crate) type FocusQuery = QueryState<Entity, With<StreamingFocus>>;

/// Loads cells streaming sources came near and unloads those they left,
/// before transforms refresh: [`crate::WorldSyncSystems::StreamCells`].
pub(crate) fn stream_cells(world: &mut World, focus: &mut FocusQuery) {
    if !world.contains_resource::<WorldCells>() {
        return;
    }
    let _profile = engine_profile::span("cpu", "world_streaming");
    let settings = world.get_resource::<WorldStreaming>().copied().map_or(
        WorldStreaming {
            enabled: false,
            ..WorldStreaming::default()
        },
        WorldStreaming::sanitized,
    );
    if !world.contains_resource::<WorldChanges>() {
        world.init_resource::<WorldChanges>();
    }
    world.resource_scope(|world, mut cells: Mut<WorldCells>| {
        let cells = &mut *cells;
        cells.stats.spawned = 0;
        cells.stats.despawned = 0;
        cells.stats.pending = 0;
        if !cells.cells.is_empty() {
            sources(world, focus, cells, settings);
            world.resource_scope(|world, mut tracked: Mut<WorldChanges>| {
                // What gameplay and changes removed or restored first.
                let (before, revision) = (tracked.revision(), cells.changes);
                let changes = tracked.bypass_change_detection();
                for cell in cells.cells.iter_mut().filter(|cell| cell.loaded) {
                    let [despawned, spawned] = cell.reconcile(world, changes, revision);
                    cells.stats.despawned += despawned;
                    cells.stats.spawned += spawned;
                }
                cells.changes = changes.revision();
                load_and_unload(world, cells, changes, settings);
                if tracked.revision() != before {
                    tracked.set_changed();
                }
            });
        }
        cells.stats.cells = cells.cells.len() as u32;
        cells.stats.loaded = cells.cells.iter().filter(|cell| cell.loaded).count() as u32;
        cells.stats.items = cells.items() as u32;
        cells.stats.entities = cells.cells.iter().map(|cell| cell.spawned).sum();
    });
}

/// This frame's streaming sources into `cells.sources`: the camera,
/// every terrain `focus`, and where the camera's motion takes it
/// `prefetch` seconds ahead.
fn sources(
    world: &World,
    focus: &mut FocusQuery,
    cells: &mut WorldCells,
    settings: WorldStreaming,
) {
    let eye = world.resource::<Camera>().eye();
    let frame = world.resource::<FrameTime>();
    let (number, elapsed) = (frame.frame, frame.elapsed_seconds);
    let motion = &mut cells.motion;
    match motion.last {
        Some((last_frame, last_elapsed, last)) if number != last_frame => {
            let seconds = (elapsed - last_elapsed) as f32;
            let velocity = if seconds > 0.0 {
                (eye - last) / seconds
            } else {
                Vec3::ZERO
            };
            motion.velocity = if velocity.is_finite() && velocity.length() <= JUMP_SPEED {
                velocity
            } else {
                Vec3::ZERO
            };
            motion.last = Some((number, elapsed, eye));
        }
        // Later extractions of a frame (paused renders) keep its first
        // position: motion is measured frame to frame.
        Some(_) => {}
        None => {
            motion.velocity = Vec3::ZERO;
            motion.last = Some((number, elapsed, eye));
        }
    }
    let flat = |point: Vec3| Vec2::new(point.x, point.z);
    cells.sources.clear();
    cells.sources.push(flat(eye));
    let ahead = eye + motion.velocity * settings.prefetch;
    if ahead.distance_squared(eye) > 1.0 {
        cells.sources.push(flat(ahead));
    }
    let globals = world.resource::<GlobalTransforms>();
    for entity in focus.iter(world) {
        if let Some(position) = globals
            .get(entity)
            .and_then(|global| global.translation())
            .filter(|position| position.is_finite())
        {
            cells.sources.push(flat(position));
        }
    }
}

/// Unloads the cells every source left and loads those one came near,
/// nearest their range's middle first, within the frame's spawn budget.
fn load_and_unload(
    world: &mut World,
    cells: &mut WorldCells,
    changes: &WorldChanges,
    settings: WorldStreaming,
) {
    let WorldCells {
        cells: list,
        sources,
        wanted,
        stats,
        ..
    } = cells;
    wanted.clear();
    for (index, cell) in list.iter_mut().enumerate() {
        cell.distance = sources
            .iter()
            .map(|&source| cell.distance_to(source))
            .fold(f32::INFINITY, min);
        let range = STREAMING_GRIDS[cell.grid as usize].range * settings.distance;
        let leaving = !settings.enabled || cell.distance > range * settings.unload;
        if cell.loaded && leaving && !settings.frozen {
            stats.despawned += cell.unload(world);
        } else if !cell.loaded && settings.enabled && cell.distance <= range {
            wanted.push((cell.distance / range, index));
        }
    }
    wanted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut loaded = 0;
    if !settings.frozen {
        for &(_, index) in wanted.iter() {
            if loaded > 0 && stats.spawned >= settings.spawns_per_frame {
                break;
            }
            stats.spawned += list[index].load(world, changes);
            loaded += 1;
        }
    }
    stats.pending = (wanted.len() - loaded) as u32;
}

/// Streamed entities spawning near the camera:
/// [`crate::Feature::WorldStreaming`].
#[derive(Default)]
pub struct WorldStreamingPlugin;

impl bevy_app::Plugin for WorldStreamingPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        crate::install(app, crate::Feature::WorldStreaming);
        app.init_resource::<WorldCells>()
            .init_resource::<WorldChanges>()
            .add_systems(
                crate::WorldSync,
                stream_cells.in_set(crate::WorldSyncSystems::StreamCells),
            );
    }
}
