#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod dev;

pub use dev::{
    MAX_CHANGES, StreamedCellView, StreamingGridView, WorldCellsView, WorldChangesControl,
    WorldChangesResult, WorldChangesView,
};

use engine_core::collections::HashMap;
use std::collections::BTreeSet;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use engine_core::math::{max, min};
use engine_core::{
    Brick, BrickStats, Bricks, Camera, EnvironmentSection, FrameTime, GlobalTransforms,
    StreamingFocus, WorldSync, WorldSyncSystems,
};
use engine_settings::{InvalidSetting, settings};

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct StreamingGrid {
    pub cell: f32,
    pub range: f32,
    pub reach: f32,
}

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

const JUMP_SPEED: f32 = 1000.0;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneWorldStreaming")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct WorldStreaming {
        pub enabled: bool = true => any,
        pub distance: f32 = 1.0 => range(0.05, 20.0),
        pub unload: f32 = 1.25 => range(1.0, 4.0),
        pub prefetch: f32 = 1.5 => range(0.0, 10.0),
        pub spawns_per_frame: u32 = 2048 => at_least(1),
        pub frozen: bool = false => any,
    }
}

impl WorldStreaming {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

impl EnvironmentSection for WorldStreaming {
    const KEY: &'static str = "world_streaming";
    const NAME: &'static str = "WorldStreaming";
    const OPT_IN: bool = false;

    fn section_off() -> Self {
        Self::off()
    }

    fn section_check(&self) -> Result<(), InvalidSetting> {
        self.validate()
    }

    fn section_sanitized(self) -> Self {
        self.sanitized()
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(WorldStreaming)
    }
}

pub trait Streamed: Send + Sync + 'static {
    fn entities(&self) -> u32;
    fn spawn(&self, world: &mut World, spawned: &mut Vec<Entity>);
}

#[derive(Clone)]
pub struct StreamedItem {
    pub id: Option<Arc<str>>,
    pub position: Vec3,
    pub reach: f32,
    pub content: Arc<dyn Streamed>,
}

#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct StreamedId(pub Arc<str>);

#[derive(Resource, Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldChanges {
    #[serde(default)]
    removed: BTreeSet<String>,
    #[serde(skip, default = "fresh_revision")]
    revision: u64,
}

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
    pub fn remove(&mut self, id: &str) -> bool {
        let added = !self.removed.contains(id) && self.removed.insert(id.to_owned());
        if added {
            self.revision = fresh_revision();
        }
        added
    }

    pub fn restore(&mut self, id: &str) -> bool {
        let restored = self.removed.remove(id);
        if restored {
            self.revision = fresh_revision();
        }
        restored
    }

    pub fn clear(&mut self) {
        if !self.removed.is_empty() {
            self.removed.clear();
            self.revision = fresh_revision();
        }
    }

    pub fn is_removed(&self, id: &str) -> bool {
        self.removed.contains(id)
    }

    pub fn removed(&self) -> impl ExactSizeIterator<Item = &str> {
        self.removed.iter().map(String::as_str)
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StreamSet(u64);

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CellStats {
    pub cells: u32,
    pub loaded: u32,
    pub pending: u32,
    pub items: u32,
    pub entities: u32,
    pub spawned: u32,
    pub despawned: u32,
}

impl BrickStats for CellStats {
    const KEY: &'static str = "world_streaming";

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(CellStats)
    }
}

struct Cell {
    grid: u8,
    key: [i32; 2],
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
    fn spawn(&mut self, world: &mut World) -> u32 {
        self.content.spawn(world, &mut self.spawned);
        if let (Some(id), Some(&root)) = (&self.id, self.spawned.first())
            && let Ok(mut root) = world.get_entity_mut(root)
        {
            root.insert(StreamedId(id.clone()));
        }
        self.spawned.len() as u32
    }

    fn despawn(&mut self, world: &mut World) -> u32 {
        let mut count = 0;
        for entity in self.spawned.drain(..) {
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
    fn distance_to(&self, point: Vec2) -> f32 {
        let side = STREAMING_GRIDS[self.grid as usize].cell;
        let low = Vec2::new(self.key[0] as f32, self.key[1] as f32) * side - self.reach;
        let high = low + side + 2.0 * self.reach;
        let outside = (low - point).max(point - high).max(Vec2::ZERO);
        outside.length()
    }

    fn unload(&mut self, world: &mut World) -> u32 {
        let count = self.items.iter_mut().map(|item| item.despawn(world)).sum();
        self.loaded = false;
        self.spawned = 0;
        count
    }

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

#[derive(Default)]
struct Motion {
    last: Option<(u64, f64, Vec3)>,
    velocity: Vec3,
}

#[derive(Resource, Default)]
pub struct WorldCells {
    cells: Vec<Cell>,
    index: HashMap<(u8, [i32; 2]), usize>,
    next_set: u64,
    motion: Motion,
    stats: CellStats,
    changes: u64,
    sources: Vec<Vec2>,
    wanted: Vec<(f32, usize)>,
}

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

    pub fn stats(&self) -> CellStats {
        self.stats
    }

    pub fn items(&self) -> usize {
        self.cells.iter().map(|cell| cell.items.len()).sum()
    }

    pub fn velocity(&self) -> Vec3 {
        self.motion.velocity
    }

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

    pub fn grids(&self) -> [[u32; 2]; 3] {
        let mut grids = [[0; 2]; 3];
        for cell in &self.cells {
            grids[cell.grid as usize][0] += 1;
            grids[cell.grid as usize][1] += u32::from(cell.loaded);
        }
        grids
    }
}

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

pub(crate) type FocusQuery = QueryState<Entity, With<StreamingFocus>>;

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
    let stats = world.resource_scope(|world, mut cells: Mut<WorldCells>| {
        let cells = &mut *cells;
        cells.stats.spawned = 0;
        cells.stats.despawned = 0;
        cells.stats.pending = 0;
        if !cells.cells.is_empty() {
            sources(world, focus, cells, settings);
            world.resource_scope(|world, mut tracked: Mut<WorldChanges>| {
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
        cells.stats
    });
    if let Some(mut kept) = world.get_resource_mut::<CellStats>() {
        *kept = stats;
    }
}

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

pub struct WorldStreamingBrick;

impl Brick for WorldStreamingBrick {
    fn register(bricks: &mut Bricks) {
        bricks
            .settings::<WorldStreaming>()
            .resource::<WorldCells>()
            .resource::<WorldChanges>()
            .resource::<CellStats>()
            .stats::<CellStats>()
            .systems(WorldSync, stream_cells.in_set(WorldSyncSystems::Stream))
            .inspector(dev::world_cells as fn(&WorldCells) -> WorldCellsView)
            .inspector(dev::world_changes as fn(&WorldChanges) -> WorldChangesView)
            .operation::<WorldChangesControl>();
    }
}
