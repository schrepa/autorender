#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod dev;
mod instance;
mod scene;
mod smoke;

pub use dev::{
    MAX_CHANGES, StreamedCellView, StreamedIdQuery, StreamedIdView, StreamingGridView,
    WorldCellsQuery, WorldCellsView, WorldChangesControl, WorldChangesQuery, WorldChangesResult,
    WorldChangesView,
};
pub use scene::{
    AlongStreamed, MAX_ALONG, StreamedAlong, StreamedObject, StreamedScene, StreamedTransform,
};

use engine_core::collections::HashMap;
use std::collections::BTreeSet;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use glam::{Vec2, Vec3};
use serde::Serialize;

use engine_core::math::{max, min};
use engine_core::{
    Brick, Bricks, Camera, FrameTime, GlobalTransforms, Runs, SceneContext, Schedule, Smoke,
    Source, Spawner, settings, stats,
};
use engine_places::streaming::StreamingFocus;

use instance::{Content, Tree};

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
    #[section("world_streaming")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneWorldStreaming")]
    pub struct WorldStreaming {
        pub enabled: bool = true => any,
        pub distance: f32 = 1.0 => range(0.05, 20.0),
        pub unload: f32 = 1.25 => range(1.0, 4.0),
        pub prefetch: f32 = 1.5 => range(0.0, 10.0),
        pub spawns_per_frame: u32 = 2048 => at_least(1),
        pub frozen: bool = false => any,
    }
}

stats! {
    #[section("world_streaming_stats")]
    #[derive(Copy, Eq)]
    pub struct CellStats {
        pub cells: u32,
        pub loaded: u32,
        pub pending: u32,
        pub items: u32,
        pub entities: u32,
        pub spawned: u32,
        pub despawned: u32,
    }
}

#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct StreamedId(pub Arc<str>);

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct WorldChanges {
    removed: BTreeSet<String>,
}

impl WorldChanges {
    pub fn remove(&mut self, id: &str) -> bool {
        !self.removed.contains(id) && self.removed.insert(id.to_owned())
    }

    pub fn restore(&mut self, id: &str) -> bool {
        self.removed.remove(id)
    }

    pub fn clear(&mut self) -> bool {
        let cleared = !self.removed.is_empty();
        self.removed.clear();
        cleared
    }

    pub fn is_removed(&self, id: &str) -> bool {
        self.removed.contains(id)
    }

    pub fn removed(&self) -> impl ExactSizeIterator<Item = &str> {
        self.removed.iter().map(String::as_str)
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
    index: usize,
    id: Arc<str>,
    spawned: Vec<Entity>,
    spawned_in: u64,
}

struct Acting<'a, 'w, 's> {
    live: &'a dyn Fn(Entity) -> bool,
    spawner: &'a mut Spawner<'w, 's, WorldStreamingBrick>,
    pass: u64,
}

impl Acting<'_, '_, '_> {
    fn despawn(&mut self, entity: Entity, fresh: bool) -> bool {
        let present = fresh || (self.live)(entity);
        if present {
            self.spawner.despawn(entity);
        }
        present
    }
}

impl Item {
    fn spawn(&mut self, tree: &Tree, act: &mut Acting<'_, '_, '_>) -> u32 {
        tree.spawn(self.index, act.spawner, &mut self.spawned);
        if let Some(&root) = self.spawned.first() {
            act.spawner.insert(root, StreamedId(self.id.clone()));
        }
        self.spawned_in = act.pass;
        self.spawned.len() as u32
    }

    fn despawn(&mut self, act: &mut Acting<'_, '_, '_>) -> u32 {
        let fresh = self.spawned_in == act.pass;
        let mut count = 0;
        for entity in self.spawned.drain(..) {
            if act.despawn(entity, fresh) {
                count += 1;
            }
        }
        count
    }

    fn removed(&self, changes: &WorldChanges) -> bool {
        changes.is_removed(&self.id)
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

    fn unload(&mut self, act: &mut Acting<'_, '_, '_>) -> u32 {
        let count = self.items.iter_mut().map(|item| item.despawn(act)).sum();
        self.loaded = false;
        self.spawned = 0;
        count
    }

    fn load(&mut self, tree: &Tree, changes: &WorldChanges, act: &mut Acting<'_, '_, '_>) -> u32 {
        let mut count = 0;
        for item in &mut self.items {
            if !item.removed(changes) {
                count += item.spawn(tree, act);
            }
        }
        self.loaded = true;
        self.spawned = count;
        count
    }

    fn reconcile(
        &mut self,
        tree: &Tree,
        act: &mut Acting<'_, '_, '_>,
        changes: &mut WorldChanges,
        (revised, removed): (&mut bool, &mut bool),
    ) -> [u32; 2] {
        let mut counts = [0, 0];
        for item in &mut self.items {
            let gone = item.spawned_in != act.pass
                && item.spawned.first().is_some_and(|&root| !(act.live)(root));
            if gone {
                counts[0] += item.despawn(act);
                let added = changes.remove(&item.id);
                *revised |= added;
                *removed |= added;
            } else if *revised {
                if item.removed(changes) {
                    counts[0] += item.despawn(act);
                } else if item.spawned.is_empty() {
                    counts[1] += item.spawn(tree, act);
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

    fn entities(&self, tree: &Tree) -> u32 {
        self.items
            .iter()
            .map(|item| tree.entities_of(item.index))
            .sum()
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
    content: Arc<Content>,
    kept: Vec<Vec<Entity>>,
    from_scene: bool,
    pass: u64,
    motion: Motion,
    stats: CellStats,
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
    fn add(&mut self) {
        for (index, item) in self.content.streamed.items.iter().enumerate() {
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
            let at = *self.index.entry((grid as u8, key)).or_insert_with(|| {
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
            let cell = &mut self.cells[at];
            let corner = Vec2::new(key[0] as f32, key[1] as f32) * side;
            let local = Vec2::new(item.position.x, item.position.z) - corner;
            let past = max(
                max(reach - local.x, reach - local.y),
                max(local.x + reach - side, local.y + reach - side),
            );
            cell.reach = max(cell.reach, past);
            cell.items.push(Item {
                index,
                id: item.id.clone(),
                spawned: Vec::new(),
                spawned_in: 0,
            });
        }
    }

    fn replace(&mut self, content: Arc<Content>, act: &mut Acting<'_, '_, '_>) {
        for item in self.cells.iter_mut().flat_map(|cell| cell.items.iter_mut()) {
            item.despawn(act);
        }
        for entity in self.kept.drain(..).flatten() {
            act.despawn(entity, false);
        }
        self.cells.clear();
        self.index.clear();
        self.content = content;
        self.add();
        let kept = &self.content.kept;
        self.kept = (0..kept.items.len())
            .map(|item| {
                let mut spawned = Vec::new();
                kept.spawn(item, act.spawner, &mut spawned);
                spawned
            })
            .collect();
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
        let tree = &self.content.streamed;
        let mut cells: Vec<CellView> = self
            .cells
            .iter()
            .map(|cell| CellView {
                grid: cell.grid,
                key: cell.key,
                items: cell.items.len() as u32,
                entities: cell.entities(tree),
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

type StreamReads<'w> = (
    Res<'w, WorldStreaming>,
    Res<'w, Camera>,
    Res<'w, FrameTime>,
    Res<'w, GlobalTransforms>,
);

type SceneReads<'w> = (Res<'w, StreamedScene>, Option<Res<'w, SceneContext>>);

type FocusQuery<'w, 's> = Query<'w, 's, Entity, With<StreamingFocus>>;

pub(crate) fn stream_cells(
    (settings, camera, frame, globals): StreamReads<'_>,
    (scene, context): SceneReads<'_>,
    (focus, live): (FocusQuery<'_, '_>, Query<Entity>),
    mut cells: ResMut<WorldCells>,
    mut tracked: ResMut<WorldChanges>,
    mut published: ResMut<CellStats>,
    mut spawner: Spawner<WorldStreamingBrick>,
) {
    let settings = *settings;
    let cells = &mut *cells;
    cells.pass += 1;
    let alive = |entity: Entity| live.contains(entity);
    let mut act = Acting {
        live: &alive,
        spawner: &mut spawner,
        pass: cells.pass,
    };
    let context_changed = context
        .as_ref()
        .map_or(cells.from_scene, |context| context.is_changed());
    if scene.is_changed() || context_changed {
        let content = context.as_deref().map_or_else(Content::default, |context| {
            instance::content(&scene, context)
        });
        cells.from_scene = context.is_some();
        cells.replace(Arc::new(content), &mut act);
    }
    let _profile = engine_core::profile::span("cpu", "world_streaming");
    cells.stats.spawned = 0;
    cells.stats.despawned = 0;
    cells.stats.pending = 0;
    if !cells.cells.is_empty() {
        sources(camera.eye(), &frame, (&*globals, &focus), cells, settings);
        let mut revised = tracked.is_changed();
        let mut removed = false;
        let changes = tracked.bypass_change_detection();
        let tree = cells.content.clone();
        for cell in cells.cells.iter_mut().filter(|cell| cell.loaded) {
            let [despawned, spawned] = cell.reconcile(
                &tree.streamed,
                &mut act,
                changes,
                (&mut revised, &mut removed),
            );
            cells.stats.despawned += despawned;
            cells.stats.spawned += spawned;
        }
        load_and_unload(cells, &tree.streamed, changes, settings, &mut act);
        if removed {
            tracked.set_changed();
        }
    }
    cells.stats.cells = cells.cells.len() as u32;
    cells.stats.loaded = cells.cells.iter().filter(|cell| cell.loaded).count() as u32;
    cells.stats.items = cells.items() as u32;
    cells.stats.entities = cells.cells.iter().map(|cell| cell.spawned).sum();
    *published = cells.stats;
}

fn sources(
    eye: Vec3,
    frame: &FrameTime,
    (globals, focus): (&GlobalTransforms, &FocusQuery<'_, '_>),
    cells: &mut WorldCells,
    settings: WorldStreaming,
) {
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
    for entity in focus {
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
    cells: &mut WorldCells,
    tree: &Tree,
    changes: &WorldChanges,
    settings: WorldStreaming,
    act: &mut Acting<'_, '_, '_>,
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
            stats.despawned += cell.unload(act);
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
            stats.spawned += list[index].load(tree, changes, act);
            loaded += 1;
        }
    }
    stats.pending = (wanted.len() - loaded) as u32;
}

pub struct WorldStreamingBrick;

impl Brick for WorldStreamingBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[Smoke {
        name: "world_streaming",
        runs: Runs::Gpu,
        run: smoke::run,
    }];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<WorldStreaming>()
            .settings::<StreamedScene>()
            .stats::<CellStats>()
            .resource::<WorldCells>()
            .resource::<WorldChanges>()
            .systems(Schedule::Stream, stream_cells)
            .operation::<WorldChangesControl>()
            .observation::<WorldCellsQuery>()
            .observation::<WorldChangesQuery>()
            .observation::<StreamedIdQuery>();
    }
}
