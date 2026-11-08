#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod dev;
mod instance;
mod keys;
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
    AssetLibrary, Brick, Bricks, FrameTime, GlobalTransforms, Interest, Request, Requests, Runs,
    SceneContext, Schedule, Smoke, Source, Spawner, Views, settings, stats,
};
use engine_places_world::streaming::{StreamedKeys, StreamingFocus};

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

enum Load {
    Unloaded,
    Loaded { _held: Vec<Interest<usize>> },
}

struct Cell {
    grid: u8,
    key: [i32; 2],
    reach: f32,
    items: Vec<usize>,
    load: Load,
    distance: f32,
}

type Items = Requests<usize, Vec<Entity>>;

struct Acting<'a, 'w, 's> {
    live: &'a dyn Fn(Entity) -> bool,
    spawner: &'a mut Spawner<'w, 's, WorldStreamingBrick>,
}

impl Acting<'_, '_, '_> {
    fn despawn(&mut self, entities: &[Entity]) -> u32 {
        let mut count = 0;
        for &entity in entities {
            if (self.live)(entity) {
                self.spawner.despawn(entity);
                count += 1;
            }
        }
        count
    }

    fn spawn(&mut self, tree: &Tree, index: usize) -> Vec<Entity> {
        let spawned = self.spawner.install(tree.sub_scene(index));
        if let Some(&root) = spawned.first() {
            self.spawner
                .insert(root, StreamedId(tree.items[index].id.clone()));
        }
        spawned
    }
}

fn miss(items: &mut Items, index: usize) -> Option<Vec<Entity>> {
    let spawned = items.get(&index).and_then(Request::ready).cloned();
    if let Some(ticket) = items.ask_again(&index) {
        items.miss(&index, ticket);
    }
    spawned
}

impl Cell {
    fn distance_to(&self, point: Vec2) -> f32 {
        let side = STREAMING_GRIDS[self.grid as usize].cell;
        let low = Vec2::new(self.key[0] as f32, self.key[1] as f32) * side - self.reach;
        let high = low + side + 2.0 * self.reach;
        let outside = (low - point).max(point - high).max(Vec2::ZERO);
        outside.length()
    }

    fn loaded(&self) -> bool {
        match self.load {
            Load::Loaded { .. } => true,
            Load::Unloaded => false,
        }
    }

    fn waiting(&self, items: &Items) -> bool {
        self.loaded()
            && self
                .items
                .iter()
                .any(|index| items.ticket(index).is_some())
    }

    fn spawned(&self, items: &Items) -> u32 {
        if !self.loaded() {
            return 0;
        }
        self.items
            .iter()
            .filter_map(|index| items.get(index).and_then(Request::ready))
            .map(|spawned| spawned.len() as u32)
            .sum()
    }

    fn entities(&self, tree: &Tree) -> u32 {
        self.items
            .iter()
            .map(|&index| tree.entities_of(index))
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
    spawns: Items,
    kept: Vec<Vec<Entity>>,
    from_scene: bool,
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
                    load: Load::Unloaded,
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
            cell.items.push(index);
        }
    }

    fn replace(&mut self, content: Arc<Content>, act: &mut Acting<'_, '_, '_>) {
        for cell in &self.cells {
            for spawned in cell
                .items
                .iter()
                .filter_map(|index| self.spawns.get(index).and_then(Request::ready))
            {
                act.despawn(spawned);
            }
        }
        for spawned in self.kept.drain(..) {
            act.despawn(&spawned);
        }
        self.cells.clear();
        self.index.clear();
        self.spawns = Items::default();
        self.content = content;
        self.add();
        let kept = &self.content.kept;
        self.kept = (0..kept.items.len())
            .map(|item| act.spawner.install(kept.sub_scene(item)))
            .collect();
    }

    fn reconcile(
        &mut self,
        changes: &mut WorldChanges,
        revised: bool,
        act: &mut Acting<'_, '_, '_>,
    ) -> (u32, bool) {
        let tree = &self.content.streamed;
        let mut removed = false;
        for cell in self.cells.iter().filter(|cell| cell.loaded()) {
            for &index in &cell.items {
                let gone = self
                    .spawns
                    .get(&index)
                    .and_then(Request::ready)
                    .and_then(|spawned| spawned.first())
                    .is_some_and(|&root| !(act.live)(root));
                if gone {
                    removed |= changes.remove(&tree.items[index].id);
                }
            }
        }
        let mut despawned = 0;
        if !revised && !removed {
            return (despawned, removed);
        }
        for cell in self.cells.iter().filter(|cell| cell.loaded()) {
            for &index in &cell.items {
                let missing = matches!(self.spawns.get(&index), Some(Request::Missing));
                if changes.is_removed(&tree.items[index].id) {
                    if !missing && let Some(spawned) = miss(&mut self.spawns, index) {
                        despawned += act.despawn(&spawned);
                    }
                } else if missing {
                    self.spawns.ask_again(&index);
                }
            }
        }
        (despawned, removed)
    }

    fn load_and_unload(
        &mut self,
        settings: WorldStreaming,
        now: f64,
        act: &mut Acting<'_, '_, '_>,
    ) {
        let WorldCells {
            cells,
            spawns,
            sources,
            wanted,
            stats,
            ..
        } = self;
        wanted.clear();
        for (index, cell) in cells.iter_mut().enumerate() {
            cell.distance = sources
                .iter()
                .map(|&source| cell.distance_to(source))
                .fold(f32::INFINITY, min);
            let range = STREAMING_GRIDS[cell.grid as usize].range * settings.distance;
            let leaving = cell.distance > range * settings.unload;
            let coming = cell.distance <= range;
            if !settings.frozen {
                match cell.load {
                    Load::Loaded { .. } if leaving => cell.load = Load::Unloaded,
                    Load::Unloaded if coming => {
                        let asked = cell.items.iter().map(|&item| spawns.ask(item)).collect();
                        cell.load = Load::Loaded { _held: asked };
                    }
                    Load::Loaded { .. } | Load::Unloaded => {}
                }
            }
            if cell.waiting(spawns) {
                wanted.push((cell.distance / range, index));
            }
        }
        wanted.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, request) in spawns.settle(now) {
            if let Request::Ready(spawned) = request {
                stats.despawned += act.despawn(&spawned);
            }
        }
    }

    fn spawn(
        &mut self,
        changes: &WorldChanges,
        settings: WorldStreaming,
        act: &mut Acting<'_, '_, '_>,
    ) {
        if settings.frozen {
            return;
        }
        let tree = &self.content.streamed;
        let mut cells = 0;
        for &(_, at) in &self.wanted {
            if cells > 0 && self.stats.spawned >= settings.spawns_per_frame {
                break;
            }
            for &index in &self.cells[at].items {
                let Some(ticket) = self.spawns.ticket(&index) else {
                    continue;
                };
                if changes.is_removed(&tree.items[index].id) {
                    self.spawns.miss(&index, ticket);
                    continue;
                }
                let spawned = act.spawn(tree, index);
                self.stats.spawned += spawned.len() as u32;
                self.spawns.ready(&index, ticket, spawned);
            }
            cells += 1;
        }
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
                loaded: cell.loaded() && !cell.waiting(&self.spawns),
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
            let loaded = cell.loaded() && !cell.waiting(&self.spawns);
            grids[cell.grid as usize][0] += 1;
            grids[cell.grid as usize][1] += u32::from(loaded);
        }
        grids
    }
}

type StreamReads<'w> = (
    Res<'w, WorldStreaming>,
    Res<'w, Views>,
    Res<'w, FrameTime>,
    Res<'w, GlobalTransforms>,
);

type SceneReads<'w> = (
    Res<'w, StreamedScene>,
    Option<Res<'w, SceneContext>>,
    Res<'w, AssetLibrary>,
);

type FocusQuery<'w, 's> = Query<'w, 's, Entity, With<StreamingFocus>>;

pub(crate) fn stream_cells(
    (settings, views, frame, globals): StreamReads<'_>,
    (scene, context, assets): SceneReads<'_>,
    (focus, live): (FocusQuery<'_, '_>, Query<Entity>),
    mut cells: ResMut<WorldCells>,
    mut tracked: ResMut<WorldChanges>,
    mut published: ResMut<CellStats>,
    mut spawner: Spawner<WorldStreamingBrick>,
) {
    let settings = *settings;
    let cells = &mut *cells;
    let alive = |entity: Entity| live.contains(entity);
    let mut act = Acting {
        live: &alive,
        spawner: &mut spawner,
    };
    let context_changed = context
        .as_ref()
        .map_or(cells.from_scene, |context| context.is_changed());
    if scene.is_changed() || context_changed {
        let content = context.as_deref().map_or_else(Content::default, |context| {
            context.scope(|| instance::content(&scene, context, &assets))
        });
        cells.from_scene = context.is_some();
        cells.replace(Arc::new(content), &mut act);
    }
    let _profile = engine_core::profile::span("cpu", "world_streaming");
    cells.stats.spawned = 0;
    cells.stats.despawned = 0;
    if !cells.cells.is_empty() {
        sources(&views, &frame, (&*globals, &focus), cells, settings);
        let revised = tracked.is_changed();
        let changes = tracked.bypass_change_detection();
        let (despawned, removed) = cells.reconcile(changes, revised, &mut act);
        cells.stats.despawned += despawned;
        cells.load_and_unload(settings, frame.real_elapsed, &mut act);
        cells.spawn(changes, settings, &mut act);
        if removed {
            tracked.set_changed();
        }
    }
    let spawns = &cells.spawns;
    cells.stats.cells = cells.cells.len() as u32;
    cells.stats.loaded = cells
        .cells
        .iter()
        .filter(|cell| cell.loaded() && !cell.waiting(spawns))
        .count() as u32;
    cells.stats.items = cells.items() as u32;
    cells.stats.entities = cells.cells.iter().map(|cell| cell.spawned(spawns)).sum();
    *published = cells.stats;
}

fn sources(
    views: &Views,
    frame: &FrameTime,
    (globals, focus): (&GlobalTransforms, &FocusQuery<'_, '_>),
    cells: &mut WorldCells,
    settings: WorldStreaming,
) {
    let eye = views.main().map_or(Vec3::ZERO, |main| main.eye);
    let (number, elapsed) = (frame.frame, frame.real_elapsed);
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
    cells
        .sources
        .extend(views.listed().iter().map(|facts| flat(facts.eye)));
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
            .when(|settings: Res<WorldStreaming>| settings.enabled)
            .settings::<WorldStreaming>()
            .settings::<StreamedScene>()
            .stats::<CellStats>()
            .resource::<WorldCells>()
            .resource::<WorldChanges>()
            .systems(Schedule::WorldSync, stream_cells)
            .operation::<WorldChangesControl>()
            .observation::<WorldCellsQuery>()
            .observation::<WorldChangesQuery>()
            .observation::<StreamedIdQuery>();
        for key in keys::RENDERED {
            bricks.join::<StreamedKeys>(key, None);
        }
    }
}
