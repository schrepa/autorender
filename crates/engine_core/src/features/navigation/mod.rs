//! Navigation agents: entities that walk over terrains to a destination.
//! When an agent's destination changes, the cheapest walk to it from where
//! it stands is searched over its terrain's walk graph, then over the
//! samples of the tiles the graph's way crosses
//! ([`engine_terrain::WalkSearch`]), from records the host hands back: each
//! tile's portals of the graph of the agent's gait (cooked, or made by a
//! job where the packs lack them) and the tiles' records, which walks read
//! the ground from. A search waits for the records it reads without holding
//! anything up and does its work in steps: every agent's search shares the
//! fixed tick's budget of [`Navigation::samples_per_tick`], so long walks
//! take several ticks. Each fixed tick a walking agent moves along its walk
//! at its speed, on the ground (the finest resident tile's height, else the
//! walk's), turning toward where it walks; it arrives at the walk's end.
//! Agents walk unparented entities; give them `InterpolateTransform` to
//! present smoothly between ticks. A session cooking asks for the portals
//! of every tile of each agent's terrain at its gait.

mod cook;
mod records;

use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_terrain::jobs::{PortalsJob, TileJob};
use engine_terrain::{
    Gait, NavOptions, NavPath, TileKey, TilePortals, WalkData, WalkGround, WalkSearch, Want,
};
use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::features::terrain::{Walked, WalkedTerrain};
use crate::math::{clamp, max};
use crate::{
    Cooking, FixedTime, JobBudget, JobFailures, JobId, Jobs, Parent, RecordKey, Terrains,
    Transform, job_output,
};
use cook::CookGraph;
use engine_settings::settings;
use records::{Resolved, WalkRecords};

settings! {
    /// Navigation.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneNavigation")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Navigation {
        /// Whether agents walk.
        pub enabled: bool = true => any,
        /// The samples of search work each fixed tick does at most over every
        /// walk being searched.
        pub samples_per_tick: u32 = 32_768 => at_least(1024),
    }
}

impl Navigation {
    /// What a world without these settings reads as: no agents walking.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

settings! {
    /// Walks its entity over the terrain under it to `destination`. Changing
    /// the destination searches anew; none stands still.
    #[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[require(NavWalk, crate::HeightHeld)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct NavAgent {
        /// Where it walks to: world x and z.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub destination: Option<[f32; 2]> = None => finite,
        /// Meters a second it walks.
        pub speed: f32 = 1.4 => range(0.01, 100.0),
        /// Degrees a second it turns at most.
        pub turn: f32 = 360.0 => range(1.0, 10_000.0),
        /// Meters its origin stands above the ground.
        pub offset: f32 = 0.0 => range(-10_000.0, 10_000.0),
        /// Meters around obstacles it keeps.
        pub radius: f32 = 0.4 => same(engine_terrain::NavOptions::rule_of("radius")),
        /// The steepest steps it climbs: rise per meter.
        pub max_slope: f32 = 0.8 => same(engine_terrain::NavOptions::rule_of("max_slope")),
        /// Meters of water it wades through at most.
        pub max_water: f32 = 0.6 => same(engine_terrain::NavOptions::rule_of("max_water")),
        /// The pyramid level it searches over: 0 the finest.
        pub level: u32 = 0 => same(engine_terrain::NavOptions::rule_of("level")),
    }
}

impl NavAgent {
    /// How its walks are searched.
    fn options(&self) -> NavOptions {
        NavOptions {
            level: self.level,
            max_slope: self.max_slope,
            radius: self.radius,
            max_water: self.max_water,
        }
    }
}

/// Where an agent is in its walk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum NavStatus {
    /// Without a destination.
    #[default]
    Idle,
    Searching,
    Walking,
    Arrived,
    /// No walk reaches the destination, or no terrain lies under it.
    Unreachable,
}

/// An agent's walk: its status, the destination it was
/// searched for, its points (world, on the ground), the next it walks to,
/// its length (meters) and the samples and portals its search took.
#[derive(Component, Clone, Debug, Default)]
pub struct NavWalk {
    pub status: NavStatus,
    pub destination: Option<[f32; 2]>,
    pub points: Vec<Vec3>,
    pub next: usize,
    pub length: f32,
    pub searched: usize,
}

/// What navigation did at the last fixed tick: agents, and
/// those searching (and of them those waiting for records or their
/// terrain's bakes), walking, arrived and unable to reach their
/// destination; the samples of search work the tick did; and the tiles'
/// grounds and portals held for walks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct NavStats {
    /// Navigation agents.
    pub agents: u32,
    /// Agents whose walk is being searched.
    pub searching: u32,
    /// Agents whose search waits for records the host has yet to hand
    /// back, or for its terrain's bakes.
    pub waiting: u32,
    /// Agents walking a found path.
    pub walking: u32,
    /// Agents at their destination.
    pub arrived: u32,
    /// Agents whose destination no walk reaches.
    pub unreachable: u32,
    /// Samples of search work the tick did.
    pub work: u64,
    /// Tiles whose ground navigation holds for walks.
    pub grounds: u32,
    /// Tiles whose portals of a walk graph navigation holds.
    pub portals: u32,
}

/// Agents' walks being searched, the records they read, walks asked of
/// navigation alone ([`NavAgents::query`]), and what navigation did at the
/// last fixed tick. Inserted by default.
#[derive(Resource, Default)]
pub struct NavAgents {
    // Each agent's search, and each walk asked for alone.
    searches: Vec<(Entity, Search)>,
    queries: Vec<WalkQuery>,
    records: WalkRecords,
    // The walk graphs cooking, while a session cooks.
    cook: Vec<CookGraph>,
    // Where the next tick's searches start, round robin, and the fixed
    // tick now.
    turn: usize,
    tick: u64,
    stats: NavStats,
    // Records walks read that failed on workers, by fixed tick.
    failures: JobFailures,
}

/// A walk being searched: waiting for its terrain's bakes, or under way.
enum Search {
    Starting {
        from: Vec2,
        to: Vec2,
        options: NavOptions,
    },
    Running(Box<Running>),
}

/// A search under way: the terrain it walks as it stood when it set out,
/// the search, the records it read, and how many of the tiles ahead of it
/// it asked for the portals of.
struct Running {
    terrain: Arc<WalkedTerrain>,
    search: WalkSearch,
    held: Held,
    asked_ahead: usize,
}

/// The tiles' grounds and portals a search read, by tile.
#[derive(Default)]
struct Held {
    grounds: BTreeMap<TileKey, Arc<WalkGround>>,
    portals: BTreeMap<TileKey, Arc<TilePortals>>,
}

impl WalkData for Held {
    fn ground(&self, key: TileKey) -> Option<Arc<WalkGround>> {
        self.grounds.get(&key).cloned()
    }

    fn portals(&self, key: TileKey) -> Option<Arc<TilePortals>> {
        self.portals.get(&key).cloned()
    }
}

/// What a walk found: its terrain's origin, and the walk (terrain-local),
/// none where no walk reaches.
type Found = (Vec3, Option<NavPath>);

impl Search {
    fn new(from: Vec2, to: Vec2, options: NavOptions) -> Self {
        Self::Starting { from, to, options }
    }

    /// Whether it waits for records or its terrain's bakes.
    fn waiting(&self) -> bool {
        match self {
            Self::Starting { .. } => true,
            Self::Running(running) => !running.search.wants().is_empty(),
        }
    }

    /// Advances the search within `budget` over `terrains` and the records
    /// held, asking `slots` for those it lacks at fixed tick `tick`: what it
    /// found once done.
    fn advance(
        &mut self,
        terrains: &Terrains,
        records: &mut WalkRecords,
        slots: &mut Option<JobBudget<'_>>,
        budget: &mut u64,
        tick: u64,
    ) -> Option<Found> {
        if let Self::Starting { from, to, options } = self {
            let terrain = match terrains.walked_terrain(*from, *to) {
                Walked::Ready(terrain) => terrain,
                Walked::Baking => return None,
                Walked::Off => return Some((Vec3::ZERO, None)),
            };
            let corner = Vec2::new(terrain.origin.x, terrain.origin.z);
            let search = WalkSearch::new(
                &terrain.pyramid,
                (*from - corner, *to - corner),
                terrains.walk_obstacles(terrain.entity),
                options,
            );
            *self = Self::Running(Box::new(Running {
                terrain: Arc::new(terrain),
                search,
                held: Held::default(),
                asked_ahead: 0,
            }));
        }
        let Self::Running(running) = self else {
            return None;
        };
        let origin = running.terrain.origin;
        loop {
            records.ask_ahead(running, slots, tick);
            if let Some(found) = running.search.step(budget, &running.held) {
                return Some((origin, found));
            }
            records.ask_ahead(running, slots, tick);
            if running.search.wants().is_empty() {
                return None;
            }
            match records.resolve(running, slots, tick) {
                Resolved::Progress => {}
                Resolved::Waiting => return None,
                Resolved::Failed => return Some((origin, None)),
            }
        }
    }
}

/// A walk asked of navigation alone: its ends (world x and z), how it
/// walks, and its search.
struct WalkQuery {
    ends: (Vec2, Vec2),
    options: NavOptions,
    search: Search,
}

/// Walks asked of navigation alone it keeps searching at most: the oldest
/// goes past them.
const QUERIES: usize = 8;

/// What a walk asked of navigation alone found so far.
#[derive(Clone, Debug, PartialEq)]
pub enum NavQuery {
    /// Its search waits for records, or its terrain's bakes: ask again.
    Searching,
    /// The walk, its points world.
    Found(NavPath),
    /// No walk reaches, or no terrain lies under both ends.
    Unreachable,
}

impl NavAgents {
    /// What navigation did at the last fixed tick.
    pub fn stats(&self) -> NavStats {
        self.stats
    }

    /// The records walks read that failed on workers since navigation
    /// started, and the latest one's error.
    pub fn job_failures(&self) -> (u64, Option<&str>) {
        let last = self.failures.last.as_ref();
        (self.failures.count, last.map(|(_, error)| error.as_str()))
    }

    /// The cheapest walk from world `ends.0` to `ends.1` (x and z) over the
    /// terrain both lie on, as `options` walk, its points world,
    /// searched at once over the records the host handed back, those it
    /// lacks asked of `jobs`; until they are back, [`NavQuery::Searching`]:
    /// asked again, the same search goes on. Its work grows with the walk's
    /// length: for planning and tools, not every frame.
    pub fn query(
        &mut self,
        terrains: &Terrains,
        jobs: &mut Jobs,
        ends: (Vec2, Vec2),
        options: &NavOptions,
    ) -> NavQuery {
        self.records.take(jobs, self.tick, &mut self.failures);
        let at = match self
            .queries
            .iter()
            .position(|query| query.ends == ends && query.options == *options)
        {
            Some(at) => at,
            None => {
                if self.queries.len() >= QUERIES {
                    self.queries.remove(0);
                }
                self.queries.push(WalkQuery {
                    ends,
                    options: *options,
                    search: Search::new(ends.0, ends.1, *options),
                });
                self.queries.len() - 1
            }
        };
        let mut slots = Some(JobBudget::of(jobs));
        let mut unlimited = u64::MAX;
        let found = self.queries[at].search.advance(
            terrains,
            &mut self.records,
            &mut slots,
            &mut unlimited,
            self.tick,
        );
        let Some((origin, found)) = found else {
            return NavQuery::Searching;
        };
        self.queries.remove(at);
        match found {
            Some(mut path) => {
                for point in &mut path.points {
                    *point = (Vec3::from_array(*point) + origin).to_array();
                }
                NavQuery::Found(path)
            }
            None => NavQuery::Unreachable,
        }
    }

    /// Stops every search, its records' jobs cancelled.
    fn stop(&mut self, mut jobs: Option<&mut Jobs>) {
        if let Some(jobs) = jobs.as_deref_mut() {
            for id in self.cooking_jobs() {
                jobs.cancel(id);
            }
        }
        self.searches.clear();
        self.queries.clear();
        self.cook.clear();
        self.records.clear(jobs);
    }
}

/// Navigation's systems.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NavSystems {
    /// Searches walks and moves agents along them, in `FixedSimulation`.
    Walk,
}

type Agents<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static NavAgent,
        &'static mut NavWalk,
        &'static mut Transform,
    ),
    Without<Parent>,
>;

/// Searches the walks of agents whose destination changed, then moves
/// walking agents along theirs; while a session cooks, cooks agents' walk
/// graphs.
pub(crate) fn walk_agents(
    time: Res<FixedTime>,
    settings: Option<Res<Navigation>>,
    terrains: Res<Terrains>,
    mut jobs: Option<ResMut<Jobs>>,
    mut cooking: Option<ResMut<Cooking>>,
    mut navigation: ResMut<NavAgents>,
    mut agents: Agents,
) {
    let settings = settings.map_or(Navigation::off(), |settings| settings.sanitized());
    navigation.stats = NavStats::default();
    navigation.tick = time.tick;
    let mut jobs = jobs.as_deref_mut();
    if !settings.enabled {
        navigation.stop(jobs);
        return;
    }
    // Records handed back since the last tick.
    if let Some(jobs) = jobs.as_deref_mut() {
        let navigation = navigation.as_mut();
        navigation
            .records
            .take(jobs, navigation.tick, &mut navigation.failures);
    }
    // New destinations start searches from where agents stand.
    for (entity, agent, mut walk, transform) in &mut agents {
        let agent = agent.sanitized();
        if walk.destination == agent.destination {
            continue;
        }
        navigation.searches.retain(|(owner, _)| *owner != entity);
        *walk = NavWalk {
            destination: agent.destination,
            ..NavWalk::default()
        };
        let Some(destination) = agent.destination else {
            continue;
        };
        let from = Vec2::new(transform.translation.x, transform.translation.z);
        let search = Search::new(from, Vec2::from(destination), agent.options());
        navigation.searches.push((entity, search));
        walk.status = NavStatus::Searching;
    }
    search(
        &settings,
        &terrains,
        jobs.as_deref_mut(),
        &mut navigation,
        &mut agents,
    );
    if let (Some(jobs), Some(cooking)) = (jobs, cooking.as_deref_mut()) {
        navigation.cook(&terrains, &agents, jobs, cooking);
    }
    let (grounds, portals) = navigation.records.held();
    navigation.stats.grounds = grounds;
    navigation.stats.portals = portals;
    // Walkers move along their walks, on the ground.
    let seconds = time.delta_seconds;
    for (_, agent, mut walk, mut transform) in &mut agents {
        navigation.stats.agents += 1;
        match walk.status {
            NavStatus::Searching => navigation.stats.searching += 1,
            NavStatus::Arrived => navigation.stats.arrived += 1,
            NavStatus::Unreachable => navigation.stats.unreachable += 1,
            NavStatus::Idle | NavStatus::Walking => {}
        }
        if walk.status != NavStatus::Walking {
            continue;
        }
        navigation.stats.walking += 1;
        let agent = agent.sanitized();
        step(&agent, &mut walk, &mut transform, &terrains, seconds);
    }
}

/// Advances the searches in progress by the tick's budget, round robin,
/// and gives finished ones' walks to their agents.
fn search(
    settings: &Navigation,
    terrains: &Terrains,
    jobs: Option<&mut Jobs>,
    navigation: &mut NavAgents,
    agents: &mut Agents,
) {
    let mut budget = u64::from(settings.samples_per_tick);
    let mut slots = jobs.map(JobBudget::of);
    let count = navigation.searches.len();
    let mut finished = Vec::new();
    for offset in 0..count {
        if budget == 0 {
            break;
        }
        let index = (navigation.turn + offset) % count;
        let (entity, search) = &mut navigation.searches[index];
        if let Some(found) = search.advance(
            terrains,
            &mut navigation.records,
            &mut slots,
            &mut budget,
            navigation.tick,
        ) {
            finished.push((index, *entity, found));
        }
    }
    navigation.turn = navigation.turn.wrapping_add(1);
    navigation.stats.work = u64::from(settings.samples_per_tick) - budget;
    navigation.stats.waiting = navigation
        .searches
        .iter()
        .filter(|(_, search)| search.waiting())
        .count() as u32;
    finished.sort_by_key(|&(index, ..)| std::cmp::Reverse(index));
    for (index, entity, (origin, found)) in finished {
        navigation.searches.swap_remove(index);
        let Ok((_, _, mut walk, _)) = agents.get_mut(entity) else {
            continue;
        };
        give(&mut walk, origin, found);
    }
}

/// Gives `walk` the walk a search found (terrain-local, of a terrain at
/// `origin`), or no walk where none reaches.
fn give(walk: &mut NavWalk, origin: Vec3, found: Option<NavPath>) {
    match found {
        Some(path) => {
            walk.searched = path.searched;
            walk.points = path
                .points
                .iter()
                .map(|point| Vec3::from_array(*point) + origin)
                .collect();
            walk.next = walk.points.len().min(1);
            walk.length = path.length;
            walk.status = NavStatus::Walking;
        }
        None => walk.status = NavStatus::Unreachable,
    }
}

/// Moves `agent` `seconds` along `walk`, on the ground, turning toward
/// where it walks.
fn step(
    agent: &NavAgent,
    walk: &mut NavWalk,
    transform: &mut Transform,
    terrains: &Terrains,
    seconds: f32,
) {
    let mut position = transform.translation;
    let mut travel = agent.speed * seconds;
    let mut heading = None;
    while travel > 0.0 && walk.next < walk.points.len() {
        let target = walk.points[walk.next];
        let to = Vec2::new(target.x - position.x, target.z - position.z);
        let distance = to.length();
        if distance > 1.0e-4 {
            heading = Some(to / distance);
        }
        if distance <= travel {
            position.x = target.x;
            position.z = target.z;
            travel -= distance;
            walk.next += 1;
        } else {
            let along = to / distance * travel;
            position.x += along.x;
            position.z += along.y;
            travel = 0.0;
        }
    }
    // On the finest resident ground, else the walk's own; and on the walk's
    // own where it runs above the ground, over a bridge's deck.
    let walked = walk_height(walk, position);
    let ground = terrains
        .height_at(position.x, position.z)
        .map_or(walked, |ground| {
            if walked > ground + DECK_ABOVE {
                walked
            } else {
                ground
            }
        });
    position.y = ground + agent.offset;
    transform.translation = position;
    if walk.next >= walk.points.len() {
        walk.status = NavStatus::Arrived;
    }
    if let Some(heading) = heading {
        // Its -z turns toward where it walks.
        let target = Quat::from_rotation_y(f32::atan2(-heading.x, -heading.y));
        let angle = transform.rotation.angle_between(target);
        let turn = agent.turn.to_radians() * seconds;
        let share = if angle > turn { turn / angle } else { 1.0 };
        transform.rotation = transform
            .rotation
            .slerp(target, max(share, 0.0))
            .normalize();
    }
}

/// Meters a walk runs above the ground beneath before agents stand on the
/// walk's own height, as on a bridge's deck: decks clear the
/// water below them by more.
const DECK_ABOVE: f32 = 1.0;

/// The walk's height at `position` between the points around its next.
fn walk_height(walk: &NavWalk, position: Vec3) -> f32 {
    let last = walk.points.len().saturating_sub(1);
    let (Some(a), Some(b)) = (
        walk.points.get(walk.next.saturating_sub(1).min(last)),
        walk.points.get(walk.next.min(last)),
    ) else {
        return position.y;
    };
    let span = Vec2::new(b.x - a.x, b.z - a.z);
    let along = Vec2::new(position.x - a.x, position.z - a.z);
    let t = if span.length_squared() > 1.0e-8 {
        clamp(along.dot(span) / span.length_squared(), 0.0, 1.0)
    } else {
        1.0
    };
    a.y + (b.y - a.y) * t
}

/// Navigation agents walking over terrain: [`crate::Feature::Navigation`].
/// Needs [`crate::TerrainPlugin`], whose terrains it searches.
#[derive(Default)]
pub struct NavigationPlugin;

impl bevy_app::Plugin for NavigationPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        crate::install(app, crate::Feature::Navigation);
        app.init_resource::<NavAgents>()
            .add_systems(
                crate::FixedSimulation,
                (walk_agents, hold_heights).chain().in_set(NavSystems::Walk),
            )
            .configure_sets(
                crate::FixedSimulation,
                NavSystems::Walk.in_set(crate::FixedSimulationSystems::Move),
            );
    }
}

/// Walking agents keep their own heights on their paths' decks: grounding
/// leaves them meanwhile.
fn hold_heights(mut agents: Query<(&NavWalk, &mut crate::HeightHeld), Changed<NavWalk>>) {
    for (walk, mut held) in &mut agents {
        let walking = walk.status == NavStatus::Walking;
        if held.0 != walking {
            held.0 = walking;
        }
    }
}
