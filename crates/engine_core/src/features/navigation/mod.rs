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
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneNavigation")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Navigation {
        pub enabled: bool = true => any,
        pub samples_per_tick: u32 = 32_768 => at_least(1024),
    }
}

impl Navigation {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

settings! {
    #[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[require(NavWalk, crate::HeightHeld)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct NavAgent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub destination: Option<[f32; 2]> = None => finite,
        pub speed: f32 = 1.4 => range(0.01, 100.0),
        pub turn: f32 = 360.0 => range(1.0, 10_000.0),
        pub offset: f32 = 0.0 => range(-10_000.0, 10_000.0),
        pub radius: f32 = 0.4 => same(engine_terrain::NavOptions::rule_of("radius")),
        pub max_slope: f32 = 0.8 => same(engine_terrain::NavOptions::rule_of("max_slope")),
        pub max_water: f32 = 0.6 => same(engine_terrain::NavOptions::rule_of("max_water")),
        pub level: u32 = 0 => same(engine_terrain::NavOptions::rule_of("level")),
    }
}

impl NavAgent {
    fn options(&self) -> NavOptions {
        NavOptions {
            level: self.level,
            max_slope: self.max_slope,
            radius: self.radius,
            max_water: self.max_water,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum NavStatus {
    #[default]
    Idle,
    Searching,
    Walking,
    Arrived,
    Unreachable,
}

#[derive(Component, Clone, Debug, Default)]
pub struct NavWalk {
    pub status: NavStatus,
    pub destination: Option<[f32; 2]>,
    pub points: Vec<Vec3>,
    pub next: usize,
    pub length: f32,
    pub searched: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct NavStats {
    pub agents: u32,
    pub searching: u32,
    pub waiting: u32,
    pub walking: u32,
    pub arrived: u32,
    pub unreachable: u32,
    pub work: u64,
    pub grounds: u32,
    pub portals: u32,
}

#[derive(Resource, Default)]
pub struct NavAgents {
    searches: Vec<(Entity, Search)>,
    queries: Vec<WalkQuery>,
    records: WalkRecords,
    cook: Vec<CookGraph>,
    turn: usize,
    tick: u64,
    stats: NavStats,
    failures: JobFailures,
}

enum Search {
    Starting {
        from: Vec2,
        to: Vec2,
        options: NavOptions,
    },
    Running(Box<Running>),
}

struct Running {
    terrain: Arc<WalkedTerrain>,
    search: WalkSearch,
    held: Held,
    asked_ahead: usize,
}

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

type Found = (Vec3, Option<NavPath>);

impl Search {
    fn new(from: Vec2, to: Vec2, options: NavOptions) -> Self {
        Self::Starting { from, to, options }
    }

    fn waiting(&self) -> bool {
        match self {
            Self::Starting { .. } => true,
            Self::Running(running) => !running.search.wants().is_empty(),
        }
    }

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

struct WalkQuery {
    ends: (Vec2, Vec2),
    options: NavOptions,
    search: Search,
}

const QUERIES: usize = 8;

#[derive(Clone, Debug, PartialEq)]
pub enum NavQuery {
    Searching,
    Found(NavPath),
    Unreachable,
}

impl NavAgents {
    pub fn stats(&self) -> NavStats {
        self.stats
    }

    pub fn job_failures(&self) -> (u64, Option<&str>) {
        let last = self.failures.last.as_ref();
        (self.failures.count, last.map(|(_, error)| error.as_str()))
    }

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

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NavSystems {
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
    if let Some(jobs) = jobs.as_deref_mut() {
        let navigation = navigation.as_mut();
        navigation
            .records
            .take(jobs, navigation.tick, &mut navigation.failures);
    }
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

const DECK_ABOVE: f32 = 1.0;

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

fn hold_heights(mut agents: Query<(&NavWalk, &mut crate::HeightHeld), Changed<NavWalk>>) {
    for (walk, mut held) in &mut agents {
        let walking = walk.status == NavStatus::Walking;
        if held.0 != walking {
            held.0 = walking;
        }
    }
}
