mod cook;
mod records;

use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::math::{clamp, max};
use engine_core::{
    Brick, BrickStats, Bricks, Cooking, EnvironmentSection, FixedSimulation,
    FixedSimulationSystems, FixedTime, GAMEPLAY_JOBS, Ground, GroundObstacle, GroundRead,
    GroundRecord, GroundSource, HeightHeld, IDLE_JOBS, JobBudget, JobFailures, JobId, Jobs, Parent,
    RecordKey, TileKey, Transform, job_output, job_priority,
};
use engine_settings::{InvalidSetting, settings};
use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::dev::{self, NavAgentView, NavAgentsView, NavGoControl, NavPathQuery};
use crate::jobs::{PortalsJob, portals_output};
use crate::{Gait, NavOptions, NavPath, TilePortals, WalkData, WalkGround, WalkSearch, Want};
use cook::CookGraph;
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
    #[require(NavWalk, HeightHeld)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct NavAgent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub destination: Option<[f32; 2]> = None => finite,
        pub speed: f32 = 1.4 => range(0.01, 100.0),
        pub turn: f32 = 360.0 => range(1.0, 10_000.0),
        pub offset: f32 = 0.0 => range(-10_000.0, 10_000.0),
        pub radius: f32 = 0.4 => same(NavOptions::rule_of("radius")),
        pub max_slope: f32 = 0.8 => same(NavOptions::rule_of("max_slope")),
        pub max_water: f32 = 0.6 => same(NavOptions::rule_of("max_water")),
        pub level: u32 = 0 => same(NavOptions::rule_of("level")),
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

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
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
    source: GroundSource,
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
        world: &World,
        records: &mut WalkRecords,
        slots: &mut Option<JobBudget<'_>>,
        budget: &mut u64,
        tick: u64,
    ) -> Option<Found> {
        if let Self::Starting { from, to, options } = self {
            let source = match ground_source(world, *from, *to) {
                GroundRead::Ready(source) => source,
                GroundRead::Waiting => return None,
                GroundRead::Off => return Some((Vec3::ZERO, None)),
            };
            let corner = Vec2::new(source.origin.x, source.origin.z);
            let search = WalkSearch::new(
                &source.pyramid,
                (*from - corner, *to - corner),
                ground_obstacles(world, source.ground),
                options,
            );
            *self = Self::Running(Box::new(Running {
                source,
                search,
                held: Held::default(),
                asked_ahead: 0,
            }));
        }
        let Self::Running(running) = self else {
            return None;
        };
        let origin = running.source.origin;
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
    pub fn job_failures(&self) -> (u64, Option<&str>) {
        let last = self.failures.last.as_ref();
        (self.failures.count, last.map(|(_, error)| error.as_str()))
    }

    pub fn query(
        &mut self,
        world: &World,
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
            world,
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

type Agents = QueryState<
    (
        Entity,
        &'static NavAgent,
        &'static mut NavWalk,
        &'static mut Transform,
    ),
    Without<Parent>,
>;

fn ground_source(world: &World, from: Vec2, to: Vec2) -> GroundRead<GroundSource> {
    world
        .get_resource::<Ground>()
        .map_or(GroundRead::Off, |ground| ground.face().source(world, from, to))
}

fn ground_obstacles(world: &World, entity: Entity) -> Vec<GroundObstacle> {
    world
        .get_resource::<Ground>()
        .map_or_else(Vec::new, |ground| ground.face().obstacles(world, entity))
}

fn scoped<R: Resource, T>(
    world: &mut World,
    run: impl FnOnce(&mut World, Option<Mut<'_, R>>) -> T,
) -> T {
    if world.contains_resource::<R>() {
        world.resource_scope(|world, resource: Mut<R>| run(world, Some(resource)))
    } else {
        run(world, None)
    }
}

pub(crate) fn walk_agents(world: &mut World, agents: &mut Agents) {
    let time = *world.resource::<FixedTime>();
    let settings = world
        .get_resource::<Navigation>()
        .map_or(Navigation::off(), |settings| settings.sanitized());
    let stats = world.resource_scope(|world, mut navigation: Mut<NavAgents>| {
        scoped::<Jobs, _>(world, |world, jobs| {
            scoped::<Cooking, _>(world, |world, cooking| {
                walk(world, agents, (&settings, time), &mut navigation, (jobs, cooking))
            })
        })
    });
    *world.resource_mut::<NavStats>() = stats;
}

fn walk(
    world: &mut World,
    agents: &mut Agents,
    (settings, time): (&Navigation, FixedTime),
    navigation: &mut NavAgents,
    (mut jobs, mut cooking): (Option<Mut<'_, Jobs>>, Option<Mut<'_, Cooking>>),
) -> NavStats {
    let mut stats = NavStats::default();
    navigation.tick = time.tick;
    let mut jobs = jobs.as_deref_mut();
    if !settings.enabled {
        navigation.stop(jobs);
        return stats;
    }
    if let Some(jobs) = jobs.as_deref_mut() {
        navigation
            .records
            .take(jobs, navigation.tick, &mut navigation.failures);
    }
    for (entity, agent, mut walk, transform) in agents.iter_mut(world) {
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
    search(settings, world, jobs.as_deref_mut(), (&mut *navigation, &mut stats), agents);
    if let (Some(jobs), Some(cooking)) = (jobs, cooking.as_deref_mut()) {
        navigation.cook(world, agents, jobs, cooking);
    }
    let (grounds, portals) = navigation.records.held();
    stats.grounds = grounds;
    stats.portals = portals;
    let seconds = time.delta_seconds;
    let mut moved = Vec::new();
    for (entity, agent, mut walk, mut transform) in agents.iter_mut(world) {
        stats.agents += 1;
        match walk.status {
            NavStatus::Searching => stats.searching += 1,
            NavStatus::Arrived => stats.arrived += 1,
            NavStatus::Unreachable => stats.unreachable += 1,
            NavStatus::Idle | NavStatus::Walking => {}
        }
        if walk.status != NavStatus::Walking {
            continue;
        }
        stats.walking += 1;
        let agent = agent.sanitized();
        moved.push((entity, step(&agent, &mut walk, &mut transform, seconds)));
    }
    for (_, step) in &mut moved {
        step.settle(world);
    }
    for (entity, step) in moved {
        if let Ok((_, _, _, mut transform)) = agents.get_mut(world, entity) {
            transform.translation = step.position;
        }
    }
    stats
}

fn search(
    settings: &Navigation,
    world: &mut World,
    jobs: Option<&mut Jobs>,
    (navigation, stats): (&mut NavAgents, &mut NavStats),
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
            world,
            &mut navigation.records,
            &mut slots,
            &mut budget,
            navigation.tick,
        ) {
            finished.push((index, *entity, found));
        }
    }
    navigation.turn = navigation.turn.wrapping_add(1);
    stats.work = u64::from(settings.samples_per_tick) - budget;
    stats.waiting = navigation
        .searches
        .iter()
        .filter(|(_, search)| search.waiting())
        .count() as u32;
    finished.sort_by_key(|&(index, ..)| std::cmp::Reverse(index));
    for (index, entity, (origin, found)) in finished {
        navigation.searches.swap_remove(index);
        let Ok((_, _, mut walk, _)) = agents.get_mut(world, entity) else {
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

struct Step {
    position: Vec3,
    walked: f32,
    offset: f32,
}

impl Step {
    fn settle(&mut self, world: &World) {
        let walked = self.walked;
        let ground = Ground::sample_in(world, self.position.x, self.position.z)
            .map_or(walked, |point| {
                if walked > point.height + DECK_ABOVE {
                    walked
                } else {
                    point.height
                }
            });
        self.position.y = ground + self.offset;
    }
}

fn step(agent: &NavAgent, walk: &mut NavWalk, transform: &mut Transform, seconds: f32) -> Step {
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
    Step {
        position,
        walked,
        offset: agent.offset,
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

fn portals_record(source: &GroundSource, key: TileKey, gait: &Gait) -> (Vec<u8>, RecordKey) {
    let job = PortalsJob {
        source: source.identity.to_vec(),
        pyramid: source.pyramid,
        key,
        gait: *gait,
    };
    let input = job.identity();
    let record = RecordKey::new(&crate::SOURCE, PortalsJob::KIND, &input);
    (input, record)
}

impl EnvironmentSection for Navigation {
    const KEY: &'static str = "navigation";
    const NAME: &'static str = "Navigation";
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
        schemars::schema_for!(Navigation)
    }
}

impl BrickStats for NavStats {
    const KEY: &'static str = "navigation";

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(NavStats)
    }
}

pub struct NavigationBrick;

impl Brick for NavigationBrick {
    fn register(bricks: &mut Bricks) {
        bricks
            .settings::<Navigation>()
            .records(&[PortalsJob::KIND])
            .resource::<NavAgents>()
            .resource::<NavStats>()
            .stats::<NavStats>()
            .systems(
                FixedSimulation,
                (walk_agents, hold_heights).chain().in_set(NavSystems::Walk),
            )
            .configure_sets(
                FixedSimulation,
                NavSystems::Walk.in_set(FixedSimulationSystems::Move),
            )
            .operation::<NavGoControl>()
            .operation::<NavPathQuery>()
            .component_inspector::<NavAgent, NavAgentView>(dev::agent_view)
            .inspector::<NavAgents, Option<NavAgentsView>>(|world, _| dev::agents_view(world));
        crate::smokes::register(bricks);
    }
}

fn hold_heights(mut agents: Query<(&NavWalk, &mut HeightHeld), Changed<NavWalk>>) {
    for (walk, mut held) in &mut agents {
        let walking = walk.status == NavStatus::Walking;
        if held.0 != walking {
            held.0 = walking;
        }
    }
}
