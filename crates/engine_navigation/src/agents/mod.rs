mod cook;
mod records;

use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{
    CookMode, Error, ErrorCode, FixedTime, GAMEPLAY_JOBS, IDLE_JOBS, JobFailures, JobKind, JobRef,
    Jobs, Parent, Records, Ticket, Transform, fields, job_output, job_priority, settings, stats,
};
use engine_math::{clamp, max};
use engine_places::ground::{
    Ground, GroundGrid, GroundObstacle, GroundRead, GroundSource, HeightHeld, Pyramid, TileKey,
};
use engine_places::navigation::{NavAgent, NavStatus, NavWalk};
use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::jobs::{
    GroundJob, PORTALS, PORTALS_NAME, PortalsJob, WALK_GROUND, WALK_GROUND_NAME, portals_output,
};
use crate::{Gait, NavOptions, NavPath, TilePortals, WalkData, WalkGround, WalkSearch, Want};
use cook::CookGraph;
use records::{Resolved, WalkRecords};

settings! {
    #[section("navigation")]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneNavigation")]
    pub struct Navigation {
        pub enabled: bool = true => any,
        pub samples_per_tick: u32 = 32_768 => at_least(1024),
    }
}

fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    #[schemars(rename = "NavAgent")]
    pub struct SceneNavAgent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub destination: Option<[f32; 2]> = NavAgent::default().destination => finite,
        pub speed: f32 = NavAgent::default().speed => range(0.01, 100.0),
        pub turn: f32 = NavAgent::default().turn => range(1.0, 10_000.0),
        pub offset: f32 = NavAgent::default().offset => range(-10_000.0, 10_000.0),
        pub radius: f32 = NavAgent::default().radius => same(NavOptions::rule_of("radius")),
        pub max_slope: f32 = NavAgent::default().max_slope
            => same(NavOptions::rule_of("max_slope")),
        pub max_water: f32 = NavAgent::default().max_water
            => same(NavOptions::rule_of("max_water")),
        pub level: u32 = NavAgent::default().level => same(NavOptions::rule_of("level")),
    }
}

impl From<NavAgent> for SceneNavAgent {
    fn from(agent: NavAgent) -> Self {
        let NavAgent {
            destination,
            speed,
            turn,
            offset,
            radius,
            max_slope,
            max_water,
            level,
        } = agent;
        Self {
            destination,
            speed,
            turn,
            offset,
            radius,
            max_slope,
            max_water,
            level,
        }
    }
}

impl From<SceneNavAgent> for NavAgent {
    fn from(agent: SceneNavAgent) -> Self {
        let SceneNavAgent {
            destination,
            speed,
            turn,
            offset,
            radius,
            max_slope,
            max_water,
            level,
        } = agent;
        Self {
            destination,
            speed,
            turn,
            offset,
            radius,
            max_slope,
            max_water,
            level,
        }
    }
}

fn sanitized(agent: &NavAgent) -> NavAgent {
    SceneNavAgent::from(*agent).sanitized().into()
}

fn options(agent: &NavAgent) -> NavOptions {
    NavOptions {
        level: agent.level,
        max_slope: agent.max_slope,
        radius: agent.radius,
        max_water: agent.max_water,
    }
}

stats! {
    #[section("navigation_stats")]
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
        #[pending] pub cooking: u32,
    }
}

#[derive(Resource, Default)]
pub struct NavAgents {
    searches: Vec<(Entity, Search)>,
    query: Option<WalkQuery>,
    records: WalkRecords,
    cook: Vec<CookGraph>,
    turn: usize,
    tick: u64,
    failures: JobFailures,
}

struct Walked {
    source: GroundSource,
    grid: JobRef,
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
    walked: Walked,
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
        ground: &Ground,
        (records, jobs): (&mut WalkRecords, &mut Jobs),
        budget: &mut u64,
        tick: u64,
    ) -> Option<Found> {
        if let Self::Starting { from, to, options } = self {
            let walked = match ground_source(ground, *from, *to) {
                GroundRead::Ready(walked) => walked,
                GroundRead::Waiting => return None,
                GroundRead::Off => return Some((Vec3::ZERO, None)),
            };
            let corner = Vec2::new(walked.source.origin.x, walked.source.origin.z);
            let search = WalkSearch::new(
                &walked.source.pyramid,
                (*from - corner, *to - corner),
                ground_obstacles(ground, walked.source.ground),
                options,
            );
            *self = Self::Running(Box::new(Running {
                walked,
                search,
                held: Held::default(),
                asked_ahead: 0,
            }));
        }
        let Self::Running(running) = self else {
            return None;
        };
        let origin = running.walked.source.origin;
        loop {
            records.ask_ahead(running, jobs, tick);
            if let Some(found) = running.search.step(budget, &running.held) {
                return Some((origin, found));
            }
            records.ask_ahead(running, jobs, tick);
            if running.search.wants().is_empty() {
                return None;
            }
            match records.resolve(running, jobs, tick) {
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
        ground: &Ground,
        (records, jobs): (&Records, &mut Jobs),
        ends: (Vec2, Vec2),
        options: &NavOptions,
    ) -> NavQuery {
        self.records
            .take(records, jobs, self.tick, &mut self.failures);
        let mut asked = self
            .query
            .take()
            .filter(|query| query.ends == ends && query.options == *options)
            .unwrap_or_else(|| WalkQuery {
                ends,
                options: *options,
                search: Search::new(ends.0, ends.1, *options),
            });
        let mut unlimited = u64::MAX;
        let found = asked.search.advance(
            ground,
            (&mut self.records, jobs),
            &mut unlimited,
            self.tick,
        );
        let Some((origin, found)) = found else {
            self.query = Some(asked);
            return NavQuery::Searching;
        };
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

    fn stop(&mut self, jobs: &mut Jobs) {
        for graph in std::mem::take(&mut self.cook) {
            for ticket in graph.flight {
                jobs.cancel(ticket);
            }
        }
        self.searches.clear();
        self.query = None;
        self.records.clear(jobs);
    }
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

fn ground_source(ground: &Ground, from: Vec2, to: Vec2) -> GroundRead<Walked> {
    match ground.0.source(from, to) {
        GroundRead::Ready(source) => ground.0.grid().map_or(GroundRead::Off, |grid| {
            GroundRead::Ready(Walked { source, grid })
        }),
        GroundRead::Waiting => GroundRead::Waiting,
        GroundRead::Off => GroundRead::Off,
    }
}

fn ground_obstacles(ground: &Ground, entity: Entity) -> Vec<GroundObstacle> {
    ground.0.obstacles(entity)
}

fn ground_input(walked: &Walked, key: TileKey) -> Vec<u8> {
    GroundJob {
        grid: walked.grid.clone(),
        source: walked.source.identity.to_vec(),
        key,
    }
    .input()
}

fn portals_input(walked: &Walked, key: TileKey, gait: &Gait) -> Vec<u8> {
    PortalsJob {
        grid: walked.grid.clone(),
        source: walked.source.identity.to_vec(),
        pyramid: walked.source.pyramid,
        key,
        gait: *gait,
    }
    .input()
}

fn made<T>(
    kind: &str,
    output: Result<Arc<[u8]>, Error>,
    decode: impl FnOnce(&[u8]) -> Result<T, Error>,
) -> Result<T, String> {
    job_output(kind, output.map_err(worker_message), decode)
}

fn worker_message(error: Error) -> String {
    if error.code == ErrorCode::InvalidRecord
        && let Some(Value::String(message)) = &error.detail.got
    {
        return message.clone();
    }
    error.to_string()
}

pub(crate) fn walk_agents(
    (time, settings, cook): (Res<FixedTime>, Res<Navigation>, Res<CookMode>),
    (ground, records): (Res<Ground>, Res<Records>),
    (mut jobs, mut cooks): (Jobs, Jobs),
    mut navigation: ResMut<NavAgents>,
    mut stats: ResMut<NavStats>,
    mut agents: Agents,
) {
    *stats = walk(
        (&*settings, *time, cook.cooking),
        (&*ground, &*records, (&mut jobs, &mut cooks)),
        &mut navigation,
        &mut agents,
    );
}

pub(crate) fn stop_walks(
    time: Res<FixedTime>,
    mut navigation: ResMut<NavAgents>,
    mut jobs: Jobs,
    mut stats: ResMut<NavStats>,
) {
    navigation.tick = time.tick;
    navigation.stop(&mut jobs);
    *stats = NavStats::default();
}

fn walk(
    (settings, time, cooking): (&Navigation, FixedTime, bool),
    (ground, records, (jobs, cooks)): (&Ground, &Records, (&mut Jobs, &mut Jobs)),
    navigation: &mut NavAgents,
    agents: &mut Agents,
) -> NavStats {
    let mut stats = NavStats::default();
    navigation.tick = time.tick;
    navigation
        .records
        .take(records, jobs, navigation.tick, &mut navigation.failures);
    for (entity, agent, mut walk, transform) in agents.iter_mut() {
        let agent = sanitized(agent);
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
        let search = Search::new(from, Vec2::from(destination), options(&agent));
        navigation.searches.push((entity, search));
        walk.status = NavStatus::Searching;
    }
    search(
        settings,
        (ground, &mut *jobs),
        (&mut *navigation, &mut stats),
        agents,
    );
    if cooking {
        let left = navigation.cook(ground, agents, (records, &mut *cooks));
        stats.cooking = u32::try_from(left).unwrap_or(u32::MAX);
    }
    let (grounds, portals) = navigation.records.held();
    stats.grounds = grounds;
    stats.portals = portals;
    let seconds = time.delta_seconds;
    let mut moved = Vec::new();
    for (entity, agent, mut walk, mut transform) in agents.iter_mut() {
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
        let agent = sanitized(agent);
        moved.push((entity, step(&agent, &mut walk, &mut transform, seconds)));
    }
    for (_, step) in &mut moved {
        step.settle(ground);
    }
    for (entity, step) in moved {
        if let Ok((_, _, _, mut transform)) = agents.get_mut(entity) {
            transform.translation = step.position;
        }
    }
    stats
}

fn search(
    settings: &Navigation,
    (ground, jobs): (&Ground, &mut Jobs),
    (navigation, stats): (&mut NavAgents, &mut NavStats),
    agents: &mut Agents,
) {
    let mut budget = u64::from(settings.samples_per_tick);
    let count = navigation.searches.len();
    let mut finished = Vec::new();
    for offset in 0..count {
        if budget == 0 {
            break;
        }
        let index = (navigation.turn + offset) % count;
        let (entity, search) = &mut navigation.searches[index];
        if let Some(found) = search.advance(
            ground,
            (&mut navigation.records, &mut *jobs),
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

struct Step {
    position: Vec3,
    walked: f32,
    offset: f32,
}

impl Step {
    fn settle(&mut self, ground: &Ground) {
        let walked = self.walked;
        let ground = ground
            .sample(Vec2::new(self.position.x, self.position.z))
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

pub(crate) fn hold_heights(mut agents: Query<(&NavWalk, &mut HeightHeld), Changed<NavWalk>>) {
    for (walk, mut held) in &mut agents {
        let walking = walk.status == NavStatus::Walking;
        if held.0 != walking {
            held.0 = walking;
        }
    }
}
