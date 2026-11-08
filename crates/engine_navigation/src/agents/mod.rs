mod cook;
mod obstacles;
mod records;

use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{
    Error, ErrorCode, Filled, FixedTime, GAMEPLAY_JOBS, GlobalTransforms,
    Interest, JobFailures, JobKind, JobRef, Jobs, Parent, RecordKey, Records, Request, Requests,
    Transform, fields, job_output, job_priority, settings, stats,
};
use engine_math::{atan2, max};
use engine_places_simulation::ground::{
    Ground, GroundGrid, GroundObstacle, GroundRead, GroundSource, Pyramid, TileKey,
};
use engine_places_simulation::navigation::{NavAgent, NavStatus, NavStep, NavWalk};
use engine_places_simulation::physics::{
    Collider, ColliderShape, Collision, QueryFilter, SensorFilter, Slid,
};
use glam::{Mat4, Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::jobs::{
    GroundJob, PORTALS, PORTALS_NAME, PortalsJob, WALK_GROUND, WALK_GROUND_NAME, portals_output,
};
use crate::{
    Gait, NavOptions, NavPath, Stepped, TilePortals, WalkData, WalkGround, WalkSearch, Want,
};
pub(crate) use cook::cook;
pub use obstacles::ObstacleColliders;
use records::{Interests, Resolved, WalkRecords};

settings! {
    #[section("navigation")]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneNavigation")]
    pub struct Navigation {
        pub enabled: bool = true => any,
        pub samples_per_tick: u32 = 32_768 => at_least(1024),
        pub obstacle_layers: u32 = 0 => any,
        pub grounds_per_tick: u32 = 1 => at_least(1),
        pub kept_grounds: u32 = 64 => any,
        pub kept_portals: u32 = 8192 => any,
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
            radius,
            max_slope,
            max_water,
            level,
        } = agent;
        Self {
            destination,
            speed,
            turn,
            radius,
            max_slope,
            max_water,
            level,
        }
    }
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
        pub refused: u32,
        pub collision_errors: u32,
        pub work: u64,
        pub grounds: u32,
        pub portals: u32,
    }
}

#[derive(Resource, Default)]
pub struct NavAgents {
    searches: Vec<(Entity, Search)>,
    query: Option<WalkQuery>,
    records: WalkRecords,
    turn: usize,
    tick: u64,
    elapsed: f64,
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
    interests: Interests,
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
        (ground, colliders): (&Ground, &ObstacleColliders),
        (records, jobs): (&mut WalkRecords, &mut Jobs),
        budget: &mut u64,
    ) -> Option<Found> {
        if let Self::Starting { from, to, options } = self {
            let walked = match ground_source(ground, *from, *to) {
                GroundRead::Ready(walked) => walked,
                GroundRead::NotLoaded => return None,
                GroundRead::Off => return Some((Vec3::ZERO, None)),
            };
            let corner = Vec2::new(walked.source.origin.x, walked.source.origin.z);
            let mut obstacles = ground_obstacles(ground, walked.source.ground);
            obstacles.extend(colliders.read(corner));
            let search = WalkSearch::new(
                &walked.source.pyramid,
                (*from - corner, *to - corner),
                obstacles,
                options,
            );
            *self = Self::Running(Box::new(Running {
                walked,
                search,
                held: Held::default(),
                interests: Interests::default(),
                asked_ahead: 0,
            }));
        }
        let Self::Running(running) = self else {
            return None;
        };
        let origin = running.walked.source.origin;
        loop {
            records.ask_ahead(running, jobs);
            let blocked = match running.search.step(budget, &running.held) {
                Stepped::Found(found) => return Some((origin, found)),
                Stepped::Spent => false,
                Stepped::Blocked => true,
            };
            records.ask_ahead(running, jobs);
            if !blocked {
                return None;
            }
            match records.resolve(running, jobs) {
                Resolved::Progress => {}
                Resolved::Waiting => return None,
                Resolved::Failed => return Some((origin, None)),
            }
        }
    }

    fn interests(self) -> Interests {
        match self {
            Self::Starting { .. } => Interests::default(),
            Self::Running(running) => running.interests,
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
        (settings, ground, colliders): (&Navigation, &Ground, &ObstacleColliders),
        (records, jobs): (&Records, &mut Jobs),
        ends: (Vec2, Vec2),
        options: &NavOptions,
    ) -> NavQuery {
        self.settle_records(settings, records);
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
        let found = asked
            .search
            .advance((ground, colliders), (&mut self.records, jobs), &mut unlimited);
        let Some((origin, found)) = found else {
            self.query = Some(asked);
            return NavQuery::Searching;
        };
        self.records.keep(asked.search.interests(), settings);
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

    fn settle_records(&mut self, settings: &Navigation, records: &Records) {
        self.records.settle(
            (records, self.elapsed, self.tick),
            settings,
            &mut self.failures,
        );
    }
}

pub(crate) fn settle_walk_records(
    (settings, records): (Res<Navigation>, Res<Records>),
    mut navigation: ResMut<NavAgents>,
) {
    navigation.settle_records(&settings, &records);
}

type Agents<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static NavAgent,
        &'static mut NavWalk,
        &'static Transform,
    ),
    Without<Parent>,
>;

fn ground_source(ground: &Ground, from: Vec2, to: Vec2) -> GroundRead<Walked> {
    match ground.0.source(from, to) {
        GroundRead::Ready(source) => ground.0.grid().map_or(GroundRead::Off, |grid| {
            GroundRead::Ready(Walked { source, grid })
        }),
        GroundRead::NotLoaded => GroundRead::NotLoaded,
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
    (time, settings): (Res<FixedTime>, Res<Navigation>),
    (ground, colliders, records): (Res<Ground>, ObstacleColliders, Res<Records>),
    (collision, collides, bodies): (Res<Collision>, Filled<Collision>, Query<&Collider>),
    mut jobs: Jobs,
    mut navigation: ResMut<NavAgents>,
    mut stats: ResMut<NavStats>,
    (mut agents, globals): (Agents, Res<GlobalTransforms>),
) {
    *stats = walk(
        (&*settings, *time),
        ((&*ground, &colliders), &*records, &mut jobs),
        (collides.on().then_some(&*collision), &bodies),
        &mut navigation,
        (&mut agents, &globals),
    );
}

fn walk(
    (settings, time): (&Navigation, FixedTime),
    ((ground, colliders), records, jobs): ((&Ground, &ObstacleColliders), &Records, &mut Jobs),
    bodies: (Option<&Collision>, &Query<&Collider>),
    navigation: &mut NavAgents,
    (agents, globals): (&mut Agents, &GlobalTransforms),
) -> NavStats {
    let mut stats = NavStats::default();
    navigation.tick = time.tick;
    navigation.elapsed = time.elapsed_seconds;
    navigation.settle_records(settings, records);
    for (entity, agent, mut walk, transform) in agents.iter_mut() {
        if let Err(invalid) = SceneNavAgent::from(*agent).validate() {
            let field = invalid.field.into_owned();
            if walk.refused.as_ref() != Some(&field) {
                navigation.searches.retain(|(owner, _)| *owner != entity);
                *walk = NavWalk {
                    status: NavStatus::Refused,
                    refused: Some(field),
                    ..NavWalk::default()
                };
            }
            continue;
        }
        if walk.status == NavStatus::Refused {
            *walk = NavWalk::default();
        }
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
        let search = Search::new(from, Vec2::from(destination), options(agent));
        navigation.searches.push((entity, search));
        walk.status = NavStatus::Searching;
    }
    search(
        settings,
        ((ground, colliders), &mut *jobs),
        (&mut *navigation, &mut stats),
        agents,
    );
    let (grounds, portals) = navigation.records.held();
    stats.grounds = grounds;
    stats.portals = portals;
    let seconds = time.delta_seconds;
    for (entity, agent, mut walk, transform) in agents.iter_mut() {
        stats.agents += 1;
        match walk.status {
            NavStatus::Searching => stats.searching += 1,
            NavStatus::Arrived => stats.arrived += 1,
            NavStatus::Unreachable => stats.unreachable += 1,
            NavStatus::Refused => stats.refused += 1,
            NavStatus::Idle | NavStatus::Walking => {}
        }
        if walk.status != NavStatus::Walking || !globals.active(entity) {
            if walk.step.is_some() {
                walk.step = None;
            }
            continue;
        }
        stats.walking += 1;
        let body = Body::of(bodies, entity, transform);
        let (next, failed) = step(agent, &mut walk, transform, seconds, body);
        stats.collision_errors += u32::from(failed);
        walk.step = Some(next);
    }
    stats
}

fn search(
    settings: &Navigation,
    ((ground, colliders), jobs): ((&Ground, &ObstacleColliders), &mut Jobs),
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
            (ground, colliders),
            (&mut navigation.records, &mut *jobs),
            &mut budget,
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
        let (_, search) = navigation.searches.swap_remove(index);
        navigation.records.keep(search.interests(), settings);
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

struct Body<'a> {
    collision: &'a Collision,
    shape: ColliderShape,
    matrix: Mat4,
    filter: QueryFilter,
}

impl<'a> Body<'a> {
    fn of(
        (collision, colliders): (Option<&'a Collision>, &Query<&Collider>),
        entity: Entity,
        transform: &Transform,
    ) -> Option<Self> {
        let collision = collision?;
        let collider = colliders.get(entity).ok()?;
        Some(Self {
            collision,
            shape: collider.shape(),
            matrix: transform.matrix(),
            filter: QueryFilter {
                layers: collider.layers(),
                sensors: SensorFilter::Exclude,
                exclude: Some(entity),
            },
        })
    }

    fn slide(&self, from: Vec3, along: Vec2) -> Result<Slid, Error> {
        let mut matrix = self.matrix;
        matrix.w_axis = from.extend(1.0);
        let motion = Vec3::new(along.x, 0.0, along.y);
        self.collision
            .0
            .slide(self.shape, matrix, motion, self.filter)
    }
}

fn step(
    agent: &NavAgent,
    walk: &mut NavWalk,
    transform: &Transform,
    seconds: f32,
    body: Option<Body<'_>>,
) -> (NavStep, bool) {
    let mut position = transform.translation;
    let mut travel = agent.speed * seconds;
    let mut heading = None;
    let mut failed = false;
    let mut blocked = None;
    while travel > 0.0 && walk.next < walk.points.len() {
        let target = walk.points[walk.next];
        let to = Vec2::new(target.x - position.x, target.z - position.z);
        let distance = to.length();
        if distance > 1.0e-4 {
            heading = Some(to / distance);
        }
        let reached = distance <= travel;
        let along = if reached { to } else { to / distance * travel };
        match body.as_ref().map(|body| body.slide(position, along)) {
            Some(Ok(slid)) if slid.hits.iter().any(Option::is_some) => {
                position.x = slid.position.x;
                position.z = slid.position.z;
                blocked = slid.hits.iter().flatten().next().map(|hit| hit.entity);
                walk.next += usize::from(reached);
                break;
            }
            Some(Err(_)) => {
                failed = true;
                break;
            }
            Some(Ok(_)) | None => {}
        }
        if reached {
            position.x = target.x;
            position.z = target.z;
            travel -= distance;
            walk.next += 1;
        } else {
            position.x += along.x;
            position.z += along.y;
            travel = 0.0;
        }
    }
    position.y = walk
        .walked(Vec2::new(position.x, position.z))
        .unwrap_or(position.y);
    if walk.next >= walk.points.len() {
        walk.status = NavStatus::Arrived;
    }
    let facing = heading.map_or(transform.rotation, |heading| {
        let target = Quat::from_rotation_y(atan2(-heading.x, -heading.y));
        let angle = transform.rotation.angle_between(target);
        let turn = agent.turn.to_radians() * seconds;
        let share = if angle > turn { turn / angle } else { 1.0 };
        transform
            .rotation
            .slerp(target, max(share, 0.0))
            .normalize()
    });
    let next = NavStep {
        to: position,
        facing,
        blocked,
    };
    (next, failed)
}
