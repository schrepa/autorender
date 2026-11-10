mod obstacles;
mod records;

use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{
    Error, ErrorCode, Filled, FixedTime, GAMEPLAY_JOBS, GlobalTransforms, Interest, JobFailures,
    Jobs, Parent, RecordKey, Records, Refusals, Request, Requests, Transform, fields, job_output,
    job_priority, settings, stats,
};

pub const NAVIGATION_PART: &str = "navigation";
use engine_math::{atan2, max};
use engine_places_simulation::ground::{Ground, GroundProbe, GroundRead};
use engine_places_simulation::navigation::{NavAgent, NavStatus, NavStep, NavWalk};
use engine_places_simulation::physics::{
    Collider, ColliderShape, Collision, QueryFilter, SensorFilter, Slid,
};
use engine_places_simulation::water::WaterBody;
use engine_tiles::{Pyramid, TILE_SAMPLES, TileKey};
use glam::{Mat4, Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::jobs::{PORTALS, PORTALS_NAME, PortalsJob, portals_output};
use crate::{
    Gait, HOLE_WORDS, NavOptions, NavPath, Stepped, TilePortals, WALK_SAMPLES, WalkData,
    WalkGround, WalkSearch, Want,
};
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
        pub floor_layers: u32 = u32::MAX => any,
        pub ground_samples_per_tick: u32 = 16_384 => at_least(1),
        pub kept_grounds: u32 = 64 => any,
        pub kept_portals: u32 = 8192 => any,
        pub walk_center: [f32; 2] = [0.0, 0.0] => range(-1.0e6, 1.0e6),
        pub walk_tiles: u32 = 64 => range(1, 4096),
        pub walk_spacing: f32 = 1.0 => range(0.015625, 1024.0),
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WalkArea {
    pub corner: Vec2,
    pub pyramid: Pyramid,
}

impl WalkArea {
    pub fn of(settings: &Navigation) -> Option<Self> {
        let pyramid = Pyramid::new([settings.walk_tiles; 2], settings.walk_spacing).ok()?;
        let corner = Vec2::from_array(settings.walk_center) - pyramid.extent() * 0.5;
        Some(Self { corner, pyramid })
    }

    fn origin(&self) -> Vec3 {
        Vec3::new(self.corner.x, 0.0, self.corner.y)
    }

    fn holds(&self, point: Vec2) -> bool {
        self.pyramid.tile_at(0, point - self.corner).is_some()
    }
}

pub(crate) struct Floors<'a> {
    ground: &'a Ground,
    waters: Vec<&'a WaterBody>,
    layers: u32,
}

impl<'a> Floors<'a> {
    pub(crate) fn new(
        ground: &'a Ground,
        waters: impl Iterator<Item = &'a WaterBody>,
        layers: u32,
    ) -> Self {
        Self {
            ground,
            waters: waters.collect(),
            layers,
        }
    }

    fn read(&self, at: Vec2) -> GroundRead<Option<f32>> {
        match self
            .ground
            .down(GroundProbe::at(at).on_layers(self.layers))
        {
            GroundRead::Ready(point) => GroundRead::Ready(point.map(|point| point.height)),
            GroundRead::NotLoaded => GroundRead::NotLoaded,
            GroundRead::Off => GroundRead::Off,
        }
    }

    fn water(&self, point: Vec3) -> Option<f32> {
        let mut found: Option<(i32, f32)> = None;
        for body in &self.waters {
            if found.is_some_and(|(precedence, _)| precedence >= body.precedence) {
                continue;
            }
            if let Some(sample) = body.shape.water(point).ready().flatten() {
                found = Some((body.precedence, sample.surface));
            }
        }
        found.map(|(_, surface)| surface)
    }

    fn layout(&self) -> u64 {
        self.ground.layout()
    }
}

struct Sampling {
    key: TileKey,
    next: usize,
    heights: Box<[f32]>,
    holes: Box<[u64]>,
    holed: bool,
    water: Option<Box<[f32]>>,
}

impl Sampling {
    fn new(key: TileKey) -> Self {
        Self {
            key,
            next: 0,
            heights: vec![0.0; WALK_SAMPLES].into_boxed_slice(),
            holes: vec![0; HOLE_WORDS].into_boxed_slice(),
            holed: false,
            water: None,
        }
    }

    fn advance(&mut self, area: &WalkArea, floors: &Floors, budget: &mut u64) -> bool {
        let side = TILE_SAMPLES as usize;
        let level = u32::from(self.key.level);
        let spacing = area.pyramid.spacing(level);
        let first = area.pyramid.first_sample(self.key);
        while self.next < WALK_SAMPLES {
            if *budget == 0 {
                return false;
            }
            *budget -= 1;
            let index = self.next;
            self.next += 1;
            let (column, row) = ((index % side) as i64, (index / side) as i64);
            let at = area.corner
                + Vec2::new((first[0] + column) as f32, (first[1] + row) as f32) * spacing;
            let Some(height) = floors.read(at).ready().flatten() else {
                self.holes[index / 64] |= 1 << (index % 64);
                self.holed = true;
                continue;
            };
            self.heights[index] = height;
            if let Some(surface) = floors.water(Vec3::new(at.x, height, at.y)) {
                self.water
                    .get_or_insert_with(|| vec![f32::NAN; WALK_SAMPLES].into_boxed_slice())
                    [index] = surface;
            }
        }
        true
    }

    fn ground(self, area: &WalkArea) -> WalkGround {
        let spacing = area.pyramid.spacing(u32::from(self.key.level));
        WalkGround::new(
            (self.key, spacing),
            self.heights,
            self.holed.then_some(self.holes),
            self.water,
        )
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

enum Search {
    Starting {
        from: Vec2,
        to: Vec2,
        options: NavOptions,
    },
    Running(Box<Running>),
}

struct Running {
    area: WalkArea,
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

struct Budgets<'b> {
    search: &'b mut u64,
    samples: &'b mut u64,
}

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
        (area, floors, colliders): (WalkArea, &Floors, &ObstacleColliders),
        (records, jobs): (&mut WalkRecords, &mut Jobs),
        budgets: &mut Budgets<'_>,
    ) -> Option<Found> {
        if let Self::Starting { from, to, options } = self {
            if !area.holds(*from) || !area.holds(*to) {
                return Some((Vec3::ZERO, None));
            }
            match floors.read(*from) {
                GroundRead::Ready(_) => {}
                GroundRead::NotLoaded => return None,
                GroundRead::Off => return Some((Vec3::ZERO, None)),
            }
            let corner = area.corner;
            let search = WalkSearch::new(
                &area.pyramid,
                (*from - corner, *to - corner),
                colliders.read(corner),
                options,
            );
            *self = Self::Running(Box::new(Running {
                area,
                search,
                held: Held::default(),
                interests: Interests::default(),
                asked_ahead: 0,
            }));
        }
        let Self::Running(running) = self else {
            return None;
        };
        let origin = running.area.origin();
        records.refresh((floors.layout(), running.area));
        loop {
            records.ask_ahead(running, (floors, jobs), &mut *budgets.samples);
            let blocked = match running.search.step(&mut *budgets.search, &running.held) {
                Stepped::Found(found) => return Some((origin, found)),
                Stepped::Spent => false,
                Stepped::Blocked => true,
            };
            records.ask_ahead(running, (floors, jobs), &mut *budgets.samples);
            if !blocked {
                return None;
            }
            match records.resolve(running, (floors, jobs), &mut *budgets.samples) {
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

    pub(crate) fn query(
        &mut self,
        (settings, floors, colliders): (&Navigation, &Floors, &ObstacleColliders),
        (records, jobs): (&Records, &mut Jobs),
        ends: (Vec2, Vec2),
        options: &NavOptions,
    ) -> NavQuery {
        self.settle_records(settings, records);
        let Some(area) = WalkArea::of(settings) else {
            return NavQuery::Unreachable;
        };
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
        let mut samples = u64::from(settings.ground_samples_per_tick);
        let found = asked.search.advance(
            (area, floors, colliders),
            (&mut self.records, jobs),
            &mut Budgets {
                search: &mut unlimited,
                samples: &mut samples,
            },
        );
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

fn portals_key(layout: u64, key: TileKey, gait: &Gait) -> Vec<u8> {
    let mut input = engine_codec::encode(&layout);
    input.extend_from_slice(&engine_codec::encode(&key));
    input.extend_from_slice(&engine_codec::encode(gait));
    input
}

fn portals_input(area: &WalkArea, ground: &WalkGround, gait: &Gait) -> Vec<u8> {
    PortalsJob {
        ground: ground.clone(),
        pyramid: area.pyramid,
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
    (ground, waters, colliders, records): (
        Res<Ground>,
        Query<&WaterBody>,
        ObstacleColliders,
        Res<Records>,
    ),
    (collision, collides, bodies): (Res<Collision>, Filled<Collision>, Query<&Collider>),
    mut jobs: Jobs,
    (mut navigation, mut refusals): (ResMut<NavAgents>, ResMut<Refusals>),
    mut stats: ResMut<NavStats>,
    (mut agents, globals): (Agents, Res<GlobalTransforms>),
) {
    let floors = Floors::new(&ground, waters.iter(), settings.floor_layers);
    *stats = walk(
        (&*settings, *time),
        ((&floors, &colliders), &*records, &mut jobs),
        (collides.on().then_some(&*collision), &bodies),
        (&mut navigation, &mut refusals),
        (&mut agents, &globals),
    );
}

fn walk(
    (settings, time): (&Navigation, FixedTime),
    ((floors, colliders), records, jobs): ((&Floors, &ObstacleColliders), &Records, &mut Jobs),
    bodies: (Option<&Collision>, &Query<&Collider>),
    (navigation, refusals): (&mut NavAgents, &mut Refusals),
    (agents, globals): (&mut Agents, &GlobalTransforms),
) -> NavStats {
    let mut stats = NavStats::default();
    navigation.tick = time.tick;
    navigation.elapsed = time.elapsed_seconds;
    navigation.settle_records(settings, records);
    for (entity, agent, mut walk, transform) in agents.iter_mut() {
        if let Err(invalid) = SceneNavAgent::from(*agent).validate() {
            let field = invalid.field.to_string();
            refusals.refuse(
                entity,
                NAVIGATION_PART,
                invalid.error(ErrorCode::InvalidValue, "$.nav_agent"),
            );
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
            refusals.accept(entity, NAVIGATION_PART);
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
        ((floors, colliders), &mut *jobs),
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
    ((floors, colliders), jobs): ((&Floors, &ObstacleColliders), &mut Jobs),
    (navigation, stats): (&mut NavAgents, &mut NavStats),
    agents: &mut Agents,
) {
    let mut budget = u64::from(settings.samples_per_tick);
    let mut samples = u64::from(settings.ground_samples_per_tick);
    let count = navigation.searches.len();
    let mut finished = Vec::new();
    if let Some(area) = WalkArea::of(settings) {
        for offset in 0..count {
            if budget == 0 {
                break;
            }
            let index = (navigation.turn + offset) % count;
            let (entity, search) = &mut navigation.searches[index];
            if let Some(found) = search.advance(
                (area, floors, colliders),
                (&mut navigation.records, &mut *jobs),
                &mut Budgets {
                    search: &mut budget,
                    samples: &mut samples,
                },
            ) {
                finished.push((index, *entity, found));
            }
        }
    } else {
        finished.extend(
            navigation
                .searches
                .iter()
                .enumerate()
                .map(|(index, (entity, _))| (index, *entity, (Vec3::ZERO, None))),
        );
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
