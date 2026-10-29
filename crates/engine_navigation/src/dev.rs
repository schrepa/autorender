use bevy_ecs::prelude::{Entity, World};
use bevy_ecs::world::Mut;
use engine_core::{
    Error, Ground, GroundRead, GroundSlot, Jobs, Operation, Slot, Transform, missing_brick,
};
use engine_settings::{ErrorCode, InvalidSetting, Rule, Setting};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{NavAgent, NavAgents, NavOptions, NavQuery, NavStats, NavStatus, NavWalk};

const COORDINATE: Rule = Rule::Range(-1.0e6, 1.0e6);
const FIRST_AGENTS: usize = 16;

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct NavAgentView {
    pub agent: NavAgent,
    pub status: NavStatus,
    pub searched_for: Option<[f32; 2]>,
    pub length: f32,
    pub points: usize,
    pub next: usize,
    pub left: f32,
    pub searched: usize,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct NavAgentsView {
    pub agents: u32,
    pub searching: u32,
    pub waiting: u32,
    pub walking: u32,
    pub arrived: u32,
    pub unreachable: u32,
    pub work: u64,
    pub grounds: u32,
    pub portals: u32,
    pub failed_jobs: u64,
    pub latest_job_failure: Option<String>,
    pub first: Vec<NavAgentRow>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct NavAgentRow {
    pub id: Option<String>,
    pub status: NavStatus,
    pub position: [f32; 3],
    pub destination: Option<[f32; 2]>,
    pub left: f32,
}

fn remaining(walk: &NavWalk, position: Vec3) -> f32 {
    let mut from = position;
    let mut left = 0.0;
    for point in walk.points.iter().skip(walk.next) {
        left += from.distance(*point);
        from = *point;
    }
    left
}

pub(crate) fn agent_view(world: &World, entity: Entity, agent: &NavAgent) -> NavAgentView {
    let agent = *agent;
    let still = NavWalk::default();
    let walk = world.get::<NavWalk>(entity).unwrap_or(&still);
    let position = world
        .get::<Transform>(entity)
        .map(|transform| transform.translation);
    NavAgentView {
        agent,
        status: walk.status,
        searched_for: walk.destination,
        length: walk.length,
        points: walk.points.len(),
        next: walk.next,
        left: position.map_or(0.0, |position| remaining(walk, position)),
        searched: walk.searched,
    }
}

pub(crate) fn agents_view(world: &World) -> Option<NavAgentsView> {
    let navigation = world.get_resource::<NavAgents>()?;
    let stats = world
        .get_resource::<NavStats>()
        .copied()
        .unwrap_or_default();
    let (failed_jobs, latest_job_failure) = navigation.job_failures();
    let latest_job_failure = latest_job_failure.map(str::to_owned);
    let mut query = world.try_query::<(Entity, &NavAgent, &NavWalk, &Transform)>()?;
    let first = query
        .iter(world)
        .take(FIRST_AGENTS)
        .map(|(entity, agent, walk, transform)| NavAgentRow {
            id: engine_core::authored_id(world, entity),
            status: walk.status,
            position: transform.translation.to_array(),
            destination: agent.destination,
            left: remaining(walk, transform.translation),
        })
        .collect();
    Some(NavAgentsView {
        agents: stats.agents,
        searching: stats.searching,
        waiting: stats.waiting,
        walking: stats.walking,
        arrived: stats.arrived,
        unreachable: stats.unreachable,
        work: stats.work,
        grounds: stats.grounds,
        portals: stats.portals,
        failed_jobs,
        latest_job_failure,
        first,
    })
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct NavGoControl {
    pub entity: String,
    #[serde(default)]
    pub destination: Option<[f32; 2]>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct NavGoResult {
    pub temporary: bool,
    pub destination: Option<[f32; 2]>,
}

impl Operation for NavGoControl {
    const NAME: &'static str = "nav_go";
    const CONTROL: bool = true;
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<Value, Error> {
        let Self {
            entity: id,
            destination,
        } = request;
        if let Some(destination) = destination
            && !destination
                .iter()
                .all(|value| value.is_finite() && value.abs() <= 1.0e6)
        {
            return Err(Error::new(ErrorCode::OutOfRange, "$.destination")
                .range(-1.0e6, 1.0e6)
                .got(destination));
        }
        let entity = engine_core::authored_entity(world, &id)
            .filter(|&entity| world.get::<NavAgent>(entity).is_some())
            .ok_or_else(|| Error::new(ErrorCode::UnknownEntity, "$.entity").got(&id))?;
        let mut agent = world.get_mut::<NavAgent>(entity).expect("agent_found");
        agent.destination = destination;
        Ok(json!(NavGoResult {
            temporary: true,
            destination,
        }))
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(NavGoControl)
    }
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum GroundKey {
    Authored(String),
    Entity(u64),
}

fn ground_key(world: &World, entity: Entity) -> GroundKey {
    engine_core::authored_id(world, entity)
        .map_or_else(|| GroundKey::Entity(entity.to_bits()), GroundKey::Authored)
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct NavPathResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terrain: Option<GroundKey>,
    pub options: NavOptions,
    pub searching: bool,
    pub walk: Option<WalkResult>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct WalkResult {
    pub points: Vec<[f64; 3]>,
    pub length: f32,
    pub level: u32,
    pub searched: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct NavPathQuery {
    pub from: [f64; 2],
    pub to: [f64; 2],
    #[serde(default)]
    pub options: Option<NavOptions>,
}

impl Operation for NavPathQuery {
    const NAME: &'static str = "nav_path";
    const CONTROL: bool = false;
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<Value, Error> {
        walk(world, request)
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(NavPathQuery)
    }
}

fn coordinate(value: f64, path: &str) -> Result<f32, Error> {
    match value.check("value", COORDINATE) {
        Ok(()) => Ok(value as f32),
        Err(invalid) => Err(out_of_range(invalid, path)),
    }
}

fn out_of_range(invalid: InvalidSetting, path: &str) -> Error {
    Error {
        code: ErrorCode::OutOfRange,
        path: path.to_owned(),
        ..Error::from(invalid)
    }
}

fn walk(
    world: &mut World,
    NavPathQuery { from, to, options }: NavPathQuery,
) -> Result<Value, Error> {
    let from = Vec2::new(
        coordinate(from[0], "$.from[0]")?,
        coordinate(from[1], "$.from[1]")?,
    );
    let to = Vec2::new(coordinate(to[0], "$.to[0]")?, coordinate(to[1], "$.to[1]")?);
    let options = options.unwrap_or_default();
    options.validate().map_err(|invalid| Error {
        code: ErrorCode::OutOfRange,
        ..Error::from(invalid).within("$.options")
    })?;
    let ground = world
        .get_resource::<Ground>()
        .cloned()
        .ok_or_else(|| missing_brick(GroundSlot::NAME, "$.op"))?;
    if !world.contains_resource::<NavAgents>() || !world.contains_resource::<Jobs>() {
        return Err(missing_brick("navigation", "$.op"));
    }
    let terrain = match ground.face().source(world, from, to) {
        GroundRead::Ready(source) => Some(ground_key(world, source.ground)),
        GroundRead::Waiting => None,
        GroundRead::Off => return Err(Error::new(ErrorCode::UnknownTerrain, "$")),
    };
    let found = world.resource_scope(|world, mut navigation: Mut<NavAgents>| {
        world.resource_scope(|world, mut jobs: Mut<Jobs>| {
            navigation.query(world, &mut jobs, (from, to), &options)
        })
    });
    let round = |value: f32| (f64::from(value) * 100.0).round() / 100.0;
    Ok(json!(NavPathResult {
        terrain,
        options,
        searching: found == NavQuery::Searching,
        walk: match found {
            NavQuery::Found(walk) => Some(WalkResult {
                points: walk.points.iter().map(|point| point.map(round)).collect(),
                length: walk.length,
                level: walk.level,
                searched: walk.searched,
            }),
            NavQuery::Searching | NavQuery::Unreachable => None,
        },
    }))
}
