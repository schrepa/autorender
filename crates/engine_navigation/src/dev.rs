use bevy_ecs::prelude::{Entity, World};
use bevy_ecs::world::Mut;
use engine_core::{Error, Jobs, Operation, Transform};
use engine_settings::{ErrorCode, Rule, Setting};
use engine_terrain::{Terrain, TerrainKey, Terrains, terrain_key};
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

pub(crate) fn agent_view(world: &World, entity: Entity) -> Option<NavAgentView> {
    let agent = *world.get::<NavAgent>(entity)?;
    let walk = world.get::<NavWalk>(entity)?;
    let position = world
        .get::<Transform>(entity)
        .map(|transform| transform.translation);
    Some(NavAgentView {
        agent,
        status: walk.status,
        searched_for: walk.destination,
        length: walk.length,
        points: walk.points.len(),
        next: walk.next,
        left: position.map_or(0.0, |position| remaining(walk, position)),
        searched: walk.searched,
    })
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
            return Err(Error::new(
                ErrorCode::OutOfRange,
                "$.destination",
                "must be finite and within 1000000 meters",
            ));
        }
        let entity = engine_core::authored_entity(world, &id)
            .filter(|&entity| world.get::<NavAgent>(entity).is_some())
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::UnknownEntity,
                    "$.entity",
                    format!("no navigation agent {id} in the scene"),
                )
            })?;
        let mut agent = world.get_mut::<NavAgent>(entity).expect("checked above");
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
pub struct NavPathResult {
    pub terrain: TerrainKey,
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
        Err(invalid) => Err(Error::new(ErrorCode::OutOfRange, path, invalid.rule)),
    }
}

fn missing(brick: &str) -> Error {
    Error::new(
        ErrorCode::MissingPlugin,
        "$.op",
        format!("the {brick} brick is not added: add it to the engine"),
    )
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
    options.validate().map_err(|invalid| {
        Error::new(
            ErrorCode::OutOfRange,
            format!("$.options.{}", invalid.field),
            invalid.rule,
        )
    })?;
    let entity = world
        .query::<(Entity, &Terrain)>()
        .iter(world)
        .map(|(entity, _)| entity)
        .next()
        .ok_or_else(|| Error::new(ErrorCode::UnknownTerrain, "$", "no terrain"))?;
    if !world.contains_resource::<Terrains>() {
        return Err(missing("terrain"));
    }
    if !world.contains_resource::<NavAgents>() || !world.contains_resource::<Jobs>() {
        return Err(missing("navigation"));
    }
    let found = world.resource_scope(|world, mut navigation: Mut<NavAgents>| {
        world.resource_scope(|world, mut jobs: Mut<Jobs>| {
            navigation.query(
                world.resource::<Terrains>(),
                &mut jobs,
                (from, to),
                &options,
            )
        })
    });
    let round = |value: f32| (f64::from(value) * 100.0).round() / 100.0;
    Ok(json!(NavPathResult {
        terrain: terrain_key(world, entity),
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
