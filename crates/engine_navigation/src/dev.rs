use bevy_ecs::prelude::*;
use engine_core::{
    Error, ErrorCode, InvalidSetting, Jobs, Operation, Records, Rule, SceneIds, Setting, Transform,
};
use engine_places::ground::{Ground, GroundRead};
use engine_places::navigation::{NavAgent, NavStatus, NavWalk};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{NavAgents, NavOptions, NavQuery, NavStats};

const COORDINATE: Rule = Rule::Range(-1.0e6, 1.0e6);
const FIRST_AGENTS: usize = 16;

pub(crate) fn schema_of<T: schemars::JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(T)).unwrap_or(Value::Null)
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct NavAgentView {
    #[schemars(with = "crate::SceneNavAgent")]
    pub agent: NavAgent,
    pub status: NavStatus,
    pub searched_for: Option<[f32; 2]>,
    pub length: f32,
    pub points: usize,
    pub next: usize,
    pub left: f32,
    pub searched: usize,
}

#[derive(Serialize, schemars::JsonSchema)]
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

#[derive(Serialize, schemars::JsonSchema)]
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

fn agent_view(agent: &NavAgent, walk: Option<&NavWalk>, at: Option<&Transform>) -> NavAgentView {
    let agent = *agent;
    let still = NavWalk::default();
    let walk = walk.unwrap_or(&still);
    let position = at.map(|transform| transform.translation);
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

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NavAgentQuery {
    pub entity: String,
}

impl Operation for NavAgentQuery {
    const NAME: &'static str = "nav_agent";
    type Params = (
        SceneIds<'static, 'static>,
        Query<
            'static,
            'static,
            (
                &'static NavAgent,
                Option<&'static NavWalk>,
                Option<&'static Transform>,
            ),
        >,
    );
    type Request = Self;

    fn run(
        (ids, agents): SystemParamItem<'_, '_, Self::Params>,
        Self { entity: id }: Self,
    ) -> Result<Value, Error> {
        let (agent, walk, at) = ids
            .entity(&id)
            .and_then(|entity| agents.get(entity).ok())
            .ok_or_else(|| Error::new(ErrorCode::UnknownEntity, "$.entity").got(&id))?;
        Ok(json!(agent_view(agent, walk, at)))
    }

    fn schema() -> Value {
        schema_of::<Self>()
    }

    fn result_schema() -> Option<Value> {
        Some(schema_of::<NavAgentView>())
    }
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NavAgentsQuery {}

impl Operation for NavAgentsQuery {
    const NAME: &'static str = "nav_agents";
    type Params = (
        Res<'static, NavStats>,
        Res<'static, NavAgents>,
        SceneIds<'static, 'static>,
        Query<
            'static,
            'static,
            (
                Entity,
                &'static NavAgent,
                &'static NavWalk,
                &'static Transform,
            ),
        >,
    );
    type Request = Self;

    fn run(
        (stats, navigation, ids, agents): SystemParamItem<'_, '_, Self::Params>,
        _: Self,
    ) -> Result<Value, Error> {
        let (failed_jobs, latest_job_failure) = navigation.job_failures();
        let latest_job_failure = latest_job_failure.map(str::to_owned);
        let first = agents
            .iter()
            .take(FIRST_AGENTS)
            .map(|(entity, agent, walk, transform)| NavAgentRow {
                id: ids.id(entity).map(str::to_owned),
                status: walk.status,
                position: transform.translation.to_array(),
                destination: agent.destination,
                left: remaining(walk, transform.translation),
            })
            .collect();
        Ok(json!(NavAgentsView {
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
        }))
    }

    fn schema() -> Value {
        schema_of::<Self>()
    }

    fn result_schema() -> Option<Value> {
        Some(schema_of::<NavAgentsView>())
    }
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NavGoControl {
    pub entity: String,
    #[serde(default)]
    pub destination: Option<[f32; 2]>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct NavGoResult {
    pub temporary: bool,
    pub destination: Option<[f32; 2]>,
}

impl Operation for NavGoControl {
    const NAME: &'static str = "nav_go";
    type Params = (
        SceneIds<'static, 'static>,
        Query<'static, 'static, &'static mut NavAgent>,
    );
    type Request = Self;

    fn run(
        (ids, mut agents): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
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
        let mut agent = ids
            .entity(&id)
            .and_then(|entity| agents.get_mut(entity).ok())
            .ok_or_else(|| Error::new(ErrorCode::UnknownEntity, "$.entity").got(&id))?;
        agent.destination = destination;
        Ok(json!(NavGoResult {
            temporary: true,
            destination,
        }))
    }

    fn schema() -> Value {
        schema_of::<Self>()
    }
}

#[derive(Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum GroundKey {
    Authored(String),
    Entity(u64),
}

fn ground_key(ids: &SceneIds, entity: Entity) -> GroundKey {
    ids.id(entity).map_or_else(
        || GroundKey::Entity(entity.to_bits()),
        |id| GroundKey::Authored(id.to_owned()),
    )
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct NavPathResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terrain: Option<GroundKey>,
    pub options: NavOptions,
    pub searching: bool,
    pub walk: Option<WalkResult>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct WalkResult {
    pub points: Vec<[f64; 3]>,
    pub length: f32,
    pub level: u32,
    pub searched: usize,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
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
    type Params = (
        Res<'static, Ground>,
        Res<'static, Records>,
        SceneIds<'static, 'static>,
        ResMut<'static, NavAgents>,
        Jobs<'static, 'static>,
    );
    type Request = Self;

    fn run(
        (ground, records, ids, mut navigation, mut jobs): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        walk(
            (&*ground, &*records, &ids),
            (&mut *navigation, &mut jobs),
            request,
        )
    }

    fn schema() -> Value {
        schema_of::<Self>()
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
    (ground, records, ids): (&Ground, &Records, &SceneIds),
    (navigation, jobs): (&mut NavAgents, &mut Jobs),
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
    let terrain = match ground.0.source(from, to) {
        GroundRead::Ready(source) => Some(ground_key(ids, source.ground)),
        GroundRead::Waiting => None,
        GroundRead::Off => return Err(Error::new(ErrorCode::UnknownTerrain, "$")),
    };
    let found = navigation.query(ground, (records, jobs), (from, to), &options);
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
