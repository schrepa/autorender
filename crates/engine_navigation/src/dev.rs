use bevy_ecs::prelude::*;
use engine_core::{
    Error, ErrorCode, Expected, Filled, InvalidSetting, Jobs, Mover, Operation, Records, Rule,
    SceneIds, Setting, Slot, Transform,
};
use engine_places_simulation::ground::Ground;
use engine_places_simulation::navigation::{NavAgent, NavStatus, NavWalk};
use engine_places_simulation::water::WaterBody;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::agents::Floors;
use crate::{NavAgents, NavOptions, NavQuery, NavStats, Navigation, ObstacleColliders};

const COORDINATE: Rule = Rule::Range(-1.0e6, 1.0e6);
const FIRST_AGENTS: usize = 16;

#[derive(Serialize, schemars::JsonSchema)]
pub struct NavAgentView {
    #[schemars(with = "crate::SceneNavAgent")]
    pub agent: NavAgent,
    pub status: NavStatus,
    pub refused: Option<String>,
    pub searched_for: Option<[f32; 2]>,
    pub length: f32,
    pub points: usize,
    pub next: usize,
    pub left: f32,
    pub searched: usize,
    pub moved_by: Option<String>,
    pub blocked_by: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct NavAgentsView {
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

fn agent_view(
    ids: &SceneIds,
    (agent, walk, at, mover): (
        &NavAgent,
        Option<&NavWalk>,
        Option<&Transform>,
        Option<&Mover>,
    ),
) -> NavAgentView {
    let agent = *agent;
    let still = NavWalk::default();
    let walk = walk.unwrap_or(&still);
    let position = at.map(|transform| transform.translation);
    let blocked = walk.step.and_then(|step| step.blocked);
    NavAgentView {
        agent,
        status: walk.status,
        refused: walk.refused.clone(),
        searched_for: walk.destination,
        length: walk.length,
        points: walk.points.len(),
        next: walk.next,
        left: position.map_or(0.0, |position| remaining(walk, position)),
        searched: walk.searched,
        moved_by: mover.map(|mover| mover.by.name().to_owned()),
        blocked_by: blocked.and_then(|entity| ids.id(entity).map(str::to_owned)),
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
                Option<&'static Mover>,
            ),
        >,
    );
    type Request = Self;
    type Reply = NavAgentView;

    fn run(
        (ids, agents): SystemParamItem<'_, '_, Self::Params>,
        Self { entity: id }: Self,
    ) -> Result<NavAgentView, Error> {
        let entity = ids.named(&id, "$.entity")?;
        let found = agents
            .get(entity)
            .map_err(|_| Error::new(ErrorCode::UnknownEntity, "$.entity").got(&id))?;
        Ok(agent_view(&ids, found))
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
    type Reply = NavAgentsView;

    fn run(
        (stats, navigation, ids, agents): SystemParamItem<'_, '_, Self::Params>,
        _: Self,
    ) -> Result<NavAgentsView, Error> {
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
        Ok(NavAgentsView {
            agents: stats.agents,
            searching: stats.searching,
            waiting: stats.waiting,
            walking: stats.walking,
            arrived: stats.arrived,
            unreachable: stats.unreachable,
            refused: stats.refused,
            collision_errors: stats.collision_errors,
            work: stats.work,
            grounds: stats.grounds,
            portals: stats.portals,
            failed_jobs,
            latest_job_failure,
            first,
        })
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
    type Reply = NavGoResult;

    fn run(
        (ids, mut agents): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<NavGoResult, Error> {
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
        let entity = ids.named(&id, "$.entity")?;
        let mut agent = agents
            .get_mut(entity)
            .map_err(|_| Error::new(ErrorCode::UnknownEntity, "$.entity").got(&id))?;
        agent.destination = destination;
        Ok(NavGoResult {
            temporary: true,
            destination,
        })
    }
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct NavPathResult {
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
        (
            Res<'static, Navigation>,
            Res<'static, Ground>,
            Query<'static, 'static, &'static WaterBody>,
        ),
        Filled<'static, Ground>,
        Res<'static, Records>,
        ResMut<'static, NavAgents>,
        Jobs<'static, 'static>,
        ObstacleColliders<'static, 'static>,
    );
    type Request = Self;
    type Reply = NavPathResult;

    fn run(
        ((settings, ground, waters), filled, records, mut navigation, mut jobs, colliders): SystemParamItem<
            '_,
            '_,
            Self::Params,
        >,
        request: Self,
    ) -> Result<NavPathResult, Error> {
        let floors = Floors::new(&ground, waters.iter(), settings.floor_layers);
        walk(
            (
                (&*settings, &floors, &colliders),
                filled.listed(),
                &*records,
            ),
            (&mut *navigation, &mut jobs),
            request,
        )
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
    ((settings, floors, colliders), listed, records): (
        (&Navigation, &Floors, &ObstacleColliders),
        bool,
        &Records,
    ),
    (navigation, jobs): (&mut NavAgents, &mut Jobs),
    NavPathQuery { from, to, options }: NavPathQuery,
) -> Result<NavPathResult, Error> {
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
    if !listed {
        return Err(Error::new(ErrorCode::MissingPlugin, "$.op")
            .expected(Expected::Present)
            .got(Ground::NAME));
    }
    let found = navigation.query(
        (settings, floors, colliders),
        (records, jobs),
        (from, to),
        &options,
    );
    let round = |value: f32| (f64::from(value) * 100.0).round() / 100.0;
    Ok(NavPathResult {
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
    })
}
