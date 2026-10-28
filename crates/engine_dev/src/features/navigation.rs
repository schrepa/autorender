use engine_core::bevy_ecs::prelude::Entity;
use engine_core::bevy_ecs::world::Mut;
use engine_core::{
    Engine, Feature, Jobs, NavAgents, NavQuery, Terrain, Terrains, Transform, glam::Vec2,
    has_feature,
};
use engine_scene::{SceneEntity, SceneInstance};
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::terrain::{TerrainKey, terrain_key};
use super::{EngineControl, EngineQuery, missing};
use crate::inspect::Inspector;
use crate::results::result;
use crate::{DevError, DevTools, InspectorDescriptor, InspectorKind, coordinate};

pub(crate) fn register(tools: &mut DevTools, schemas: &Value) {
    tools
        .insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "NavAgent"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<engine_core::NavAgent>(entity).is_some())
            }),
            read: Box::new(|world, entity, _context| {
                let entity = entity?;
                let agent = *world.get::<engine_core::NavAgent>(entity)?;
                let walk = world.get::<engine_core::NavWalk>(entity)?;
                let position = world
                    .get::<Transform>(entity)
                    .map(|transform| transform.translation);
                let left = position.map_or(0.0, |position| {
                    let mut from = position;
                    let mut left = 0.0;
                    for point in walk.points.iter().skip(walk.next) {
                        left += from.distance(*point);
                        from = *point;
                    }
                    left
                });
                Some(json!(crate::views::NavAgentView {
                    agent,
                    status: walk.status,
                    searched_for: walk.destination,
                    length: walk.length,
                    points: walk.points.len(),
                    next: walk.next,
                    left,
                    searched: walk.searched,
                }))
            }),
        })
        .expect("built-in descriptor is bounded");
    tools
        .insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "NavAgents"),
            kind: InspectorKind::Resource,
            has: Box::new(|world, _| world.get_resource::<engine_core::NavAgents>().is_some()),
            read: Box::new(|world, _, _| {
                let navigation = world.get_resource::<engine_core::NavAgents>()?;
                let stats = navigation.stats();
                let (failed_jobs, latest_job_failure) = navigation.job_failures();
                let latest_job_failure = latest_job_failure.map(str::to_owned);
                let mut query = world.try_query::<(
                    Entity,
                    &engine_core::NavAgent,
                    &engine_core::NavWalk,
                    &Transform,
                )>()?;
                let first = query
                    .iter(world)
                    .take(16)
                    .map(|(entity, agent, walk, transform)| {
                        let mut from = transform.translation;
                        let mut left = 0.0;
                        for point in walk.points.iter().skip(walk.next) {
                            left += from.distance(*point);
                            from = *point;
                        }
                        crate::views::NavAgentRow {
                            id: world
                                .get::<SceneEntity>(entity)
                                .map(|metadata| metadata.authored_id.clone()),
                            status: walk.status,
                            position: transform.translation.to_array(),
                            destination: agent.destination,
                            left,
                        }
                    })
                    .collect();
                Some(json!(crate::views::NavAgentsView {
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
            }),
        })
        .expect("built-in descriptor is bounded");
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn schemas(schemas: &mut serde_json::Map<String, Value>) {
    schemas.insert(
        "NavAgent".into(),
        super::schema::<crate::views::NavAgentView>(),
    );
    schemas.insert(
        "NavAgents".into(),
        super::schema::<crate::views::NavAgentsView>(),
    );
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct NavGoControl {
    pub entity: String,
    #[serde(default)]
    pub destination: Option<[f32; 2]>,
}

impl EngineControl for NavGoControl {
    fn changed(&self) -> &'static str {
        "agent sent"
    }

    fn apply(&self, engine: &mut Engine, scene: Option<&SceneInstance>) -> Result<Value, DevError> {
        nav_go(engine, scene, &self.entity, self.destination)
    }
}

fn nav_go(
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    id: &str,
    destination: Option<[f32; 2]>,
) -> Result<Value, DevError> {
    if let Some(destination) = destination
        && !destination
            .iter()
            .all(|value| value.is_finite() && value.abs() <= 1.0e6)
    {
        return Err(DevError::new(
            ErrorCode::OutOfRange,
            "$.destination",
            "must be finite and within 1000000 meters",
        ));
    }
    let world = engine.app.world_mut();
    if !has_feature(world, Feature::Navigation) {
        return Err(missing(Feature::Navigation));
    }
    let entity = scene
        .and_then(|scene| crate::inspect::authored_entity(world, scene, id))
        .filter(|&entity| world.get::<engine_core::NavAgent>(entity).is_some())
        .ok_or_else(|| {
            DevError::new(
                ErrorCode::UnknownEntity,
                "$.entity",
                format!("no navigation agent {id} in the scene"),
            )
        })?;
    let mut agent = world
        .get_mut::<engine_core::NavAgent>(entity)
        .expect("checked above");
    agent.destination = destination;
    Ok(result::nav_go(NavGoResult {
        temporary: true,
        destination,
    }))
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct NavGoResult {
    pub temporary: bool,
    pub destination: Option<[f32; 2]>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct NavPathResult {
    pub terrain: TerrainKey,
    pub options: engine_core::terrain_data::NavOptions,
    pub searching: bool,
    pub walk: Option<WalkResult>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct WalkResult {
    pub points: Vec<[f64; 3]>,
    pub length: f32,
    pub level: u32,
    pub searched: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct NavPathQuery {
    pub from: [f64; 2],
    pub to: [f64; 2],
    #[serde(default)]
    pub options: Option<engine_core::terrain_data::NavOptions>,
}

impl EngineQuery for NavPathQuery {
    fn answer(&self, engine: &mut Engine, _: Option<&SceneInstance>) -> Result<Value, DevError> {
        walk(engine, self.from, self.to, self.options)
    }
}

fn walk(
    engine: &mut Engine,
    from: [f64; 2],
    to: [f64; 2],
    options: Option<engine_core::terrain_data::NavOptions>,
) -> Result<Value, DevError> {
    let from = Vec2::new(
        coordinate(from[0], "$.from[0]")?,
        coordinate(from[1], "$.from[1]")?,
    );
    let to = Vec2::new(coordinate(to[0], "$.to[0]")?, coordinate(to[1], "$.to[1]")?);
    let options = options.unwrap_or_default();
    options.validate().map_err(|invalid| {
        DevError::new(
            ErrorCode::OutOfRange,
            format!("$.options.{}", invalid.field),
            invalid.rule,
        )
    })?;
    let world = engine.app.world_mut();
    let entity = world
        .query::<(Entity, &Terrain)>()
        .iter(world)
        .map(|(entity, _)| entity)
        .next()
        .ok_or_else(|| DevError::new(ErrorCode::UnknownTerrain, "$", "no terrain"))?;
    if !has_feature(world, Feature::Terrain) || !world.contains_resource::<Terrains>() {
        return Err(missing(Feature::Terrain));
    }
    if !has_feature(world, Feature::Navigation)
        || !world.contains_resource::<NavAgents>()
        || !world.contains_resource::<Jobs>()
    {
        return Err(missing(Feature::Navigation));
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
    Ok(result::nav_path(NavPathResult {
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
