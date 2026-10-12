//! Navigation's development tools: the inspectors of agents and of their
//! walks, sending agents and the query of walks.

use engine_core::bevy_ecs::prelude::Entity;
use engine_core::{
    Engine, Feature, Terrain, TerrainSource, Terrains, Transform, glam::Vec2, has_feature,
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

/// Registers navigation's inspectors.
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
                let stats = world.get_resource::<engine_core::NavAgents>()?.stats();
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
                    walking: stats.walking,
                    arrived: stats.arrived,
                    unreachable: stats.unreachable,
                    work: stats.work,
                    first,
                }))
            }),
        })
        .expect("built-in descriptor is bounded");
}

/// Adds the schemas of navigation's inspectors.
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

/// Send navigation agent `entity` (authored ID) walking to world
/// `destination` `[x, z]`, or stop it with null, and
/// render while paused; steps then walk it. Not a source change.
/// Returns its status.
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

/// Sends navigation agent `id` in `scene` to `destination` (world x and
/// z), or stops it.
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

/// What `nav_go` answers: where the agent now walks.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct NavGoResult {
    /// Controls change the session's world, not the scene's source.
    pub temporary: bool,
    /// Where it walks (world x and z); null where it stopped.
    pub destination: Option<[f32; 2]>,
}

/// What `nav_path` answers: the walk found, if one reaches.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct NavPathResult {
    /// The terrain walked.
    pub terrain: TerrainKey,
    /// How the walk weighed the ground.
    pub options: engine_core::terrain_data::NavOptions,
    /// The walk; null where none reaches.
    pub walk: Option<WalkResult>,
}

/// A walk over the terrain.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct WalkResult {
    /// Its points, world space, rounded to centimeters.
    pub points: Vec<[f64; 3]>,
    /// Its length, meters.
    pub length: f32,
    /// The pyramid level walked.
    pub level: u32,
    /// Samples searched.
    pub searched: usize,
}

/// The shortest walk over a generated terrain from world `from` to `to`
/// `[x, z]`: across ground an agent stands on (no holes,
/// no water deeper than `max_water`, no closer than `radius` to placed
/// objects that collide on resident tiles), each step no steeper than
/// `max_slope` (rise per meter), pulled taut and following the ground.
/// It walks pyramid `level` (0 by default) unless the area within
/// `margin` meters of the ends holds more than `max_samples` samples,
/// then the finest level within. Returns its points (world, rounded to
/// centimeters), length, the level walked and samples searched, or a
/// null walk where none reaches.
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

/// A navigation walk over the first generated terrain from `from` to `to`.
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
        .find(|(_, terrain)| matches!(terrain.source, TerrainSource::Generated(_)))
        .map(|(entity, _)| entity)
        .ok_or_else(|| DevError::new(ErrorCode::UnknownTerrain, "$", "no generated terrain"))?;
    let walk = world
        .get_resource::<Terrains>()
        .filter(|_| has_feature(world, Feature::Terrain))
        .ok_or_else(|| missing(Feature::Terrain))?
        .walk(entity, from, to, &options);
    let round = |value: f32| (f64::from(value) * 100.0).round() / 100.0;
    Ok(result::nav_path(NavPathResult {
        terrain: terrain_key(world, entity),
        options,
        walk: walk.map(|walk| WalkResult {
            points: walk.points.iter().map(|point| point.map(round)).collect(),
            length: walk.length,
            level: walk.level,
            searched: walk.searched,
        }),
    }))
}
