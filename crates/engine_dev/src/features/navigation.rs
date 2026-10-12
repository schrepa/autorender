//! Navigation's development tools: the inspectors of agents and of their
//! walks.

use engine_core::{Engine, Feature, Transform, has_feature};
use engine_core::bevy_ecs::prelude::Entity;
use engine_scene::{SceneEntity, SceneInstance};
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{EngineControl, missing};
use crate::inspect::Inspector;
use crate::{DevError, DevTools, InspectorDescriptor, InspectorKind};

/// Registers navigation's inspectors.
pub(crate) fn register(tools: &mut DevTools, schemas: &Value) {
    tools.insert(Inspector {
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
    tools.insert(Inspector {
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
    schemas.insert("NavAgent".into(), super::schema::<crate::views::NavAgentView>());
    schemas.insert("NavAgents".into(), super::schema::<crate::views::NavAgentsView>());
}

/// Send navigation agent `entity` (authored ID) walking to world
/// `destination` `[x, z]`, or stop it with null, and
/// render while paused; steps then walk it. Not a source change.
/// Returns its status.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct NavGoControl {
    pub version: u32,
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
    Ok(json!({ "temporary": true, "destination": destination }))
}
