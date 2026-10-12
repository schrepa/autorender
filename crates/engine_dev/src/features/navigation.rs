//! Navigation's development tools: the inspectors of agents and of their
//! walks.

use engine_core::Transform;
use engine_core::bevy_ecs::prelude::Entity;
use engine_scene::SceneEntity;
use serde_json::{Value, json};

use crate::inspect::Inspector;
use crate::{DevTools, InspectorDescriptor, InspectorKind};

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
