mod queries;
mod results;

use engine_collision::{Collider, CollisionWorld};
use serde_json::Value;

pub use queries::{
    CollisionEventsQuery, CollisionFilterRequest, CollisionSensorFilter, CollisionShapeRequest,
    OverlapQuery, RaycastQuery, ShapeCastQuery,
};
pub(crate) use queries::{native_error, world_value, world_view};
pub use results::{
    CastHit, CollisionEventsResult, HitEntity, OverlapResult, PlacedHit, RayHit, RaycastResult,
    SensorEventKindView, SensorEventRow, ShapeCastResult,
};

use crate::inspect::Inspector;
use crate::{DevTools, InspectorDescriptor, InspectorKind};

pub(crate) fn register(tools: &mut DevTools, schemas: &Value) {
    tools
        .insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "Collider"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<Collider>(entity).is_some())
            }),
            read: Box::new(|world, entity, _context| queries::collider_value(world, entity?)),
        })
        .expect("built-in descriptor is bounded");
    tools
        .register_resource::<CollisionWorld>(
            InspectorDescriptor::generated(schemas, "CollisionWorld"),
            world_value,
        )
        .expect("built-in descriptor is bounded");
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn schemas(schemas: &mut serde_json::Map<String, Value>) {
    schemas.insert(
        "Collider".into(),
        super::schema::<crate::views::ColliderView>(),
    );
    schemas.insert(
        "CollisionWorld".into(),
        super::schema::<crate::views::CollisionWorldView>(),
    );
}
