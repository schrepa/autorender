//! Physics' development tools: the inspectors of rigid bodies and of the
//! physics world.

use engine_core::PhysicsStats;
use engine_physics::{BodyProblem, BodyWater, PhysicsWorld, RigidBody};
use serde::Serialize;
use serde_json::{Value, json};

use crate::inspect::Inspector;
use crate::{DevTools, InspectorDescriptor, InspectorKind};

/// A rigid body as authored, and what physics did with it at the last
/// fixed tick: whether the solver holds it (not with a problem, nor before
/// its first tick), its mass (kilograms) and volume (cubic meters), its
/// velocity (meters a second) and spin (radians a second about the world's
/// axes), whether it sleeps or holds still where the ground under it is not
/// resident, and the water it is in.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct RigidBodyView {
    pub body: RigidBody,
    pub simulated: bool,
    pub problem: Option<BodyProblem>,
    pub mass: f32,
    pub volume: f32,
    pub velocity: [f32; 3],
    pub spin: [f32; 3],
    pub sleeping: bool,
    pub held: bool,
    pub water: Option<BodyWater>,
}

/// The physics world at the last fixed tick: its bodies, the world's solids
/// it holds for them, their contacts and the water samples they took.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(transparent)]
pub struct PhysicsWorldView(pub PhysicsStats);

/// Registers physics' inspectors.
pub(crate) fn register(tools: &mut DevTools, schemas: &Value) {
    tools
        .insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "RigidBody")
                .with_default(json!(RigidBody::default())),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<RigidBody>(entity).is_some())
            }),
            read: Box::new(|world, entity, _context| {
                let entity = entity?;
                let body = *world.get::<RigidBody>(entity)?;
                let status = world
                    .get_resource::<PhysicsWorld>()
                    .and_then(|physics| physics.status(entity));
                let simulated = status.is_some_and(|status| status.problem.is_none());
                let status = status.unwrap_or_default();
                Some(json!(RigidBodyView {
                    body,
                    simulated,
                    problem: status.problem,
                    mass: status.mass,
                    volume: status.volume,
                    velocity: status.velocity.to_array(),
                    spin: status.spin.to_array(),
                    sleeping: status.sleeping,
                    held: status.held,
                    water: status.water,
                }))
            }),
        })
        .expect("built-in descriptor is bounded");
    tools
        .register_resource::<PhysicsWorld>(
            InspectorDescriptor::generated(schemas, "PhysicsWorld"),
            |physics| json!(PhysicsWorldView(physics.stats())),
        )
        .expect("built-in descriptor is bounded");
}

/// Adds the schemas of physics' inspectors.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn schemas(schemas: &mut serde_json::Map<String, Value>) {
    schemas.insert("RigidBody".into(), super::schema::<RigidBodyView>());
    schemas.insert("PhysicsWorld".into(), super::schema::<PhysicsWorldView>());
}
