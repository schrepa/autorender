use bevy_ecs::prelude::{Entity, World};
use serde::Serialize;

use crate::{BodyProblem, BodyWater, PhysicsStats, PhysicsWorld, RigidBody};

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
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

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(transparent)]
pub struct PhysicsWorldView(pub PhysicsStats);

impl PhysicsWorldView {
    pub fn of(physics: &PhysicsWorld) -> Self {
        Self(physics.stats())
    }
}

pub fn rigid_body_view(world: &World, entity: Entity) -> Option<RigidBodyView> {
    let body = *world.get::<RigidBody>(entity)?;
    let status = world
        .get_resource::<PhysicsWorld>()
        .and_then(|physics| physics.status(entity));
    let simulated = status.is_some_and(|status| status.problem.is_none());
    let status = status.unwrap_or_default();
    Some(RigidBodyView {
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
    })
}
