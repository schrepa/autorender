use bevy_ecs::prelude::*;
use engine_core::{Error, Operation};
use engine_places::physics::RigidBody;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::body::{BodyProblem, BodyWater};
use crate::world::PhysicsWorld;

#[derive(Serialize)]
struct BodyView {
    entity: u64,
    body: RigidBody,
    simulated: bool,
    problem: Option<BodyProblem>,
    mass: f32,
    volume: f32,
    velocity: [f32; 3],
    spin: [f32; 3],
    sleeping: bool,
    held: bool,
    water: Option<BodyWater>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BodiesRequest {}

pub(crate) struct PhysicsBodies;

impl Operation for PhysicsBodies {
    const NAME: &'static str = "physics_bodies";
    type Params = (
        Res<'static, PhysicsWorld>,
        Query<'static, 'static, (Entity, &'static RigidBody)>,
    );
    type Request = BodiesRequest;

    fn run(
        (physics, bodies): SystemParamItem<'_, '_, Self::Params>,
        _: BodiesRequest,
    ) -> Result<Value, Error> {
        let mut views: Vec<BodyView> = bodies
            .iter()
            .map(|(entity, body)| {
                let status = physics.status(entity);
                let simulated = status.is_some_and(|status| status.problem.is_none());
                let status = status.unwrap_or_default();
                BodyView {
                    entity: entity.to_bits(),
                    body: *body,
                    simulated,
                    problem: status.problem,
                    mass: status.mass,
                    volume: status.volume,
                    velocity: status.velocity.to_array(),
                    spin: status.spin.to_array(),
                    sleeping: status.sleeping,
                    held: status.held,
                    water: status.water,
                }
            })
            .collect();
        views.sort_by_key(|view| view.entity);
        Ok(json!({ "bodies": views }))
    }
}
