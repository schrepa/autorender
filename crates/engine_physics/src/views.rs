use bevy_ecs::prelude::*;
use engine_core::{Error, ErrorCode, Expected, Operation, SceneIds};
use engine_places::physics::RigidBody;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::body::{BodyProblem, BodyWater};
use crate::world::PhysicsWorld;

#[derive(Serialize)]
struct RigidBodyView {
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
pub(crate) struct BodyRequest {
    entity: String,
}

pub(crate) struct PhysicsBody;

impl Operation for PhysicsBody {
    const NAME: &'static str = "physics_body";
    type Params = (
        Res<'static, PhysicsWorld>,
        SceneIds<'static, 'static>,
        Query<'static, 'static, &'static RigidBody>,
    );
    type Request = BodyRequest;

    fn run(
        (physics, ids, bodies): SystemParamItem<'_, '_, Self::Params>,
        request: BodyRequest,
    ) -> Result<Value, Error> {
        let entity = ids.entity(&request.entity).ok_or_else(|| {
            Error::new(ErrorCode::UnknownEntity, "$.entity")
                .expected(Expected::Entity)
                .got(&request.entity)
        })?;
        let Ok(body) = bodies.get(entity) else {
            return Ok(Value::Null);
        };
        let status = physics.status(entity);
        let simulated = status.is_some_and(|status| status.problem.is_none());
        let status = status.unwrap_or_default();
        let view = RigidBodyView {
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
        };
        Ok(json!(view))
    }
}
