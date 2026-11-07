use bevy_ecs::prelude::*;
use engine_core::{
    Ask, Asked, Error, ErrorCode, Expected, FixedTime, FrameTime, Moment, Mover, Operation,
    SceneIds,
};
use engine_places_simulation::physics::{BodyAsk, BodyChange, BodyPush, RigidBody};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::body::{BodyProblem, BodyWater, check_change, check_push};
use crate::INVALID_RIGID_BODY;
use crate::world::PhysicsWorld;

#[derive(Serialize)]
struct RigidBodyView {
    body: RigidBody,
    moved_by: Option<&'static str>,
    simulated: bool,
    problem: Option<BodyProblem>,
    mass: f32,
    volume: f32,
    velocity: [f32; 3],
    spin: [f32; 3],
    sleeping: bool,
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
        Query<'static, 'static, (&'static RigidBody, Option<&'static Mover>)>,
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
        let Ok((body, mover)) = bodies.get(entity) else {
            return Ok(Value::Null);
        };
        let status = physics.status(entity);
        let simulated = status
            .as_ref()
            .is_some_and(|status| status.problem.is_none());
        let status = status.unwrap_or_default();
        let view = RigidBodyView {
            body: *body,
            moved_by: mover.map(|mover| mover.by.name()),
            simulated,
            problem: status.problem,
            mass: status.mass,
            volume: status.volume,
            velocity: status.velocity.to_array(),
            spin: status.spin.to_array(),
            sleeping: status.sleeping,
            water: status.water,
        };
        Ok(json!(view))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PushRequest {
    entity: String,
    push: BodyPush,
}

pub(crate) struct BodyPushOperation;

impl Operation for BodyPushOperation {
    const NAME: &'static str = "body_push";
    type Params = (
        SceneIds<'static, 'static>,
        Query<'static, 'static, &'static mut BodyPush, With<RigidBody>>,
    );
    type Request = PushRequest;

    fn run(
        (ids, mut bodies): SystemParamItem<'_, '_, Self::Params>,
        request: PushRequest,
    ) -> Result<Value, Error> {
        let entity = ids.entity(&request.entity).ok_or_else(|| {
            Error::new(ErrorCode::UnknownEntity, "$.entity")
                .expected(Expected::Entity)
                .got(&request.entity)
        })?;
        check_push(&request.push, "$.push")?;
        let mut push = bodies.get_mut(entity).map_err(|_| {
            Error::new(INVALID_RIGID_BODY, "$.entity").got(&request.entity)
        })?;
        *push = request.push;
        Ok(json!(request.push))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AskRequest {
    entity: String,
    ask: BodyChange,
}

pub(crate) struct BodyAskOperation;

impl Operation for BodyAskOperation {
    const NAME: &'static str = BodyAsk::NAME;
    type Params = (
        SceneIds<'static, 'static>,
        Query<'static, 'static, (), With<RigidBody>>,
        ResMut<'static, Asked<BodyAsk>>,
        (Res<'static, FixedTime>, Res<'static, FrameTime>),
    );
    type Request = AskRequest;

    fn run(
        (ids, bodies, mut asked, (fixed, frame)): SystemParamItem<'_, '_, Self::Params>,
        request: AskRequest,
    ) -> Result<Value, Error> {
        let entity = ids.entity(&request.entity).ok_or_else(|| {
            Error::new(ErrorCode::UnknownEntity, "$.entity")
                .expected(Expected::Entity)
                .got(&request.entity)
        })?;
        check_change(request.ask, "$.ask")?;
        if !bodies.contains(entity) {
            return Err(Error::new(INVALID_RIGID_BODY, "$.entity").got(&request.entity));
        }
        asked.ask(
            BodyAsk {
                body: entity,
                change: request.ask,
            },
            Moment {
                tick: fixed.tick,
                frame: frame.frame,
            },
        );
        Ok(json!({ "ask": request.ask, "at": BodyAsk::AT }))
    }
}
