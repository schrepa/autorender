use engine_core::{Error, ErrorCode, InvalidSetting, Rule, Setting};
use engine_places::physics::{BodyPush, RigidBody};
use engine_places::water::WaterKind;
use glam::Vec3;
use serde::Serialize;
use serde_json::{Value, json};

const DENSITY: Rule = Rule::range(1.0, 100_000.0);
const FRICTION: Rule = Rule::range(0.0, 10.0);
const RESTITUTION: Rule = Rule::range(0.0, 1.0);
const DRAG: Rule = Rule::range(0.0, 10.0);
const MOTION: Rule = Rule::range(-1000.0, 1000.0);
const RULES: [(&str, Rule); 8] = [
    ("motion", Rule::any()),
    ("density", DENSITY),
    ("friction", FRICTION),
    ("restitution", RESTITUTION),
    ("drag", DRAG),
    ("velocity", MOTION),
    ("spin", MOTION),
    ("fast", Rule::any()),
];
const PUSH: Rule = Rule::range(-1.0e9, 1.0e9);
const PUSH_RULES: [(&str, Rule); 4] = [
    ("force", PUSH),
    ("torque", PUSH),
    ("impulse", PUSH),
    ("angular_impulse", PUSH),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BodyProblem {
    NoCollider,
    NotRoot,
    NoTransform,
    BadScale,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(crate) struct BodyWater {
    pub kind: WaterKind,
    pub submerged: f32,
    pub current: [f32; 3],
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct BodyStatus {
    pub problem: Option<BodyProblem>,
    pub mass: f32,
    pub volume: f32,
    pub velocity: Vec3,
    pub spin: Vec3,
    pub sleeping: bool,
    pub held: bool,
    pub water: Option<BodyWater>,
}

fn ruled(body: &RigidBody) -> Result<(), InvalidSetting> {
    body.density.check("density", DENSITY)?;
    body.friction.check("friction", FRICTION)?;
    body.restitution.check("restitution", RESTITUTION)?;
    body.drag.check("drag", DRAG)?;
    body.velocity.check("velocity", MOTION)?;
    body.spin.check("spin", MOTION)
}

pub(crate) fn check_body(body: &RigidBody, path: &str) -> Result<(), Error> {
    ruled(body).map_err(|invalid| invalid.error(ErrorCode::InvalidValue, path))
}

pub(crate) fn body_schema() -> Value {
    let mut schema = schemars::schema_for!(RigidBody);
    for (field, rule) in RULES {
        engine_core::state(&mut schema, field, rule);
    }
    serde_json::to_value(schema).unwrap_or(Value::Null)
}

pub(crate) fn body_example() -> Value {
    json!({})
}

pub(crate) fn sanitized(body: RigidBody) -> RigidBody {
    let defaults = RigidBody::default();
    RigidBody {
        motion: body.motion,
        density: body.density.kept(DENSITY, defaults.density),
        friction: body.friction.kept(FRICTION, defaults.friction),
        restitution: body.restitution.kept(RESTITUTION, defaults.restitution),
        drag: body.drag.kept(DRAG, defaults.drag),
        velocity: body.velocity.kept(MOTION, defaults.velocity),
        spin: body.spin.kept(MOTION, defaults.spin),
        fast: body.fast,
    }
}

fn push_ruled(push: &BodyPush) -> Result<(), InvalidSetting> {
    push.force.check("force", PUSH)?;
    push.torque.check("torque", PUSH)?;
    push.impulse.check("impulse", PUSH)?;
    push.angular_impulse.check("angular_impulse", PUSH)
}

pub(crate) fn check_push(push: &BodyPush, path: &str) -> Result<(), Error> {
    push_ruled(push).map_err(|invalid| invalid.error(ErrorCode::InvalidValue, path))
}

pub(crate) fn push_schema() -> Value {
    let mut schema = schemars::schema_for!(BodyPush);
    for (field, rule) in PUSH_RULES {
        engine_core::state(&mut schema, field, rule);
    }
    serde_json::to_value(schema).unwrap_or(Value::Null)
}

pub(crate) fn push_example() -> Value {
    json!({"force": [0.0, 500.0, 0.0]})
}

pub(crate) fn sanitized_push(push: BodyPush) -> BodyPush {
    let none = [0.0; 3];
    BodyPush {
        force: push.force.kept(PUSH, none),
        torque: push.torque.kept(PUSH, none),
        impulse: push.impulse.kept(PUSH, none),
        angular_impulse: push.angular_impulse.kept(PUSH, none),
    }
}
