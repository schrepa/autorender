use engine_core::math::clamp;
use engine_places::physics::RigidBody;
use engine_places::water::WaterKind;
use glam::Vec3;
use serde::Serialize;

const DENSITY: (f32, f32) = (1.0, 100_000.0);
const FRICTION: (f32, f32) = (0.0, 10.0);
const RESTITUTION: (f32, f32) = (0.0, 1.0);
const DRAG: (f32, f32) = (0.0, 10.0);
const MOTION: (f32, f32) = (-1000.0, 1000.0);

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

fn within(value: f32, (low, high): (f32, f32)) -> bool {
    (low..=high).contains(&value)
}

fn kept(value: f32, (low, high): (f32, f32), default: f32) -> f32 {
    if value.is_finite() {
        clamp(value, low, high)
    } else {
        default
    }
}

pub(crate) fn valid(body: &RigidBody) -> bool {
    within(body.density, DENSITY)
        && within(body.friction, FRICTION)
        && within(body.restitution, RESTITUTION)
        && within(body.drag, DRAG)
        && body.velocity.iter().all(|&value| within(value, MOTION))
        && body.spin.iter().all(|&value| within(value, MOTION))
}

pub(crate) fn sanitized(body: RigidBody) -> RigidBody {
    let defaults = RigidBody::default();
    let each = |values: [f32; 3], defaults: [f32; 3]| {
        [0, 1, 2].map(|axis| kept(values[axis], MOTION, defaults[axis]))
    };
    RigidBody {
        motion: body.motion,
        density: kept(body.density, DENSITY, defaults.density),
        friction: kept(body.friction, FRICTION, defaults.friction),
        restitution: kept(body.restitution, RESTITUTION, defaults.restitution),
        drag: kept(body.drag, DRAG, defaults.drag),
        velocity: each(body.velocity, defaults.velocity),
        spin: each(body.spin, defaults.spin),
        fast: body.fast,
    }
}
