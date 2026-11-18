use glam::{Mat3, Vec3};

use super::effects::{EffectId, Effects, Living, State};
use super::numbers::{AbilityId, Delivery, TrapShape};
use super::shapes::flat;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PropPose {
    Flying {
        centre: Vec3,
        velocity_per_tick: Vec3,
        resting: bool,
    },
    Placed {
        contact: Vec3,
        normal: Vec3,
        facing: Vec3,
    },
    Wall {
        base: Vec3,
        axes: Mat3,
        half_extents: Vec3,
    },
    Wire {
        a: Vec3,
        b: Vec3,
        normal_a: Vec3,
        normal_b: Vec3,
    },
    Falling {
        target: Vec3,
        impact_tick: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PropState {
    pub effect: EffectId,
    pub ability: AbilityId,
    pub team: usize,
    pub born: u64,
    pub pose: PropPose,
}

fn facing_on(normal: Vec3, aim: Vec3) -> Vec3 {
    let along = aim - normal * aim.dot(normal);
    along
        .try_normalize()
        .or_else(|| flat(aim))
        .or_else(|| normal.cross(Vec3::X).try_normalize())
        .unwrap_or(Vec3::Z)
}

fn lift_of(living: &Living) -> f32 {
    match living.delivery {
        Delivery::Place(place) => place.lift_m,
        Delivery::Caster | Delivery::Throw(_) => 0.0,
    }
}

fn pose_of(living: &Living) -> Option<PropPose> {
    let aim = living.aim.direction;
    let lift = lift_of(living);
    let placed = |point: Vec3, normal: Vec3| PropPose::Placed {
        contact: point - normal * lift,
        normal,
        facing: facing_on(normal, aim),
    };
    match &living.state {
        State::Flight(flight) => Some(PropPose::Flying {
            centre: flight.position,
            velocity_per_tick: if flight.resting {
                Vec3::ZERO
            } else {
                flight.velocity
            },
            resting: flight.resting,
        }),
        State::Smoke { centre, .. } | State::Zone { centre, .. } => Some(PropPose::Flying {
            centre: *centre,
            velocity_per_tick: Vec3::ZERO,
            resting: true,
        }),
        State::Wall(wall) => Some(PropPose::Wall {
            base: wall.centre - Vec3::Y * wall.half_extents.y,
            axes: wall.axes,
            half_extents: wall.half_extents,
        }),
        State::Trap(trap) => Some(match trap.spec.shape {
            TrapShape::Radius(_) => placed(trap.a, trap.normal_a),
            TrapShape::Wire { .. } => PropPose::Wire {
                a: trap.a,
                b: trap.b,
                normal_a: trap.normal_a,
                normal_b: trap.normal_b,
            },
        }),
        State::Sensor(sensor) => Some(placed(sensor.at, sensor.normal)),
        State::Strike {
            centre, warn_ticks, ..
        } => Some(PropPose::Falling {
            target: *centre,
            impact_tick: living.born + u64::from(*warn_ticks),
        }),
        State::Pending
        | State::Glare(_)
        | State::Heal { .. }
        | State::Dash { .. }
        | State::Transform { .. }
        | State::Done => None,
    }
}

impl Effects {
    pub fn props(&self) -> Vec<PropState> {
        self.livings
            .iter()
            .filter_map(|living| {
                Some(PropState {
                    effect: living.id,
                    ability: living.ability,
                    team: living.caster.team,
                    born: living.born,
                    pose: pose_of(living)?,
                })
            })
            .collect()
    }
}
