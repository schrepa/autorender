use bevy_ecs::prelude::*;
use engine_core::{
    Asks, Clock, FrameTime, GlobalTransforms, HandOver, MoveInput, Mover, Moves, Pace, Views,
    mover_of,
};
use engine_math::{atan2, clamp};
use glam::{Quat, Vec3};

use crate::{FlyCamera, FlyCameraBrick};

const SENSITIVITY: f32 = 0.003;
const PITCH_LIMIT: f32 = 1.48;

#[derive(Resource, Default)]
pub(crate) enum Rig {
    #[default]
    Taking,
    Holding,
}

struct Flight {
    yaw: f32,
    pitch: f32,
}

impl Flight {
    fn of(rotation: Quat) -> Self {
        let forward = rotation * Vec3::NEG_Z;
        let yaw = atan2(-forward.x, -forward.z);
        let pitch = clamp(forward.y, -1.0, 1.0).asin();
        Self {
            yaw: if yaw.is_finite() { yaw } else { 0.0 },
            pitch: if pitch.is_finite() {
                clamp(pitch, -PITCH_LIMIT, PITCH_LIMIT)
            } else {
                0.0
            },
        }
    }

    #[must_use]
    fn turned(self, input: &MoveInput) -> Self {
        Self {
            yaw: (self.yaw - input.look_right_pixels * SENSITIVITY) % std::f32::consts::TAU,
            pitch: clamp(
                self.pitch - input.look_down_pixels * SENSITIVITY,
                -PITCH_LIMIT,
                PITCH_LIMIT,
            ),
        }
    }

    fn rotation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch)
    }
}

fn steered(input: &MoveInput) -> bool {
    [
        input.move_forward,
        input.move_right,
        input.move_up,
        input.look_right_pixels,
        input.look_down_pixels,
    ]
    .iter()
    .any(|&value| value != 0.0)
}

pub(crate) fn fly(
    (time, input, settings, globals): (
        Res<FrameTime>,
        Res<MoveInput>,
        Res<FlyCamera>,
        Res<GlobalTransforms>,
    ),
    views: Res<Views>,
    mut rig: ResMut<Rig>,
    mut handover: Asks<HandOver>,
    mut moves: Moves<FlyCameraBrick>,
) {
    let Some(camera) = views.main().map(|main| main.camera) else {
        return;
    };
    match *rig {
        Rig::Taking => {
            handover.send(HandOver {
                node: camera,
                to: Mover {
                    by: mover_of::<FlyCameraBrick>(),
                    pace: Pace::Frame,
                },
            });
            *rig = Rig::Holding;
            return;
        }
        Rig::Holding => {}
    }
    if !steered(&input) {
        return;
    }
    let clock = globals.word(camera).map_or(Clock::Real, |word| word.clock);
    let seconds = time.seconds_on(clock) as f32;
    let Some(mut transform) = moves.get_mut(camera) else {
        return;
    };
    let flight = Flight::of(transform.rotation).turned(&input);
    let rotation = flight.rotation();
    let movement = rotation * Vec3::NEG_Z * input.move_forward
        + rotation * Vec3::X * input.move_right
        + Vec3::Y * input.move_up;
    transform.translation += movement.normalize_or_zero() * seconds * settings.speed;
    transform.rotation = rotation;
}
