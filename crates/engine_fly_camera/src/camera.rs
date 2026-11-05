use bevy_ecs::prelude::*;
use engine_core::{Camera, FrameTime, MoveInput};
use engine_math::{atan2, clamp};
use glam::{Quat, Vec3};

use crate::FlyCamera;

const SENSITIVITY: f32 = 0.003;

pub(crate) struct Flight {
    yaw: f32,
    pitch: f32,
}

impl Flight {
    fn new(yaw: f32, pitch: f32) -> Self {
        let yaw = if yaw.is_finite() {
            yaw % std::f32::consts::TAU
        } else {
            0.0
        };
        let pitch = if pitch.is_finite() {
            clamp(pitch, -1.48, 1.48)
        } else {
            0.0
        };
        Self { yaw, pitch }
    }

    fn from_camera(camera: &Camera) -> Self {
        let direction = (camera.target - camera.position).normalize_or_zero();
        Self::new(atan2(-direction.x, -direction.z), direction.y.asin())
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
    time: Res<FrameTime>,
    input: Res<MoveInput>,
    settings: Res<FlyCamera>,
    mut flight: Local<Option<Flight>>,
    mut camera: ResMut<Camera>,
) {
    let flight = if camera.is_changed() {
        flight.insert(Flight::from_camera(&camera))
    } else {
        flight.get_or_insert_with(|| Flight::from_camera(&camera))
    };
    if settings.steered_only && !steered(&input) {
        return;
    }
    flight.yaw = (flight.yaw - input.look_right_pixels * SENSITIVITY) % std::f32::consts::TAU;
    flight.pitch = clamp(
        flight.pitch - input.look_down_pixels * SENSITIVITY,
        -1.48,
        1.48,
    );
    let rotation = Quat::from_rotation_y(flight.yaw) * Quat::from_rotation_x(flight.pitch);
    let forward = rotation * -Vec3::Z;
    let right = rotation * Vec3::X;
    let movement =
        forward * input.move_forward + right * input.move_right + Vec3::Y * input.move_up;
    camera.position += movement.normalize_or_zero() * time.delta_seconds * settings.speed;
    camera.target = camera.position + forward;
}
