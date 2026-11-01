use bevy_ecs::prelude::*;
use engine_core::{Camera, FrameTime, MoveInput};
use engine_math::clamp;
use glam::{Quat, Vec3};

use crate::FlyCamera;

const SENSITIVITY: f32 = 0.003;

#[derive(Default)]
pub(crate) struct Flight {
    yaw: f32,
    pitch: f32,
    written: Option<Camera>,
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
        Self {
            yaw,
            pitch,
            written: None,
        }
    }

    fn from_camera(camera: &Camera) -> Self {
        let direction = (camera.target - camera.position).normalize_or_zero();
        Self::new((-direction.x).atan2(-direction.z), direction.y.asin())
    }
}

pub(crate) fn fly(
    time: Res<FrameTime>,
    input: Res<MoveInput>,
    settings: Res<FlyCamera>,
    mut flight: Local<Flight>,
    mut camera: ResMut<Camera>,
) {
    if flight.written != Some(*camera) {
        *flight = Flight::from_camera(&camera);
    }
    flight.yaw = (flight.yaw - input.look_yaw * SENSITIVITY) % std::f32::consts::TAU;
    flight.pitch = clamp(flight.pitch - input.look_pitch * SENSITIVITY, -1.48, 1.48);
    let rotation = Quat::from_rotation_y(flight.yaw) * Quat::from_rotation_x(flight.pitch);
    let forward = rotation * -Vec3::Z;
    let right = rotation * Vec3::X;
    let movement =
        forward * input.move_forward + right * input.move_right + Vec3::Y * input.move_up;
    camera.position += movement.normalize_or_zero() * time.delta_seconds * settings.speed;
    camera.target = camera.position + forward;
    flight.written = Some(*camera);
}
