use bevy_ecs::prelude::*;
use engine_core::{
    Camera, Error, ErrorCode, FrameTime, MoveInput, Operation, Poses, Rule, Setting,
};
use engine_math::{clamp, exp, max, min, sin, sin_cos};
use engine_places::physics::{
    ColliderShape, Collision, QueryFilter, SensorFilter, ShapeCastOptions,
};
use glam::{Mat4, Quat, Vec3};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{CameraTarget, FollowCamera};

const SENSITIVITY: f32 = 0.003;
const STRENGTH: Rule = Rule::range(0.0, 10.0);
const SECONDS: Rule = Rule::range(0.0, 60.0);
const SHAKE_RATES: [f32; 3] = [37.0, 53.0, 71.0];

#[derive(Resource, Default)]
pub(crate) struct Orbit {
    started: bool,
    yaw: f32,
    pitch: f32,
    focus: Vec3,
    reach: f32,
}

impl Orbit {
    fn start(&mut self, camera: &Camera, focus: Vec3) {
        let direction = (camera.target - camera.position).normalize_or_zero();
        self.started = true;
        self.yaw = (-direction.x).atan2(-direction.z);
        self.pitch = direction.y.asin();
        self.focus = focus;
        self.reach = 1.0;
    }
}

#[derive(Resource, Default)]
pub(crate) struct Shake {
    strength: f32,
    seconds: f32,
    left: f32,
}

impl Shake {
    fn offset(&mut self, seconds: f32) -> Vec3 {
        if self.left <= 0.0 {
            return Vec3::ZERO;
        }
        self.left = max(self.left - seconds, 0.0);
        let fade = self.left / self.seconds;
        let at = self.seconds - self.left;
        let (across, up) = sin_cos(at * SHAKE_RATES[0]);
        let along = sin(at * SHAKE_RATES[2]);
        let bob = sin(at * SHAKE_RATES[1]);
        Vec3::new(across, up * bob, along) * self.strength * fade
    }
}

fn eased(seconds: f32, smoothing: f32) -> f32 {
    if smoothing > 0.0 {
        1.0 - exp(-seconds / smoothing)
    } else {
        1.0
    }
}

fn clear_reach(
    collision: &Collision,
    settings: &FollowCamera,
    (focus, back): (Vec3, Vec3),
    entity: Entity,
) -> f32 {
    let hit = collision.0.cast_shape(
        ColliderShape::Sphere {
            radius: settings.collision_radius,
        },
        Mat4::from_translation(focus),
        back,
        QueryFilter {
            exclude: Some(entity),
            sensors: SensorFilter::Exclude,
            ..QueryFilter::default()
        },
        ShapeCastOptions {
            stop_at_penetration: false,
            target_distance: 0.0,
        },
    );
    match hit {
        Ok(Some(hit)) => clamp(hit.fraction, 0.0, 1.0),
        Ok(None) | Err(_) => 1.0,
    }
}

pub(crate) fn follow(
    (time, input, settings): (Res<FrameTime>, Res<MoveInput>, Res<FollowCamera>),
    (poses, collision): (Res<Poses>, Res<Collision>),
    targets: Query<Entity, With<CameraTarget>>,
    mut orbit: ResMut<Orbit>,
    mut shake: ResMut<Shake>,
    mut camera: ResMut<Camera>,
) {
    let Some(entity) = targets.iter().min() else {
        return;
    };
    let Some(pose) = poses.of(entity) else {
        return;
    };
    let goal = Vec3::from(pose.translation) + Vec3::Y * settings.height;
    if !orbit.started {
        orbit.start(&camera, goal);
    }
    let blend = eased(time.delta_seconds, settings.smoothing);
    let (low, high) = (
        min(settings.pitch_min, settings.pitch_max),
        max(settings.pitch_min, settings.pitch_max),
    );
    orbit.yaw = (orbit.yaw - input.look_right_pixels * SENSITIVITY) % std::f32::consts::TAU;
    orbit.pitch = clamp(orbit.pitch - input.look_down_pixels * SENSITIVITY, low, high);
    orbit.focus = orbit.focus.lerp(goal, blend);
    let rotation = Quat::from_rotation_y(orbit.yaw) * Quat::from_rotation_x(orbit.pitch);
    let back = rotation * Vec3::Z * settings.distance;
    let clear = clear_reach(&collision, &settings, (orbit.focus, back), entity);
    orbit.reach = if clear < orbit.reach {
        clear
    } else {
        orbit.reach + (clear - orbit.reach) * blend
    };
    let offset = shake.offset(time.delta_seconds);
    let position = orbit.focus + back * orbit.reach + offset;
    let target = orbit.focus + offset;
    if camera.position != position || camera.target != target {
        camera.position = position;
        camera.target = target;
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CameraShake {
    strength: f32,
    seconds: f32,
}

impl Operation for CameraShake {
    const NAME: &'static str = "camera_shake";
    type Params = ResMut<'static, Shake>;
    type Request = Self;

    fn run(
        mut shake: SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        request
            .strength
            .check("strength", STRENGTH)
            .and_then(|()| request.seconds.check("seconds", SECONDS))
            .map_err(|invalid| invalid.error(ErrorCode::InvalidValue, "$"))?;
        shake.strength = request.strength;
        shake.seconds = request.seconds;
        shake.left = request.seconds;
        Ok(json!({"strength": request.strength, "seconds": request.seconds}))
    }
}
