use bevy_ecs::prelude::*;
use engine_core::{
    Asks, CameraHandoff, CameraNode, Clock, Error, ErrorCode, FrameTime, GlobalTransforms,
    MoveInput, Mover, Moves, Operation, Pace, Poses, Rule, Setting, mover_of,
};
use engine_math::{atan2, clamp, exp, max, min, sin, sin_cos};
use engine_places::sweeps::Sweeps;
use glam::{Quat, Vec3};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{CameraTarget, FollowCamera, FollowCameraBrick};

const SENSITIVITY: f32 = 0.003;
const STRENGTH: Rule = Rule::range(0.0, 10.0);
const SECONDS: Rule = Rule::range(0.0, 60.0);
const SHAKE_RATES: [f32; 3] = [37.0, 53.0, 71.0];

#[derive(Clone, Copy)]
pub(crate) struct Follow {
    target: Entity,
    yaw: f32,
    pitch: f32,
    focus: Vec3,
    reach: f32,
}

impl Follow {
    fn start(target: Entity, rotation: Quat, focus: Vec3) -> Self {
        let forward = rotation * Vec3::NEG_Z;
        Self {
            target,
            yaw: atan2(-forward.x, -forward.z),
            pitch: clamp(forward.y, -1.0, 1.0).asin(),
            focus,
            reach: 1.0,
        }
    }

    fn rotation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch)
    }
}

#[derive(Resource, Default)]
pub(crate) enum Orbit {
    #[default]
    Taking,
    Waiting,
    Following(Follow),
}

#[derive(Resource, Default)]
pub(crate) enum Shake {
    #[default]
    Still,
    Shaking {
        strength: f32,
        seconds: f32,
        left: f32,
    },
}

impl Shake {
    fn offset(&mut self, elapsed: f32) -> Vec3 {
        let Self::Shaking {
            strength,
            seconds,
            left,
        } = *self
        else {
            return Vec3::ZERO;
        };
        let left = max(left - elapsed, 0.0);
        *self = if left > 0.0 {
            Self::Shaking {
                strength,
                seconds,
                left,
            }
        } else {
            Self::Still
        };
        let fade = left / seconds;
        let at = seconds - left;
        let (across, up) = sin_cos(at * SHAKE_RATES[0]);
        let along = sin(at * SHAKE_RATES[2]);
        let bob = sin(at * SHAKE_RATES[1]);
        Vec3::new(across, up * bob, along) * strength * fade
    }
}

fn eased(seconds: f32, smoothing: f32) -> f32 {
    if smoothing > 0.0 {
        1.0 - exp(-seconds / smoothing)
    } else {
        1.0
    }
}

pub(crate) fn follow(
    (time, input, settings): (Res<FrameTime>, Res<MoveInput>, Res<FollowCamera>),
    (globals, poses, sweeps): (Res<GlobalTransforms>, Poses, Res<Sweeps>),
    (cameras, targets): (
        Query<Entity, With<CameraNode>>,
        Query<Entity, With<CameraTarget>>,
    ),
    mut handoff: Asks<CameraHandoff>,
    (mut orbit, mut shake): (ResMut<Orbit>, ResMut<Shake>),
    mut moves: Moves<FollowCameraBrick>,
) {
    let Ok(camera) = cameras.single() else {
        return;
    };
    let held = match *orbit {
        Orbit::Taking => {
            handoff.send(CameraHandoff {
                camera,
                to: Mover {
                    by: mover_of::<FollowCameraBrick>(),
                    pace: Pace::Frame,
                },
            });
            *orbit = Orbit::Waiting;
            return;
        }
        Orbit::Waiting => None,
        Orbit::Following(follow) => Some(follow),
    };
    let Some(target) = targets.iter().min() else {
        *orbit = Orbit::Waiting;
        return;
    };
    let Some(pose) = poses.drawn_pose(target) else {
        return;
    };
    let goal = pose.w_axis.truncate() + Vec3::Y * settings.height;
    let clock = globals.word(camera).map_or(Clock::Real, |word| word.clock);
    let seconds = time.seconds_on(clock) as f32;
    let Some(mut transform) = moves.get_mut(camera) else {
        return;
    };
    let mut follow = match held {
        Some(follow) if follow.target == target => follow,
        Some(_) | None => Follow::start(target, transform.rotation, goal),
    };
    let blend = eased(seconds, settings.smoothing);
    let (low, high) = (
        min(settings.pitch_min, settings.pitch_max),
        max(settings.pitch_min, settings.pitch_max),
    );
    follow.yaw = (follow.yaw - input.look_right_pixels * SENSITIVITY) % std::f32::consts::TAU;
    follow.pitch = clamp(
        follow.pitch - input.look_down_pixels * SENSITIVITY,
        low,
        high,
    );
    follow.focus = follow.focus.lerp(goal, blend);
    let rotation = follow.rotation();
    let back = rotation * Vec3::Z * settings.distance;
    let clear = sweeps
        .0
        .sphere(settings.collision_radius, follow.focus, back, target)
        .map_or(1.0, |fraction| clamp(fraction, 0.0, 1.0));
    follow.reach = if clear < follow.reach {
        clear
    } else {
        follow.reach + (clear - follow.reach) * blend
    };
    let position = follow.focus + back * follow.reach + shake.offset(seconds);
    if transform.translation != position || transform.rotation != rotation {
        transform.translation = position;
        transform.rotation = rotation;
    }
    *orbit = Orbit::Following(follow);
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
        *shake = if request.seconds > 0.0 {
            Shake::Shaking {
                strength: request.strength,
                seconds: request.seconds,
                left: request.seconds,
            }
        } else {
            Shake::Still
        };
        Ok(json!({"strength": request.strength, "seconds": request.seconds}))
    }
}
