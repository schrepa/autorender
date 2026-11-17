use bevy_ecs::prelude::*;
use engine_core::{
    Asks, Binding, BindingSource, Button, Clock, Contexts, CursorMode, CursorRequests, Error,
    ErrorCode, FrameTime, GlobalTransforms, HandOver, Hold, Input, MouseButton, Mover, Moves,
    Operation, Pace, Player, Poses, Rule, Setting, Stick, Verb, VerbId, VerbShape, Views, mover_of,
};
use engine_math::{atan2, clamp, exp, max, min, sin, sin_cos};
use engine_places::queries::{CastFilter, CastShape, SceneQueries};
use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::{CameraTarget, FollowCamera, FollowCameraBrick};

const CONTEXT: &str = "follow_camera";
const DRAG: &str = "follow_camera.drag";
const CAPTURED: &str = "follow_camera.captured";

pub(crate) const LOOK: Verb = Verb {
    name: "follow_camera.look",
    shape: VerbShape::Delta2,
    bindings: &[
        Binding::of(
            "follow_camera.look",
            BindingSource::MouseMotion,
            Some(Button::Mouse(MouseButton::Left)),
            Some(DRAG),
        ),
        Binding::of(
            "follow_camera.look",
            BindingSource::TouchMotion,
            None,
            Some(DRAG),
        ),
        Binding::of(
            "follow_camera.look",
            BindingSource::MouseMotion,
            None,
            Some(CAPTURED),
        ),
    ],
};

pub(crate) const TURN: Verb = Verb {
    name: "follow_camera.turn",
    shape: VerbShape::Value2,
    bindings: &[
        Binding::new("follow_camera.turn", BindingSource::Stick(Stick::Right)).dead_zone(0.15),
    ],
};

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

pub(crate) struct Controls {
    look: VerbId,
    turn: VerbId,
    looking: (CursorMode, Option<Hold>),
    _context: Hold,
    _cursor: Hold,
}

fn context_of(mode: CursorMode) -> Option<&'static str> {
    match mode {
        CursorMode::Drag => Some(DRAG),
        CursorMode::Captured => Some(CAPTURED),
        CursorMode::Free => None,
    }
}

impl Controls {
    fn look_in(&mut self, mode: CursorMode, contexts: &mut Contexts) {
        if self.looking.0 != mode {
            self.looking = (
                mode,
                context_of(mode).map(|context| contexts.ask(CONTEXT, context, false)),
            );
        }
    }

    fn turn(&self, input: &Input, settings: &FollowCamera, real_seconds: f32) -> Vec2 {
        let turned = input.value2(self.turn, Player::ONE);
        let unit = Vec2::new(clamp(turned.x, -1.0, 1.0), clamp(turned.y, -1.0, 1.0));
        let looked = input.value2(self.look, Player::ONE);
        Vec2::new(
            looked.x * settings.look_sensitivity + unit.x * settings.turn_speed * real_seconds,
            looked.y * settings.look_sensitivity - unit.y * settings.turn_speed * real_seconds,
        )
    }
}

#[derive(Resource, Default)]
pub(crate) enum Orbit {
    #[default]
    Taking,
    Waiting(Controls),
    Following(Follow, Controls),
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

fn kept(held: Option<Follow>, controls: Controls) -> Orbit {
    match held {
        Some(follow) => Orbit::Following(follow, controls),
        None => Orbit::Waiting(controls),
    }
}

fn taken(
    input: &Input,
    (contexts, cursor): (&mut Contexts, &mut CursorRequests),
) -> Option<Controls> {
    Some(Controls {
        look: input.find(LOOK.name)?,
        turn: input.find(TURN.name)?,
        looking: (CursorMode::Free, None),
        _context: contexts.ask(CONTEXT, CONTEXT, false),
        _cursor: cursor.ask(CONTEXT, CursorMode::Drag),
    })
}

pub(crate) fn follow(
    (time, input, settings): (Res<FrameTime>, Res<Input>, Res<FollowCamera>),
    (globals, poses, queries): (Res<GlobalTransforms>, Poses, Res<SceneQueries>),
    (views, targets): (Res<Views>, Query<Entity, With<CameraTarget>>),
    (mut contexts, mut cursor, mut handover): (
        ResMut<Contexts>,
        ResMut<CursorRequests>,
        Asks<HandOver>,
    ),
    (mut orbit, mut shake): (ResMut<Orbit>, ResMut<Shake>),
    mut moves: Moves<FollowCameraBrick>,
) {
    let Some(camera) = views.main().map(|main| main.camera) else {
        return;
    };
    let (held, mut controls) = match std::mem::take(&mut *orbit) {
        Orbit::Taking => {
            let Some(controls) = taken(&input, (&mut *contexts, &mut *cursor)) else {
                return;
            };
            handover.send(HandOver {
                node: camera,
                to: Mover {
                    by: mover_of::<FollowCameraBrick>(),
                    pace: Pace::Frame,
                },
            });
            *orbit = Orbit::Waiting(controls);
            return;
        }
        Orbit::Waiting(controls) => (None, controls),
        Orbit::Following(follow, controls) => (Some(follow), controls),
    };
    controls.look_in(cursor.applied(), &mut contexts);
    let Some(target) = targets.iter().min() else {
        *orbit = Orbit::Waiting(controls);
        return;
    };
    let Some(pose) = poses.drawn_pose(target) else {
        *orbit = kept(held, controls);
        return;
    };
    let goal = pose.w_axis.truncate() + Vec3::Y * settings.height;
    let clock = globals.word(camera).map_or(Clock::Real, |word| word.clock);
    let seconds = time.seconds_on(clock) as f32;
    let Some(mut transform) = moves.get_mut(camera) else {
        *orbit = kept(held, controls);
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
    let turned = controls.turn(&input, &settings, time.real_seconds as f32);
    follow.yaw = (follow.yaw - turned.x) % std::f32::consts::TAU;
    follow.pitch = clamp(follow.pitch - turned.y, low, high);
    follow.focus = follow.focus.lerp(goal, blend);
    let rotation = follow.rotation();
    let back = rotation * Vec3::Z * settings.distance;
    let clear = queries
        .0
        .cast(
            CastShape::Sphere {
                radius: settings.collision_radius,
            },
            follow.focus,
            back,
            CastFilter {
                ignore: Some(target),
                ..CastFilter::default()
            },
        )
        .map_or(1.0, |hit| clamp(hit.distance / settings.distance, 0.0, 1.0));
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
    *orbit = Orbit::Following(follow, controls);
}

#[derive(Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CameraShake {
    strength: f32,
    seconds: f32,
}

impl Operation for CameraShake {
    const NAME: &'static str = "camera_shake";
    type Params = ResMut<'static, Shake>;
    type Request = Self;
    type Reply = Self;

    fn run(mut shake: SystemParamItem<'_, '_, Self::Params>, request: Self) -> Result<Self, Error> {
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
        Ok(request)
    }
}
