use bevy_ecs::prelude::*;
use engine_core::{
    Asks, Binding, BindingSource, Button, Clock, Contexts, CursorMode, CursorRequests,
    FrameTime, GlobalTransforms, HandOver, Hold, Input, MouseButton, Mover,
    Moves, Pace, Player, Stick, Trigger, Verb, VerbId, VerbShape, Views, mover_of,
};
use engine_math::{atan2, clamp};
use glam::{Quat, Vec2, Vec3};

use crate::{FlyCamera, FlyCameraBrick};

const PITCH_LIMIT: f32 = 1.48;
const CONTEXT: &str = "fly_camera";
const DRAG: &str = "fly_camera.drag";
const CAPTURED: &str = "fly_camera.captured";

pub(crate) const MOVE: Verb = Verb {
    name: "fly_camera.move",
    shape: VerbShape::Value2,
    bindings: &[
        Binding::new(
            "fly_camera.move",
            BindingSource::keys("KeyD", "KeyA", "KeyW", "KeyS"),
        ),
        Binding::new(
            "fly_camera.move",
            BindingSource::keys("ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown"),
        ),
        Binding::new("fly_camera.move", BindingSource::Stick(Stick::Left)).dead_zone(0.15),
    ],
};

pub(crate) const RISE: Verb = Verb {
    name: "fly_camera.rise",
    shape: VerbShape::Value,
    bindings: &[
        Binding::new("fly_camera.rise", BindingSource::key("KeyE")),
        Binding::new("fly_camera.rise", BindingSource::key("KeyQ")).scaled(-1.0),
        Binding::new("fly_camera.rise", BindingSource::Trigger(Trigger::Right)),
        Binding::new("fly_camera.rise", BindingSource::Trigger(Trigger::Left)).scaled(-1.0),
    ],
};

pub(crate) const LOOK: Verb = Verb {
    name: "fly_camera.look",
    shape: VerbShape::Delta2,
    bindings: &[
        Binding::of(
            "fly_camera.look",
            BindingSource::MouseMotion,
            Some(Button::Mouse(MouseButton::Left)),
            Some(DRAG),
        ),
        Binding::of(
            "fly_camera.look",
            BindingSource::TouchMotion,
            None,
            Some(DRAG),
        ),
        Binding::of(
            "fly_camera.look",
            BindingSource::MouseMotion,
            None,
            Some(CAPTURED),
        ),
    ],
};

pub(crate) const TURN: Verb = Verb {
    name: "fly_camera.turn",
    shape: VerbShape::Value2,
    bindings: &[Binding::new("fly_camera.turn", BindingSource::Stick(Stick::Right)).dead_zone(0.15)],
};

pub(crate) struct Controls {
    verbs: [VerbId; 4],
    looking: (CursorMode, Option<Hold>),
    _context: Hold,
    _cursor: Hold,
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
}

fn context_of(mode: CursorMode) -> Option<&'static str> {
    match mode {
        CursorMode::Drag => Some(DRAG),
        CursorMode::Captured => Some(CAPTURED),
        CursorMode::Free => None,
    }
}

#[derive(Resource, Default)]
pub(crate) enum Rig {
    #[default]
    Taking,
    Holding(Controls),
}

struct Steer {
    moving: Vec2,
    rising: f32,
    looking: Vec2,
    turning: Vec2,
}

impl Steer {
    fn of(input: &Input, [moving, rising, looking, turning]: [VerbId; 4]) -> Self {
        let unit = |value: f32| clamp(value, -1.0, 1.0);
        let moved = input.value2(moving, Player::ONE);
        let turned = input.value2(turning, Player::ONE);
        Self {
            moving: Vec2::new(unit(moved.x), unit(moved.y)),
            rising: unit(input.value(rising, Player::ONE)),
            looking: input.value2(looking, Player::ONE),
            turning: Vec2::new(unit(turned.x), unit(turned.y)),
        }
    }

    fn steered(&self) -> bool {
        self.moving != Vec2::ZERO
            || self.rising != 0.0
            || self.looking != Vec2::ZERO
            || self.turning != Vec2::ZERO
    }
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
    fn turned(self, steer: &Steer, settings: &FlyCamera, real_seconds: f32) -> Self {
        let turn = settings.turn_speed * real_seconds;
        let sensitivity = settings.look_sensitivity;
        Self {
            yaw: (self.yaw - steer.looking.x * sensitivity - steer.turning.x * turn)
                % std::f32::consts::TAU,
            pitch: clamp(
                self.pitch - steer.looking.y * sensitivity + steer.turning.y * turn,
                -PITCH_LIMIT,
                PITCH_LIMIT,
            ),
        }
    }

    fn rotation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch)
    }
}

pub(crate) fn fly(
    (time, input, settings, globals): (
        Res<FrameTime>,
        Res<Input>,
        Res<FlyCamera>,
        Res<GlobalTransforms>,
    ),
    (views, mut contexts, mut cursor): (Res<Views>, ResMut<Contexts>, ResMut<CursorRequests>),
    mut rig: ResMut<Rig>,
    mut handover: Asks<HandOver>,
    mut moves: Moves<FlyCameraBrick>,
) {
    let Some(camera) = views.main().map(|main| main.camera) else {
        return;
    };
    let verbs = match &mut *rig {
        Rig::Holding(controls) => {
            controls.look_in(cursor.applied(), &mut contexts);
            controls.verbs
        }
        Rig::Taking => {
            let [Some(moving), Some(rising), Some(looking), Some(turning)] =
                [MOVE, RISE, LOOK, TURN].map(|verb| input.find(verb.name))
            else {
                return;
            };
            handover.send(HandOver {
                node: camera,
                to: Mover {
                    by: mover_of::<FlyCameraBrick>(),
                    pace: Pace::Frame,
                },
            });
            *rig = Rig::Holding(Controls {
                verbs: [moving, rising, looking, turning],
                looking: (CursorMode::Free, None),
                _context: contexts.ask(CONTEXT, CONTEXT, false),
                _cursor: cursor.ask(CONTEXT, CursorMode::Drag),
            });
            return;
        }
    };
    let steer = Steer::of(&input, verbs);
    if !steer.steered() {
        return;
    }
    let clock = globals.word(camera).map_or(Clock::Real, |word| word.clock);
    let seconds = time.seconds_on(clock) as f32;
    let Some(mut transform) = moves.get_mut(camera) else {
        return;
    };
    let flight = Flight::of(transform.rotation).turned(&steer, &settings, time.real_seconds as f32);
    let rotation = flight.rotation();
    let movement = rotation * Vec3::NEG_Z * steer.moving.y
        + rotation * Vec3::X * steer.moving.x
        + Vec3::Y * steer.rising;
    transform.translation += movement.normalize_or_zero() * seconds * settings.speed;
    transform.rotation = rotation;
}
