use bevy_ecs::prelude::*;
use engine_core::math::max;
use engine_core::{
    Asks, Binding, BindingSource, Devices, Input, MouseButton, PadButton, Player, TouchPhase, Verb,
    VerbId, VerbShape,
};
use engine_places_rendering::screens::{ScreenAsk, Screens};
use engine_places_rendering::ui::{UiAction, UiEvents, UiKind};
use glam::Vec2;

use crate::tree::UiState;

const WHEEL_STEP: f32 = 40.0;
const CROSS_WEIGHT: f32 = 2.0;

pub(crate) const VERBS: [Verb; 6] = [
    Verb {
        name: "ui.up",
        shape: VerbShape::Button,
        bindings: &[
            Binding::new("ui.up", BindingSource::key("ArrowUp")),
            Binding::new("ui.up", BindingSource::Pad(PadButton::Up)),
        ],
    },
    Verb {
        name: "ui.down",
        shape: VerbShape::Button,
        bindings: &[
            Binding::new("ui.down", BindingSource::key("ArrowDown")),
            Binding::new("ui.down", BindingSource::key("Tab")),
            Binding::new("ui.down", BindingSource::Pad(PadButton::Down)),
        ],
    },
    Verb {
        name: "ui.left",
        shape: VerbShape::Button,
        bindings: &[
            Binding::new("ui.left", BindingSource::key("ArrowLeft")),
            Binding::new("ui.left", BindingSource::Pad(PadButton::Left)),
        ],
    },
    Verb {
        name: "ui.right",
        shape: VerbShape::Button,
        bindings: &[
            Binding::new("ui.right", BindingSource::key("ArrowRight")),
            Binding::new("ui.right", BindingSource::Pad(PadButton::Right)),
        ],
    },
    Verb {
        name: "ui.accept",
        shape: VerbShape::Button,
        bindings: &[
            Binding::new("ui.accept", BindingSource::key("Enter")),
            Binding::new("ui.accept", BindingSource::Pad(PadButton::South)),
        ],
    },
    Verb {
        name: "ui.back",
        shape: VerbShape::Button,
        bindings: &[
            Binding::new("ui.back", BindingSource::key("Escape")),
            Binding::new("ui.back", BindingSource::Pad(PadButton::East)),
        ],
    },
];

#[derive(Clone, Copy, Default)]
pub(crate) struct Menu {
    up: bool,
    down: bool,
    left: bool,
    right: bool,
    accept: bool,
    back: bool,
}

impl Menu {
    pub(crate) fn of(input: &Input, verbs: Option<[VerbId; 6]>) -> Self {
        let Some(verbs) = verbs else {
            return Self::default();
        };
        let [up, down, left, right, accept, back] = verbs
            .map(|verb| (1..=input.players()).any(|player| input.pressed(verb, Player(player))));
        Self {
            up,
            down,
            left,
            right,
            accept,
            back,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Pointer {
    position_pixels: Option<Vec2>,
    inside: bool,
    held: bool,
    pressed: bool,
    released: bool,
    wheel_notches: Vec2,
}

impl Pointer {
    pub(crate) fn of(devices: &Devices, reach: bool) -> Self {
        if let Some(touch) = devices.touches.first() {
            return Self {
                position_pixels: Some(touch.position_pixels),
                inside: reach,
                held: matches!(touch.phase, TouchPhase::Began | TouchPhase::Moved),
                pressed: touch.phase == TouchPhase::Began,
                released: touch.phase == TouchPhase::Ended,
                wheel_notches: Vec2::ZERO,
            };
        }
        let mouse = &devices.mouse;
        let left = MouseButton::Left;
        Self {
            position_pixels: mouse.position_pixels,
            inside: mouse.inside && reach,
            held: mouse.buttons.held.contains(&left),
            pressed: mouse.buttons.pressed.contains(&left),
            released: mouse.buttons.released.contains(&left),
            wheel_notches: mouse.wheel_notches,
        }
    }
}

type Steer<'a, 'w, 's> = (&'a Screens, &'a mut Asks<'w, 's, ScreenAsk>);

pub(crate) fn run(
    state: &mut UiState,
    (pointer, reach): (&Pointer, bool),
    menu: &Menu,
    (screens, asks): Steer<'_, '_, '_>,
    events: &mut ResMut<'_, UiEvents>,
) {
    for id in std::mem::take(&mut state.pending) {
        if let Some(at) = state.named(&id).filter(|&at| reach && state.pressable(at)) {
            activate(state, at, asks, events);
        }
    }
    hover(state, pointer);
    press(state, pointer, asks, events);
    if state.wanting {
        navigate(state, menu, (screens, asks), events);
    }
    wheel(state, pointer);
}

fn activate(
    state: &mut UiState,
    at: u32,
    asks: &mut Asks<'_, '_, ScreenAsk>,
    events: &mut ResMut<'_, UiEvents>,
) {
    let Some(action) = state.nodes[at as usize].node.action.clone() else {
        return;
    };
    match action {
        UiAction::Go(name) => asks.send(ScreenAsk::Go(name)),
        UiAction::Push(name) => asks.send(ScreenAsk::Push(name)),
        UiAction::Pop => asks.send(ScreenAsk::Pop),
        UiAction::Emit(name) => {
            events.emit(&name);
            state.events += 1;
        }
    }
}

fn spot(state: &UiState, pointer: &Pointer) -> Option<Vec2> {
    let ratio = max(state.metrics[2], f32::EPSILON);
    pointer
        .position_pixels
        .filter(|_| pointer.inside)
        .map(|at| at / ratio)
}

pub(crate) fn over(state: &UiState, pointer: &Pointer) -> bool {
    spot(state, pointer).is_some_and(|point| state.over(point))
}

fn hover(state: &mut UiState, pointer: &Pointer) {
    state.hovered = spot(state, pointer).and_then(|point| state.top_at(point, UiKind::Button));
}

fn press(
    state: &mut UiState,
    pointer: &Pointer,
    asks: &mut Asks<'_, '_, ScreenAsk>,
    events: &mut ResMut<'_, UiEvents>,
) {
    if pointer.pressed {
        state.pressed = state.hovered;
    }
    if pointer.released {
        let pressed = state.pressed.take();
        if let Some(at) = pressed.filter(|_| pressed == state.hovered) {
            activate(state, at, asks, events);
        }
    } else if !pointer.held {
        state.pressed = None;
    }
}

fn navigate(
    state: &mut UiState,
    menu: &Menu,
    (screens, asks): Steer<'_, '_, '_>,
    events: &mut ResMut<'_, UiEvents>,
) {
    let ways = [
        (menu.up, Vec2::NEG_Y),
        (menu.down, Vec2::Y),
        (menu.left, Vec2::NEG_X),
        (menu.right, Vec2::X),
    ];
    for (asked, way) in ways {
        if asked {
            move_focus(state, way);
        }
    }
    if menu.accept
        && let Some(at) = state.focused
    {
        activate(state, at, asks, events);
    }
    if menu.back && screens.stack().len() > 1 {
        asks.send(ScreenAsk::Pop);
    }
}

fn move_focus(state: &mut UiState, way: Vec2) {
    let from = state.focused.map(|at| state.nodes[at as usize].center());
    let mut best: Option<(f32, u32)> = None;
    for (at, flat) in state.nodes.iter().enumerate() {
        if flat.node.kind != UiKind::Button || flat.at[2] <= 0.0 || flat.at[3] <= 0.0 {
            continue;
        }
        let score = match from {
            None => at as f32,
            Some(from) => {
                let delta = flat.center() - from;
                let along = delta.dot(way);
                if along <= 0.0 {
                    continue;
                }
                along + (delta - way * along).length() * CROSS_WEIGHT
            }
        };
        if best.is_none_or(|(kept, _)| score < kept) {
            best = Some((score, at as u32));
        }
    }
    if let Some((_, at)) = best {
        state.focused = Some(at);
        state.reveal(at);
    }
}

fn wheel(state: &mut UiState, pointer: &Pointer) {
    let turn = pointer.wheel_notches.x + pointer.wheel_notches.y;
    if turn == 0.0 {
        return;
    }
    let list = spot(state, pointer).and_then(|point| state.top_at(point, UiKind::List));
    if let Some(list) = list {
        let scale = state.metrics[3];
        state.scroll_by(list, -turn * WHEEL_STEP * scale);
    }
}
