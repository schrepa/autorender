use bevy_ecs::prelude::*;
use engine_core::{MenuInput, PointerInput, Screens};
use engine_places::ui::{UiAction, UiEvents, UiKind};
use glam::Vec2;

use crate::tree::UiState;

const WHEEL_STEP: f32 = 40.0;
const CROSS_WEIGHT: f32 = 2.0;

pub(crate) fn run(
    state: &mut UiState,
    pointer: &PointerInput,
    menu: &MenuInput,
    screens: &mut ResMut<'_, Screens>,
    events: &mut ResMut<'_, UiEvents>,
) {
    for id in std::mem::take(&mut state.pending) {
        if let Some(at) = state.named(&id) {
            activate(state, at, screens, events);
        }
    }
    if state.buttons {
        hover(state, pointer);
        press(state, pointer, screens, events);
    }
    if state.wanting {
        navigate(state, menu, screens, events);
    }
    wheel(state, pointer);
}

fn activate(
    state: &mut UiState,
    at: u32,
    screens: &mut ResMut<'_, Screens>,
    events: &mut ResMut<'_, UiEvents>,
) {
    let Some(action) = state.nodes[at as usize].node.action.clone() else {
        return;
    };
    match action {
        UiAction::Go(name) => screens.go(name),
        UiAction::Push(name) => screens.push(name),
        UiAction::Pop => {
            screens.pop();
        }
        UiAction::Emit(name) => {
            events.emit(&name);
            state.events += 1;
        }
    }
}

fn hover(state: &mut UiState, pointer: &PointerInput) {
    let hovered = pointer
        .position
        .filter(|_| pointer.inside)
        .and_then(|point| state.top_at(point, UiKind::Button));
    if hovered == state.hovered {
        return;
    }
    state.hovered = hovered;
    state.revision += 1;
    if hovered.is_some() {
        state.focused = hovered;
    }
}

fn press(
    state: &mut UiState,
    pointer: &PointerInput,
    screens: &mut ResMut<'_, Screens>,
    events: &mut ResMut<'_, UiEvents>,
) {
    if pointer.pressed {
        state.pressed = state.hovered;
        state.revision += 1;
    }
    if pointer.released {
        let pressed = state.pressed.take();
        state.revision += 1;
        if let Some(at) = pressed.filter(|_| pressed == state.hovered) {
            activate(state, at, screens, events);
        }
    } else if !pointer.held && state.pressed.take().is_some() {
        state.revision += 1;
    }
}

fn navigate(
    state: &mut UiState,
    menu: &MenuInput,
    screens: &mut ResMut<'_, Screens>,
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
        activate(state, at, screens, events);
    }
    if menu.back && screens.stack().len() > 1 {
        screens.pop();
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
        state.revision += 1;
        state.reveal(at);
    }
}

fn wheel(state: &mut UiState, pointer: &PointerInput) {
    let turn = pointer.wheel.x + pointer.wheel.y;
    if turn == 0.0 {
        return;
    }
    let list = pointer
        .position
        .filter(|_| pointer.inside)
        .and_then(|point| state.top_at(point, UiKind::List));
    if let Some(list) = list {
        let scale = state.metrics[3];
        state.scroll_by(list, -turn * WHEEL_STEP * scale);
    }
}
