use bevy_ecs::prelude::*;
use engine_core::{Cursor, CursorRequests, Error, ErrorCode, Events, Operation};
use engine_places_rendering::ui::{UiEvents, UiKind};
use serde::{Deserialize, Serialize};

use crate::tree::{Flat, UiState};

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct UiPress {
    id: String,
}

fn pressable_ids(state: &UiState, reach: bool) -> Vec<String> {
    (0..state.nodes.len() as u32)
        .filter(|&at| reach && state.pressable(at))
        .filter_map(|at| state.nodes[at as usize].node.id.clone())
        .collect()
}

impl Operation for UiPress {
    const NAME: &'static str = "ui_press";
    type Params = (ResMut<'static, UiState>, Res<'static, CursorRequests>);
    type Request = Self;
    type Reply = UiPressed;

    fn run(
        (mut state, cursor): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<UiPressed, Error> {
        let reach = crate::reaches(&cursor);
        let pressed = state
            .named(&request.id)
            .is_some_and(|at| reach && state.pressable(at));
        if !pressed {
            return Err(Error::new(ErrorCode::InvalidValue, "$.id")
                .got(&request.id)
                .variants(pressable_ids(&state, reach)));
        }
        state.pending.push(request.id.clone());
        Ok(UiPressed {
            id: request.id,
            applies_on: "next_step",
        })
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub(crate) struct UiPressed {
    id: String,
    applies_on: &'static str,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub(crate) struct UiNodeView {
    id: Option<String>,
    kind: UiKind,
    parent: Option<u32>,
    rect: [f32; 4],
    text: String,
    fill: Option<f32>,
    state: &'static str,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub(crate) struct UiTreeView {
    nodes: Vec<UiNodeView>,
    events: Vec<String>,
}

fn state_of(state: &UiState, at: u32) -> &'static str {
    let here = Some(at);
    if state.pressed == here {
        "pressed"
    } else if state.hovered == here {
        "hovered"
    } else if state.focused == here {
        "focused"
    } else {
        "idle"
    }
}

fn node_view(state: &UiState, at: u32, flat: &Flat) -> UiNodeView {
    UiNodeView {
        id: flat.node.id.clone(),
        kind: flat.node.kind,
        parent: flat.parent,
        rect: flat.at,
        text: flat.text.clone(),
        fill: (flat.node.kind == UiKind::Bar).then_some(flat.fill),
        state: state_of(state, at),
    }
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct UiTree {}

impl Operation for UiTree {
    const NAME: &'static str = "ui_tree";
    type Params = (Res<'static, UiState>, Res<'static, UiEvents>);
    type Request = Self;
    type Reply = UiTreeView;

    fn run(
        (state, events): SystemParamItem<'_, '_, Self::Params>,
        _: Self,
    ) -> Result<UiTreeView, Error> {
        let nodes: Vec<UiNodeView> = state
            .nodes
            .iter()
            .enumerate()
            .map(|(at, flat)| node_view(&state, at as u32, flat))
            .collect();
        let raised = match events.since(&mut Cursor::default()) {
            Events::Listed(listed) => listed,
            Events::Missed(_) => &[],
        };
        Ok(UiTreeView {
            nodes,
            events: raised.to_vec(),
        })
    }
}
