use bevy_ecs::prelude::*;
use engine_core::{Error, ErrorCode, Operation};
use engine_places::ui::{UiEvents, UiKind};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::tree::{Flat, UiState};

fn schema_of<T: schemars::JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(T)).unwrap_or(Value::Null)
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct UiPress {
    id: String,
}

impl Operation for UiPress {
    const NAME: &'static str = "ui_press";
    type Params = ResMut<'static, UiState>;
    type Request = Self;

    fn run(
        mut state: SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        let known = state.nodes.iter().any(|flat| {
            flat.node.action.is_some() && flat.node.id.as_deref() == Some(request.id.as_str())
        });
        if !known {
            let ids = state
                .nodes
                .iter()
                .filter(|flat| flat.node.action.is_some())
                .filter_map(|flat| flat.node.id.clone());
            return Err(Error::new(ErrorCode::InvalidValue, "$.id")
                .got(&request.id)
                .variants(ids));
        }
        state.pending.push(request.id.clone());
        Ok(json!({ "id": request.id, "applies_on": "next_step" }))
    }

    fn schema() -> Value {
        schema_of::<Self>()
    }
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

fn node_view(state: &UiState, at: u32, flat: &Flat) -> Value {
    json!({
        "id": flat.node.id,
        "kind": flat.node.kind,
        "parent": flat.parent,
        "rect": flat.at,
        "text": flat.text,
        "fill": (flat.node.kind == UiKind::Bar).then_some(flat.fill),
        "state": state_of(state, at),
    })
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct UiTree {}

impl Operation for UiTree {
    const NAME: &'static str = "ui_tree";
    type Params = (Res<'static, UiState>, Res<'static, UiEvents>);
    type Request = Self;

    fn run(
        (state, events): SystemParamItem<'_, '_, Self::Params>,
        _: Self,
    ) -> Result<Value, Error> {
        let nodes: Vec<Value> = state
            .nodes
            .iter()
            .enumerate()
            .map(|(at, flat)| node_view(&state, at as u32, flat))
            .collect();
        Ok(json!({ "nodes": nodes, "events": events.names() }))
    }

    fn schema() -> Value {
        schema_of::<Self>()
    }
}
