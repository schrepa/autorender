use std::any::TypeId;

use bevy_ecs::prelude::*;
use engine_core::Joined;
use engine_render::{DebugMode, DebugModes, DebugShown};
use serde_json::Value;

use crate::gpu::{
    DEBUG_VIEW_CARDS, DEBUG_VIEW_KINDS, DEBUG_VIEW_LEVELS, DEBUG_VIEW_NORMALS, DEBUG_VIEW_OVERDRAW,
    DebugViewPiece,
};
use crate::probe::Probe;
use crate::{DebugView, DebugViewBrick};

pub(crate) const OWN: [(&str, u32); 5] = [
    ("kinds", DEBUG_VIEW_KINDS),
    ("levels", DEBUG_VIEW_LEVELS),
    ("cards", DEBUG_VIEW_CARDS),
    ("overdraw", DEBUG_VIEW_OVERDRAW),
    ("normals", DEBUG_VIEW_NORMALS),
];

pub(crate) fn member(name: &'static str) -> DebugMode {
    DebugMode {
        name,
        decode: drawn,
    }
}

fn drawn(_: [u32; 4], _: [f32; 4]) -> Value {
    Value::Null
}

fn kinds() -> DebugShown {
    DebugShown {
        owner: Some(TypeId::of::<DebugViewBrick>()),
        local: 0,
    }
}

fn code(shown: DebugShown) -> u32 {
    if shown.owner != Some(TypeId::of::<DebugViewBrick>()) {
        return 0;
    }
    OWN.get(shown.local as usize).map_or(0, |(_, code)| *code)
}

pub(crate) fn publish(
    view: Res<DebugView>,
    probe: Res<Probe>,
    modes: Joined<DebugModes>,
    mut shown: ResMut<DebugShown>,
    mut piece: ResMut<DebugViewPiece>,
) {
    let selected = if view.showing() {
        modes
            .owned()
            .find(|(_, _, mode)| mode.name == view.view.as_str())
            .map(|(owner, local, _)| DebugShown {
                owner: Some(owner),
                local,
            })
    } else {
        None
    };
    let next = match probe.forces() {
        Some(false) => DebugShown::default(),
        Some(true) => selected.unwrap_or_else(kinds),
        None => selected.unwrap_or_default(),
    };
    shown.set_if_neq(next);
    piece.set_if_neq(DebugViewPiece {
        view: [code(next), u32::from(view.shade), 0, 0],
    });
}
