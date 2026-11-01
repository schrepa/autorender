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

type Drawn = (TypeId, u32);

pub(crate) fn member(name: &'static str) -> DebugMode {
    DebugMode {
        name,
        decode: no_fields,
    }
}

pub(crate) fn name(mode: &DebugMode) -> &'static str {
    mode.name
}

fn no_fields(_: [u32; 4], _: [f32; 4]) -> Value {
    Value::Null
}

fn named_mode(view: &DebugView, modes: &Joined<'_, DebugModes>) -> Option<Drawn> {
    if !view.showing() {
        return None;
    }
    modes
        .owned()
        .find(|(_, _, mode)| mode.name == view.view)
        .map(|(owner, local, _)| (owner, local))
}

fn code(drawn: Option<Drawn>) -> u32 {
    match drawn {
        Some((owner, local)) if owner == TypeId::of::<DebugViewBrick>() => {
            OWN.get(local as usize).map_or(0, |(_, code)| *code)
        }
        Some(_) | None => 0,
    }
}

pub(crate) fn publish(
    view: Res<DebugView>,
    probe: Res<Probe>,
    modes: Joined<DebugModes>,
    mut shown: ResMut<DebugShown>,
    mut piece: ResMut<DebugViewPiece>,
) {
    let chosen = named_mode(&view, &modes);
    let drawn = match probe.forces() {
        Some(false) => None,
        Some(true) => Some(chosen.unwrap_or((TypeId::of::<DebugViewBrick>(), 0))),
        None => chosen,
    };
    let coming = probe.coming();
    shown.set_if_neq(DebugShown {
        owner: drawn.map(|(owner, _)| owner),
        local: drawn.map_or(0, |(_, local)| local),
        next_debug: coming.unwrap_or(chosen.is_some()),
        next_whole: coming.is_some(),
    });
    piece.set_if_neq(DebugViewPiece {
        view: [code(drawn), u32::from(view.shade), 0, 0],
    });
}
