use std::any::TypeId;

use bevy_ecs::prelude::*;
use engine_core::{DrawKind, Interest, Joined, WholeFrames};
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

pub(crate) const WHOLE: &str = "debug_view";

#[derive(Resource, Default)]
pub(crate) struct Whole(Option<Interest<&'static str>>);

pub(crate) fn member(name: &'static str) -> DebugMode {
    DebugMode {
        name,
        decode: no_fields,
    }
}

pub(crate) fn name(mode: &DebugMode) -> &'static str {
    mode.name
}

fn no_fields(_: Option<DrawKind>, _: [u32; 4], _: [f32; 4]) -> Value {
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
    (view, probe, modes): (Res<DebugView>, Res<Probe>, Joined<DebugModes>),
    (mut whole, mut frames): (ResMut<Whole>, ResMut<WholeFrames>),
    mut shown: ResMut<DebugShown>,
    mut piece: ResMut<DebugViewPiece>,
) {
    let drawn = named_mode(&view, &modes);
    if drawn.is_some() != whole.0.is_some() {
        whole.0 = drawn.map(|_| frames.ask(WHOLE));
    }
    shown.set_if_neq(DebugShown {
        owner: drawn.map(|(owner, _)| owner),
        local: drawn.map_or(0, |(_, local)| local),
    });
    piece.set_if_neq(DebugViewPiece {
        view: [code(drawn), u32::from(view.shade), 0, 0],
        probe: probe.pixel(),
    });
}
