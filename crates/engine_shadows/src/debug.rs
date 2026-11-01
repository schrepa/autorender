use std::any::TypeId;

use engine_places::shadows::{DEBUG_SHADOWED, DEBUG_SUN_BEYOND, DEBUG_SUN_HORIZON, DEBUG_SUN_LEVEL};
use engine_render::debug::DEBUG_SOURCE_SHIFT;
use engine_render::{DebugMode, DebugShown};
use serde::Serialize;
use serde_json::Value;

use crate::ShadowsBrick;
use crate::constants::{DEBUG_SHOWN_CASTERS, DEBUG_SHOWN_SHADOWS};

pub(crate) const MODES: [(&str, u32); 2] = [
    ("shadows", DEBUG_SHOWN_SHADOWS),
    ("casters", DEBUG_SHOWN_CASTERS),
];

pub(crate) fn member(name: &'static str) -> DebugMode {
    DebugMode { name, decode: sun }
}

pub(crate) fn shown(shown: DebugShown) -> u32 {
    if shown.owner != Some(TypeId::of::<ShadowsBrick>()) {
        return 0;
    }
    MODES.get(shown.local as usize).map_or(0, |(_, code)| *code)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum SunLookup {
    None,
    Level,
    Horizon,
    Beyond,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct SunProbe {
    sun: SunLookup,
    sun_index: u32,
    shadowed: bool,
}

fn sun_lookup(source: u32) -> (SunLookup, u32) {
    match source {
        DEBUG_SUN_HORIZON => (SunLookup::Horizon, 0),
        DEBUG_SUN_BEYOND => (SunLookup::Beyond, 0),
        level if level >= DEBUG_SUN_LEVEL => (SunLookup::Level, level - DEBUG_SUN_LEVEL),
        _ => (SunLookup::None, 0),
    }
}

fn sun([record, ..]: [u32; 4], _: [f32; 4]) -> Value {
    let sun = (record >> DEBUG_SOURCE_SHIFT) & 0xff;
    let (lookup, sun_index) = sun_lookup(sun & (DEBUG_SHADOWED - 1));
    serde_json::to_value(SunProbe {
        sun: lookup,
        sun_index,
        shadowed: sun & DEBUG_SHADOWED != 0,
    })
    .unwrap_or(Value::Null)
}
