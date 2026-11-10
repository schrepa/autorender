use engine_core::math::half;
use engine_core::{BitValue, DrawKind, GpuBits};
use engine_render::DebugMode;
use serde::Serialize;
use serde_json::Value;

use crate::gpu::{DEBUG_SPAN_STEPS, Shown, WaterProbeDetail, WaterProbeState};
use crate::jobs::gpu::{WATER_SPAN_LEAST, WATER_SPAN_MOST};

pub(crate) fn modes() -> impl Iterator<Item = DebugMode> {
    <Shown as BitValue>::VALUES
        .iter()
        .map(|&(_, name)| DebugMode {
            name,
            decode: probe,
            fields: engine_core::schema_of::<Probed>,
        })
}

pub(crate) fn shown(local: u32) -> u32 {
    <Shown as BitValue>::VALUES
        .get(local as usize)
        .map_or(0, |&(value, _)| value)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum WaterReach {
    Pool,
    Plane,
    Steps,
    Cascade,
    Falls,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, schemars::JsonSchema)]
struct WaterProbe {
    depth: f32,
    speed: f32,
    froude: f32,
    turbulence: f32,
    white: f32,
    reach: Option<WaterReach>,
    span: f32,
    eddies: f32,
    roughness: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, schemars::JsonSchema)]
struct Probed {
    water: Option<WaterProbe>,
}

fn water_reach(code: u32) -> Option<WaterReach> {
    [
        WaterReach::Pool,
        WaterReach::Plane,
        WaterReach::Steps,
        WaterReach::Cascade,
        WaterReach::Falls,
    ]
    .into_iter()
    .zip(1..)
    .find(|&(_, reach)| reach == code)
    .map(|(reach, _)| reach)
}

fn water_span(code: u8) -> f32 {
    if code == 0 {
        return 0.0;
    }
    WATER_SPAN_LEAST
        * (WATER_SPAN_MOST / WATER_SPAN_LEAST).powf(f32::from(code - 1) / DEBUG_SPAN_STEPS)
}

fn water_probe((object, detail): (u32, u32), (froudes, state): (u32, u32)) -> WaterProbe {
    let share = |byte: u8| f32::from(byte) / 255.0;
    let halves = |pair: u32| (half(pair as u16), half((pair >> 16) as u16));
    let (depth, speed) = halves(object);
    let (froude, eddies) = halves(froudes);
    let detail = WaterProbeDetail::unpack(detail).unwrap_or_default();
    let state = WaterProbeState::unpack(state).unwrap_or_default();
    WaterProbe {
        depth,
        speed,
        froude,
        turbulence: share(state.turbulence),
        white: share(state.white),
        reach: water_reach(u32::from(state.reach)),
        span: water_span(detail.span),
        eddies,
        roughness: share(detail.roughness),
    }
}

fn probe(kind: Option<DrawKind>, ids: [u32; 4], color: [f32; 4]) -> Value {
    let [_, object, detail, froudes] = ids;
    let state = color[0].to_bits();
    let falls = WaterProbeDetail::unpack(detail).is_some_and(|detail| detail.falls);
    let water = (kind == Some(crate::water::WATER_KIND) && !falls)
        .then(|| water_probe((object, detail), (froudes, state)));
    serde_json::to_value(Probed { water }).unwrap_or(Value::Null)
}
