use engine_render::DebugMode;
use serde::Serialize;
use serde_json::Value;

use crate::gpu::{DEBUG_SPAN_STEPS, WATER_SPAN_LEAST, WATER_SPAN_MOST};

pub(crate) const NAMES: [&str; 9] = [
    "water_depth",
    "water_speed",
    "water_froude",
    "water_turbulence",
    "water_white",
    "water_reach",
    "water_span",
    "water_eddies",
    "water_roughness",
];

pub(crate) fn modes() -> impl Iterator<Item = DebugMode> {
    NAMES.into_iter().map(|name| DebugMode {
        name,
        decode: probe,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum WaterReach {
    Pool,
    Plane,
    Steps,
    Cascade,
    Falls,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
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

fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * fraction * 2.0_f32.powi(-24),
        31 => sign * f32::INFINITY,
        _ => sign * (1.0 + fraction / 1024.0) * 2.0_f32.powi(exponent - 15),
    }
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
    WATER_SPAN_LEAST * (WATER_SPAN_MOST / WATER_SPAN_LEAST).powf(f32::from(code - 1) / DEBUG_SPAN_STEPS)
}

fn probe(ids: [u32; 4], color: [f32; 4]) -> Value {
    let [_, object, detail, froudes] = ids;
    let state = color[0].to_bits();
    let share = |word: u32, byte: u32| f32::from(((word >> (byte * 8)) & 0xff) as u8) / 255.0;
    let halves = |pair: u32| (half(pair as u16), half((pair >> 16) as u16));
    let (depth, speed) = halves(object);
    let (froude, eddies) = halves(froudes);
    serde_json::to_value(WaterProbe {
        depth,
        speed,
        froude,
        turbulence: share(state, 0),
        white: share(state, 1),
        reach: water_reach(state >> 24),
        span: water_span((detail & 0xff) as u8),
        eddies,
        roughness: share(detail, 1),
    })
    .unwrap_or(Value::Null)
}
