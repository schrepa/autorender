import package::face::debug::{DEBUG_KIND_SKY, DEBUG_KIND_WATER, DEBUG_OTHER, DEBUG_SKY, DebugShade,
    DebugTargets};
import package::generated::pieces::water::piece;
import package::generated::water::{BREAK_FULL, BREAK_ONSET, DEBUG_SPAN_STEPS};

const DEBUG_WATER_DEPTH: u32 = 0u;
const DEBUG_WATER_SPEED: u32 = 1u;
const DEBUG_WATER_FROUDE: u32 = 2u;
const DEBUG_WATER_TURBULENCE: u32 = 3u;
const DEBUG_WATER_WHITE: u32 = 4u;
const DEBUG_WATER_SPAN: u32 = 6u;
const DEBUG_WATER_EDDY: u32 = 7u;
const DEBUG_WATER_ROUGH: u32 = 8u;

const DEBUG_WATER_DEEPEST: f32 = 2.0;
const DEBUG_WATER_FASTEST: f32 = 6.0;
const DEBUG_WATER_FROUDE_TOP: f32 = 3.0;
const DEBUG_WATER_CRITICAL: f32 = 0.04;
const DEBUG_WATER_DEPTHS = array<vec3<f32>, 3>(
    vec3<f32>(0.88, 0.96, 0.88), vec3<f32>(0.2, 0.62, 0.88), vec3<f32>(0.04, 0.09, 0.4),
);
const DEBUG_WATER_SPEEDS = array<vec3<f32>, 3>(
    vec3<f32>(0.08, 0.03, 0.25), vec3<f32>(0.8, 0.15, 0.3), vec3<f32>(1.0, 0.92, 0.35),
);
const DEBUG_WATER_TURBULENCE = array<vec3<f32>, 3>(
    vec3<f32>(0.02, 0.2, 0.22), vec3<f32>(0.1, 0.6, 0.45), vec3<f32>(0.85, 1.0, 0.4),
);
const DEBUG_WATER_CLEAR = vec3<f32>(0.0, 0.12, 0.35);
const DEBUG_WATER_SPANS = array<vec3<f32>, 3>(
    vec3<f32>(0.3, 0.05, 0.45), vec3<f32>(0.1, 0.62, 0.58), vec3<f32>(0.98, 0.95, 0.6),
);
const DEBUG_WATER_EDDIES = array<vec3<f32>, 3>(
    vec3<f32>(0.04, 0.05, 0.12), vec3<f32>(0.95, 0.5, 0.1), vec3<f32>(1.0, 1.0, 0.95),
);
const DEBUG_WATER_ROUGHNESS = array<vec3<f32>, 3>(
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.75, 0.42, 0.22), vec3<f32>(1.0, 1.0, 1.0),
);
const DEBUG_REACHES = array<vec3<f32>, 6>(
    vec3<f32>(0.5, 0.35, 0.65), vec3<f32>(0.15, 0.35, 0.95), vec3<f32>(0.1, 0.8, 0.85),
    vec3<f32>(0.45, 0.85, 0.2), vec3<f32>(0.98, 0.6, 0.1), vec3<f32>(0.9, 0.12, 0.2),
);

fn debug_ramp(stops: array<vec3<f32>, 3>, t: f32) -> vec3<f32> {
    let along = saturate(t) * 2.0;
    if along < 1.0 {
        return mix(stops[0], stops[1], along);
    }
    return mix(stops[1], stops[2], along - 1.0);
}

fn debug_froude_color(froude: f32) -> vec3<f32> {
    if abs(froude - 1.0) < DEBUG_WATER_CRITICAL {
        return vec3<f32>(0.0);
    }
    if froude < 1.0 {
        return mix(vec3<f32>(0.08, 0.2, 0.7), vec3<f32>(0.65, 0.92, 1.0), saturate(froude));
    }
    return mix(vec3<f32>(1.0, 0.9, 0.3), vec3<f32>(0.8, 0.08, 0.08),
        saturate((froude - 1.0) / (DEBUG_WATER_FROUDE_TOP - 1.0)));
}

fn debug_eddies_color(froude: f32) -> vec3<f32> {
    let calm = saturate(froude / BREAK_ONSET);
    let breaking = saturate((froude - BREAK_ONSET) / (BREAK_FULL - BREAK_ONSET));
    return debug_ramp(DEBUG_WATER_EDDIES, 0.5 * (calm + breaking));
}

fn debug_water_color(view: u32, ids: vec4<u32>, numbers: vec2<u32>) -> vec3<f32> {
    let motion = unpack2x16float(ids.y);
    let body = unpack4x8unorm(ids.z);
    let froudes = unpack2x16float(numbers.x);
    let state = unpack4x8unorm(numbers.y);
    switch view {
        case DEBUG_WATER_DEPTH: {
            return debug_ramp(DEBUG_WATER_DEPTHS, sqrt(max(motion.x, 0.0) / DEBUG_WATER_DEEPEST));
        }
        case DEBUG_WATER_SPEED: {
            return debug_ramp(DEBUG_WATER_SPEEDS, motion.y / DEBUG_WATER_FASTEST);
        }
        case DEBUG_WATER_FROUDE: {
            return debug_froude_color(froudes.x);
        }
        case DEBUG_WATER_TURBULENCE: {
            return debug_ramp(DEBUG_WATER_TURBULENCE, state.x);
        }
        case DEBUG_WATER_WHITE: {
            return mix(DEBUG_WATER_CLEAR, vec3<f32>(1.0), state.y);
        }
        case DEBUG_WATER_SPAN: {
            let steps = round(body.x * (DEBUG_SPAN_STEPS + 1.0));
            return debug_ramp(DEBUG_WATER_SPANS, max(steps - 1.0, 0.0) / DEBUG_SPAN_STEPS);
        }
        case DEBUG_WATER_EDDY: {
            return debug_eddies_color(froudes.y);
        }
        case DEBUG_WATER_ROUGH: {
            return debug_ramp(DEBUG_WATER_ROUGHNESS, body.y);
        }
        default: {
            return DEBUG_REACHES[min(numbers.y >> 24u, 5u)];
        }
    }
}

fn shade_kind(shade: DebugShade, targets: DebugTargets) -> DebugShade {
    let shown = piece().debug;
    if shown.x == 0u {
        return shade;
    }
    let kind = targets.ids.x & 0xfu;
    var color = select(DEBUG_OTHER, DEBUG_SKY, kind == DEBUG_KIND_SKY);
    if kind == DEBUG_KIND_WATER {
        color = debug_water_color(shown.y, targets.ids, targets.terrain);
    }
    return DebugShade(color, false);
}
