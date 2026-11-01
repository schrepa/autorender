import package::engine::view::eye_world;
import package::face::debug::{DEBUG_KIND_FALLS, DEBUG_KIND_WATER, DebugTargets, debug_targets};
import package::face::lighting::sun_debug;
import package::generated::water::{DEBUG_SPAN_STEPS, DRY, WATER_GRAVITY, WATER_SPAN_LEAST,
    WATER_SPAN_MOST};
import package::place::water::WaterAt;
import package::water::falls::{FallOutput, shade_fall};
import package::water::surface::{WaterOutput, shade_water, water, water_drawn, water_eddy_froude,
    water_on_surface};

fn debug_span_code(span: f32) -> f32 {
    if span <= 0.0 {
        return 0.0;
    }
    let octaves = log2(WATER_SPAN_MOST / WATER_SPAN_LEAST);
    let steps = clamp(round(log2(span / WATER_SPAN_LEAST) / octaves * DEBUG_SPAN_STEPS), 0.0,
        DEBUG_SPAN_STEPS);
    return (steps + 1.0) / (DEBUG_SPAN_STEPS + 1.0);
}

fn debug_water(point: vec3<f32>, pixel: vec2<f32>, at: WaterAt) -> DebugTargets {
    let shade = shade_water(point, pixel, at, false);
    if shade.coverage < 0.5 && water.debug.x == 0u { // optics: water covers as its shading sees it
        discard;
    }
    let world = eye_world(point);
    let normal = shade.shaded.normal;
    let depth = at.depth;
    let speed = length(at.words.flow);
    let froude = speed / sqrt(WATER_GRAVITY * max(depth, DRY));
    let eddies = water_eddy_froude(speed, length(at.slope), depth);
    var targets = debug_targets(DEBUG_KIND_WATER, pack2x16float(vec2<f32>(depth, speed)),
        pack4x8unorm(vec4<f32>(debug_span_code(at.words.span), shade.shaded.roughness,
            0.0, 0.0)), normal, sun_debug(world, normal, dpdx(world), dpdy(world), false));
    targets.terrain = vec2<u32>(pack2x16float(vec2<f32>(froude, eddies)),
        pack4x8unorm(vec4<f32>(at.words.state.xy, 0.0, 0.0))
            | (min(at.words.reach, 0xffu) << 24u));
    return targets;
}

@fragment
fn fs_debug_water(input: WaterOutput) -> DebugTargets {
    let at = water_drawn(input.relative);
    return debug_water(water_on_surface(input.relative, at), input.clip_position.xy, at);
}

@fragment
fn fs_debug_fall(input: FallOutput) -> DebugTargets {
    let shade = shade_fall(input);
    if shade.shown < 0.5 { // optics: falls show as their shading sees them
        discard;
    }
    let world = eye_world(input.relative);
    let normal = shade.shaded.normal;
    return debug_targets(DEBUG_KIND_FALLS, 0u, 0u, normal,
        sun_debug(world, normal, dpdx(world), dpdy(world), false));
}
