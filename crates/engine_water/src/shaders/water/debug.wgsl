import package::engine::math::GRAVITY;
import package::engine::surfaces::SURFACE_WATER;
import package::engine::view::eye_world;
import package::face::debug::{DebugTargets, debug_record, debug_targets};
import package::face::heightfield::use_terrain;
import package::face::lighting::sun_debug;
import package::face::sun_blockers::SunAt;
import package::generated::kinds::KIND_WATER;
import package::generated::pieces::water::piece;
import package::generated::water::{DEBUG_SPAN_STEPS, DRY, WATER_SPAN_LEAST, WATER_SPAN_MOST,
    water_probe_detail, water_probe_state};
import package::place::water::WaterAt;
import package::water::falls::{FallOutput, shade_fall};
import package::water::surface::{WaterOutput, shade_water, water_drawn, water_eddy_froude,
    water_on_surface};

fn debug_span_code(span: f32) -> u32 {
    if span <= 0.0 {
        return 0u;
    }
    let octaves = log2(WATER_SPAN_MOST / WATER_SPAN_LEAST);
    let steps = clamp(round(log2(span / WATER_SPAN_LEAST) / octaves * DEBUG_SPAN_STEPS), 0.0,
        DEBUG_SPAN_STEPS);
    return u32(steps) + 1u;
}

fn debug_byte(share: f32) -> u32 {
    return u32(round(saturate(share) * 255.0));
}

fn water_sun(world: vec3<f32>, normal: vec3<f32>) -> u32 {
    return sun_debug(SunAt(world, normal, dpdx(world), dpdy(world), 0.0, SURFACE_WATER, 0u,
        vec4<f32>(0.0)));
}

fn debug_water(point: vec3<f32>, pixel: vec2<f32>, at: WaterAt) -> DebugTargets {
    let shade = shade_water(point, pixel, at, false);
    if shade.coverage < 0.5 && piece().debug.x == 0u { // optics: water covers as its shading sees it
        discard;
    }
    let world = eye_world(point);
    let normal = shade.point.normal;
    let depth = at.depth;
    let speed = length(at.words.flow);
    let froude = speed / sqrt(GRAVITY * max(depth, DRY));
    let eddies = water_eddy_froude(speed, length(at.slope), depth);
    var targets = debug_targets(debug_record(KIND_WATER, 0u, false, false, 0u),
        pack2x16float(vec2<f32>(depth, speed)),
        water_probe_detail(debug_span_code(at.words.span), debug_byte(shade.point.roughness),
            false),
        normal, water_sun(world, normal));
    targets.words = vec2<u32>(pack2x16float(vec2<f32>(froude, eddies)),
        water_probe_state(debug_byte(at.words.state.x), debug_byte(at.words.state.y),
            min(at.words.reach, 0xffu)));
    return targets;
}

@fragment
fn fs_debug_water(input: WaterOutput) -> DebugTargets {
    use_terrain(input.terrain);
    let at = water_drawn(input.relative);
    return debug_water(water_on_surface(input.relative, at), input.clip_position.xy, at);
}

@fragment
fn fs_debug_fall(input: FallOutput) -> DebugTargets {
    use_terrain(input.terrain);
    let shade = shade_fall(input);
    if shade.shown < 0.5 { // optics: falls show as their shading sees them
        discard;
    }
    let world = eye_world(input.relative);
    let normal = shade.point.normal;
    return debug_targets(debug_record(KIND_WATER, 0u, false, false, 0u), 0u,
        water_probe_detail(0u, 0u, true), normal, water_sun(world, normal));
}
