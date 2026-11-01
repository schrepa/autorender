fn caster_box(center: vec3<f32>, half: vec3<f32>) -> array<vec3<f32>, 2> {
    return array<vec3<f32>, 2>(center, center);
}

fn caster_regions() -> u32 {
    return 0u;
}

fn caster_in_region(region: u32, low: vec3<f32>, high: vec3<f32>) -> bool {
    return false;
}

fn caster_in_sky(center: vec3<f32>, half: vec3<f32>) -> bool {
    return false;
}

fn caster_levels() -> u32 {
    return 0u;
}

fn caster_span(level: u32, low: vec2<f32>, high: vec2<f32>) -> vec4<i32> {
    return vec4<i32>(0, 0, -1, -1);
}

fn mark_caster(level: u32, tile: vec2<i32>, moving: bool) -> u32 {
    return 0u;
}

fn route(world: vec3<f32>, tile: u32) -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn outside_route(position: vec2<f32>, tile: u32) -> bool {
    return true;
}
